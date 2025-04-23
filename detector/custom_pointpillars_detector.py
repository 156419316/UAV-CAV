import torch
import torch.nn as nn
import numpy as np
import os
from detector.pointpillars.model.pointpillar import PointPillars
#from detector.fusion_module import UAVBEVProjector, BEVFusionModule
from utils.visualize import visualize_fused_feature

##用于融合后的特征图的检测头
class FusionHead(nn.Module):
    def __init__(self, in_channel=384, n_anchors=6, n_classes=3):
        super(FusionHead, self).__init__()
        self.n_anchors = n_anchors
        self.n_classes = n_classes

        # 分类分支
        self.cls_branch = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, n_anchors * n_classes, kernel_size=1)
        )

        # 回归分支
        self.reg_branch = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, n_anchors * 7, kernel_size=1)
        )

        # 方向分支
        self.dir_branch = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, n_anchors * 2, kernel_size=1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: [B, C, H, W]
        returns:
            cls_pred: [B, n_anchors * n_classes, H, W]
            reg_pred: [B, n_anchors * 7, H, W]
            dir_cls_pred: [B, n_anchors * 2, H, W]
        """
        cls_pred = self.cls_branch(x)
        reg_pred = self.reg_branch(x)
        dir_cls_pred = self.dir_branch(x)
        return cls_pred, reg_pred, dir_cls_pred

class CustomPointPillarsDetector(nn.Module):
    def __init__(self, model_path, device='cuda'):
        super().__init__()
        self.model_path = model_path
        self.device = torch.device(device)
        self.model = PointPillars(nclasses=3).to(self.device)
            
        # === 新增 BEVFusion 风格融合模块 ===
        #self.uav_bev_encoder = UAVBEVProjector(img_channels=3, bev_channels=64, bev_size=(248, 216)).to(self.device)
        #self.fusion_module = BEVFusionModule(lidar_channels=384, uav_channels=64, out_channels=384).to(self.device)
        #self.fusion_head = FusionHead(in_channel=384, n_anchors=6, n_classes=self.model.nclasses).to(self.device)
        
    def load_model(self):
    
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

    def forward(self, points):
        
        self.load_model()
        batched_pts = [points.to(self.device)]
        results = self.model(batched_pts, mode='test')

        preds = []
        for box, score, label in zip(results[0]['lidar_bboxes'],
                                     results[0]['scores'],
                                     results[0]['labels']):
            pred = {
                'name': ('Pedestrian'if label == 0 else 'Cyclist' if label == 1 else 'Car'),
                'translation': box[:3].tolist(),
                'size': box[3:6].tolist(),
                'rotation_y': box[6],
                'score': float(score)
            }
            preds.append(pred)
        return preds

    """ def forward_with_fusion(self, points, uav_image, sample_token=None):
        "
        融合 UAV 图像信息后的推理接口
        Args:
            points: Tensor [N, 4]
            uav_image: Tensor [1, 3, 256, 256]
        Returns:
            preds: KITTI 格式预测结果
        "
        batched_pts = [points.to(self.device)]
        uav_image = uav_image.to(self.device)

        with torch.no_grad():
            # 1. 点云前向
            pillars, coors_batch, npoints_per_pillar = self.model.pillar_layer(batched_pts)
            pillar_features = self.model.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
            xs = self.model.backbone(pillar_features)
            lidar_bev_feat = self.model.neck(xs)  # [1, 384, 248, 216]

            # 2. 图像编码 → UAV BEV 表征
            uav_bev_feat = self.uav_bev_encoder(uav_image)  # [1, 64, 248, 216]

            # 3. BEVFusion 融合
            fused_feat = self.fusion_module(lidar_bev_feat, uav_bev_feat)  # [1, 384, 248, 216]

            # 4. 可视化融合特征
            #f sample_token:
            #   visualize_fused_feature(fused_feat, sample_token)

            # 5. 检测头
            cls_pred, box_pred, dir_pred = self.fusion_head(fused_feat)
            feature_map_size = torch.tensor(list(cls_pred.size()[-2:]), device=self.device)
            anchors = self.model.anchors_generator.get_multi_anchors(feature_map_size)
            batched_anchors = [anchors]

            results = self.model.get_predicted_bboxes(cls_pred, box_pred, dir_pred, batched_anchors)

        # 6. 格式化输出
        preds = []
        boxes = results[0]['lidar_bboxes']
        scores = results[0]['scores']
        labels = results[0]['labels']
        for box, score, label in zip(boxes, scores, labels):
            pred = {
                'name': 'Car',
                'translation': box[:3].tolist(),
                'size': box[3:6].tolist(),
                'rotation_y': box[6],
                'score': float(score)
            }
            preds.append(pred)
        return preds """
