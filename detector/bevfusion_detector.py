import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .fusion_module  import ImageBEVEncoder
from .fusion_module  import BEVFeatureFusion
from .pointpillars.model.pointpillar import PillarLayer, PillarEncoder, Backbone, Neck
from .pointpillars.utils import anchors2bboxes, limit_period, nms_cuda

#BEV检测头
class BEVDetectionHead(nn.Module):
    def __init__(self, in_channels=384, nclasses=3):
        super().__init__()
        self.cls_head = nn.Conv2d(in_channels, nclasses * 2, 1)
        self.reg_head = nn.Conv2d(in_channels, nclasses * 7, 1)
        self.dir_head = nn.Conv2d(in_channels, nclasses * 2, 1)

    def forward(self, x):
        cls = self.cls_head(x)
        reg = self.reg_head(x)
        direction = self.dir_head(x)
        return cls, reg, direction
    


class BEVFusionDetector(nn.Module):
    def __init__(self, voxel_cfg, nclasses=3, device='cuda'):
        super().__init__()
        self.device = torch.device(device)

        # 点云编码
        self.pillar_layer = PillarLayer(**voxel_cfg)

        self.pillar_encoder = PillarEncoder(
            voxel_size=voxel_cfg['voxel_size'],
            point_cloud_range=voxel_cfg['point_cloud_range'],
            in_channel=9,
            out_channel=64
            )
        self.backbone = Backbone(in_channel=64, out_channels=[64, 128, 256], layer_nums=[3, 5, 5]).to(self.device)
        self.neck = Neck(in_channels=[64, 128, 256], upsample_strides=[1, 2, 4], out_channels=[128, 128, 128]).to(self.device)

        # 图像编码 → BEV
        self.image_bev_encoder = ImageBEVEncoder(out_channels=384).to(self.device)

        # BEV融合
        self.fusion = BEVFeatureFusion(in_channels=384 * 2, out_channels=384).to(self.device)

        # 检测头
        self.head = BEVDetectionHead(in_channels=384, nclasses=nclasses).to(self.device)

        # 参数
        self.score_thr = 0.1
        self.nms_thr = 0.01
        self.nms_pre = 100
        self.max_num = 50
        self.nclasses = nclasses

    def forward(self, points, image=None):
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer([points])
        feat_lidar = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar).to(self.device)
        xs = self.backbone(feat_lidar)
        bev_lidar = self.neck(xs).to(self.device)

        if image is not None:
            bev_img = self.image_bev_encoder(image)
            fused = self.fusion(bev_lidar, bev_img)
        else:
            fused = bev_lidar

        cls_pred, box_pred, dir_pred = self.head(fused)
        return self.post_process(cls_pred[0], box_pred[0], dir_pred[0])

    def post_process(self, cls_pred, box_pred, dir_pred):
        cls_pred = cls_pred.permute(1, 2, 0).reshape(-1, self.nclasses)
        box_pred = box_pred.permute(1, 2, 0).reshape(-1, 7)
        dir_pred = dir_pred.permute(1, 2, 0).reshape(-1, 2)

        anchors = torch.zeros_like(box_pred)
        scores = torch.sigmoid(cls_pred)
        dir_labels = torch.argmax(dir_pred, dim=-1)

        inds = scores.max(1)[0].topk(self.nms_pre)[1]
        scores = scores[inds]
        box_pred = box_pred[inds]
        dir_labels = dir_labels[inds]
        anchors = anchors[inds]

        boxes = anchors2bboxes(anchors, box_pred)
        boxes[:, -1] = limit_period(boxes[:, -1].detach().cpu(), 1, torch.pi).to(boxes)
        boxes[:, -1] += (1 - dir_labels) * torch.pi

        box2d = torch.cat([boxes[:, :2] - boxes[:, 3:5] / 2,
                           boxes[:, :2] + boxes[:, 3:5] / 2,
                           boxes[:, 6:]], dim=-1)

        final_boxes, final_scores, final_labels = [], [], []
        for i in range(self.nclasses):
            mask = scores[:, i] > self.score_thr
            if mask.sum() == 0:
                continue
            cur_boxes = box2d[mask]
            cur_scores = scores[mask, i]
            cur_3d_boxes = boxes[mask]
            keep = nms_cuda(cur_boxes, cur_scores, self.nms_thr)
            final_boxes.append(cur_3d_boxes[keep])
            final_scores.append(cur_scores[keep])
            final_labels.append(torch.zeros_like(cur_scores[keep], dtype=torch.long) + i)

        if len(final_boxes) == 0:
            return []

        final_boxes = torch.cat(final_boxes, dim=0)
        final_scores = torch.cat(final_scores, dim=0)
        final_labels = torch.cat(final_labels, dim=0)

        if final_boxes.size(0) > self.max_num:
            idx = final_scores.topk(self.max_num)[1]
            final_boxes = final_boxes[idx]
            final_scores = final_scores[idx]
            final_labels = final_labels[idx]

        preds = []
        for b, s, l in zip(final_boxes, final_scores, final_labels):
            preds.append({
                'name': 'Car',
                'translation': b[:3].tolist(),
                'size': b[3:6].tolist(),
                'rotation_y': b[6].item(),
                'score': s.item()
            })
        return preds

