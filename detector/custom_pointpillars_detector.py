import torch
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('/home/lancegan/Datas/Codes/Python/P1'))
from detector.pointpillars.model.pointpillar import PointPillars 


class CustomPointPillarsDetector:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device)
        self.model = PointPillars().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def load_bin(self, bin_path):
        """
        从 .bin 文件中读取 LiDAR 点云
        """
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        return torch.tensor(points, dtype=torch.float32)

    def forward(self, points):
        """
        输入：
            points: [N, 4] LiDAR 点云 tensor
        输出：
            pred_list: list[dict]，每个 dict 包含 KITTI 格式的预测框
        """
        batched_pts = [points.to(self.device)]  # 单帧 batch 输入
        results = self.model(batched_pts, mode='test')

        preds = []
        for box, score, label in zip(results[0]['lidar_bboxes'],
                                     results[0]['scores'],
                                     results[0]['labels']):
            pred = {
                'name': 'Car',  # 可拓展 label-to-name 映射
                'translation': box[:3].tolist(),
                'size': box[3:6].tolist(),
                'rotation_y': box[6],
                'score': float(score)
            }
            preds.append(pred)

        return preds
