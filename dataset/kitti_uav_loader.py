import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class KITTILoader(Dataset):
    def __init__(self, root_dir, split='training', grid_size=8):
        """
        root_dir: KITTI 根路径，如 /path/to/kitti
        split: 'training' or 'testing'
        """
        self.root_dir = os.path.join(root_dir, split)
        self.grid_size = grid_size

        self.image_dir = os.path.join(self.root_dir, 'image_2')
        self.lidar_dir = os.path.join(self.root_dir, 'velodyne')
        self.label_dir = os.path.join(self.root_dir, 'label_2') if split == 'training' else None

        self.frame_ids = sorted([f.split('.')[0] for f in os.listdir(self.image_dir) if f.endswith('.png')])

        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]
        sample_token = frame_id

        # === UAV 特征图 ===
        image_path = os.path.join(self.image_dir, f"{frame_id}.png")
        img = Image.open(image_path).convert('RGB')
        uav_feat = self.img_transform(img).unsqueeze(0)  # [1, 3, 256, 256]

        # === 地面感知输入 ===
        ground_inputs = {
            'camera': torch.stack([self.img_transform(img) for _ in range(6)])
        }

        # === LiDAR 点云 ===
        lidar_path = os.path.join(self.lidar_dir, f"{frame_id}.bin")
        lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        ground_inputs['lidar'] = torch.tensor(lidar, dtype=torch.float32)  # [N, 4]

        # === 3D 标签（仅 training split 有） ===
        gt_boxes = []
        if self.label_dir:
            label_path = os.path.join(self.label_dir, f"{frame_id}.txt")
            with open(label_path, 'r') as f:
                for line in f:
                    elems = line.strip().split()
                    cls = elems[0]
                    if cls == 'DontCare':
                        continue
                    x, y, z = float(elems[11]), float(elems[12]), float(elems[13])
                    h, w, l = float(elems[8]), float(elems[9]), float(elems[10])
                    ry = float(elems[14])
                    box = {
                        'translation': [x, y, z],
                        'size': [w, l, h],
                        'rotation': [0, 0, np.sin(ry / 2), np.cos(ry / 2)],
                        'name': cls
                    }
                    gt_boxes.append(box)

        return uav_feat, ground_inputs, gt_boxes, sample_token

    def apply_mask(self, feature_map, mask):
        """
        将策略输出的掩码应用到 UAV 特征图上
        参数:
            feature_map: [1, C, H, W]
            mask: [1, grid, grid]
        返回:
            masked_feat: [1, C, H, W]
        """
        B, C, H, W = feature_map.shape
        mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest')
        return feature_map * mask
