import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageBEVEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=384):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU()
        )
        self.to_bev = nn.AdaptiveAvgPool2d((248, 216))  # 默认输出大小

    def forward(self, img):
        feat = self.encoder(img)
        bev_feat = self.to_bev(feat)
        return bev_feat

class BEVFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU()
        )

    def forward(self, bev_lidar, bev_img):
        x = torch.cat([bev_lidar, bev_img], dim=1)  # [B, 768, H, W]
        return self.fuse_conv(x)
    
 