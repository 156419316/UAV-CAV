import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_size=256):
        super().__init__()
        self.resize = nn.AdaptiveAvgPool2d((out_size, out_size))  # 统一到 256x256
        self.expand_dim = nn.Identity()  # 用于 [C,H,W] → [1,C,H,W]

    def forward(self, x):
        if x.dim() == 3:  # [C,H,W] → [1,C,H,W]
            x = x.unsqueeze(0)
        # 转换数据类型为 float，并归一化到 [0, 1]
        if x.dtype == torch.uint8:  # 检查是否是 Byte 类型
            x = x.float() / 255.0
        x = self.resize(x)
        return x

# ----------------------------
# 1. CBAM模块（通道 + 空间注意力）
# ----------------------------
class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.channel_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        ca = self.channel_fc(x)
        x = x * ca

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        return x * sa

# ----------------------------
# 2. Pyramid Transformer（多尺度融合）
# ----------------------------
class PyramidTransformer(nn.Module):
    def __init__(self, dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.size()
        outputs = []
        for scale in [H, H // 2, H // 4]:
            feat = F.adaptive_avg_pool2d(x, (scale, scale))  # [B, C, h, w]
            feat = feat.flatten(2).permute(2, 0, 1)  # → [HW, B, C]
            for layer in self.layers:
                feat = layer(feat)
            feat = feat.permute(1, 0, 2)  # → [B, HW, C]
            pooled = self.norm(feat).mean(dim=1)
            outputs.append(pooled)
        return torch.cat(outputs, dim=-1)  # [B, C*3]


# ----------------------------
# 3. UAV策略网络（Actor-Critic）
# ----------------------------
class UAVPolicy(nn.Module):
    def __init__(self, grid_size=8, embed_dim=256, in_channels=3):
        super().__init__()
        
        self.feature_extractor = FeatureExtractor(in_channels, out_size=256)
        
        self.grid_size = grid_size

        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            CBAM(embed_dim)
        )

        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))  # [B, C, G, G]
        self.actor = nn.Conv2d(embed_dim, 1, kernel_size=1)       # [B, 1, G, G]
        self.critic_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # [B, C, 1, 1]
            nn.Flatten(),                # [B, C]
            nn.Linear(embed_dim, 1)     # [B, 1]
        )
        
    def apply_mask(self, feature_map, mask):
        """
        将策略输出的掩码应用到 UAV 特征图上
        参数:
            feature_map: [C, H, W]
            mask: [grid, grid]
        返回:
            masked_feat: [C, H, W]
        """
        # 如果输入是 [C, H, W]，需要在第 0 维增加批量维度
        if feature_map.dim() == 3:
            feature_map = feature_map.unsqueeze(0)  # [C, H, W] → [1, C, H, W]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # [grid, grid] → [1, grid, grid]

        B, C, H, W = feature_map.shape
        mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest')  # [1, 1, H, W]
        masked_feat = feature_map * mask  # [1, C, H, W]

        return masked_feat.squeeze(0)  # 去掉批量维度，返回 [C, H, W]

    def forward(self, x):  # x: [B, C_in, H, W]
        x = self.feature_extractor(x)
        x = self.cnn_encoder(x)          # [B, C, H, W]
        pooled = self.pool(x)            # [B, C, G, G]
        probs = torch.sigmoid(self.actor(pooled)).flatten(1)  # [B, G*G]
        value = self.critic_fc(x)        # [B, 1]
        return probs, value