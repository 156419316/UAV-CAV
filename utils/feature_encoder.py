import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small

class FeatureEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        """
        in_ch: 输入通道数
        out_ch: 输出通道数
        """
        super().__init__()
        # Backbone提取局部特征
        self.backbone = nn.Sequential
        (
            mobilenet_v3_small(pretrained=True).features,
        )
        backbone_out_ch = 576  # MobileNetV3-Small的输出通道数

        # Transformer捕捉全局上下文
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=backbone_out_ch, nhead=8),
            num_layers=2
        )

        # 空间注意力
        self.attention = nn.Sequential(
            nn.Conv2d(backbone_out_ch, out_ch, 3, padding=1),
            nn.Sigmoid()  # 输出0-1的注意力权重
        )

    def forward(self, x):
        x = self.backbone(x)           # [B, 576, 20, 20] (假设输入为 640x640)
        b, c, h, w = x.shape

        # Transformer处理全局关系
        x_flat = x.view(b, c, -1).permute(2, 0, 1)  # [H*W, B, C]
        x_trans = self.transformer(x_flat)          # [H*W, B, C]
        x_trans = x_trans.permute(1, 2, 0).view(b, c, h, w)

        # 空间注意力加权
        attn = self.attention(x_trans)  # [B, out_ch, H, W]
        return x_trans * attn           # 加权后的特征