import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def visualize_probs(probs, grid_size=8, step=None, save_path=None):
    """
    可视化策略输出的传输概率（heatmap）
    probs: tensor [grid_size^2]
    """
    heatmap = probs.detach().cpu().view(grid_size, grid_size).numpy()
    plt.figure(figsize=(5, 5))
    plt.imshow(heatmap, cmap='viridis')
    plt.title(f'Transmission Probabilities (Step {step})')
    plt.colorbar()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def visualize_cbam_attention(attn_map, step=None, save_path=None):
    """
    可视化 CBAM 空间注意力热图
    attn_map: tensor [1, 1, H, W]
    """
    heatmap = attn_map.squeeze().detach().cpu().numpy()
    plt.figure(figsize=(5, 5))
    plt.imshow(heatmap, cmap='hot')
    plt.title(f'CBAM Spatial Attention (Step {step})')
    plt.colorbar()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def visualize_fused_feature(fused_feat, sample_token=None, save_dir='vis/fusion'):
    """
    可视化融合后的 BEV 特征图的通道平均热力图
    Args:
        fused_feat: Tensor [B, C, H, W]
        sample_token: optional identifier
        save_dir: folder to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)

    # 通道平均，得到 [H, W] 热力图
    bev_map = fused_feat.mean(dim=1).squeeze().detach().cpu().numpy()

    plt.figure(figsize=(6, 5))
    plt.imshow(bev_map, cmap='viridis')
    plt.colorbar()
    plt.title(f'Fused BEV Feature (Token: {sample_token})')
    plt.tight_layout()

    # 保存图像
    filename = f"fused_{sample_token}.png" if sample_token else "fused.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()