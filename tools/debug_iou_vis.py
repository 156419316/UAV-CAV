import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset.kitti_uav_loader import KITTILoader
from detector.bevfusion_detector import BEVFusionDetector
from utils.kitti_eval import KITTIEvaluator
from utils.kitti_helper import wrap_predictions_for_kitti
from detector.pointpillars.utils.process import iou2d

os.makedirs("debug_vis_rgb", exist_ok=True)

# === 配置路径 ===
DEVICE = 'cuda'
DATA_ROOT = "/home/lancegan/Datas/Codes/Python/KITTI"
LABEL_DIR = os.path.join(DATA_ROOT, "training/label_2")
IMAGE_DIR = os.path.join(DATA_ROOT, "training/image_2")

# === 初始化组件 ===
dataset = KITTILoader(root_dir=DATA_ROOT, split='training')
detector = BEVFusionDetector(model_path="models/epoch_160.pth", device=DEVICE)
evaluator = KITTIEvaluator(label_dir=LABEL_DIR)

# === IOU 计算逻辑 ===
def iou_debug(preds, gts):
    if not preds or not gts:
        return []

    pred_bboxes = []
    gt_bboxes = []
    for pred in preds:
        x, y, z = pred['translation']
        l, w, h = pred['size']
        pred_bboxes.append([x - l/2, y - w/2, x + l/2, y + w/2])

    for gt in gts:
        x, y, z = gt['translation']
        l, w, h = gt['size']
        gt_bboxes.append([x - l/2, y - w/2, x + l/2, y + w/2])

    ious = iou2d(torch.tensor(pred_bboxes), torch.tensor(gt_bboxes))
    max_ious = ious.max(dim=1)[0].cpu().numpy()
    print("预测框数量:", len(preds))
    print("GT框数量:", len(gts))
    print("最大IoU分布:", max_ious)
    print("IoU > 0.5 的比例:", np.mean(max_ious > 0.5))
    return max_ious

# === 可视化绘制 ===
def draw_on_image(image, boxes, color, label, thickness=2):
    for box in boxes:
        if 'bbox' in box:
            x1, y1, x2, y2 = map(int, box['bbox'])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# === 主循环：调试前3个样本 ===
for idx in range(3):
    uav_feat, ground_inputs, gt_boxes, token = dataset[idx]
    lidar = ground_inputs['lidar']
    uav_image = ground_inputs.get('image', uav_feat)  # 保底使用 UAV 特征图

    # --- 推理：融合版本 ---
    pred = detector(points=lidar, image=uav_image)
    pred_dict = wrap_predictions_for_kitti({token: pred})
    gt_dict = evaluator.get_gt_dict([token])

    preds = pred_dict[token]
    gts = gt_dict[token]

    print(f"\n[DEBUG] Sample: {token}")
    iou_debug(preds, gts)

    # --- 原始图像可视化 ---
    img_path = os.path.join(IMAGE_DIR, f"{token}.png")
    image = cv2.imread(img_path)

    draw_on_image(image, preds, (0, 0, 255), "PRED")  # 红色
    draw_on_image(image, gts, (0, 255, 0), "GT")      # 绿色

    out_path = os.path.join("debug_vis_rgb", f"{token}.jpg")
    cv2.imwrite(out_path, image)
    print(f"[Saved] {out_path}")
