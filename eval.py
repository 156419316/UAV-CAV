import torch
from tqdm import tqdm

from policy.uav_policy import UAVPolicy, sample_action
from detector.custom_pointpillars_detector import CustomPointPillarsDetector
from dataset.kitti_uav_loader import KITTILoader
from utils.kitti_helper import wrap_predictions_for_kitti
from utils.kitti_eval import KITTIEvaluator

GRID_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 初始化策略和检测器 ===
policy = UAVPolicy(grid_size=GRID_SIZE).to(DEVICE)
policy.load_state_dict(torch.load('logs/uav_rl/best_policy.pth'))
policy.eval()

detector = CustomPointPillarsDetector(model_path='checkpoints/pointpillar.pth')
dataset = KITTILoader(root_dir='/path/to/kitti', split='training')
evaluator = KITTIEvaluator(label_dir='/path/to/kitti/training/label_2')

all_preds_base, all_preds_aug = {}, {}

for idx in tqdm(range(len(dataset))):
    uav_feat, ground_inputs, gt_boxes, sample_token = dataset[idx]
    uav_feat = uav_feat.to(DEVICE)

    # 推理1：无 UAV
    pred_base = detector.forward(ground_inputs['lidar'])
    all_preds_base[sample_token] = pred_base

    # 推理2：使用策略选择特征区域
    with torch.no_grad():
        probs, _ = policy(uav_feat)
        action, _ = sample_action(probs)
        masked_feat = dataset.apply_mask(uav_feat, action.view(1, GRID_SIZE, GRID_SIZE))
        pred_aug = detector.forward(ground_inputs['lidar'])  # 同样保留原始推理

    all_preds_aug[sample_token] = pred_aug

# 评估
ap_base, _ = evaluator.evaluate(wrap_predictions_for_kitti(all_preds_base))
ap_aug, _ = evaluator.evaluate(wrap_predictions_for_kitti(all_preds_aug))

print("==== Evaluation Summary ====")
print(f"Baseline 3D AP: {ap_base:.4f}")
print(f"UAV-enhanced 3D AP: {ap_aug:.4f}")
print(f"AP Gain: {ap_aug - ap_base:.4f}")
