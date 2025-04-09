import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from policy.uav_policy import UAVPolicy
from policy.sample import sample_action
from detector.custom_pointpillars_detector import CustomPointPillarsDetector
from dataset.kitti_uav_loader import KITTILoader
from utils.kitti_helper import wrap_predictions_for_kitti
from utils.kitti_eval import KITTIEvaluator
from utils.rl_buffer import PPOBuffer
from utils.logger import RLLogger
from utils.visualize import visualize_probs

import sys
import os

GRID_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 初始化组件 ===
policy = UAVPolicy(grid_size=GRID_SIZE).to(DEVICE)
optimizer = Adam(policy.parameters(), lr=1e-4)

# === 数据集 ===
dataset = KITTILoader(root_dir='/home/lancegan/Datas/Codes/Python/KITTI', split='training')
detector = CustomPointPillarsDetector(model_path='models/epoch_160.pth')
evaluator = KITTIEvaluator(label_dir='/home/lancegan/Datas/Codes/Python/KITTI/training/label_2')
buffer = PPOBuffer()
logger = RLLogger(log_dir='logs/uav_rl/')

for episode in range(100):
    for idx in tqdm(range(len(dataset))):
        uav_feat, ground_inputs, gt_boxes, sample_token = dataset[idx]
        uav_feat = uav_feat.to(DEVICE)

       # -------- 推理策略 with no_grad，节省显存 --------
        with torch.no_grad():
            probs, values = policy(uav_feat)
            
        action, log_prob = sample_action(probs)
        masked_uav_feat = dataset.apply_mask(uav_feat, action.view(1, GRID_SIZE, GRID_SIZE))

        # 点云推理（Base & Aug，暂用相同输入）
        points = ground_inputs['lidar']
        pred_base = detector.forward(points)
        pred_aug = detector.forward(points)  # 可替换为增强后融合版本

        # 构造预测字典并包裹成 KITTI 评估格式
        pred_dict_base = wrap_predictions_for_kitti({sample_token: pred_base})
        pred_dict_aug = wrap_predictions_for_kitti({sample_token: pred_aug})

        # 自动评估，使用 evaluator 的内部 GT
        ap_base, _ = evaluator.evaluate(pred_dict_base)
        ap_aug, _ = evaluator.evaluate(pred_dict_aug)

        # 计算带宽成本并组合 reward
        trans_ratio = action.sum().item() / action.numel()
        reward_value = ap_aug - ap_base - 0.2 * trans_ratio

        # 安全构造 reward tensor，避免 NaN
        reward = torch.tensor([reward_value], dtype=torch.float32, device=DEVICE)

        # 可视化调试输出
        print(f"[Sample {sample_token}] AP_base: {ap_base:.4f}, AP_aug: {ap_aug:.4f}, Trans: {trans_ratio:.3f}, Reward: {reward.item():.4f}")
        # 缓存
        buffer.store(uav_feat, action, reward, log_prob, values)
         # 显式释放变量
        del uav_feat, probs, values, masked_uav_feat, pred_base, pred_aug
        torch.cuda.empty_cache()

     # ---- PPO 更新阶段 ----
    policy.train()
    
    states, actions, rewards, log_probs, values = buffer.get()
    states = torch.cat(states).to(DEVICE)
    actions = torch.stack(actions).to(DEVICE)
    log_probs = torch.stack(log_probs).to(DEVICE)
    returns = torch.stack(rewards).to(DEVICE)
    values = torch.cat(values).to(DEVICE)

    new_probs, new_values = policy(states)
    dist = torch.distributions.Bernoulli(new_probs)
    new_log_probs = dist.log_prob(actions)
    ratio = torch.exp(new_log_probs - log_probs)
    advantage = (returns - values).detach()

    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 0.8, 1.2) * advantage
    actor_loss = -torch.min(surr1, surr2).mean()
    critic_loss = F.mse_loss(new_values, returns)
    loss = actor_loss + 0.5 * critic_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    logger.log_scalar('loss/total', loss.item(), episode)
    logger.log_scalar('reward/mean', returns.mean().item(), episode)
    logger.log_scalar('transmission/ratio', trans_ratio, episode)
    logger.close()
    buffer.clear()

    if episode % 5 == 0:
        visualize_probs(probs[0], grid_size=GRID_SIZE, step=episode)

    print(f"[Episode {episode}] Loss: {loss.item():.4f}, AP Gain: {ap_aug - ap_base:.4f}")
