import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from policy.uav_policy import UAVPolicy
from policy.sample import sample_action
#from detector.bevfusion_detector import BEVFusionDetector
#from detector.custom_pointpillars_detector import CustomPointPillarsDetector
#from dataset.kitti_uav_loader import KITTILoader
#from utils.kitti_helper import wrap_predictions_for_kitti
#from utils.kitti_eval import KITTIEvaluator
from detector.pointpillars.dataset import Kitti, get_dataloader
from detector.pointpillars.model import PointPillars
from detector.pointpillars.dataset import wrap2kitti,do_eval_all,do_eval_batch
from utils.rl_buffer import PPOBuffer
from utils.logger import RLLogger
from utils.visualize import visualize_probs
from utils import keep_bbox_from_image_range, keep_bbox_from_lidar_range

import sys
import os

GRID_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 1
NUM_WORKERS = 1
NCCLASS = 3

# 设置随机种子
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# === 初始化组件 ===
policy = UAVPolicy(grid_size=GRID_SIZE).to(DEVICE)
optimizer = Adam(policy.parameters(), lr=1e-4)

dataset = Kitti(data_root='/home/lancegan/Datas/Codes/Python/KITTI',split='val')
dataloader = get_dataloader(dataset=dataset, 
                            batch_size=BATCH, 
                            num_workers=NUM_WORKERS,
                            shuffle=False)                       
CLASSES = Kitti.CLASSES 
LABEL2CLASSES = {v:k for k, v in CLASSES.items()}

detector = PointPillars(nclasses=NCCLASS).cuda(DEVICE)
detector.load_state_dict(torch.load('models/epoch_160.pth'))
pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)#点云的范围

buffer = PPOBuffer()
# logger = RLLogger(log_dir='logs/uav_rl/')

# === 奖励函数封装（归一化） ===
def compute_reward(ap_aug, ap_base, trans_ratio, method='tanh'):
    raw = ap_aug - ap_base
    penalty = 0.2 * trans_ratio
    if method == 'linear':
        norm = (raw + 0.2) / 0.4
        norm = max(0.0, min(norm, 1.0))
    else:  # 'tanh'
        norm = torch.tanh(torch.tensor(raw / 0.1)).item()
    return norm - penalty

if __name__ == '__main__':

    detector.eval()
    for episode in range(100):
        
        for i,data_dict in enumerate(tqdm(dataloader)):
            #将数据加载到GPU
            for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()
            
            # 获取数据  
            batched_pts = data_dict['batched_pts']
            batched_image = data_dict['batched_image']
            
            batched_gt_bboxes = data_dict['batched_gt_bboxes']  
            batched_gt_result = data_dict['batched_gt_result']         
            batched_labels = data_dict['batched_labels']
            batched_difficulty = data_dict['batched_difficulty']
            
           
            batch_pts_results = detector(batched_pts=batched_pts, 
                                        mode='val',
                                        batched_gt_bboxes=batched_gt_bboxes, 
                                        batched_gt_labels=batched_labels)
            
            #将检测结果转化为kitti格式
            wrapped_results = wrap2kitti(batch_pts_results,
                                        data_dict=data_dict,
                                        pcd_limit_range=pcd_limit_range,
                                        LABEL2CLASSES=LABEL2CLASSES)
            
            eval_results = do_eval_batch(wrapped_results,batched_gt_result,CLASSES)
            
            for j in(range(len(batched_pts))):
            # -------- 策略网络输出 --------
                uav_feat = batched_image[j]
                with torch.no_grad():
                    probs, values = policy(uav_feat)
                    
                action, log_prob = sample_action(probs)
                mask = action.view(1, GRID_SIZE, GRID_SIZE)
                masked_uav_feat = policy.apply_mask(uav_feat, mask)  # [1, 3, 256, 256]
                
                ap_base = eval_results['AP_3D']
                
                #目前还没有把无人机特征图融合到点云特征图中
                #只能在aug_base基础上上下波动来模拟融合特征图后的检测性能
                
                ap_aug = ap_base+np.random.uniform(-1,1)*np.random.uniform(0,0.5)
                trans_ratio = action.sum().item() / action.numel()
                reward_val = compute_reward(ap_aug, ap_base, trans_ratio)
                reward = torch.tensor([reward_val], dtype=torch.float32, device=DEVICE)
                print(f" AP_base: {ap_base:.4f}, AP_aug: {ap_aug:.4f}, Trans: {trans_ratio:.3f}, Reward: {reward.item():.4f}")
                buffer.store(uav_feat, action, reward, log_prob, values)
                
                # 清理显存
                del uav_feat, probs, values, action, log_prob, mask, masked_uav_feat, reward
                torch.cuda.empty_cache()
                
            del batched_pts, batched_image, batched_gt_bboxes, batched_gt_result, batched_labels, batched_difficulty
            torch.cuda.empty_cache()
            
        # # -------- PPO 策略更新 --------
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

        # -------- 日志记录 --------
        # logger.log_scalar('loss/total', loss.item(), episode)
        # logger.log_scalar('reward/mean', returns.mean().item(), episode)
        # logger.log_scalar('transmission/ratio', trans_ratio, episode)
        # logger.close()
        
        #清除缓存
        buffer.clear()


            # # -------- PPO 策略更新 --------
            # policy.train()

            # states, actions, rewards, log_probs, values = buffer.get()
            # states = torch.cat(states).to(DEVICE)
            # actions = torch.stack(actions).to(DEVICE)
            # log_probs = torch.stack(log_probs).to(DEVICE)
            # returns = torch.stack(rewards).to(DEVICE)
            # values = torch.cat(values).to(DEVICE)

            # new_probs, new_values = policy(states)
            # dist = torch.distributions.Bernoulli(new_probs)
            # new_log_probs = dist.log_prob(actions)
            # ratio = torch.exp(new_log_probs - log_probs)
            # advantage = (returns - values).detach()

            # surr1 = ratio * advantage
            # surr2 = torch.clamp(ratio, 0.8, 1.2) * advantage
            # actor_loss = -torch.min(surr1, surr2).mean()
            # critic_loss = F.mse_loss(new_values, returns)
            # loss = actor_loss + 0.5 * critic_loss

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # # -------- 日志记录 --------
            # logger.log_scalar('loss/total', loss.item(), episode)
            # logger.log_scalar('reward/mean', returns.mean().item(), episode)
            # logger.log_scalar('transmission/ratio', trans_ratio, episode)
            # logger.close()
            # buffer.clear()

            # # 可视化策略分布
            # if episode % 5 == 0:
            #     visualize_probs(probs[0], grid_size=GRID_SIZE, step=episode)

            # print(f"[Episode {episode}] Loss: {loss.item():.4f}, AP Gain: {ap_aug - ap_base:.4f}")
