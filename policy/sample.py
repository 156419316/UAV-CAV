import torch
from torch.distributions import Bernoulli

def sample_action(probs):
    """
    输入:probs [B, grid^2]
    输出：
      action: [B, grid^2],0/1传输决策
      log_prob: [B, grid^2],对应 log概率
    """
    dist = Bernoulli(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action, log_prob
