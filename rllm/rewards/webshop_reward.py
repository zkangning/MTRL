# Copyright 2025 RLLM Team
# Webshop Reward Function for RLLM Multi-Task Training
#
# Webshop 奖励计算逻辑说明：
# 原始 webshop 环境的 get_reward() 返回 0.0 ~ 1.0 的连续分数，基于：
# 1. 类型匹配 (r_type): 产品类型是否匹配 (0.0 ~ 1.0)
# 2. 属性匹配 (r_att): 产品属性匹配数 / 目标属性数
# 3. 选项匹配 (r_option): 选项匹配数 / 目标选项数
# 4. 价格匹配 (r_price): 价格是否在预算内 (0 或 1)
#
# 最终公式: total_reward = (attr_matches + option_matches + price_match) / total_criteria * r_type

import re
import logging
from dataclasses import dataclass
from typing import Any, Dict

from rllm.rewards.reward_types import RewardConfig, RewardOutput

logger = logging.getLogger(__name__)


@dataclass
class WebshopRewardConfig(RewardConfig):
    """Configuration for Webshop reward calculation."""
    
    # 完美完成任务的奖励
    success_reward: float = 1.0
    
    # 失败时的奖励
    failure_reward: float = 0.0
    
    # 是否使用稀疏奖励（只在 episode 结束时给奖励）
    sparse_reward: bool = True


class RewardWebshopFn:
    """
    Reward function for Webshop shopping tasks.
    
    使用二元奖励：只有完美完成任务（task_score == 1.0）才给 1.0，否则给 0.0。
    这与其他任务（math, code）的奖励设计保持一致。
    
    奖励计算基于：
    1. task_score: 环境返回的原始匹配分数 (0.0 ~ 1.0)
       - 综合考虑产品类型、属性、选项、价格的匹配程度
    2. 只在 episode 结束时（done=True）给予奖励
    3. 只有 task_score == 1.0 时才给 1.0 奖励
    """
    
    def __init__(self, config: WebshopRewardConfig = None):
        self.config = config or WebshopRewardConfig()
    
    def __call__(self, task_info: Dict[str, Any], action: str) -> RewardOutput:
        """
        Calculate reward for a Webshop action.
        
        Args:
            task_info: Dictionary containing task information and environment state
                - done: 是否结束
                - task_score: 原始匹配分数 (0.0 ~ 1.0)
                - won: 是否完美完成 (task_score == 1.0)
            action: The agent's action/response
            
        Returns:
            RewardOutput with binary reward (0.0 or 1.0)
        """
        reward = 0.0
        metadata = {}
        is_correct = False
        
        # Extract relevant info from environment
        done = task_info.get("done", False)
        task_score = task_info.get("task_score", 0.0)
        won = task_info.get("won", False)
        
        # Ensure task_score is in valid range
        task_score = max(0.0, min(1.0, float(task_score)))
        
        metadata["task_score"] = task_score
        metadata["won"] = won
        
        # Calculate reward only when episode is done (sparse reward)
        if done:
            # 使用二元奖励：只有完美匹配才给满分
            if won or task_score >= 1.0:
                reward = self.config.success_reward
                is_correct = True
                metadata["completion"] = "perfect"
            else:
                reward = self.config.failure_reward
                is_correct = False
                if task_score > 0:
                    metadata["completion"] = "partial"
                else:
                    metadata["completion"] = "failed"
        else:
            # Episode 未结束，稀疏奖励模式下返回 0
            reward = 0.0
            metadata["completion"] = "in_progress"
        
        metadata["reward"] = reward
        
        return RewardOutput(
            reward=reward,
            metadata=metadata,
            is_correct=is_correct
        )


def webshop_reward_fn(task_info: Dict[str, Any], action: str) -> RewardOutput:
    """
    Wrapper function for Webshop reward calculation.
    
    This function follows the RewardFunction protocol used by RLLM.
    
    Args:
        task_info: Dictionary containing task information
        action: The agent's action/response
        
    Returns:
        RewardOutput with calculated reward
    """
    from rllm.agents.agent import Action
    
    config = WebshopRewardConfig()
    reward_fn = RewardWebshopFn(config)
    
    # Handle Action object
    if isinstance(action, Action):
        action = str(action.action)
    
    return reward_fn(task_info, action)
