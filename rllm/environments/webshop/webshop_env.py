# Copyright 2025 RLLM Team
# Webshop Environment adapted for RLLM Multi-Task Training
#
# Based on the original implementation from verl-agent (GiGPO) team.

import logging
import re
from typing import Any, Dict, Tuple, Callable, Optional

from rllm.environments.base.base_env import BaseEnv
from rllm.rewards.reward_types import RewardOutput

logger = logging.getLogger(__name__)


# System prompt for Webshop task
WEBSHOP_SYSTEM_PROMPT = """You are a shopping assistant helping users find and purchase products on a web shopping platform.

You will be given a shopping task with specific requirements (e.g., product type, price range, features).
You need to navigate the website by searching for products and clicking on items to find the best match.

Available actions:
- search[keywords]: Search for products using keywords
- click[element]: Click on a button or link (e.g., click[Buy Now], click[< Prev], click[item_name])

You must respond in the following format:
<think>Your reasoning about what action to take next</think>
<action>Your action (e.g., search[red shoes] or click[Buy Now])</action>

Important:
- Read the instruction carefully to understand what product features are required
- Use search to find relevant products
- Click on products to view details and select options
- Click "Buy Now" when you find a product that matches all requirements
"""


def parse_webshop_action(response: str) -> Tuple[str, bool]:
    """
    Parse the action from model response.
    Expected format: <think>...</think><action>...</action>
    
    Returns:
        Tuple of (action_string, is_valid)
    """
    response_lower = response.lower()
    
    # Try to extract action from <action>...</action> tags
    start_tag = "<action>"
    end_tag = "</action>"
    start_idx = response_lower.find(start_tag)
    end_idx = response_lower.find(end_tag)
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        action = response[start_idx + len(start_tag):end_idx].strip()
        
        # Check for <think>...</think> tags
        think_start = response_lower.find("<think>")
        think_end = response_lower.find("</think>")
        has_think = think_start != -1 and think_end != -1
        
        # Check for Chinese characters (invalid)
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', response))
        
        is_valid = has_think and not has_chinese
        return action, is_valid
    
    # Fallback: return last 50 chars as action (invalid format)
    return response[-50:] if len(response) > 50 else response, False


class WebshopEnvironment(BaseEnv):
    """
    Webshop Environment adapted for RLLM multi-task training.
    
    This environment wraps the WebAgentTextEnv from the webshop package
    and provides a standard RLLM interface.
    """
    
    def __init__(
        self,
        reward_fn: Optional[Callable] = None,
        max_steps: int = 15,
        observation_mode: str = "text",
        seed: int = 42,
        webshop_path: Optional[str] = None,
        task: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Initialize the Webshop Environment.
        
        Args:
            reward_fn: Reward function to use (optional, env provides its own reward)
            max_steps: Maximum number of steps per episode
            observation_mode: Observation mode ('text' or 'html')
            seed: Random seed
            webshop_path: Path to webshop package (if not in PYTHONPATH)
            task: Initial task dictionary
        """
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.observation_mode = observation_mode
        self.seed = seed
        self.webshop_path = webshop_path
        self.task = task or {}
        self.kwargs = kwargs
        
        # Environment state
        self._env = None
        self._current_step = 0
        self._done = False
        self._instruction_text = ""
        self._cumulative_reward = 0.0
        self._last_info = {}
        
        # Lazy initialization flag
        self._initialized = False
    
    def _lazy_init(self):
        """Lazily initialize the underlying webshop environment."""
        if self._initialized:
            return
            
        try:
            import sys
            import os
            
            # Add webshop path if provided
            if self.webshop_path:
                sys.path.insert(0, self.webshop_path)
            else:
                # Try default path relative to rllm
                default_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                    "agent_system", "environments", "env_package", "webshop", "webshop"
                )
                if os.path.exists(default_path):
                    sys.path.insert(0, default_path)
            
            # Import webshop environment
            import gym
            from web_agent_site.envs import WebAgentTextEnv
            
            # Create the environment
            self._env = gym.make(
                'WebAgentTextEnv-v0',
                observation_mode=self.observation_mode,
                seed=self.seed,
                **self.kwargs
            )
            self._initialized = True
            logger.info("WebshopEnvironment initialized successfully.")
            
        except ImportError as e:
            logger.error(f"Failed to import webshop environment: {e}")
            logger.error("Please ensure the webshop package is installed or provide webshop_path.")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize webshop environment: {e}")
            raise
    
    def reset(self, task: Dict[str, Any] = None) -> Tuple[Any, Dict]:
        """
        Reset the environment for a new episode.
        
        Args:
            task: Task dictionary containing goal_idx or instruction
            
        Returns:
            Tuple of (observation, info)
        """
        self._lazy_init()
        
        if task is not None:
            self.task = task
        
        # Reset state
        self._current_step = 0
        self._done = False
        self._cumulative_reward = 0.0
        
        # Get goal index from task if available
        goal_idx = None
        if self.task:
            goal_idx = self.task.get("goal_idx") or self.task.get("session_idx")
        
        # Reset underlying environment
        if goal_idx is not None:
            obs, info = self._env.reset(session=goal_idx)
        else:
            obs, info = self._env.reset()
        
        info = info or {}
        
        # Get instruction text
        self._instruction_text = self._env.get_instruction_text() if hasattr(self._env, 'get_instruction_text') else ""
        
        # Get available actions
        available_actions = {}
        if hasattr(self._env, 'get_available_actions'):
            available_actions = self._env.get_available_actions()
        
        # Build observation with instruction
        full_observation = self._build_observation(obs, available_actions)
        
        # Build info dict
        info.update({
            "instruction": self._instruction_text,
            "available_actions": available_actions,
            "step": self._current_step,
            "max_steps": self.max_steps,
        })
        self._last_info = info
        
        return full_observation, info
    
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to execute (can be raw string or Action object)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        self._lazy_init()
        
        # Extract action string
        if hasattr(action, 'action'):
            action_str = str(action.action)
        else:
            action_str = str(action)
        
        # Parse action from model response
        parsed_action, is_valid_format = parse_webshop_action(action_str)
        
        self._current_step += 1
        
        # Execute action in environment
        # 注意：原始环境的 step() 返回的 reward 就是 task_score (0.0 ~ 1.0)
        obs, raw_reward, done, info = self._env.step(parsed_action)
        info = dict(info or {})
        
        # Get available actions
        available_actions = {}
        if hasattr(self._env, 'get_available_actions'):
            available_actions = self._env.get_available_actions()
        
        # 保存原始的 task_score (0.0 ~ 1.0)
        # 这是 webshop 环境计算的匹配分数，综合考虑了类型、属性、选项、价格
        task_score = float(raw_reward) if raw_reward is not None else 0.0
        task_score = max(0.0, min(1.0, task_score))  # 确保在 [0, 1] 范围内
        
        # 判断是否完美完成（task_score == 1.0）
        won = done and task_score >= 1.0
        info['won'] = won
        info['task_score'] = task_score
        
        # 使用二元奖励：只有完美完成才给 1.0，否则给 0.0
        # 这与其他任务（math, code）的奖励设计保持一致
        if done:
            reward = 1.0 if won else 0.0
        else:
            reward = 0.0
        
        # Check max steps
        if self._current_step >= self.max_steps:
            done = True
        
        self._done = done
        self._cumulative_reward += reward  # 累积二元奖励
        
        # Build observation
        full_observation = self._build_observation(obs, available_actions)
        
        # Build info dict
        info.update({
            "instruction": self._instruction_text,
            "available_actions": available_actions,
            "step": self._current_step,
            "max_steps": self.max_steps,
            "task_score": task_score,  # 原始匹配分数 (0.0 ~ 1.0)
            "is_valid_format": is_valid_format,
            "parsed_action": parsed_action,
            "cumulative_reward": self._cumulative_reward,
            "done": done,  # 添加 done 状态，供 reward_fn 使用
        })
        self._last_info = info
        
        return full_observation, reward, done, info
    
    def _build_observation(self, obs: str, available_actions: Dict) -> str:
        """Build the full observation string for the agent."""
        parts = []
        
        # Add instruction
        if self._instruction_text:
            parts.append(f"Instruction: {self._instruction_text}")
        
        # Add current page observation
        parts.append(f"\nCurrent Page:\n{obs}")
        
        # Add available actions hint
        if available_actions:
            clickables = available_actions.get("clickables", [])
            if clickables:
                parts.append(f"\nAvailable clickable elements: {', '.join(clickables[:20])}")
                if len(clickables) > 20:
                    parts.append(f"... and {len(clickables) - 20} more")
            if available_actions.get("has_search_bar"):
                parts.append("\nSearch bar is available. Use search[keywords] to search.")
        
        return "\n".join(parts)
    
    def close(self):
        """Close the environment."""
        if self._env is not None:
            self._env.close()
            self._env = None
            self._initialized = False
    
    @staticmethod
    def from_dict(env_args: dict) -> "WebshopEnvironment":
        """Create environment from dictionary configuration."""
        # 提取已知参数
        known_keys = {"reward_fn", "max_steps", "observation_mode", "seed", "webshop_path", "task"}
        # 其余参数作为 kwargs 传递给底层环境（如 num_products, file_path, attr_path 等）
        extra_kwargs = {k: v for k, v in env_args.items() if k not in known_keys}
        
        return WebshopEnvironment(
            reward_fn=env_args.get("reward_fn"),
            max_steps=env_args.get("max_steps", 15),
            observation_mode=env_args.get("observation_mode", "text"),
            seed=env_args.get("seed", 42),
            webshop_path=env_args.get("webshop_path"),
            task=env_args.get("task"),
            **extra_kwargs
        )
    
    @staticmethod
    def is_multithread_safe() -> bool:
        """Webshop environment is NOT multithread safe due to Flask server."""
        return False
