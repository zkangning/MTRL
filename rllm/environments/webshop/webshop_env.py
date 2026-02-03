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


# System prompt for Webshop task (kept for backward compatibility)
# The main system prompt is now in rllm/agents/webshop_agent.py
WEBSHOP_SYSTEM_PROMPT = """You are an expert autonomous agent operating in the WebShop e-commerce environment.

Your goal is to find and purchase a product that matches ALL the requirements specified in the instruction.

## Available Actions (MUST use exact format)
1. search[keywords]: Search for products using keywords
   - Example: search[red running shoes]
   - Use simple, relevant keywords (2-4 words work best)
   
2. click[element]: Click on any button, link, or option
   - Navigation: click[next >], click[< prev], click[back to search]
   - Products: click[B07HRFSNL4] (use the product ASIN code)
   - Options: click[large], click[red], click[size: medium]
   - Purchase: click[buy now]

## Response Format
You MUST follow this exact format:
1. First, reason about the current situation inside <think> </think> tags
2. Then, provide your action inside <action> </action> tags

## Example
<think>
The instruction asks for red shoes under $50. I see a search bar is available.
</think>
<action>search[red shoes]</action>

## Important Rules
1. ALWAYS use click[element] format for ALL clicks - never just write the element name
2. Use simple search keywords (2-4 words work best)
3. Click on product ASINs to view details, select options, then click[buy now]
4. If no matching products after browsing 3 pages, click[back to search] and try different keywords
"""


# Common navigation button patterns that should be converted to click[...] format
NAVIGATION_BUTTONS = {
    "next >": "next >",
    "next>": "next >",
    "next": "next >",
    "< prev": "< prev",
    "<prev": "< prev",
    "prev": "< prev",
    "previous": "< prev",
    "back to search": "back to search",
    "back": "back to search",
    "buy now": "buy now",
    "buy": "buy now",
    "purchase": "buy now",
    "add to cart": "buy now",
    "description": "description",
    "features": "features",
    "reviews": "reviews",
}


def normalize_action_to_click(action: str) -> str:
    """
    Normalize common navigation commands to click[...] format.
    
    For example:
    - "next >" -> "click[next >]"
    - "< prev" -> "click[< prev]"
    - "back to search" -> "click[back to search]"
    """
    action_lower = action.strip().lower()
    
    # Check if it's already in click[...] or search[...] format
    if action_lower.startswith("click[") or action_lower.startswith("search["):
        return action
    
    # Check if it matches a known navigation button
    for pattern, normalized in NAVIGATION_BUTTONS.items():
        if action_lower == pattern or action_lower == pattern.replace(" ", ""):
            return f"click[{normalized}]"
    
    # Check if it looks like a product ASIN (e.g., B07HRFSNL4)
    if re.match(r'^[Bb]\d{2}[A-Za-z0-9]{7}$', action.strip()):
        return f"click[{action.strip().lower()}]"
    
    # If it's a short string without brackets, assume it's a click target
    if len(action) < 50 and "[" not in action:
        return f"click[{action.strip()}]"
    
    return action


def parse_webshop_action(response: str) -> Tuple[str, bool]:
    """
    Parse the action from model response.
    Expected format: <think>...</think><action>...</action>
    Also supports direct search[...] or click[...] format.
    
    This function also normalizes common navigation commands:
    - "next >" -> "click[next >]"
    - "< prev" -> "click[< prev]"
    - "back to search" -> "click[back to search]"
    
    Returns:
        Tuple of (action_string, is_valid)
    """
    if not response:
        return "", False
    
    response_lower = response.lower()
    
    # Check for <think>...</think> tags
    think_start = response_lower.find("<think>")
    think_end = response_lower.find("</think>")
    has_think = think_start != -1 and think_end != -1
    
    # Check for Chinese characters (invalid for webshop which uses English)
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', response))
    
    # Try to extract action from <action>...</action> tags
    start_tag = "<action>"
    end_tag = "</action>"
    start_idx = response_lower.find(start_tag)
    end_idx = response_lower.find(end_tag)
    
    action = None
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        # Extract action content using original case
        action = response[start_idx + len(start_tag):end_idx].strip()
    
    # If no action tag found, try to find search[...] or click[...] patterns
    if not action:
        search_match = re.search(r'search\[([^\]]+)\]', response, re.IGNORECASE)
        if search_match:
            action = f"search[{search_match.group(1)}]"
        else:
            click_match = re.search(r'click\[([^\]]+)\]', response, re.IGNORECASE)
            if click_match:
                action = f"click[{click_match.group(1)}]"
    
    if action:
        # Normalize the action format
        action = action.strip()
        
        # First, try to normalize navigation commands to click[...] format
        action = normalize_action_to_click(action)
        
        # Ensure proper format: search[...] or click[...]
        search_match = re.search(r'search\[([^\]]+)\]', action, re.IGNORECASE)
        if search_match:
            # Return normalized search action
            is_valid = has_think and not has_chinese
            return f"search[{search_match.group(1)}]", is_valid
        
        click_match = re.search(r'click\[([^\]]+)\]', action, re.IGNORECASE)
        if click_match:
            # Return normalized click action
            is_valid = has_think and not has_chinese
            return f"click[{click_match.group(1)}]", is_valid
        
        # Action found but not in expected format - try to normalize it
        normalized = normalize_action_to_click(action)
        is_valid = has_think and not has_chinese
        return normalized, is_valid
    
    # Last resort: return the last part of response (invalid format)
    return response[-100:] if len(response) > 100 else response, False


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
        
        # Add available actions hint - this is CRITICAL for the agent
        if available_actions:
            clickables = available_actions.get("clickables", [])
            if clickables:
                # Categorize clickables for better understanding
                navigation_buttons = []
                product_asins = []
                options = []
                other_buttons = []
                
                for item in clickables:
                    item_lower = item.lower()
                    if item_lower in ['next >', '< prev', 'back to search', 'buy now']:
                        navigation_buttons.append(item)
                    elif re.match(r'^b\d{2}[a-z0-9]{7}$', item_lower):
                        product_asins.append(item)
                    elif any(x in item_lower for x in ['size', 'color', 'small', 'medium', 'large', 'xl', 'xxl']):
                        options.append(item)
                    else:
                        other_buttons.append(item)
                
                parts.append("\n--- Available Actions ---")
                
                # Show navigation buttons first (most important)
                if navigation_buttons:
                    nav_formatted = [f"click[{b}]" for b in navigation_buttons]
                    parts.append(f"Navigation: {', '.join(nav_formatted)}")
                
                # Show product ASINs (important for product selection)
                if product_asins:
                    # Show up to 10 products
                    displayed_products = product_asins[:10]
                    asin_formatted = [f"click[{a}]" for a in displayed_products]
                    parts.append(f"Products ({len(product_asins)} total): {', '.join(asin_formatted)}")
                    if len(product_asins) > 10:
                        parts.append(f"  ... and {len(product_asins) - 10} more products")
                
                # Show options (size, color, etc.)
                if options:
                    opt_formatted = [f"click[{o}]" for o in options[:20]]
                    parts.append(f"Options: {', '.join(opt_formatted)}")
                
                # Show other buttons
                if other_buttons:
                    other_formatted = [f"click[{b}]" for b in other_buttons[:10]]
                    parts.append(f"Other: {', '.join(other_formatted)}")
            
            if available_actions.get("has_search_bar"):
                parts.append("\nSearch bar available: Use search[keywords] to search.")
        
        # Add action format reminder
        parts.append("\nRemember: Use search[keywords] or click[element] format for your action.")
        
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
