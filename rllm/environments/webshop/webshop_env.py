# Copyright 2025 RLLM Team
# Webshop Environment adapted for RLLM Multi-Task Training
# Tool-call version
#
# Based on the original implementation from verl-agent (GiGPO) team.

import json
import logging
import random
import re
import string
import threading
import uuid
from typing import Any, Dict, Tuple, Callable, Optional, List

from rllm.environments.base.base_env import BaseEnv
from rllm.rewards.reward_types import RewardOutput

logger = logging.getLogger(__name__)


# Tool definitions for Webshop task
WEBSHOP_TOOLS = [
    {
        "name": "search",
        "description": "Search for products with the given query. Use this only if a [Search] button appears in the observation. Note: If you wish to search and there's no [Search] button, click the [Back to Search] button instead.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for products (use 2-4 simple keywords)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "click",
        "description": "Click on a button or product. Use this only if a [button] is present in the observation. When you have identified the most suitable product, click the [Buy Now] button on its product page to finish the shopping task.",
        "parameters": {
            "type": "object",
            "properties": {
                "button": {
                    "type": "string",
                    "description": "Name of the button or product ASIN, don't add '[]' around the button name"
                }
            },
            "required": ["button"]
        }
    }
]


def format_tools_for_prompt(tools: List[Dict]) -> str:
    """Format tool definitions for inclusion in system prompt."""
    tool_strs = []
    for tool in tools:
        params = tool.get("parameters", {}).get("properties", {})
        param_strs = []
        for param_name, param_info in params.items():
            param_strs.append(f'"{param_name}": "{param_info.get("description", "")}"')
        params_json = "{" + ", ".join(param_strs) + "}"
        tool_strs.append(f"- {tool['name']}: {tool['description']}\n  Parameters: {params_json}")
    return "\n".join(tool_strs)


# System prompt for Webshop task (Tool-call version)
WEBSHOP_SYSTEM_PROMPT = f"""You are a shopping assistant in the WebShop environment.

Your goal is to find and purchase a product that matches ALL requirements in the instruction.

## Available Tools
{format_tools_for_prompt(WEBSHOP_TOOLS)}

## Response Format
You MUST respond with:
1. Your reasoning inside <think> </think> tags
2. A tool call inside <tool_call> </tool_call> tags in JSON format

Example response:
<think>
I need to search for the product first.
</think>
<tool_call>
{{"name": "search", "arguments": {{"query": "red running shoes"}}}}
</tool_call>

## Important Rules
1. Use simple search keywords (2-4 words)
2. Click on product ASINs (like B07HRFSNL4) to view details
3. Select required options (size, color) before clicking Buy Now
4. If no matching products after browsing 3 pages, click Back to Search and try different keywords
"""


# Common navigation button patterns for normalization
NAVIGATION_BUTTONS = {
    "next >": "Next >",
    "next>": "Next >",
    "next": "Next >",
    "< prev": "< Prev",
    "<prev": "< Prev",
    "prev": "< Prev",
    "previous": "< Prev",
    "back to search": "Back to Search",
    "back": "Back to Search",
    "buy now": "Buy Now",
    "buy": "Buy Now",
    "purchase": "Buy Now",
    "add to cart": "Buy Now",
    "description": "Description",
    "features": "Features",
    "reviews": "Reviews",
    "search": "Search",
}


def parse_tool_call(response: str) -> Tuple[str, bool]:
    """
    Parse the tool call from model response.
    
    Expected format: 
    <think>...</think>
    <tool_call>{"name": "search", "arguments": {"query": "..."}}</tool_call>
    
    Also supports legacy format: search[...] or click[...]
    
    Returns:
        Tuple of (action_string in format "search[query]" or "click[button]", is_valid)
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
    
    action = None
    
    # Try to extract tool call from <tool_call>...</tool_call> tags
    tool_call_match = re.search(
        r'<tool_call>\s*(.*?)\s*</tool_call>',
        response,
        re.DOTALL | re.IGNORECASE
    )
    
    if tool_call_match:
        tool_call_str = tool_call_match.group(1).strip()
        try:
            # Parse JSON
            tool_call = json.loads(tool_call_str)
            tool_name = tool_call.get("name", "").lower()
            arguments = tool_call.get("arguments", {})
            
            if tool_name == "search":
                query = arguments.get("query", "")
                action = f"search[{query}]"
            elif tool_name == "click":
                button = arguments.get("button", "")
                # Normalize button name
                button_lower = button.lower().strip()
                if button_lower in NAVIGATION_BUTTONS:
                    button = NAVIGATION_BUTTONS[button_lower]
                action = f"click[{button}]"
        except json.JSONDecodeError:
            logger.debug(f"Failed to parse tool call JSON: {tool_call_str}")
    
    # Fallback: try to find JSON pattern directly (without tags)
    if not action:
        json_match = re.search(
            r'\{\s*"name"\s*:\s*"(search|click)".*?\}',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if json_match:
            try:
                # Find the complete JSON object
                json_str = json_match.group(0)
                # Handle nested braces
                brace_count = 0
                end_idx = 0
                for i, c in enumerate(json_str):
                    if c == '{':
                        brace_count += 1
                    elif c == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                if end_idx > 0:
                    json_str = json_str[:end_idx]
                
                tool_call = json.loads(json_str)
                tool_name = tool_call.get("name", "").lower()
                arguments = tool_call.get("arguments", {})
                
                if tool_name == "search":
                    query = arguments.get("query", "")
                    action = f"search[{query}]"
                elif tool_name == "click":
                    button = arguments.get("button", "")
                    button_lower = button.lower().strip()
                    if button_lower in NAVIGATION_BUTTONS:
                        button = NAVIGATION_BUTTONS[button_lower]
                    action = f"click[{button}]"
            except json.JSONDecodeError:
                pass
    
    # Fallback: try legacy format search[...] or click[...]
    if not action:
        search_match = re.search(r'search\[([^\]]+)\]', response, re.IGNORECASE)
        if search_match:
            action = f"search[{search_match.group(1)}]"
        else:
            click_match = re.search(r'click\[([^\]]+)\]', response, re.IGNORECASE)
            if click_match:
                button = click_match.group(1)
                button_lower = button.lower().strip()
                if button_lower in NAVIGATION_BUTTONS:
                    button = NAVIGATION_BUTTONS[button_lower]
                action = f"click[{button}]"
    
    # Fallback: try to extract action from <action>...</action> tags (legacy)
    if not action:
        action_match = re.search(
            r'<action>\s*(.*?)\s*</action>',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if action_match:
            action_content = action_match.group(1).strip()
            # Try to parse as search[...] or click[...]
            search_match = re.search(r'search\[([^\]]+)\]', action_content, re.IGNORECASE)
            if search_match:
                action = f"search[{search_match.group(1)}]"
            else:
                click_match = re.search(r'click\[([^\]]+)\]', action_content, re.IGNORECASE)
                if click_match:
                    button = click_match.group(1)
                    button_lower = button.lower().strip()
                    if button_lower in NAVIGATION_BUTTONS:
                        button = NAVIGATION_BUTTONS[button_lower]
                    action = f"click[{button}]"
                else:
                    # Assume it's a click target
                    button_lower = action_content.lower().strip()
                    if button_lower in NAVIGATION_BUTTONS:
                        action = f"click[{NAVIGATION_BUTTONS[button_lower]}]"
                    else:
                        action = f"click[{action_content}]"
    
    if action:
        is_valid = has_think and not has_chinese
        return action, is_valid
    
    # Last resort: return the last part of response (invalid format)
    return response[-100:] if len(response) > 100 else response, False


class WebshopEnvironment(BaseEnv):
    """
    Webshop Environment adapted for RLLM multi-task training.
    Tool-call version.
    
    This environment wraps the WebAgentTextEnv from the webshop package
    and provides a standard RLLM interface with tool-call format.
    
    Thread Safety Note:
    -------------------
    Each WebshopEnvironment instance maintains its own WebAgentTextEnv,
    which includes an independent SimServer and SimBrowser. This means:
    
    1. When shared environment mode is DISABLED (default, recommended):
       - Each parallel task gets its own WebshopEnvironment instance
       - Each instance has independent SimServer/SimBrowser state
       - n_parallel_agents > 1 is SAFE
    
    2. When shared environment mode was ENABLED (deprecated):
       - Multiple tasks would share one WebAgentTextEnv
       - Session pollution could occur due to non-thread-safe SimBrowser
       - This mode is now disabled in CompositeEnvironment
    
    Current Status: Safe for parallel execution with independent instances.
    """
    
    # Thread-local storage for session management
    _local = threading.local()
    
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
        
        # Unique instance ID for session isolation
        self._instance_id = str(uuid.uuid4())[:8]
        
        # Lazy initialization flag
        self._initialized = False
        
        # Lock for thread-safe reset/step operations
        self._lock = threading.Lock()
    
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
            # 使用 human_goals=False 加载合成 goals（synthetic goals）
            # 合成 goals 基于产品属性和选项组合自动生成
            # 对于 1000 产品的小数据集，会生成数千个 goals
            env_kwargs = {
                'human_goals': False,  # 使用合成的 goals（synthetic goals）
                **self.kwargs  # 允许外部覆盖
            }
            self._env = gym.make(
                'WebAgentTextEnv-v0',
                observation_mode=self.observation_mode,
                seed=self.seed,
                **env_kwargs
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
    
    def _generate_unique_session_id(self, goal_idx: int = None) -> str:
        """
        Generate a unique session ID to ensure session isolation in shared environments.
        
        The session ID format: {goal_idx}_{instance_id}_{random_suffix}
        This ensures that even if multiple tasks have the same goal_idx, they will
        have different session IDs and won't interfere with each other.
        
        Args:
            goal_idx: The goal index for this task
            
        Returns:
            A unique session ID string
        """
        goal_part = str(goal_idx) if goal_idx is not None else "rand"
        random_suffix = ''.join(random.choices(string.ascii_lowercase, k=6))
        return f"{goal_part}_{self._instance_id}_{random_suffix}"
    
    def reset(self, task: Dict[str, Any] = None) -> Tuple[Any, Dict]:
        """
        Reset the environment for a new episode.
        
        This method is designed to be efficient for shared environments:
        - The underlying webshop environment (with loaded product data) is reused
        - Only the episode state is reset
        - Each reset generates a unique session ID for isolation
        
        Args:
            task: Task dictionary containing goal_idx or instruction
            
        Returns:
            Tuple of (observation, info)
        """
        # Use lock to ensure thread-safe reset
        with self._lock:
            # Lazy init only happens once (first call)
            self._lazy_init()
            
            if task is not None:
                self.task = task
            
            # Reset episode state (soft reset)
            self._current_step = 0
            self._done = False
            self._cumulative_reward = 0.0
            self._instruction_text = ""
            self._last_info = {}
            
            # Get goal index from task if available
            goal_idx = None
            if self.task:
                goal_idx = self.task.get("goal_idx") or self.task.get("session_idx")
            
            # Get the number of available goals to prevent index out of range
            # The goals are stored in self._env.server.goals
            num_goals = 0
            try:
                if hasattr(self._env, 'server') and hasattr(self._env.server, 'goals'):
                    num_goals = len(self._env.server.goals)
                elif hasattr(self._env, 'unwrapped') and hasattr(self._env.unwrapped, 'server'):
                    num_goals = len(self._env.unwrapped.server.goals)
            except Exception as e:
                logger.warning(f"Failed to get number of goals: {e}")
            
            # Apply modulo to goal_idx to prevent index out of range
            # This ensures that even if the data generator produces goal_idx values
            # larger than available goals, we still get a valid index
            effective_goal_idx = goal_idx
            if goal_idx is not None and num_goals > 0:
                effective_goal_idx = goal_idx % num_goals
                if goal_idx != effective_goal_idx:
                    logger.debug(
                        f"Adjusted goal_idx from {goal_idx} to {effective_goal_idx} "
                        f"(num_goals={num_goals})"
                    )
            
            # Generate unique session ID for isolation in shared environments
            # The session ID includes goal_idx, instance_id, and a random suffix
            unique_session_id = self._generate_unique_session_id(effective_goal_idx)
            
            # Configure the underlying environment to use our unique session
            # Note: We set session_prefix to ensure the session_id includes goal info
            # but remains unique across concurrent tasks
            if hasattr(self._env, 'session_prefix'):
                self._env.session_prefix = f"g{effective_goal_idx if effective_goal_idx is not None else 'r'}_"
            
            # Reset underlying environment with the effective goal index
            # The underlying env will create a session based on goal_idx
            if effective_goal_idx is not None:
                obs, info = self._env.reset(session=effective_goal_idx)
            else:
                obs, info = self._env.reset()
            
            # Store the actual session ID for debugging
            self._current_session = self._env.session if hasattr(self._env, 'session') else None
            
            info = info or {}
            
            # Get instruction text
            self._instruction_text = self._env.get_instruction_text() if hasattr(self._env, 'get_instruction_text') else ""
            
            # Get available actions
            available_actions = {}
            if hasattr(self._env, 'get_available_actions'):
                available_actions = self._env.get_available_actions()
            
            # Build observation with instruction (tool-call format)
            full_observation = self._build_observation(obs, available_actions)
            
            # Build info dict
            info.update({
                "instruction": self._instruction_text,
                "available_actions": available_actions,
                "step": self._current_step,
                "max_steps": self.max_steps,
                "session_id": self._current_session,
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
        # Use lock to ensure thread-safe step
        with self._lock:
            self._lazy_init()
            
            # Extract action string
            if hasattr(action, 'action'):
                action_str = str(action.action)
            else:
                action_str = str(action)
            
            # Parse tool call from model response
            parsed_action, is_valid_format = parse_tool_call(action_str)
            
            self._current_step += 1
            
            # Execute action in environment
            obs, raw_reward, done, info = self._env.step(parsed_action)
            info = dict(info or {})
            
            # Get available actions
            available_actions = {}
            if hasattr(self._env, 'get_available_actions'):
                available_actions = self._env.get_available_actions()
            
            # Calculate task score
            task_score = float(raw_reward) if raw_reward is not None else 0.0
            task_score = max(0.0, min(1.0, task_score))
            
            # Check if task is completed successfully
            won = done and task_score >= 1.0
            info['won'] = won
            info['task_score'] = task_score
            
            # Binary reward: 1.0 for perfect completion, 0.0 otherwise
            if done:
                reward = 1.0 if won else 0.0
            else:
                reward = 0.0
            
            # Check max steps
            if self._current_step >= self.max_steps:
                done = True
            
            self._done = done
            self._cumulative_reward += reward
            
            # Build observation (tool-call format)
            full_observation = self._build_observation(obs, available_actions)
            
            # Build info dict
            info.update({
                "instruction": self._instruction_text,
                "available_actions": available_actions,
                "step": self._current_step,
                "max_steps": self.max_steps,
                "task_score": task_score,
                "is_valid_format": is_valid_format,
                "parsed_action": parsed_action,
                "cumulative_reward": self._cumulative_reward,
                "done": done,
                "session_id": getattr(self, '_current_session', None),
            })
            self._last_info = info
            
            return full_observation, reward, done, info
    
    def _build_observation(self, obs: str, available_actions: Dict) -> str:
        """
        Build the full observation string for the agent.
        Uses [button] format for clickable elements to match tool-call style.
        """
        parts = []
        
        # Add instruction
        if self._instruction_text:
            parts.append(f"Instruction: {self._instruction_text}")
        
        # Add current page observation
        parts.append(f"\nCurrent Page:\n{obs}")
        
        # Add available actions in [button] format for tool-call style
        if available_actions:
            clickables = available_actions.get("clickables", [])
            if clickables:
                # Categorize clickables
                navigation_buttons = []
                product_asins = []
                options = []
                other_buttons = []
                
                for item in clickables:
                    item_lower = item.lower()
                    if item_lower in ['next >', '< prev', 'back to search', 'buy now', 'search']:
                        navigation_buttons.append(item)
                    elif re.match(r'^b\d{2}[a-z0-9]{7}$', item_lower):
                        product_asins.append(item.upper())  # ASINs in uppercase
                    elif any(x in item_lower for x in ['size', 'color', 'small', 'medium', 'large', 'xl', 'xxl']):
                        options.append(item)
                    else:
                        other_buttons.append(item)
                
                parts.append("\n--- Available Buttons ---")
                
                # Show navigation buttons with [button] format
                if navigation_buttons:
                    nav_formatted = [f"[{b.title() if b.lower() not in ['< prev', 'next >'] else b}]" for b in navigation_buttons]
                    parts.append(f"Navigation: {' '.join(nav_formatted)}")
                
                # Show product ASINs
                if product_asins:
                    displayed_products = product_asins[:10]
                    asin_formatted = [f"[{a}]" for a in displayed_products]
                    parts.append(f"Products ({len(product_asins)} total): {' '.join(asin_formatted)}")
                    if len(product_asins) > 10:
                        parts.append(f"  ... and {len(product_asins) - 10} more products")
                
                # Show options
                if options:
                    opt_formatted = [f"[{o}]" for o in options[:20]]
                    parts.append(f"Options: {' '.join(opt_formatted)}")
                
                # Show other buttons
                if other_buttons:
                    other_formatted = [f"[{b}]" for b in other_buttons[:10]]
                    parts.append(f"Other: {' '.join(other_formatted)}")
            
            if available_actions.get("has_search_bar"):
                parts.append("\n[Search] button available - use search tool with query")
        
        return "\n".join(parts)
    
    def close(self):
        """
        Close the environment.
        
        Note: For shared environments managed by SharedEnvironmentManager,
        this method should NOT be called directly. The manager will handle cleanup.
        """
        if self._env is not None:
            try:
                self._env.close()
            except Exception as e:
                logger.warning(f"Error closing webshop environment: {e}")
            self._env = None
            self._initialized = False
    
    def soft_reset(self):
        """
        Soft reset: only reset episode state without reinitializing the environment.
        
        This is useful for shared environments where we want to reuse the
        initialized environment (with loaded product data) for multiple episodes.
        """
        self._current_step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._instruction_text = ""
        self._last_info = {}
    
    @staticmethod
    def from_dict(env_args: dict) -> "WebshopEnvironment":
        """Create environment from dictionary configuration."""
        known_keys = {"reward_fn", "max_steps", "observation_mode", "seed", "webshop_path", "task"}
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
        """
        Check if the environment is safe for multi-threaded execution.
        
        With shared environment mode DISABLED (the default), each parallel task
        gets its own independent WebshopEnvironment instance, making it fully
        thread-safe.
        
        Returns True to allow parallel execution.
        """
        return True


# Export tool definitions
def get_webshop_tools() -> List[Dict]:
    """Get the tool definitions for Webshop environment."""
    return WEBSHOP_TOOLS
