from typing import Any, Dict, Tuple
from rllm.environments.base.single_turn_env import SingleTurnEnvironment 
from rllm.rewards.reward_types import RewardOutput
from rllm.rewards.reward_fn import zero_reward

class ToolCallEnvironment(SingleTurnEnvironment):
    """
    Environment for Tool Call tasks.
    Returns a dictionary observation to allow the Agent to separate System and User roles.
    """

    def __init__(self, task: Dict[str, Any] = None, reward_fn=None, **kwargs):
        # SingleTurnEnvironment 内部会自动设置 max_turns=1
        super().__init__(task=task, reward_fn=reward_fn, **kwargs)

    def reset(self, **kwargs) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Constructs the observation from the task data.
        Returns a DICT containing instruction and input separately.
        """
        # 1. 解析数据
        instruction = self.task.get("instruction", "")
        user_input = self.task.get("input", "")
        
        # 2. 构造结构化 Observation
        # 不在 Env 层做拼接，而是透传原始结构，由 Agent 决定如何组装 Prompt
        observation = {
            "instruction": instruction, # 将作为 System Prompt
            "input": user_input         # 将作为 User Message
        }
        
        # 3. 准备 Info (用于 Reward 计算和日志)
        ground_truth = self.task.get("output", "")
        info = {
            "ground_truth": ground_truth,
            "instruction": instruction,
            "input": user_input,
            "original_task": self.task
        }
        
        return observation, info

    def get_reward_and_next_obs(self, task: dict, action: Any) -> Tuple[float, Dict]:
        """
        Compute the reward based on the task and action.
        """
        task_info = {
            "ground_truth": task.get("output", ""),
            "instruction": task.get("instruction", ""),
            "input": task.get("input", ""),
            "task_type": "tool_call" 
        }

        # 调用 Reward Function
        reward_output = self.reward_fn(task_info=task_info, action=action)

        if hasattr(reward_output, "reward"):
            return reward_output.reward, {}
        else:
            return float(reward_output), {}

    @staticmethod
    def from_dict(env_args: dict) -> "ToolCallEnvironment":
        reward_fn = env_args.pop("reward_fn", None)
        task = env_args.get("task", env_args)
        return ToolCallEnvironment(task=task, reward_fn=reward_fn)
