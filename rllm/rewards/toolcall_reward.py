import json
import math
import re
import logging
from collections import Counter
from typing import List, Dict, Any

from rllm.rewards.reward_types import RewardConfig, RewardOutput

logger = logging.getLogger(__name__)


class ToolCallRewardFn:
    """
    Reward function for Tool Call tasks.
    It extracts <tool_call> blocks, parses the JSON arguments,
    and compares them against the ground truth ignoring order.
    
    Note: Length penalty is applied at trainer level:
    - Kimi K1.5 style: rllm.length_penalty.enable=True
    - Simple length penalty: rllm.simple_length_penalty.enable=True
    """

    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()

    def extract_tool_calls(self, text: str) -> List[Dict]:
        """Extracts JSON objects inside <tool_call> tags."""
        if not text:
            return []
        pattern = r"<tool_call>\n(.*?)\n</tool_call>"
        matches = re.findall(pattern, text, re.DOTALL)
        result = []
        for match in matches:
            try:
                # 尝试解析 JSON
                json_obj = json.loads(match.strip())
                result.append(json_obj)
            except json.JSONDecodeError:
                # 如果解析失败，忽略该条目 (或者根据需求给予惩罚)
                continue
        return result

    def convert_to_hashable(self, data: Any):
        """Recursively converts dicts and lists to hashable types (frozenset/tuple) for comparison."""
        if isinstance(data, dict):
            return frozenset((key, self.convert_to_hashable(value)) for key, value in data.items())
        elif isinstance(data, list):
            # 对于列表，我们假设列表内的顺序也不重要（如果工具参数是列表且顺序重要，这里需要改为 tuple）
            # 根据你的参考代码逻辑：compare_parsed_content 对列表进行了 Counter 计数，意味着列表顺序不重要
            return frozenset(self.convert_to_hashable(item) for item in data)
        else:
            return data

    def compare_parsed_content(self, parsed1: List[Dict], parsed2: List[Dict]) -> bool:
        """
        Compares two lists of dicts ignoring order of list elements and dict keys.
        """
        counter1 = Counter(self.convert_to_hashable(item) for item in parsed1)
        counter2 = Counter(self.convert_to_hashable(item) for item in parsed2)
        return counter1 == counter2

    def __call__(self, task_info: dict, action: str) -> RewardOutput:
        """
        Compute reward for a single response.
        
        Base reward:
        - 1.0 if tool calls match ground truth
        - 0.0 otherwise
        
        Note: Length penalty is applied at trainer level for proper batch handling.
        
        Args:
            task_info: Task information containing ground truth
            action: Model's response
            
        Returns:
            RewardOutput with computed reward
        """
        model_response = action
        ground_truth_str = task_info.get("ground_truth", "") or task_info.get("output", "")

        # 1. 提取 Ground Truth 中的工具调用
        gt_tools = self.extract_tool_calls(ground_truth_str)
        
        # 2. 提取 Model Response 中的工具调用
        pred_tools = self.extract_tool_calls(model_response)

        # 3. 如果 GT 为空 (纯文本回复)，检查 Model 是否也为空
        if not gt_tools:
            if pred_tools:
                # GT 没有工具调用，而模型输出了工具调用 -> 0分
                return RewardOutput(reward=0.0, is_correct=False)
            else:
                # 都是纯文本，此处简单处理为1.0
                return RewardOutput(reward=1.0, is_correct=True)

        # 4. 比较工具调用是否一致
        # 长度不一致直接 0 分
        if len(gt_tools) != len(pred_tools):
            return RewardOutput(reward=0.0, is_correct=False)

        is_match = self.compare_parsed_content(gt_tools, pred_tools)

        if is_match:
            return RewardOutput(reward=1.0, is_correct=True)
        else:
            return RewardOutput(reward=0.0, is_correct=False)


def compute_length_reward_kimi(
    response_len: int,
    is_correct: bool,
    min_len: int,
    max_len: int
) -> float:
    """
    Compute length reward following modified Kimi K1.5's approach.
    
    Formula:
        λ = 0.5 - (len(i) - min_len) / (max_len - min_len)
        
        If correct (r=1): len_reward = λ
        If incorrect (r=0): len_reward = 0  # No length penalty for incorrect answers
    
    Key insight: Length penalty should ONLY differentiate among CORRECT answers.
    Applying length penalty to incorrect answers causes reward hacking where
    the model learns "short wrong is better than long wrong", leading to
    degenerate short outputs.
    
    Args:
        response_len: Length of the current response
        is_correct: Whether the response is correct
        min_len: Minimum length among all sampled responses for this problem
        max_len: Maximum length among all sampled responses for this problem
        
    Returns:
        Length reward value in range [-0.5, 0.5] for correct, 0 for incorrect
    """
    # If all responses have the same length, no length reward
    if max_len == min_len:
        return 0.0
    
    # Compute λ = 0.5 - (len(i) - min_len) / (max_len - min_len)
    # λ ranges from 0.5 (shortest) to -0.5 (longest)
    normalized_len = (response_len - min_len) / (max_len - min_len)
    lambda_val = 0.5 - normalized_len
    
    if is_correct:
        # For correct answers: use λ directly
        # Shorter responses get positive reward, longer get negative
        return lambda_val
    else:
        # For incorrect answers: NO length reward/penalty
        # This prevents reward hacking where model learns to output short wrong answers
        return 0.0


def compute_simple_length_penalty(
    response_len: int,
    baseline_len: int = 1024,
    coefficient: float = 0.15,
    max_penalty: float = 0.3
) -> float:
    """
    Compute simple length penalty based on log ratio.
    
    Formula:
        penalty = coefficient * log(response_len / baseline_len)
        penalty = min(max_penalty, penalty)
    
    This penalty should be SUBTRACTED from the reward:
        final_reward = base_reward - penalty
    
    Effect:
        - Short responses (< baseline): penalty < 0, so reward increases
        - Long responses (> baseline): penalty > 0, so reward decreases
    
    Args:
        response_len: Length of the response
        baseline_len: Baseline length for comparison (default: 1024)
        coefficient: Scaling coefficient (default: 0.15)
        max_penalty: Maximum penalty value (default: 0.3)
        
    Returns:
        Length penalty value (can be negative for short responses)
    """
    import math
    
    if response_len <= 0:
        return 0.0
    
    penalty = coefficient * math.log(response_len / baseline_len)
    penalty = min(max_penalty, penalty)
    
    return penalty
