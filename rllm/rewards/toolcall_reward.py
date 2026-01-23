import json
import re
import logging
from collections import Counter
from typing import List, Dict, Any, Union

from rllm.rewards.reward_types import RewardConfig, RewardOutput

logger = logging.getLogger(__name__)

class ToolCallRewardFn:
    """
    Reward function for Tool Call tasks.
    It extracts <tool_call> blocks, parses the JSON arguments, 
    and compares them against the ground truth ignoring order.
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
        model_response = action
        ground_truth_str = task_info.get("ground_truth", "") or task_info.get("output", "")

        # 1. 提取 Ground Truth 中的工具调用
        gt_tools = self.extract_tool_calls(ground_truth_str)
        
        # 2. 提取 Model Response 中的工具调用
        pred_tools = self.extract_tool_calls(model_response)

        # 3. 如果 GT 为空 (纯文本回复)，检查 Model 是否也为空
        if not gt_tools:
            # 如果 GT 没有工具调用，而模型输出了工具调用 -> 0分
            if pred_tools:
                return RewardOutput(reward=0.0, is_correct=False)
            else:
                # 都是纯文本，此处简单处理为1.0，或者你可以加入文本相似度匹配
                # 根据任务描述，重点是 Tool Call，这里假设 GT 为空时不需要 Call
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


