import copy
from typing import Any, Dict, List

from rllm.agents.agent import Action, BaseAgent, Trajectory

# 引入四种具体的 Agent
from rllm.agents.bfcl_agent import BFCLReadyAgent
from rllm.agents.math_agent import MathAgent
from rllm.agents.code_agent import CompetitionCodingAgent
from rllm.agents.tool_agent import ToolAgent, MCPToolAgent, ToolCallAgent

class CompositeAgent(BaseAgent):
    """
    五合一 Agent: 动态路由到 BFCL, Math, Code, Search, 或 ToolCall Agent
    """

    def __init__(
        self, 
        bfcl_agent_args: Dict = {}, 
        math_agent_args: Dict = {},
        code_agent_args: Dict = {},
        search_agent_args: Dict = {},
        tool_call_agent_args: Dict = {}  # [新增]
    ):
        # 1. 初始化子 Agent
        self.bfcl_agent = BFCLReadyAgent(**bfcl_agent_args)
        self.math_agent = ToolAgent(**math_agent_args)
        self.code_agent = CompetitionCodingAgent(**code_agent_args)
        self.search_agent = MCPToolAgent(**search_agent_args)
        # [新增] 初始化 Tool Call Agent
        self.tool_call_agent = ToolCallAgent(**tool_call_agent_args)
        
        # 2. 状态管理
        self.active_agent = self.math_agent # 默认值
        self.active_type = "math"

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        根据 info['task_type'] 切换大脑
        """
        if info and "task_type" in info:
            task_type = info["task_type"]
            
            # 如果任务类型发生变化，切换 Agent 指针
            if task_type != self.active_type:
                self.active_type = task_type
                if task_type == "bfcl":
                    self.active_agent = self.bfcl_agent
                elif task_type == "math":
                    self.active_agent = self.math_agent
                elif task_type == "code":
                    self.active_agent = self.code_agent
                elif task_type == "search":
                    self.active_agent = self.search_agent
                # [新增] 路由到 Tool Call Agent
                elif task_type == "tool_call":
                    self.active_agent = self.tool_call_agent
        
        # 将感知数据转发给当前激活的 Agent
        self.active_agent.update_from_env(observation, reward, done, info, **kwargs)

    def update_from_model(self, response: str, **kwargs) -> Action:
        return self.active_agent.update_from_model(response, **kwargs)

    def reset(self):
        self.bfcl_agent.reset()
        self.math_agent.reset()
        self.code_agent.reset()
        self.search_agent.reset()
        self.tool_call_agent.reset() # [新增]

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """
        返回对话历史，移除历史 assistant 消息中的 <think>...</think> 内容。
        
        注意：只移除 messages[:-1] 中的 <think> 内容，保留最后一条消息完整。
        这样设计的原因：
        1. Prompt 构建时（最后一条是 user 消息）：所有历史 assistant 消息的 <think> 被移除，减少上下文长度
        2. Token 计算时（最后一条是 assistant 消息）：当前步骤的 assistant 消息保持完整，确保训练时 <think> token 被正确更新
        """
        messages = copy.deepcopy(self.active_agent.chat_completions)
        
        # 对除最后一条之外的所有 assistant 消息移除 <think> 内容
        for msg in messages[:-1]:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # 移除 <think>...</think> 内容（包括 </think> 标签本身）
                if "</think>" in content:
                    _, sep, after = content.partition("</think>")
                    if sep:
                        msg["content"] = after.strip()
        
        return messages

    @chat_completions.setter
    def chat_completions(self, messages: List[Dict[str, str]]):
        """
        设置对话历史（用于初始 prompt 截断场景）。
        
        直接覆盖当前激活 Agent 的内部消息列表。
        注意：这会丢失原有的 system prompt 等信息，仅用于紧急截断场景。
        """
        if hasattr(self.active_agent, 'messages'):
            self.active_agent.messages = messages
        else:
            raise AttributeError(f"Active agent {type(self.active_agent).__name__} does not have 'messages' attribute")

    @property
    def trajectory(self) -> Trajectory:
        return self.active_agent.trajectory
