from typing import Any, Dict, List
from rllm.agents.agent import Action, BaseAgent, Trajectory

# 引入四种具体的 Agent
from rllm.agents.bfcl_agent import BFCLReadyAgent
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
        return self.active_agent.chat_completions

    @property
    def trajectory(self) -> Trajectory:
        return self.active_agent.trajectory
