import logging
from typing import Any, Dict, List, Optional
from rllm.agents.agent import Action, BaseAgent, Trajectory

logger = logging.getLogger(__name__)


class CompositeAgent(BaseAgent):
    """
    七合一 Agent: 动态路由到 BFCL, Math, Code, Search, ToolCall, LocalSearch, 或 Webshop Agent
    
    使用懒加载模式：只有在首次使用某个任务类型时才实例化对应的 Agent，
    避免不必要的初始化开销。
    """

    def __init__(
        self,
        bfcl_agent_args: Dict = None,
        math_agent_args: Dict = None,
        code_agent_args: Dict = None,
        search_agent_args: Dict = None,
        tool_call_agent_args: Dict = None,
        local_search_agent_args: Dict = None,
        webshop_agent_args: Dict = None
    ):
        """
        初始化 CompositeAgent。
        
        注意：此时不会实例化任何子 Agent，只保存配置参数。
        子 Agent 会在首次使用时懒加载。
        
        Args:
            bfcl_agent_args: BFCL Agent 配置
            math_agent_args: Math Agent 配置
            code_agent_args: Code Agent 配置
            search_agent_args: Search Agent 配置
            tool_call_agent_args: Tool Call Agent 配置
            local_search_agent_args: Local Search Agent 配置
            webshop_agent_args: Webshop Agent 配置
        """
        # 保存配置参数（不立即实例化）
        self._bfcl_agent_args = bfcl_agent_args or {}
        self._math_agent_args = math_agent_args or {}
        self._code_agent_args = code_agent_args or {}
        self._search_agent_args = search_agent_args or {}
        self._tool_call_agent_args = tool_call_agent_args or {}
        self._local_search_agent_args = local_search_agent_args or {}
        self._webshop_agent_args = webshop_agent_args or {}
        
        # Agent 实例缓存（懒加载）
        self._agent_cache: Dict[str, Optional[BaseAgent]] = {
            "bfcl": None,
            "math": None,
            "code": None,
            "search": None,
            "tool_call": None,
            "local_search": None,
            "webshop": None,
        }
        
        # 当前激活的 Agent
        self.active_agent: Optional[BaseAgent] = None
        self.active_type: Optional[str] = None
        
        # 记录已初始化的 Agent 类型（用于日志）
        self._initialized_agents: set = set()

    def _get_or_create_agent(self, task_type: str) -> BaseAgent:
        """
        获取或创建指定类型的 Agent（懒加载核心逻辑）。
        
        Args:
            task_type: 任务类型
            
        Returns:
            对应的 Agent 实例
        """
        # 如果已缓存，直接返回
        if self._agent_cache.get(task_type) is not None:
            return self._agent_cache[task_type]
        
        # 否则，按需创建
        logger.info(f"[CompositeAgent] Lazy-loading agent for task_type='{task_type}'...")
        
        agent = None
        
        if task_type == "bfcl":
            from rllm.agents.bfcl_agent import BFCLReadyAgent
            agent = BFCLReadyAgent(**self._bfcl_agent_args)
            
        elif task_type == "math":
            from rllm.agents.tool_agent import ToolAgent
            agent = ToolAgent(**self._math_agent_args)
            
        elif task_type == "code":
            from rllm.agents.code_agent import CompetitionCodingAgent
            agent = CompetitionCodingAgent(**self._code_agent_args)
            
        elif task_type == "search":
            from rllm.agents.tool_agent import MCPToolAgent
            agent = MCPToolAgent(**self._search_agent_args)
            
        elif task_type == "tool_call":
            from rllm.agents.tool_agent import ToolCallAgent
            agent = ToolCallAgent(**self._tool_call_agent_args)
            
        elif task_type == "local_search":
            from rllm.agents.tool_agent import ToolAgent
            agent = ToolAgent(**self._local_search_agent_args)
            
        elif task_type == "webshop":
            from rllm.agents.webshop_agent import WebshopAgent
            agent = WebshopAgent(**self._webshop_agent_args)
            
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
        
        # 缓存并返回
        self._agent_cache[task_type] = agent
        self._initialized_agents.add(task_type)
        logger.info(f"[CompositeAgent] ✅ Agent '{task_type}' initialized successfully.")
        
        return agent

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        根据 info['task_type'] 切换 Agent，并将感知数据转发给当前激活的 Agent。
        
        Args:
            observation: 环境观察
            reward: 奖励
            done: 是否结束
            info: 信息字典，必须包含 'task_type'
            **kwargs: 其他参数
        """
        if info and "task_type" in info:
            task_type = info["task_type"]
            
            # 如果任务类型发生变化，切换 Agent
            if task_type != self.active_type:
                self.active_type = task_type
                try:
                    self.active_agent = self._get_or_create_agent(task_type)
                except Exception as e:
                    logger.error(f"Failed to create agent for task_type='{task_type}': {e}")
                    raise
        else:
            # 如果没有 task_type，使用默认的 math agent
            if self.active_type is None:
                logger.warning("No task_type in info, defaulting to 'math'")
                self.active_type = "math"
                self.active_agent = self._get_or_create_agent("math")
        
        # 将感知数据转发给当前激活的 Agent
        if self.active_agent:
            self.active_agent.update_from_env(observation, reward, done, info, **kwargs)
        else:
            raise RuntimeError("No active agent available")

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        将模型响应转发给当前激活的 Agent。
        
        Args:
            response: 模型响应
            **kwargs: 其他参数
            
        Returns:
            Action 对象
        """
        if not self.active_agent:
            raise RuntimeError("No active agent. Call update_from_env() first.")
        
        return self.active_agent.update_from_model(response, **kwargs)

    def reset(self):
        """重置所有已初始化的 Agent。"""
        for task_type, agent in self._agent_cache.items():
            if agent is not None:
                try:
                    agent.reset()
                    logger.debug(f"[CompositeAgent] Reset agent: {task_type}")
                except Exception as e:
                    logger.warning(f"[CompositeAgent] Failed to reset {task_type}: {e}")
        
        # 重置当前状态
        self.active_agent = None
        self.active_type = None

    def close(self):
        """关闭所有已初始化的 Agent（如果有 close 方法）。"""
        for task_type, agent in self._agent_cache.items():
            if agent is not None and hasattr(agent, 'close'):
                try:
                    agent.close()
                    logger.debug(f"[CompositeAgent] Closed agent: {task_type}")
                except Exception as e:
                    logger.warning(f"[CompositeAgent] Failed to close {task_type}: {e}")
        
        # 清空缓存
        self._agent_cache = {k: None for k in self._agent_cache}
        self._initialized_agents.clear()
        self.active_agent = None
        self.active_type = None

    def get_initialized_agents(self) -> set:
        """返回已初始化的 Agent 类型集合（用于调试）。"""
        return self._initialized_agents.copy()

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """返回当前激活 Agent 的聊天历史。"""
        if not self.active_agent:
            return []
        return self.active_agent.chat_completions

    @property
    def trajectory(self) -> Trajectory:
        """返回当前激活 Agent 的轨迹。"""
        if not self.active_agent:
            return Trajectory()
        return self.active_agent.trajectory