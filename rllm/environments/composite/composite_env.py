import json
import logging
from typing import Any, Tuple, Dict, Optional
from rllm.environments.base.base_env import BaseEnv

# 确保 Logger 级别为 INFO，否则看不到 Log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompositeEnvironment(BaseEnv):
    """
    多任务组合环境: BFCL + Math + Code + Search + ToolCall + LocalSearch + Webshop
    
    使用懒加载模式：只有在首次使用某个任务类型时才实例化对应的环境，
    避免不必要的初始化开销（如 MCP Server 启动、Webshop 数据加载等）。
    
    充当数据适配器，负责将 Dataset 中的扁平化数据解包为各子环境所需的原始格式。
    """
    
    def __init__(
        self,
        bfcl_args: Dict[str, Any] = None,
        math_args: Dict[str, Any] = None,
        code_args: Dict[str, Any] = None,
        search_args: Dict[str, Any] = None,
        tool_call_args: Dict[str, Any] = None,
        local_search_args: Dict[str, Any] = None,
        webshop_args: Dict[str, Any] = None,
        task: Dict[str, Any] = None
    ):
        """
        初始化 CompositeEnvironment。
        
        注意：此时不会实例化任何子环境，只保存配置参数。
        子环境会在首次使用时懒加载。
        
        Args:
            bfcl_args: BFCL 环境配置
            math_args: Math 环境配置
            code_args: Code 环境配置
            search_args: Search (MCP) 环境配置
            tool_call_args: Tool Call 环境配置
            local_search_args: Local Search 环境配置
            webshop_args: Webshop 环境配置
            task: 初始任务（可选）
        """
        self.task = task or {}
        
        # 保存配置参数（不立即实例化）
        self._bfcl_args = bfcl_args or {}
        self._math_args = math_args or {}
        self._code_args = code_args or {}
        self._search_args = search_args or {}
        self._tool_call_args = tool_call_args or {}
        self._local_search_args = local_search_args or {}
        self._webshop_args = webshop_args or {}
        
        # 环境实例缓存（懒加载）
        self._env_cache: Dict[str, Optional[BaseEnv]] = {
            "bfcl": None,
            "math": None,
            "code": None,
            "search": None,
            "tool_call": None,
            "local_search": None,
            "webshop": None,
        }
        
        # 当前激活的环境
        self.active_env: Optional[BaseEnv] = None
        self.active_task_type: Optional[str] = None
        
        # 记录已初始化的环境类型（用于日志）
        self._initialized_envs: set = set()

    def _get_or_create_env(self, task_type: str) -> BaseEnv:
        """
        获取或创建指定类型的环境（懒加载核心逻辑）。
        
        Args:
            task_type: 任务类型
            
        Returns:
            对应的环境实例
        """
        # 如果已缓存，直接返回
        if self._env_cache.get(task_type) is not None:
            return self._env_cache[task_type]
        
        # 否则，按需创建
        logger.info(f"[CompositeEnv] Lazy-loading environment for task_type='{task_type}'...")
        
        env = None
        
        if task_type == "bfcl":
            from rllm.environments.tools.bfcl_env_v2 import BFCLEnvironment
            env = BFCLEnvironment.from_dict(self._bfcl_args)
            
        elif task_type == "math":
            from rllm.environments.tools.tool_env import ToolEnvironment
            env = self._create_env_with_reward_fn(
                ToolEnvironment, self._math_args
            )
            
        elif task_type == "code":
            from rllm.environments.base.single_turn_env import SingleTurnEnvironment
            env = self._create_env_with_reward_fn(
                SingleTurnEnvironment, self._code_args
            )
            
        elif task_type == "search":
            from rllm.environments.tools.mcp_env import MCPEnvironment
            if not self._search_args or "reward_fn" not in self._search_args:
                raise RuntimeError(
                    "Search Environment requires 'reward_fn' in search_args. "
                    "Please provide search_args with reward_fn to use search tasks."
                )
            env = MCPEnvironment(**self._search_args)
            
        elif task_type == "tool_call":
            from rllm.environments.tools.toolcall_env import ToolCallEnvironment
            env = self._create_env_with_reward_fn(
                ToolCallEnvironment, self._tool_call_args
            )
            
        elif task_type == "local_search":
            from rllm.environments.tools.tool_env import ToolEnvironment
            if not self._local_search_args or "reward_fn" not in self._local_search_args:
                raise RuntimeError(
                    "Local Search Environment requires 'reward_fn' in local_search_args."
                )
            env = self._create_env_with_reward_fn(
                ToolEnvironment, self._local_search_args
            )
            
        elif task_type == "webshop":
            from rllm.environments.webshop.webshop_env import WebshopEnvironment
            if not self._webshop_args or "reward_fn" not in self._webshop_args:
                raise RuntimeError(
                    "Webshop Environment requires 'reward_fn' in webshop_args."
                )
            env = self._create_env_with_reward_fn(
                WebshopEnvironment, self._webshop_args
            )
            
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
        
        # 缓存并返回
        self._env_cache[task_type] = env
        self._initialized_envs.add(task_type)
        logger.info(f"[CompositeEnv] ✅ Environment '{task_type}' initialized successfully.")
        
        return env

    def _create_env_with_reward_fn(self, env_class, args: Dict[str, Any]) -> BaseEnv:
        """
        辅助方法：创建带有 reward_fn 的环境。
        
        Args:
            env_class: 环境类
            args: 环境参数
            
        Returns:
            环境实例
        """
        if not args:
            return env_class.from_dict({})
            
        if "reward_fn" in args:
            r_fn = args["reward_fn"]
            env_args = args.copy()
            env_args.pop("reward_fn", None)
            return env_class(reward_fn=r_fn, **env_args)
        else:
            return env_class.from_dict(args)

    def _unpack_task_data(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        通用数据解包函数：解析 JSON 字符串并扁平化合并到顶层
        """
        if not task:
            return {}

        # 复制一份以防修改原数据
        final_data = task.copy()

        # 定义需要尝试解析和展开的字段名
        fields_to_unpack = ["original_data", "extra_info"]

        for field in fields_to_unpack:
            if field in final_data:
                raw_content = final_data[field]
                parsed_content = {}

                if isinstance(raw_content, str):
                    try:
                        parsed_content = json.loads(raw_content)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON in field '{field}'")
                        continue
                elif isinstance(raw_content, dict):
                    parsed_content = raw_content

                if parsed_content:
                    final_data.update(parsed_content)
                    # 处理 Math 数据集中常见的双层嵌套
                    if "extra_info" in parsed_content and isinstance(parsed_content["extra_info"], str):
                        try:
                            nested_extra = json.loads(parsed_content["extra_info"])
                            final_data.update(nested_extra)
                        except:
                            pass

        return final_data

    def reset(self, task: Dict[str, Any] = None) -> Tuple[Any, Dict]:
        """
        重置环境，根据任务类型路由到对应的子环境。
        
        Args:
            task: 任务数据字典
            
        Returns:
            (observation, info) 元组
        """
        raw_task = task if task is not None else self.task
        current_task = self._unpack_task_data(raw_task)
        
        # 获取任务类型，默认为 "math"（而非 "bfcl"，因为 math 更常用）
        self.active_task_type = current_task.get("task_type", "math")
        
        # 懒加载对应的环境
        try:
            self.active_env = self._get_or_create_env(self.active_task_type)
        except Exception as e:
            logger.error(f"Failed to create environment for task_type='{self.active_task_type}': {e}")
            raise
        
        # 根据任务类型执行 reset
        if self.active_task_type == "bfcl":
            if "task_id" in current_task:
                self.active_env.task_id = current_task["task_id"]
            obs, info = self.active_env.reset()
            
        elif self.active_task_type == "math":
            self.active_env.task = current_task
            obs, info = self.active_env.reset()
            
        elif self.active_task_type == "code":
            obs, info = self.active_env.reset(task=current_task)

        elif self.active_task_type == "search":
            if hasattr(self.active_env, 'task'):
                self.active_env.task = current_task
            obs, info = self.active_env.reset()

        elif self.active_task_type == "tool_call":
            self.active_env.task = current_task
            obs, info = self.active_env.reset()

        elif self.active_task_type == "local_search":
            self.active_env.task = current_task
            obs, info = self.active_env.reset()

        elif self.active_task_type == "webshop":
            self.active_env.task = current_task
            obs, info = self.active_env.reset(task=current_task)

        else:
            # Fallback to math
            logger.warning(f"Unknown task_type: {self.active_task_type}, falling back to Math")
            self.active_task_type = "math"
            self.active_env = self._get_or_create_env("math")
            self.active_env.task = current_task
            obs, info = self.active_env.reset()

        info = info or {}
        info["task_type"] = self.active_task_type
        return obs, info

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """
        执行一步动作。
        
        Args:
            action: 动作
            
        Returns:
            (observation, reward, done, info) 元组
        """
        if not self.active_env:
            raise RuntimeError("Environment not reset. Call reset() first.")
            
        obs, reward, done, info = self.active_env.step(action)
        info = info or {}
        info["task_type"] = self.active_task_type
        return obs, reward, done, info

    def close(self):
        """关闭所有已初始化的子环境。"""
        for task_type, env in self._env_cache.items():
            if env is not None and hasattr(env, 'close'):
                try:
                    env.close()
                    logger.debug(f"[CompositeEnv] Closed environment: {task_type}")
                except Exception as e:
                    logger.warning(f"[CompositeEnv] Failed to close {task_type}: {e}")
        
        # 清空缓存
        self._env_cache = {k: None for k in self._env_cache}
        self._initialized_envs.clear()
        self.active_env = None
        self.active_task_type = None

    def get_initialized_envs(self) -> set:
        """返回已初始化的环境类型集合（用于调试）。"""
        return self._initialized_envs.copy()

    @staticmethod
    def is_multithread_safe() -> bool:
        """
        CompositeEnvironment 是否线程安全。
        
        由于某些子环境（如 Webshop、MCP）不是线程安全的，
        CompositeEnvironment 保守地返回 True，让 AgentExecutionEngine 可以运行。
        
        注意：如果使用 webshop 任务，建议设置 n_parallel_agents=1。
        """
        return True

    @staticmethod
    def from_dict(env_args: dict) -> "CompositeEnvironment":
        """从字典创建 CompositeEnvironment。"""
        return CompositeEnvironment(
            bfcl_args=env_args.get("bfcl_args"),
            math_args=env_args.get("math_args"),
            code_args=env_args.get("code_args"),
            search_args=env_args.get("search_args"),
            tool_call_args=env_args.get("tool_call_args"),
            local_search_args=env_args.get("local_search_args"),
            webshop_args=env_args.get("webshop_args"),
            task=env_args
        )
