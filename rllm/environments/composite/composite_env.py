import json
import logging
from typing import Any, Tuple, Dict, Optional
from rllm.environments.base.base_env import BaseEnv

# 确保 Logger 级别为 INFO，否则看不到 Log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 需要使用共享环境的任务类型（初始化开销大的环境）
# NOTE: Webshop removed from shared types due to thread safety issues
# The underlying SimBrowser is not thread-safe and causes session pollution
# between concurrent tasks. Each task should get its own WebshopEnvironment instance.
SHARED_ENV_TYPES = set()  # {"webshop"}  # Disabled until thread safety is resolved


class CompositeEnvironment(BaseEnv):
    """
    多任务组合环境: BFCL + Math + Code + Search + ToolCall + LocalSearch + Webshop
    
    使用懒加载模式：只有在首次使用某个任务类型时才实例化对应的环境，
    避免不必要的初始化开销（如 MCP Server 启动、Webshop 数据加载等）。
    
    对于初始化开销大的环境（如 Webshop），使用 SharedEnvironmentManager 进行
    跨实例共享，避免每个任务都重新加载数据。
    
    充当数据适配器，负责将 Dataset 中的扁平化数据解包为各子环境所需的原始格式。
    """
    
    # 类级别的共享环境管理器（延迟初始化）
    _shared_manager = None
    
    @classmethod
    def _get_shared_manager(cls):
        """获取共享环境管理器（延迟初始化）"""
        if cls._shared_manager is None:
            from rllm.environments.shared_env_manager import get_shared_env_manager
            cls._shared_manager = get_shared_env_manager()
        return cls._shared_manager
    
    def __init__(
        self,
        bfcl_args: Dict[str, Any] = None,
        math_args: Dict[str, Any] = None,
        code_args: Dict[str, Any] = None,
        search_args: Dict[str, Any] = None,
        tool_call_args: Dict[str, Any] = None,
        local_search_args: Dict[str, Any] = None,
        webshop_args: Dict[str, Any] = None,
        task: Dict[str, Any] = None,
        use_shared_envs: bool = True,  # 是否使用共享环境
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
            use_shared_envs: 是否对重型环境使用共享实例（默认 True）
        """
        self.task = task or {}
        self.use_shared_envs = use_shared_envs
        
        # 保存配置参数（不立即实例化）
        self._bfcl_args = bfcl_args or {}
        self._math_args = math_args or {}
        self._code_args = code_args or {}
        self._search_args = search_args or {}
        self._tool_call_args = tool_call_args or {}
        self._local_search_args = local_search_args or {}
        self._webshop_args = webshop_args or {}
        
        # 环境实例缓存（懒加载）- 仅用于非共享环境
        self._env_cache: Dict[str, Optional[BaseEnv]] = {
            "bfcl": None,
            "math": None,
            "code": None,
            "search": None,
            "tool_call": None,
            "local_search": None,
            "webshop": None,
        }
        
        # 记录哪些环境是从共享管理器获取的（close 时不关闭）
        self._shared_env_types: set = set()
        
        # 当前激活的环境
        self.active_env: Optional[BaseEnv] = None
        self.active_task_type: Optional[str] = None
        
        # 记录已初始化的环境类型（用于日志）
        self._initialized_envs: set = set()

    def _get_or_create_env(self, task_type: str) -> BaseEnv:
        """
        获取或创建指定类型的环境（懒加载核心逻辑）。
        
        对于 SHARED_ENV_TYPES 中的环境类型，使用 SharedEnvironmentManager 获取共享实例。
        这样可以避免每个任务都重新初始化重型环境（如 Webshop 的商品数据加载）。
        
        Args:
            task_type: 任务类型
            
        Returns:
            对应的环境实例
        """
        # 如果已缓存（本地或共享），直接返回
        if self._env_cache.get(task_type) is not None:
            return self._env_cache[task_type]
        
        # 检查是否应该使用共享环境
        use_shared = self.use_shared_envs and task_type in SHARED_ENV_TYPES
        
        if use_shared:
            # 尝试从共享管理器获取
            env = self._get_shared_env(task_type)
            if env is not None:
                self._env_cache[task_type] = env
                self._shared_env_types.add(task_type)
                self._initialized_envs.add(task_type)
                return env
        
        # 否则，按需创建新实例
        # logger.info(f"[CompositeEnv] Lazy-loading environment for task_type='{task_type}'...")
        
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
        # logger.info(f"[CompositeEnv] ✅ Environment '{task_type}' initialized successfully.")
        
        return env
    
    def _get_shared_env(self, task_type: str) -> Optional[BaseEnv]:
        """
        从共享管理器获取环境实例。
        
        Args:
            task_type: 任务类型
            
        Returns:
            共享的环境实例，如果不支持共享则返回 None
        """
        manager = self._get_shared_manager()
        
        if task_type == "webshop":
            from rllm.environments.webshop.webshop_env import WebshopEnvironment
            if not self._webshop_args or "reward_fn" not in self._webshop_args:
                raise RuntimeError(
                    "Webshop Environment requires 'reward_fn' in webshop_args."
                )
            env = manager.get_or_create_env(
                env_type="webshop",
                env_class=WebshopEnvironment,
                env_args=self._webshop_args
            )
            # logger.info(f"[CompositeEnv] ✅ Using shared environment for '{task_type}'.")
            return env
        
        # 其他类型暂不支持共享
        return None

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
        
        【新增功能】自动将任务级别的配置参数注入到 info 中：
        - task_max_prompt_length: 该任务类型的最大 prompt 长度
        - task_max_response_length: 该任务类型的最大 response 长度
        - task_max_steps: 该任务类型的最大交互步数
        
        执行引擎会在运行时读取这些参数，实现动态配置。
        
        Args:
            task: 任务数据字典
            
        Returns:
            (observation, info) 元组
        """
        raw_task = task if task is not None else self.task
        current_task = self._unpack_task_data(raw_task)
        
        # 获取任务类型，默认为 "math"（而非 "bfcl"，因为 math 更常用）
        self.active_task_type = current_task.get("task_type", "math")
        
        # 【新增】提取任务级别的配置参数
        self._task_max_prompt_length = current_task.get("task_max_prompt_length")
        self._task_max_response_length = current_task.get("task_max_response_length")
        self._task_max_steps = current_task.get("task_max_steps")
        
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
        
        # 【新增】将任务级别的配置参数注入到 info 中
        # 执行引擎会在每个 trajectory 开始时读取这些参数
        if self._task_max_prompt_length is not None:
            info["task_max_prompt_length"] = self._task_max_prompt_length
        if self._task_max_response_length is not None:
            info["task_max_response_length"] = self._task_max_response_length
        if self._task_max_steps is not None:
            info["task_max_steps"] = self._task_max_steps
            
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
        """
        关闭所有已初始化的子环境。
        
        注意：共享环境不会被关闭，它们由 SharedEnvironmentManager 管理。
        """
        for task_type, env in self._env_cache.items():
            if env is not None:
                # 跳过共享环境（由 SharedEnvironmentManager 管理）
                if task_type in self._shared_env_types:
                    logger.debug(f"[CompositeEnv] Skipping shared environment: {task_type}")
                    continue
                    
                if hasattr(env, 'close'):
                    try:
                        env.close()
                        logger.debug(f"[CompositeEnv] Closed environment: {task_type}")
                    except Exception as e:
                        logger.warning(f"[CompositeEnv] Failed to close {task_type}: {e}")
        
        # 清空缓存（但不影响共享管理器中的环境）
        self._env_cache = {k: None for k in self._env_cache}
        self._initialized_envs.clear()
        self._shared_env_types.clear()
        self.active_env = None
        self.active_task_type = None

    def get_initialized_envs(self) -> set:
        """返回已初始化的环境类型集合（用于调试）。"""
        return self._initialized_envs.copy()
    
    def get_shared_envs(self) -> set:
        """返回使用共享实例的环境类型集合（用于调试）。"""
        return self._shared_env_types.copy()

    @staticmethod
    def is_multithread_safe() -> bool:
        """
        CompositeEnvironment 是否线程安全。
        
        由于共享环境模式已禁用，每个并行任务会获得独立的 CompositeEnvironment
        和子环境实例，因此支持 n_parallel_agents > 1 的并行执行。
        
        Returns:
            True - 支持并行执行
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
            task=env_args,
            use_shared_envs=env_args.get("use_shared_envs", True),
        )
    
    @classmethod
    def pre_initialize_shared_envs(cls, env_args: dict):
        """
        预初始化共享数据（可选的优化方法）。
        
        在处理任务之前调用此方法，可以提前加载 Webshop 的只读数据：
        - 产品数据 (all_products, product_item_dict, product_prices)
        - 搜索引擎 (LuceneSearcher)
        - 目标列表 (goals)
        
        这样后续创建的 WebshopEnvironment 实例可以直接复用这些数据，
        将每个实例的创建时间从 ~3-5s 降低到 <0.1s。
        
        Args:
            env_args: 环境配置字典
            
        Example:
            CompositeEnvironment.pre_initialize_shared_envs({
                "webshop_args": {"reward_fn": webshop_reward_fn, "max_steps": 15}
            })
        """
        webshop_args = env_args.get("webshop_args", {})
        if webshop_args:
            logger.info("[CompositeEnv] Pre-loading shared Webshop data...")
            try:
                # 触发共享数据加载
                # 这会加载产品数据、搜索引擎和目标列表到进程级缓存中
                from web_agent_site.envs import get_shared_webshop_data
                from web_agent_site.utils import DEFAULT_FILE_PATH, DEFAULT_ATTR_PATH
                
                shared_data = get_shared_webshop_data()
                shared_data.get_or_load_data(
                    file_path=DEFAULT_FILE_PATH,
                    attr_path=DEFAULT_ATTR_PATH,
                    num_products=webshop_args.get('num_products', 1000),
                    human_goals=webshop_args.get('human_goals', False),
                    seed=webshop_args.get('seed', 42)
                )
                logger.info("[CompositeEnv] Shared Webshop data pre-loaded successfully.")
            except ImportError:
                logger.warning("[CompositeEnv] Could not import webshop module for pre-loading. "
                              "Data will be loaded on first use.")
            except Exception as e:
                logger.warning(f"[CompositeEnv] Failed to pre-load Webshop data: {e}. "
                              "Data will be loaded on first use.")
    
    @classmethod
    def cleanup_shared_envs(cls):
        """
        清理所有共享环境。
        
        在所有任务完成后调用此方法，释放共享环境占用的资源。
        """
        if cls._shared_manager is not None:
            cls._shared_manager.close_all()
            logger.info("[CompositeEnv] Cleaned up all shared environments.")
