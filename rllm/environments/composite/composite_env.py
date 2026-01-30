import json
import logging
from typing import Any, Tuple, Dict
from rllm.environments.base.base_env import BaseEnv
from rllm.environments.tools.bfcl_env_v2 import BFCLEnvironment
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.environments.tools.mcp_env import MCPEnvironment
# [新增] 引入 ToolCallEnvironment
from rllm.environments.tools.toolcall_env import ToolCallEnvironment
# [新增] 引入 WebshopEnvironment
from rllm.environments.webshop.webshop_env import WebshopEnvironment

# 确保 Logger 级别为 INFO，否则看不到 Log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompositeEnvironment(BaseEnv):
    """
    多任务组合环境: BFCL + Math + Code + Search + ToolCall + LocalSearch + Webshop
    充当数据适配器，负责将 Dataset 中的扁平化数据解包为各子环境所需的原始格式。
    """
    
    def __init__(
        self,
        bfcl_args: Dict[str, Any],
        math_args: Dict[str, Any],
        code_args: Dict[str, Any],
        search_args: Dict[str, Any] = {},
        tool_call_args: Dict[str, Any] = {},
        local_search_args: Dict[str, Any] = {},  # [新增] Local Search 参数
        webshop_args: Dict[str, Any] = {},  # [新增] Webshop 参数
        task: Dict[str, Any] = None
    ):
        self.task = task or {}
        
        # --- 1. BFCL Environment ---
        self.bfcl_env = BFCLEnvironment.from_dict(bfcl_args)
        
        # --- 2. Math Environment ---
        if "reward_fn" in math_args:
            r_fn = math_args["reward_fn"]
            m_args = math_args.copy()
            m_args.pop("reward_fn", None)
            self.math_env = ToolEnvironment(reward_fn=r_fn, **m_args)
        else:
            self.math_env = ToolEnvironment.from_dict(math_args)

        # --- 3. Code Environment ---
        if "reward_fn" in code_args:
            r_fn = code_args["reward_fn"]
            c_args = code_args.copy()
            c_args.pop("reward_fn", None)
            self.code_env = SingleTurnEnvironment(reward_fn=r_fn, **c_args)
        else:
            self.code_env = SingleTurnEnvironment.from_dict(code_args)

        # --- 4. Search Environment ---
        if "reward_fn" in search_args:
            self.search_env = MCPEnvironment(**search_args)
        else:
            # 搜索环境通常需要显式初始化，这里暂时允许跳过（如果search_num=0）
            self.search_env = None

        # --- 5. Tool Call Environment ---
        if "reward_fn" in tool_call_args:
            r_fn = tool_call_args["reward_fn"]
            t_args = tool_call_args.copy()
            t_args.pop("reward_fn", None)
            self.tool_call_env = ToolCallEnvironment(reward_fn=r_fn, **t_args)
        else:
            self.tool_call_env = ToolCallEnvironment.from_dict(tool_call_args)

        # --- 6. Local Search Environment [新增] ---
        # 使用 ToolEnvironment，与 Math 类似，但使用 local_search 工具
        if "reward_fn" in local_search_args:
            r_fn = local_search_args["reward_fn"]
            ls_args = local_search_args.copy()
            ls_args.pop("reward_fn", None)
            self.local_search_env = ToolEnvironment(reward_fn=r_fn, **ls_args)
        else:
            self.local_search_env = None

        # --- 7. Webshop Environment [新增] ---
        if "reward_fn" in webshop_args:
            r_fn = webshop_args["reward_fn"]
            ws_args = webshop_args.copy()
            ws_args.pop("reward_fn", None)
            self.webshop_env = WebshopEnvironment(reward_fn=r_fn, **ws_args)
        else:
            self.webshop_env = None
        
        self.active_env = None
        self.active_task_type = None

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
        raw_task = task if task is not None else self.task
        current_task = self._unpack_task_data(raw_task)
        
        # 路由核心逻辑
        self.active_task_type = current_task.get("task_type", "bfcl")
        
        if self.active_task_type == "bfcl":
            self.active_env = self.bfcl_env
            if "task_id" in current_task:
                self.active_env.task_id = current_task["task_id"]
            obs, info = self.active_env.reset()
            
        elif self.active_task_type == "math":
            self.active_env = self.math_env
            self.active_env.task = current_task
            obs, info = self.active_env.reset() 
            
        elif self.active_task_type == "code":
            self.active_env = self.code_env
            obs, info = self.active_env.reset(task=current_task)

        elif self.active_task_type == "search":
            if not self.search_env:
                raise RuntimeError("Search Environment not initialized via search_args")
            self.active_env = self.search_env
            if hasattr(self.active_env, 'task'):
                self.active_env.task = current_task
            obs, info = self.active_env.reset()

        # Tool Call 路由
        elif self.active_task_type == "tool_call":
            self.active_env = self.tool_call_env
            # ToolCallEnvironment 的 reset 签名支持传入 task 或使用 internal self.task
            # 这里我们显式传入
            self.active_env.task = current_task
            obs, info = self.active_env.reset()

        # [新增] Local Search 路由
        elif self.active_task_type == "local_search":
            if not self.local_search_env:
                raise RuntimeError("Local Search Environment not initialized via local_search_args")
            self.active_env = self.local_search_env
            self.active_env.task = current_task
            obs, info = self.active_env.reset()

        # [新增] Webshop 路由
        elif self.active_task_type == "webshop":
            if not self.webshop_env:
                raise RuntimeError("Webshop Environment not initialized via webshop_args")
            self.active_env = self.webshop_env
            self.active_env.task = current_task
            obs, info = self.active_env.reset(task=current_task)

        else:
            logger.warning(f"Unknown task_type: {self.active_task_type}, falling back to Math")
            self.active_env = self.math_env
            obs, info = self.active_env.reset(task=current_task)

        info["task_type"] = self.active_task_type
        return obs, info

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        if not self.active_env:
            raise RuntimeError("Environment not reset. Call reset() first.")
            
        obs, reward, done, info = self.active_env.step(action)
        info["task_type"] = self.active_task_type
        return obs, reward, done, info

    def close(self):
        # 关闭所有子环境
        for env in [self.bfcl_env, self.math_env, self.code_env, self.search_env, self.tool_call_env, self.local_search_env, self.webshop_env]:
            if env and hasattr(env, 'close'):
                env.close()

    @staticmethod
    def from_dict(env_args: dict) -> "CompositeEnvironment":
        return CompositeEnvironment(
            bfcl_args=env_args.get("bfcl_args", {}),
            math_args=env_args.get("math_args", {}),
            code_args=env_args.get("code_args", {}),
            search_args=env_args.get("search_args", {}),
            tool_call_args=env_args.get("tool_call_args", {}),
            local_search_args=env_args.get("local_search_args", {}),
            webshop_args=env_args.get("webshop_args", {}),  # [新增]
            task=env_args
        )
