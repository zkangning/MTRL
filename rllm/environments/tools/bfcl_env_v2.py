import json
import uuid
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
import random
import time
import os  # 新增：缺失的导入
import requests  # 新增：缺失的导入
from datetime import datetime # 新增：缺失的导入
import re

from rllm.environments.base.base_env import BaseEnv
from rllm.agents.agent import Action


instruction = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out. If the result of tool calls has fulfilled the user's request, summary the answer.

**Important Notes**
1. When the tool call has fulfilled the user's request, please provide a concise summary in plain text without extra tool calls. If no tool is suitable, state that explicitly. If the user's input lacks required parameters, ask for clarification.
2. During each tool invocation, it is important to carefully examine the corresponding tool's description and constraints. Ensure that the required fields of the tool are strictly satisfied, that parameter types conform to the definitions. If function calls uses the default parameter value, it is not necessary to specify the value during the call.
3. If the user's request cannot be completed througha one-time function call, or if the parameters of subsequent function calls depend on the results of previous calls, then decompose it into multi-step calls. You only need to return the result of the first step. The use of fictitious parameters or placeholder is strictly prohibited.
4. In multi-turn dialogs, if you encounter an error and the task remains unfinished, retry with more necessary tool calls until completion. Based on the tool feedback, reflect on if understanding or selectioin of tool is wrong, what tool calling step is missing, and how to achieve the task goal from now on. (e.g., File system tools are limited to the current directory. No path is allowed. Operations on any given file should first enter its corresponding directory.)"""

# instruction = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
# If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out. If the result of tool calls has fulfilled the user's request, summary the answer."""

logger = logging.getLogger(__name__)

def safe_log(msg: str):
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] {msg}\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass  # 防止日志写失败影响RL主进程

def retry_call(
    fn: Callable,
    max_retry: int = 3,
    min_backoff: float = 3.0,
    max_backoff: float = 10.0,
    fail_return: Any = None,
    err_prefix: str = "",
    instance_id: str = "",
    action_name: str = ""
):
    last_exception = None
    for i in range(max_retry):
        try:
            res = fn()
            if i>0:
                safe_log(f"{err_prefix} {action_name} [instance={instance_id}] succeed at try {i+1}/{max_retry}")
            return res
        except Exception as e:
            last_exception = e
            safe_log(f"{err_prefix} {action_name} [instance={instance_id}] retry {i+1}/{max_retry} failed: {e}")
            if i + 1 == max_retry:
                safe_log(f"{err_prefix} {action_name} [instance={instance_id}] max retries exceeded, fallback used.")
                return fail_return
            wait = random.uniform(min_backoff, max_backoff)
            time.sleep(wait)
    return fail_return

class EnvClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.timeout = 150.0+random.uniform(50, 200)

    def _make_request(
        self,
        endpoint: str,
        env_type: str = "default",
        task_id: str = None,
        instance_id: str = None,
        messages: Dict[str, Any] = None,
        params: Dict[str, Any] = None,
    ) -> Dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        data = {
            "env_type": env_type,
            "task_id": task_id,
            "instance_id": instance_id,
            "messages": messages or {},
            "params": params or {},
        }
        try:
            response = requests.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            safe_log(
                f"[{endpoint}] _make_request failed (instance={instance_id}): {e}, data: {data}"
            )
            raise Exception(
                f"Request failed: {str(e)}, data: {data}"
            )

    def get_env_profile(
        self,
        env_type: str,
        split: str = "train",
        params: Optional[dict] = None,
        max_retry: int = 3
    ) -> List[str]:
        def call():
            response = self._make_request(
                endpoint="/get_env_profile",
                env_type=env_type,
                params={"split": split, **(params or {})},
            )
            if "data" in response:
                return response["data"]
            elif "task_ids" in response:
                return response["task_ids"]
            else:
                return []
        return retry_call(
            call,
            max_retry=max_retry,
            fail_return=[],
            err_prefix="[get_env_profile]",
            action_name="get_env_profile"
        )

    def get_tools_info(
        self, instance_id: str, messages: Dict = {}, params: Dict = {}, max_retry: int = 3
    ) -> Any:
        def call():
            response = self._make_request(
                endpoint="get_info",
                instance_id=instance_id,
                messages=messages,
                params=params,
            )
            return response.get("data", None)
        return retry_call(
            call,
            max_retry=max_retry,
            fail_return=None,
            err_prefix=f"[get_tools_info]",
            instance_id=instance_id,
            action_name="get_tools_info"
        )

    def create_instance(
        self,
        env_type: str,
        task_id: str,
        instance_id: Optional[str] = None,
        params: Optional[Dict] = None,
        max_retry: int = 3
    ) -> dict:
        fallback = {
            "state": [{"role": "system", "content": "create query failed, this is a empty task."},
                {"role": "user", "content": "create failed, this is a empty task,please close this task."}],
            "reward": 0,
            "is_terminated": False,
            "info": {"instance_id": instance_id or "", "task_id": task_id or ""},
        }
        def call():
            r = self._make_request(
                endpoint="create",
                env_type=env_type,
                task_id=task_id,
                instance_id=instance_id,
                params=params,
            )
            return r["data"]
        return retry_call(
            call,
            max_retry=max_retry,
            fail_return=fallback,
            err_prefix=f"[create_instance]",
            instance_id=instance_id,
            action_name="create_instance"
        )

    def step(
        self,
        instance_id: str,
        action: Dict = {},
        params: Dict = {},
        max_retry: int = 3,
    ) -> dict:
        fallback = {
            "state": [{"role": "assistant", "content": "Step failed (timeout or exception),please retry"}],
            "reward": 0,
            "is_terminated": False,
            "info": {"instance_id": instance_id or "", "task_id": ""},
        }
        def call():
            resp = self._make_request(
                endpoint="step",
                instance_id=instance_id,
                messages=action,
                params=params
            )
            # import pdb; pdb.set_trace()
            return resp["data"]
        return retry_call(
            call,
            max_retry=max_retry,
            fail_return=fallback,
            err_prefix=f"[step]",
            instance_id=instance_id,
            action_name="step"
        )

    def evaluate(
        self,
        instance_id: str,
        messages: Dict = {},
        params: Dict = {},
        max_retry: int = 3,
    ) -> float:
        def call():
            resp = self._make_request(
                endpoint="evaluate",
                instance_id=instance_id,
                messages=messages,
                params=params,
            )
            return resp.get("data", 0.0)
        return retry_call(
            call,
            max_retry=max_retry,
            fail_return=0.0,
            err_prefix=f"[evaluate]",
            instance_id=instance_id,
            action_name="evaluate"
        )

    def release_instance(self, instance_id: str, max_retry: int = 3) -> bool:
        def call():
            resp = self._make_request(endpoint="release", instance_id=instance_id)
            return resp.get("success", False)
        return retry_call(
            call,
            max_retry=max_retry,
            fail_return=False,
            err_prefix=f"[release_instance]",
            instance_id=instance_id,
            action_name="release_instance"
        )

class BFCLEnvironment(BaseEnv):
    """
    针对 BFCL 多轮对话和 Step-level Reward 优化的环境封装
    """

    def __init__(
        self, 
        base_url: str = "http://localhost:8801", 
        env_type: str = "bfcl",
        split: str = "train",
        max_steps: int = 20, 
        system_prompt: str = instruction,
        task_id: Optional[str] = None
    ):
        self.client = EnvClient(base_url=base_url)
        self.env_type = env_type
        self.split = split
        self.max_steps = max_steps
        
        self.task_id = task_id
        self.instance_id = None
        self.current_step = 0
        self.custom_system_prompt = system_prompt

    def reset(self) -> Tuple[Any, Dict]:
        self.current_step = 0
        
        if not self.task_id:
            tasks = self.client.get_env_profile(self.env_type, split=self.split)
            if not tasks: raise RuntimeError("No tasks available")
            self.task_id = tasks[0]

        init_res = self.client.create_instance(self.env_type, self.task_id)
        # import pdb; pdb.set_trace()
        if not init_res: raise RuntimeError("Failed to create instance")
        
        self.instance_id = init_res["info"]["instance_id"]
        
        state_messages = init_res.get("state", [])
        user_content = ""
        system_content = ""
        
        for msg in state_messages:
            if msg.get("role") == "system": system_content = msg.get("content")
            elif msg.get("role") == "user": user_content = msg.get("content")
        
        self.bfcl_system_prompt = self.custom_system_prompt + "\n\n" + system_content
        
        observation = {"question": user_content}
        info = {
            "instance_id": self.instance_id,
            "task_id": self.task_id,
            "system_prompt": self.bfcl_system_prompt
        }
        return observation, info

    def step(self, action: List[Dict] | str | Dict) -> Tuple[Dict, float, bool, Dict]:
        
        self.current_step += 1
        reward = 0.0
        done = False
        info = {}
        
        bfcl_message = self._convert_action_to_bfcl_message(action)
        
        # 预处理 Action
        is_finish_action = False
        tool_calls_in_action = []
        if isinstance(action, str): is_finish_action = True
        elif isinstance(action, list):
            for c in action:
                if c.get("function", {}).get("name") == "finish": is_finish_action = True
                else: tool_calls_in_action.append(c)

        try:
            step_res = self.client.step(
                instance_id=self.instance_id,
                action=bfcl_message
            )
        except Exception as e:
            logger.error(f"BFCL Step failed: {e}")
            return {"error": str(e)}, 0.0, True, {"error": str(e)}

        data = step_res if "state" in step_res else step_res.get("data", {})
        if not data: return {}, 0.0, True, {"error": "Empty data from env"}

        raw_state = data.get("state", [])
        is_terminated = data.get("is_terminated", False)
        
        # [MOD] 解析奖励
        # 1. 单步一致性奖励 (Server 每一轮都会返回，代表 Agent 这一步走得对不对)
        step_consistency_reward = float(data.get("reward", 0.0))
        
        # 2. 最终任务完成指标 (Server 只在 is_terminated=True 时返回 0 或 1)
        server_info = data.get("info", {})
        final_indicator_score = float(server_info.get("final_score", 0.0)) * 2 # 如果最终通过，那么

        # 构建 Observation
        next_obs = {}
        if raw_state:
            last_msg = raw_state[-1]
            if last_msg.get("role") == "user":
                content_str = last_msg.get("content", "")
                if is_finish_action:
                    if not is_terminated: next_obs = {"question": content_str}
                else:
                    matches = re.findall(r"<tool_call>(.*?)</tool_call>", content_str, re.DOTALL)
                    parsed_outputs = [m.strip() for m in matches]
                    tool_outputs = {}
                    for idx, call in enumerate(tool_calls_in_action):
                        call_id = call.get("id")
                        if idx < len(parsed_outputs): tool_outputs[call_id] = parsed_outputs[idx]
                        elif len(parsed_outputs) == 1: tool_outputs[call_id] = parsed_outputs[0]
                        else: tool_outputs[call_id] = '{"error": "No output"}'
                    
                    if not tool_outputs and tool_calls_in_action and content_str:
                         tool_outputs[tool_calls_in_action[0]["id"]] = content_str
                    next_obs = {"tool_outputs": tool_outputs}

        # [MOD] 奖励计算逻辑
        if is_terminated:
            done = True
            # 回合结束时的总奖励 = 当前步一致性(如果这步是工具调用) + 最终任务完成度
            reward = step_consistency_reward + final_indicator_score
            info["step_consistency_score"] = step_consistency_reward
            info["final_eval_score"] = final_indicator_score
        elif self.current_step >= self.max_steps:
            done = True
            reward = step_consistency_reward # 超时，只给当前步奖励
        else:
            reward = step_consistency_reward # 中间步，给一致性奖励
            
        return next_obs, reward, done, info

    def close(self):
        if self.instance_id:
            self.client.release_instance(self.instance_id)
            self.instance_id = None

    def _convert_action_to_bfcl_message(self, action: List[Dict] | str) -> Dict:
        if isinstance(action, str): return {"role": "assistant", "content": action}
        if isinstance(action, list):
            for call in action:
                if call.get("function", {}).get("name") == "finish":
                    args = call["function"]["arguments"]
                    if isinstance(args, str):
                        try: args = json.loads(args)
                        except: args = {"response": args}
                    return {"role": "assistant", "content": args.get("response", "")}
            xml_content = ""
            for call in action:
                func = call.get("function", {})
                name = func.get("name")
                args = func.get("arguments")
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except: pass
                tool_obj = {"name": name, "arguments": args}
                xml_content += f"<tool_call>\n{json.dumps(tool_obj)}\n</tool_call>\n"
            return {"role": "assistant", "content": xml_content}
        return {"role": "assistant", "content": str(action)}

    @staticmethod
    def from_dict(env_args: dict) -> "BFCLEnvironment":
        return BFCLEnvironment(
            base_url=env_args.get("base_url", "http://localhost:8801"),
            env_type=env_args.get("env_type", "bfcl"),
            max_steps=env_args.get("max_steps", 10),
            task_id=env_args.get("task_id")
        )
