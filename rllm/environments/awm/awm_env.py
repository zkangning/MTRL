"""
AWM Environment for RLLM

This environment wraps AWM-generated virtual environments (FastAPI + MCP Server)
for use in RLLM's agentic RL training pipeline.
"""

import asyncio
import contextlib
import io
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from rllm.environments.base.base_env import BaseEnv
from rllm.rewards.reward_types import RewardOutput

# AWM core imports
from awm.core.db import create_sqlite_database
from awm.core.server import format_raw_code_to_lines
from awm.tools import normalize_scenario_name as normalize_awm_name

logger = logging.getLogger(__name__)

# Thread lock to protect isolated_mcp_env (which mutates os.environ globally)
_MCP_ENV_LOCK = threading.Lock()


class AWMMCPConnectionManager:
    """
    Manages connection to AWM MCP Server.
    
    与 awm.core.agent.MCPToolExecutor 对齐:
    - __init__ 中创建 MCPApp 和 Agent（在 isolated_mcp_env 中，需持锁）
    - list_tools / call_tool 每次都 async with app.run() + async with agent
    - 使用 contextlib.redirect_stderr 屏蔽 mcp_agent 内部日志噪音
    """
    
    def __init__(self, mcp_url: str, timeout: float = 60.0):
        from mcp_agent.app import MCPApp
        from mcp_agent.agents.agent import Agent
        from mcp_agent.config import Settings, MCPSettings, MCPServerSettings, LoggerSettings
        from awm.tools import isolated_mcp_env
        
        self.mcp_url = mcp_url
        self.timeout = timeout
        self._tools: List[Dict] = []
        
        # 创建 MCPApp/Agent 需要在 isolated_mcp_env 中，
        # 且 isolated_mcp_env 修改 os.environ，必须加锁
        with _MCP_ENV_LOCK:
            with isolated_mcp_env():
                settings = Settings(
                    execution_engine="asyncio",
                    logger=LoggerSettings(
                        type="none",
                        transports=["none"],
                        progress_display=False,
                        level="error",
                    ),
                    mcp=MCPSettings(
                        servers={
                            "mcp_server": MCPServerSettings(
                                transport='streamable_http',
                                url=self.mcp_url,
                            ),
                        }
                    ),
                )
                self._app = MCPApp(name="awm_agent", settings=settings)
                self._agent = Agent(name="executor", server_names=["mcp_server"])
    
    async def list_tools(self) -> List[Dict]:
        """List available tools from MCP server."""
        if self._tools:
            return self._tools
        
        with contextlib.redirect_stderr(io.StringIO()):
            async with self._app.run():
                async with self._agent:
                    result = await asyncio.wait_for(
                        self._agent.list_tools(), timeout=self.timeout
                    )
                    self._tools = []
                    for t in result.tools:
                        tool_info = {
                            "name": t.name,
                            "description": t.description or "",
                            "inputSchema": t.inputSchema or {},
                        }
                        self._tools.append(tool_info)
                    logger.info(f"AWM MCP: Loaded {len(self._tools)} tools from {self.mcp_url}")
                    return self._tools
    
    async def call_tool(self, tool_name: str, arguments: Dict) -> str:
        """Call a tool on the MCP server."""
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                async with self._app.run():
                    async with self._agent:
                        result = await asyncio.wait_for(
                            self._agent.call_tool(tool_name, arguments),
                            timeout=self.timeout,
                        )
                        # Extract text from result content
                        parts = []
                        for c in result.content:
                            if hasattr(c, 'text'):
                                parts.append(c.text)
                            else:
                                parts.append(str(c))
                        
                        text = "\n".join(parts)
                        
                        if result.isError:
                            return f"Error: {text}"
                        return text
                    
        except asyncio.TimeoutError:
            return f"Error: Tool call timed out after {self.timeout}s"
        except Exception as e:
            return f"Error: {e}"


class AWMEnvironment(BaseEnv):
    """
    AWM Environment for RLLM Agentic RL Training.
    """
    
    AWM_SYSTEM_PROMPT = """You are an AI assistant interacting with a virtual environment through tools.

Your goal is to complete the given task by using the available tools. You should:
1. First call `list_tools` to see what tools are available
2. Then use `call_tool` to interact with the environment
3. Analyze the results and plan your next steps
4. Continue until the task is complete

## Available Meta-Tools

1. **list_tools**
   - Description: List all available MCP tools for the current environment
   - Arguments: None
   - Output: A list of environment-specific tools and their descriptions

2. **call_tool**
   - Description: Call an environment-specific tool
   - Arguments:
     - tool_name: str, the name of the tool to call (without mcp_tool_ prefix)
     - arguments: dict, the arguments for the tool
   - Output: The result of the tool call

## Response Format

For each step, respond with:
1. Your reasoning inside <think> </think> tags
2. A tool call inside <tool_call> </tool_call> tags in JSON format:

<tool_call>
{"name": "list_tools", "arguments": {}}
</tool_call>

or

<tool_call>
{"name": "call_tool", "arguments": {"tool_name": "search_products", "arguments": {"query": "laptop"}}}
</tool_call>

When you have completed the task, provide your final answer directly without any tool calls.
"""
    
    def __init__(
        self,
        scenario_name: str,
        task_description: str,
        env_code: str,
        db_path: Optional[str] = None,
        db_schema: Optional[dict] = None,
        db_sample: Optional[dict] = None,
        verifier_code: Optional[str] = None,
        max_steps: int = 30,
        reward_fn=None,
        server_host: str = "127.0.0.1",
        server_start_timeout: float = 30.0,
        **kwargs
    ):
        super().__init__()
        
        self.scenario_name = scenario_name
        self.task_description = task_description
        self.env_code = env_code
        self.db_path = db_path
        self.db_schema = db_schema
        self.db_sample = db_sample
        self.verifier_code = verifier_code
        self.max_steps = max_steps
        self.reward_fn = reward_fn
        self.server_host = server_host
        self.server_start_timeout = server_start_timeout
        
        # Server management
        self.server_port: Optional[int] = None
        self.server_process: Optional[subprocess.Popen] = None
        self.temp_dir: Optional[str] = None
        self.mcp_manager: Optional[AWMMCPConnectionManager] = None
        self.current_db_path: Optional[str] = None
        
        # State tracking
        self.current_step = 0
        self.history: List[Dict[str, Any]] = []
        self.done = False
        self.available_tools: List[Dict] = []
        
    def _get_random_port(self) -> int:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    def _wait_for_server(self, port: int, timeout: float = 30.0) -> bool:
        """
        Wait for the FastAPI server to be fully ready.
        
        使用 HTTP GET 请求检查（而非仅 TCP socket），确保 FastAPI 已完成路由挂载。
        先用 TCP socket 快速检测端口绑定，再用 HTTP 请求验证应用就绪。
        """
        import socket
        import urllib.request
        import urllib.error
        
        start_time = time.time()
        tcp_ready = False
        
        while time.time() - start_time < timeout:
            # 检查进程是否已经退出（提前失败检测）
            if self.server_process and self.server_process.poll() is not None:
                logger.error(f"Server process exited with code {self.server_process.returncode}")
                return False
            
            if not tcp_ready:
                # Phase 1: TCP port check (fast)
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(1.0)
                        s.connect((self.server_host, port))
                        tcp_ready = True
                        logger.debug(f"TCP port {port} is open, checking HTTP readiness...")
                except (socket.timeout, ConnectionRefusedError, OSError):
                    time.sleep(0.3)
                    continue
            
            # Phase 2: HTTP health check — 请求 /docs (FastAPI 自带) 来确认应用就绪
            try:
                url = f"http://{self.server_host}:{port}/docs"
                req = urllib.request.Request(url, method='GET')
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        logger.debug(f"Server on port {port} is ready (HTTP 200)")
                        return True
            except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
                time.sleep(0.5)
        
        return False
    
    def _start_server(self):
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix=f"awm_env_{normalize_awm_name(self.scenario_name)}_")
        
        # Handle database initialization
        if self.db_path and os.path.exists(self.db_path):
            # Copy existing database
            self.current_db_path = os.path.join(self.temp_dir, "environment.db")
            shutil.copy2(self.db_path, self.current_db_path)
            os.chmod(self.current_db_path, 0o644)
            logger.info(f"Using existing database copied from {self.db_path}")
        elif self.db_schema:
            # Create database from schema using awm.core.db
            logger.info(f"Creating fresh database from schema for {self.scenario_name}...")
            # We combine db_schema and db_sample into one schema dict if they are separate
            full_schema = self.db_schema.copy()
            if self.db_sample:
                # Add sample data to tables
                for table in full_schema.get("tables", []):
                    table_name = table.get("name")
                    if table_name in self.db_sample:
                        table["examples"] = self.db_sample[table_name]
            
            db_path, successful, failed, errors = create_sqlite_database(
                self.scenario_name, full_schema, self.temp_dir
            )
            self.current_db_path = db_path
            if failed > 0:
                logger.warning(f"Database creation had {failed} failures: {errors}")
        else:
            raise ValueError("Either db_path or db_schema must be provided for AWMEnvironment")
        
        # Get available port
        self.server_port = self._get_random_port()
        
        # Modify environment code using shared logic
        modified_code = self._modify_env_code(self.env_code, self.current_db_path, self.server_port)
        
        # Write server code to temp file
        server_path = os.path.join(self.temp_dir, "server.py")
        with open(server_path, 'w') as f:
            f.write(modified_code)
        
        # Start server process
        logger.info(f"Starting AWM MCP server for {self.scenario_name} on port {self.server_port}...")
        
        # 使用独立的环境变量副本 — 必须加锁，因为其他线程的
        # isolated_mcp_env() 可能正在修改 os.environ
        with _MCP_ENV_LOCK:
            env = os.environ.copy()
        env['PORT'] = str(self.server_port)
        env['HOST'] = self.server_host
        env['DATABASE_PATH'] = f"sqlite:///{self.current_db_path}"
        
        self.server_process = subprocess.Popen(
            [sys.executable, server_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True
        )
        
        # Wait for server to be ready (HTTP-level check)
        if not self._wait_for_server(self.server_port, self.server_start_timeout):
            # 尝试读取 server 输出来帮助调试
            if self.server_process:
                try:
                    stdout, _ = self.server_process.communicate(timeout=2)
                    if stdout:
                        logger.error(f"Server stdout:\n{stdout[:2000]}")
                except:
                    pass
            self._cleanup_server()
            raise RuntimeError(f"AWM MCP server failed to start within {self.server_start_timeout}s")
        
        # Initialize MCP connection manager (constructor acquires _MCP_ENV_LOCK)
        mcp_url = f"http://{self.server_host}:{self.server_port}/mcp"
        self.mcp_manager = AWMMCPConnectionManager(mcp_url)
        
        # Fetch tools (with retry) to verify MCP endpoint is working
        max_retries = 3
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    tools = loop.run_until_complete(self.mcp_manager.list_tools())
                finally:
                    loop.close()
                
                if tools:
                    logger.info(f"MCP connection verified: {len(tools)} tools on port {self.server_port}")
                    break
                else:
                    last_error = "list_tools returned empty"
                    logger.warning(f"MCP list_tools returned empty (attempt {attempt}/{max_retries})")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"MCP connection attempt {attempt}/{max_retries} failed: {e}")
            
            if attempt < max_retries:
                time.sleep(1.0 * attempt)  # progressive backoff
        else:
            self._cleanup_server()
            raise RuntimeError(f"Failed to initialize MCP connection after {max_retries} attempts: {last_error}")
    
    def _modify_env_code(self, code: str, db_path: str, port: int) -> str:
        """
        Modify generated environment code using awm.core.server patterns.
        """
        new_code = ['import warnings', 'warnings.filterwarnings("ignore", category=DeprecationWarning)']
        
        for line in code.split("\n"):
            # Replace database connection
            if 'create_engine(' in line:
                left = line.split('create_engine(')[0]
                sql_path = f"'sqlite:///{db_path}'"
                right = f"create_engine({sql_path}, connect_args={{'check_same_thread': False}})"
                line = f"{left}{right}"
            
            # Modify uvicorn.run to use environment variables and mount MCP
            if 'uvicorn.run(app' in line:
                # Use format_raw_code_to_lines from awm.core.server
                raw_code = f"""
                import os
                host = os.environ.get('HOST', '{self.server_host}')
                port = os.environ.get('PORT', {port})
                print(f'Server starting on port={{port}}')
                """
                lines = format_raw_code_to_lines(raw_code, indent=4)
                
                raw_code_mcp = f"""
                from fastapi_mcp import FastApiMCP
                mcp = FastApiMCP(app)
                mcp.mount_http()
                print("MCP server enabled")
                """
                lines += format_raw_code_to_lines(raw_code_mcp, indent=4)
                
                new_code.extend(lines)
                line = f'    uvicorn.run(app, host=host, port=int(port))'
            
            new_code.append(line)
        
        return "\n".join(new_code)
    
    def _cleanup_server(self):
        """Clean up server process and temporary files."""
        if self.server_process:
            try:
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                self.server_process.wait(timeout=5)
            except:
                try:
                    os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
                except:
                    pass
            self.server_process = None
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except:
                pass
            self.temp_dir = None
        
        self.mcp_manager = None
        self.server_port = None
        self.current_db_path = None
    
    def reset(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._cleanup_server()
        
        self.current_step = 0
        self.history = []
        self.done = False
        self.available_tools = []
        
        self._start_server()
        
        observation = {
            "system_prompt": self.AWM_SYSTEM_PROMPT,
            "task": self.task_description,
            "scenario": self.scenario_name,
        }
        
        info = {
            "scenario": self.scenario_name,
            "task": self.task_description,
            "max_steps": self.max_steps,
        }
        
        return observation, info
    
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.current_step += 1
        tool_calls = self._parse_tool_calls(action)
        
        if not tool_calls:
            self.done = True
            reward = self._compute_final_reward(action)
            info = {"step": self.current_step, "action": action, "final_answer": action}
            return {}, reward, self.done, info
        
        results = []
        for tc in tool_calls:
            result = self._execute_tool_call(tc)
            results.append(result)
        
        if self.current_step >= self.max_steps:
            self.done = True
            reward = self._compute_final_reward(action)
        else:
            reward = 0.0
        
        observation = {"tool_results": results, "step": self.current_step}
        info = {"step": self.current_step, "action": action, "tool_calls": tool_calls, "results": results}
        self.history.append(info)
        
        return observation, reward, self.done, info
    
    def _parse_tool_calls(self, action: str) -> List[Dict[str, Any]]:
        import re
        tool_calls = []
        pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        matches = re.findall(pattern, action, re.DOTALL)
        for match in matches:
            try:
                data = json.loads(match.strip())
                if isinstance(data, list) and len(data) > 0:
                    data = data[0]
                if isinstance(data, dict):
                    tool_calls.append({"name": data.get("name", ""), "arguments": data.get("arguments", {})})
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool call JSON")
        return tool_calls
    
    def _run_async(self, coro):
        """Helper: run an async coroutine from sync context, safely managing the event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool call.
        
        Compatible with AWM native implementation (awm/core/agent.py).
        """
        name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})
        tool_call_id = tool_call.get("id", "")
        
        if name == "list_tools":
            try:
                tools = self._run_async(self.mcp_manager.list_tools())
                self.available_tools = tools
                from rllm.agents.awm_prompts import format_awm_tools_for_prompt
                formatted_tools = format_awm_tools_for_prompt(tools)
                return {
                    "tool": "list_tools",
                    "tool_call_id": tool_call_id,
                    "result": formatted_tools,
                    "success": True
                }
            except Exception as e:
                return {
                    "tool": "list_tools",
                    "tool_call_id": tool_call_id,
                    "result": f"Error listing tools: {e}",
                    "success": False
                }
        
        elif name == "call_tool":
            tool_name, tool_args = self._parse_call_tool_arguments(arguments)
            
            try:
                result = self._run_async(self.mcp_manager.call_tool(tool_name, tool_args))
                return {
                    "tool": tool_name,
                    "tool_call_id": tool_call_id,
                    "arguments": tool_args,
                    "result": result,
                    "success": not result.startswith("Error:")
                }
            except Exception as e:
                return {
                    "tool": tool_name,
                    "tool_call_id": tool_call_id,
                    "arguments": tool_args,
                    "result": f"Error: {e}",
                    "success": False
                }
        
        return {"tool": name, "tool_call_id": tool_call_id, "result": f"Unknown tool: {name}", "success": False}
    
    def _parse_call_tool_arguments(self, arguments: Any) -> tuple[str, dict]:
        """
        Parse call_tool arguments matching AWM native implementation.
        
        Args:
            arguments: Can be dict, str, or None
            
        Returns:
            Tuple of (tool_name, tool_args)
        """
        from awm.tools import tools_robust_json_loads
        
        if isinstance(arguments, str):
            arguments = tools_robust_json_loads(arguments)
        if not isinstance(arguments, dict):
            return "", {}
        
        tool_name = arguments.get("tool_name", "")
        inner_args = arguments.get("arguments", {})
        
        # Handle mcp_tool_ prefix
        if tool_name.startswith("mcp_tool_"):
            tool_name = tool_name[len("mcp_tool_"):]
        
        # Parse inner arguments if string
        if isinstance(inner_args, str):
            parsed = tools_robust_json_loads(inner_args) if inner_args.strip() else {}
            if isinstance(parsed, dict):
                inner_args = parsed
            else:
                inner_args = {}
        
        if not isinstance(inner_args, dict):
            inner_args = {}
        
        return tool_name, inner_args
    
    def _compute_final_reward(self, final_answer: str) -> float:
        if self.reward_fn:
            task_info = {
                "scenario": self.scenario_name,
                "task": self.task_description,
                "verifier_code": self.verifier_code,
                "db_path": self.current_db_path or self.db_path,
                "history": self.history,
                "final_answer": final_answer,
            }
            reward_output = self.reward_fn(task_info, final_answer)
            if isinstance(reward_output, RewardOutput):
                return reward_output.reward
            return float(reward_output)
        return 0.0
    
    def close(self):
        self._cleanup_server()
    
    @staticmethod
    def is_multithread_safe() -> bool:
        """
        AWM environments are multithread-safe.
        
        Each AWM task creates a fully independent environment instance via from_dict(),
        with its own temporary directory, random port, subprocess server, and MCP
        connection. There is no shared mutable state between instances, so concurrent
        execution across threads is safe.
        """
        return True

    @staticmethod
    def from_dict(env_args: Dict[str, Any]) -> "AWMEnvironment":
        """
        Create AWMEnvironment from task dictionary.
        
        【独立管道】数据流（绕开 DatasetRegistry / apply_verl_postprocessing）:
        
        1. load_awm_dataset() 生成:
           extra_info = {
               "index": int,
               "scenario": str, "task": str,
               "env_code": str,
               "db_schema": str (JSON), "db_sample": str (JSON),
               "verifier_code": str, "max_steps": int, "task_type": "awm"
           }
        
        2. init_envs_and_agents() 调用:
           env_args[i] = extra_info dict (parsed from JSON if str)
           self.env_class.from_dict({**env_args[i], **base_env_args})
        
        3. 因此本方法收到的 env_args 顶层 key 包括:
           - AWM 字段: scenario, task, env_code, db_schema, db_sample, verifier_code, ...
           - base_env_args: reward_fn, server_host, server_start_timeout, ...
        
        The pop() pattern is consistent with other rllm environments (e.g. ToolEnvironment).
        Since the dict is a fresh merge ({**task, **self.env_args}), pop() is safe.
        """
        # Pop base env_args (from trainer's env_args)
        reward_fn = env_args.pop("reward_fn", None)
        server_host = env_args.pop("server_host", "127.0.0.1")
        server_start_timeout = env_args.pop("server_start_timeout", 30.0)
        
        # ============================================================
        # 提取 AWM 字段 — 兼容两种格式:
        #   格式 A (独立管道): 字段直接在 env_args 顶层
        #   格式 B (旧管道):   字段嵌套在 env_args["extra_info"] 中
        # ============================================================
        extra_info = env_args.pop("extra_info", None)
        if extra_info is not None:
            if isinstance(extra_info, str):
                try:
                    extra_info = json.loads(extra_info)
                except json.JSONDecodeError:
                    extra_info = {}
            if isinstance(extra_info, dict):
                # 旧管道兼容: 如果 extra_info 里有 original_data，从中恢复
                if "original_data" in extra_info and "db_schema" not in extra_info:
                    try:
                        original = json.loads(extra_info["original_data"])
                        if isinstance(original, dict):
                            # original_data 中可能还有一层 extra_info
                            nested_extra = original.get("extra_info", {})
                            if isinstance(nested_extra, str):
                                nested_extra = json.loads(nested_extra)
                            if isinstance(nested_extra, dict) and "db_schema" in nested_extra:
                                extra_info = nested_extra
                            elif "db_schema" in original:
                                extra_info = original
                    except (json.JSONDecodeError, TypeError):
                        pass
        else:
            extra_info = {}
        
        # 合并: 顶层 env_args 的 AWM 字段优先于 extra_info 中的
        def _get(key, *alt_keys):
            """从 env_args 或 extra_info 中获取值，支持备选 key。"""
            val = env_args.pop(key, None)
            if val is not None:
                return val
            for k in alt_keys:
                val = env_args.pop(k, None)
                if val is not None:
                    return val
            val = extra_info.get(key)
            if val is not None:
                return val
            for k in alt_keys:
                val = extra_info.get(k)
                if val is not None:
                    return val
            return None

        # 提取任务描述
        task_description = _get("task", "prompt") or ""
        
        # 提取场景名
        scenario_name = _get("scenario") or "unknown"
        
        # 提取环境代码
        env_code = _get("env_code") or ""
        
        # 提取数据库 schema（可能是 JSON 字符串或 dict）
        db_schema = _get("db_schema", "schema")
        if isinstance(db_schema, str):
            try:
                db_schema = json.loads(db_schema)
            except (json.JSONDecodeError, TypeError):
                pass
        
        # 提取数据库样本数据
        db_sample = _get("db_sample", "sample_data")
        if isinstance(db_sample, str):
            try:
                db_sample = json.loads(db_sample)
            except (json.JSONDecodeError, TypeError):
                pass
        
        # 提取验证代码
        verifier_code = _get("verifier_code") or ""
        
        # max_steps
        max_steps = _get("max_steps", "task_max_steps") or 30
        
        return AWMEnvironment(
            scenario_name=scenario_name,
            task_description=task_description,
            env_code=env_code,
            db_path=None,  # DB is created dynamically from schema during reset()
            db_schema=db_schema,
            db_sample=db_sample,
            verifier_code=verifier_code,
            max_steps=int(max_steps),
            reward_fn=reward_fn,
            server_host=server_host,
            server_start_timeout=server_start_timeout,
        )
