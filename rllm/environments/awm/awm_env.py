"""
AWM Environment for RLLM

This environment wraps AWM-generated virtual environments (FastAPI + MCP Server)
for use in RLLM's agentic RL training pipeline.
"""

import asyncio
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


class AWMMCPConnectionManager:
    """
    Manages connection to AWM MCP Server.
    """
    
    def __init__(self, mcp_url: str, timeout: float = 60.0):
        self.mcp_url = mcp_url
        self.timeout = timeout
        self._tools: List[Dict] = []
        self._app = None
        self._agent = None
        
    async def initialize(self):
        """Initialize connection to MCP server and fetch tools."""
        try:
            from mcp_agent.app import MCPApp
            from mcp_agent.agents.agent import Agent
            from mcp_agent.config import Settings, MCPSettings, MCPServerSettings, LoggerSettings
            from awm.tools import isolated_mcp_env
            
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
                        
        except Exception as e:
            logger.error(f"Failed to initialize AWM MCP connection: {e}")
            raise
    
    async def list_tools(self) -> List[Dict]:
        """List available tools from MCP server."""
        return self._tools
    
    async def call_tool(self, tool_name: str, arguments: Dict) -> str:
        """Call a tool on the MCP server."""
        try:
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
        import socket
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1.0)
                    s.connect((self.server_host, port))
                    return True
            except (socket.timeout, ConnectionRefusedError):
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
        
        # Wait for server to be ready
        if not self._wait_for_server(self.server_port, self.server_start_timeout):
            self._cleanup_server()
            raise RuntimeError(f"AWM MCP server failed to start within {self.server_start_timeout}s")
        
        # Initialize MCP connection
        mcp_url = f"http://{self.server_host}:{self.server_port}/mcp"
        self.mcp_manager = AWMMCPConnectionManager(mcp_url)
        
        # Initialize connection (run async init in sync context)
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.mcp_manager.initialize())
            loop.close()
        except Exception as e:
            self._cleanup_server()
            raise RuntimeError(f"Failed to initialize MCP connection: {e}")
    
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
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                tools = loop.run_until_complete(self.mcp_manager.list_tools())
                loop.close()
                self.available_tools = tools
                # Use the awm_prompts formatter for consistency
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
            # Parse arguments matching native implementation (parse_call_tool_arguments)
            tool_name, tool_args = self._parse_call_tool_arguments(arguments)
            
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.mcp_manager.call_tool(tool_name, tool_args))
                loop.close()
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
        AWM environments are NOT multithread-safe.
        
        Each AWM task requires starting an independent FastAPI+MCP subprocess server
        with unique port assignment. The server lifecycle (start/stop/cleanup) involves
        OS-level process management that must not be shared across threads.
        """
        return False

    @staticmethod
    def from_dict(env_args: Dict[str, Any]) -> "AWMEnvironment":
        """
        Create AWMEnvironment from task dictionary.
        
        Called by the execution engine with: {**task, **self.env_args}
        where task = {prompt, response, task_type, data_source, extra_info}
        and env_args = {reward_fn, max_steps, server_host, server_start_timeout, ...}
        
        The pop() pattern is consistent with other rllm environments (e.g. ToolEnvironment).
        Since the dict is a fresh merge ({**task, **self.env_args}), pop() is safe.
        """
        # Pop env_args-level parameters (from trainer's env_args)
        reward_fn = env_args.pop("reward_fn", None)
        server_host = env_args.pop("server_host", "127.0.0.1")
        server_start_timeout = env_args.pop("server_start_timeout", 30.0)
        
        # Parse extra_info from task data
        extra_info = env_args.pop("extra_info", "{}")
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except json.JSONDecodeError:
                extra_info = {}
        
        # Get task description from prompt (standard field) or extra_info
        task_description = env_args.pop("prompt", "")
        if not task_description:
            task_description = extra_info.get("task", "")
        
        # Extract scenario name
        scenario_name = extra_info.get("scenario", "unknown")
        
        # Extract environment code
        env_code = extra_info.get("env_code", "")
        
        # Extract database info (support both field naming conventions)
        db_schema = extra_info.get("db_schema") or extra_info.get("schema")
        db_sample = extra_info.get("db_sample") or extra_info.get("sample_data")
        
        # Extract verifier code
        verifier_code = extra_info.get("verifier_code")
        
        # max_steps priority: extra_info.task_max_steps > extra_info.max_steps > default 30
        max_steps = extra_info.get("task_max_steps", extra_info.get("max_steps", 30))
        
        return AWMEnvironment(
            scenario_name=scenario_name,
            task_description=task_description,
            env_code=env_code,
            db_path=None,  # DB is created dynamically from schema during reset()
            db_schema=db_schema,
            db_sample=db_sample,
            verifier_code=verifier_code,
            max_steps=max_steps,
            reward_fn=reward_fn,
            server_host=server_host,
            server_start_timeout=server_start_timeout,
        )
