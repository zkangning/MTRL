"""
AWM Environment for RLLM

This environment wraps AWM-generated virtual environments (FastAPI + MCP Server)
for use in RLLM's agentic RL training pipeline.

Server lifecycle follows awm/core/env.py::test_run_specific_env():
  1. Write env_config to temp jsonl
  2. Copy / create database to temp dir
  3. Launch `python -m awm.core.server` as subprocess
  4. Wait for server readiness via HTTP + MCP tool verification

NOTE on threading:  Original AWM uses ProcessPoolExecutor (each env in its own
process).  RLLM uses ThreadPoolExecutor (all envs share one process).  Therefore
awm/tools.py::wait_for_server() (which calls asyncio.run() in a loop) cannot be
used directly — it creates event-loop-bound asyncio.Lock conflicts.  We use a
pure-sync HTTP readiness check instead, then verify MCP via AWMMCPConnectionManager.
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
from awm.tools import (
    normalize_scenario_name as normalize_awm_name,
    get_random_available_port,
    tools_jsonl_save,
)

logger = logging.getLogger(__name__)

# Thread lock to protect isolated_mcp_env (which mutates os.environ globally)
_MCP_ENV_LOCK = threading.Lock()


class AWMMCPConnectionManager:
    """
    Manages connection to AWM MCP Server.

    Mirrors awm.core.agent.MCPToolExecutor exactly:
    - __init__ creates MCPApp + Agent inside isolated_mcp_env (needs lock)
    - list_tools / call_tool: each call does `async with app.run()` + `async with agent`
    - contextlib.redirect_stderr suppresses mcp_agent internal log noise
    """

    def __init__(self, mcp_url: str, timeout: float = 60.0):
        from mcp_agent.app import MCPApp
        from mcp_agent.agents.agent import Agent
        from mcp_agent.config import Settings, MCPSettings, MCPServerSettings, LoggerSettings
        from awm.tools import isolated_mcp_env

        self.mcp_url = mcp_url
        self.timeout = timeout
        self._tools: List[Dict] = []

        # isolated_mcp_env mutates os.environ globally -> must hold lock
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

    Server lifecycle strictly follows awm/core/env.py::test_run_specific_env().
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
        server_start_timeout: float = 60.0,
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
        self.server_log_file = None
        self.server_log_path: Optional[str] = None
        self.temp_dir: Optional[str] = None
        self.mcp_manager: Optional[AWMMCPConnectionManager] = None
        self.current_db_path: Optional[str] = None

        # State tracking
        self.current_step = 0
        self.history: List[Dict[str, Any]] = []
        self.done = False
        self.available_tools: List[Dict] = []

    # ------------------------------------------------------------------
    # Server lifecycle — mirrors awm/core/env.py::test_run_specific_env()
    # ------------------------------------------------------------------

    def _read_server_log(self, max_bytes: int = 50000) -> str:
        """
        Read the server log file content, flushing the write handle first.
        
        Args:
            max_bytes: Maximum bytes to read (default 50KB). Set to -1 for no limit.
        """
        try:
            if self.server_log_file and not self.server_log_file.closed:
                self.server_log_file.flush()
            if self.server_log_path and os.path.exists(self.server_log_path):
                file_size = os.path.getsize(self.server_log_path)
                with open(self.server_log_path, "r") as f:
                    content = f.read()
                    if max_bytes > 0 and len(content) > max_bytes:
                        # Show last N bytes when truncating
                        truncated_msg = f"\n... [TRUNCATED: showing last {max_bytes} of {file_size} bytes] ...\n"
                        return truncated_msg + content[-max_bytes:]
                    return content
        except Exception as e:
            return f"<failed to read server log: {e}>"
        return "<no server log available>"

    def _get_diagnostic_info(self) -> str:
        """
        Collect comprehensive diagnostic information for debugging server startup issues.
        """
        diag = []
        diag.append(f"\n{'=' * 60}")
        diag.append(f"AWM SERVER DIAGNOSTIC REPORT")
        diag.append(f"Scenario: {self.scenario_name}")
        diag.append(f"{'=' * 60}")
        
        # Server process info
        diag.append(f"\n[PROCESS INFO]")
        if self.server_process:
            poll_result = self.server_process.poll()
            diag.append(f"  PID: {self.server_process.pid}")
            diag.append(f"  Status: {'running' if poll_result is None else f'exited (code={poll_result})'}")
        else:
            diag.append(f"  No server process")
        
        # Network info
        diag.append(f"\n[NETWORK INFO]")
        diag.append(f"  Host: {self.server_host}")
        diag.append(f"  Port: {self.server_port}")
        
        # Check if port is in use
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                result = s.connect_ex((self.server_host, self.server_port))
                diag.append(f"  Port reachable: {'yes' if result == 0 else 'no'}")
        except Exception as e:
            diag.append(f"  Port check error: {e}")
        
        # Temp directory info
        diag.append(f"\n[TEMP DIRECTORY]")
        diag.append(f"  Path: {self.temp_dir}")
        if self.temp_dir and os.path.exists(self.temp_dir):
            diag.append(f"  Exists: yes")
            try:
                files = os.listdir(self.temp_dir)
                diag.append(f"  Contents: {files}")
                for f in files:
                    fpath = os.path.join(self.temp_dir, f)
                    if os.path.isfile(fpath):
                        diag.append(f"    - {f}: {os.path.getsize(fpath)} bytes")
            except Exception as e:
                diag.append(f"  Error listing: {e}")
        else:
            diag.append(f"  Exists: no")
        
        # Database info
        diag.append(f"\n[DATABASE INFO]")
        diag.append(f"  Path: {self.current_db_path}")
        if self.current_db_path and os.path.exists(self.current_db_path):
            diag.append(f"  Exists: yes")
            diag.append(f"  Size: {os.path.getsize(self.current_db_path)} bytes")
        else:
            diag.append(f"  Exists: no")
        
        # Environment code info
        diag.append(f"\n[ENV CODE INFO]")
        diag.append(f"  Code length: {len(self.env_code)} chars")
        if self.env_code:
            diag.append(f"  Has 'uvicorn.run': {'yes' if 'uvicorn.run' in self.env_code else 'NO - CRITICAL!'}")
            diag.append(f"  Has 'create_engine': {'yes' if 'create_engine' in self.env_code else 'no'}")
            diag.append(f"  Has 'FastAPI': {'yes' if 'FastAPI' in self.env_code else 'no'}")
        
        diag.append(f"\n{'=' * 60}")
        
        return "\n".join(diag)

    def _wait_for_server_http(self, port: int, timeout: float) -> bool:
        """
        Wait for the FastAPI server to be ready using pure-sync HTTP checks.

        Unlike awm/tools.py::wait_for_server() which uses asyncio.run() (incompatible
        with ThreadPoolExecutor), this uses only stdlib urllib — no asyncio, no
        event-loop-bound locks, fully thread-safe.

        Check flow: TCP connect -> HTTP GET /docs -> return True.
        Also checks for early process exit.
        """
        import socket
        import urllib.request
        import urllib.error

        start_time = time.time()
        tcp_ready = False
        http_attempts = 0
        last_http_error = None
        
        logger.info(f"[{self.scenario_name}] Waiting for server HTTP readiness (timeout={timeout}s)...")

        while time.time() - start_time < timeout:
            elapsed = time.time() - start_time
            
            # Early crash detection
            if self.server_process and self.server_process.poll() is not None:
                rc = self.server_process.returncode
                log_content = self._read_server_log()
                diag_info = self._get_diagnostic_info()
                logger.error(
                    f"[{self.scenario_name}] SERVER PROCESS CRASHED (exit code {rc}) after {elapsed:.1f}s on port {port}.\n"
                    f"{diag_info}\n"
                    f"--- SERVER LOG (server.log) ---\n{log_content}\n--- END SERVER LOG ---"
                )
                return False

            if not tcp_ready:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(1.0)
                        s.connect((self.server_host, port))
                        tcp_ready = True
                        logger.info(f"[{self.scenario_name}] TCP port {port} is open after {elapsed:.1f}s, checking HTTP...")
                except (socket.timeout, ConnectionRefusedError, OSError) as e:
                    # Log progress every 10 seconds
                    if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                        logger.debug(f"[{self.scenario_name}] Still waiting for TCP port {port} ({elapsed:.0f}s elapsed): {e}")
                    time.sleep(0.3)
                    continue

            # HTTP health check — /docs is always available on FastAPI
            http_attempts += 1
            try:
                url = f"http://{self.server_host}:{port}/docs"
                req = urllib.request.Request(url, method='GET')
                with urllib.request.urlopen(req, timeout=3) as resp:
                    if resp.status == 200:
                        logger.info(f"[{self.scenario_name}] Server HTTP ready on port {port} after {elapsed:.1f}s ({http_attempts} HTTP attempts)")
                        return True
            except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as e:
                last_http_error = str(e)
                # Log progress every 10 seconds or every 5 attempts
                if http_attempts % 5 == 0 or int(elapsed) % 10 == 0:
                    logger.debug(f"[{self.scenario_name}] HTTP check attempt {http_attempts} failed ({elapsed:.0f}s elapsed): {e}")
                time.sleep(0.5)

        # Timeout - collect comprehensive diagnostic info
        elapsed = time.time() - start_time
        log_content = self._read_server_log()
        diag_info = self._get_diagnostic_info()
        
        proc_status = "running" if (self.server_process and self.server_process.poll() is None) else "exited"
        if self.server_process and self.server_process.poll() is not None:
            proc_status = f"exited (code={self.server_process.returncode})"
        
        logger.error(
            f"[{self.scenario_name}] SERVER STARTUP TIMEOUT after {elapsed:.1f}s\n"
            f"  - Process status: {proc_status}\n"
            f"  - TCP ready: {tcp_ready}\n"
            f"  - HTTP attempts: {http_attempts}\n"
            f"  - Last HTTP error: {last_http_error}\n"
            f"{diag_info}\n"
            f"--- SERVER LOG (server.log) ---\n{log_content}\n--- END SERVER LOG ---"
        )
        return False

    def _start_server(self):
        """
        Start AWM MCP server — follows awm/core/env.py::test_run_specific_env():

        1. Create temp dir
        2. Prepare database (copy or create from schema)
        3. Write env_config as jsonl (so awm.core.server can read it)
        4. Launch `python -m awm.core.server` as subprocess
        5. Sleep 3s then check if process crashed (same as original)
        6. Wait for HTTP readiness (thread-safe, no asyncio)
        7. Verify MCP connectivity via AWMMCPConnectionManager.list_tools()
        """
        scenario_norm = normalize_awm_name(self.scenario_name)
        self.temp_dir = tempfile.mkdtemp(prefix=f"awm_env_{scenario_norm}_")

        # ── Step 1: Prepare database ──
        if self.db_path and os.path.exists(self.db_path):
            self.current_db_path = os.path.join(self.temp_dir, f"{scenario_norm}.db")
            shutil.copyfile(self.db_path, self.current_db_path)
            os.chmod(self.current_db_path, 0o644)
            logger.info(f"[{self.scenario_name}] Copied existing database from {self.db_path}")
        elif self.db_schema:
            logger.info(f"[{self.scenario_name}] Creating database from schema...")
            full_schema = self.db_schema.copy()
            if self.db_sample:
                for table in full_schema.get("tables", []):
                    table_name = table.get("name")
                    if table_name and table_name in self.db_sample:
                        table["examples"] = self.db_sample[table_name]

            db_path, successful, failed, errors = create_sqlite_database(
                self.scenario_name, full_schema, self.temp_dir
            )
            self.current_db_path = db_path
            if failed > 0:
                logger.warning(f"[{self.scenario_name}] Database creation had {failed} failures: {errors}")
        else:
            raise ValueError("Either db_path or db_schema must be provided for AWMEnvironment")

        # ── Step 2: Write env_config as jsonl (same format as awm/core/env.py) ──
        env_config = {
            "scenario": self.scenario_name,
            "db_path": self.current_db_path,
            "full_code": self.env_code,
        }
        temp_env_json = os.path.join(self.temp_dir, "env_config.jsonl")
        tools_jsonl_save([env_config], temp_env_json)

        # ── Step 3: Allocate port ──
        self.server_port = get_random_available_port()

        # Temp server path for awm.core.server to write the modified code
        temp_server_path = os.path.join(self.temp_dir, "temp_server.py")

        # ── Step 4: Launch subprocess — identical to awm/core/env.py:161-174 ──
        logger.info(f"[{self.scenario_name}] Starting AWM server on port {self.server_port}...")

        self.server_log_path = os.path.join(self.temp_dir, "server.log")
        self.server_log_file = open(self.server_log_path, "w")

        self.server_process = subprocess.Popen(
            [
                sys.executable, '-m', 'awm.core.server',
                '--port', str(self.server_port),
                '--scenario', scenario_norm,
                '--db_path', self.current_db_path,
                '--temp_server_path', temp_server_path,
                '--envs_load_path', temp_env_json,
            ],
            stdout=self.server_log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )

        logger.info(f"[{self.scenario_name}] Server process started (pid={self.server_process.pid})")

        # ── Step 5: Initial crash check (same 3s sleep as original) ──
        logger.info(f"[{self.scenario_name}] Waiting 3s for initial startup...")
        time.sleep(3)

        if self.server_process.poll() is not None:
            rc = self.server_process.returncode
            log_content = self._read_server_log()
            diag_info = self._get_diagnostic_info()
            logger.error(
                f"[{self.scenario_name}] SERVER CRASHED ON STARTUP (exit code {rc})\n"
                f"{diag_info}\n"
                f"--- SERVER LOG (server.log) ---\n{log_content}\n--- END SERVER LOG ---"
            )
            self._kill_server_process()
            raise RuntimeError(
                f"AWM server crashed on startup for '{self.scenario_name}' (exit code {rc}). "
                f"Temp dir preserved for debugging: {self.temp_dir}. "
                f"Check server.log in temp dir for details."
            )

        # ── Step 6: Wait for HTTP readiness (thread-safe, no asyncio) ──
        if not self._wait_for_server_http(self.server_port, self.server_start_timeout):
            # Note: _wait_for_server_http already logs detailed diagnostic info on failure
            self._kill_server_process()
            raise RuntimeError(
                f"AWM server HTTP not ready within {self.server_start_timeout}s "
                f"for scenario '{self.scenario_name}'. Temp dir: {self.temp_dir}. "
                f"Check the logs above for detailed diagnostic information."
            )

        # ── Step 7: Initialize MCP connection manager & verify tools ──
        mcp_url = f"http://{self.server_host}:{self.server_port}/mcp"
        self.mcp_manager = AWMMCPConnectionManager(mcp_url)

        # Verify MCP connectivity with retry
        max_retries = 3
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                tools = self._run_async(self.mcp_manager.list_tools())
                if tools:
                    logger.info(
                        f"[{self.scenario_name}] MCP verified: {len(tools)} tools on port {self.server_port}"
                    )
                    return  # Success!
                else:
                    last_error = "list_tools returned empty"
                    logger.warning(f"[{self.scenario_name}] MCP list_tools empty (attempt {attempt}/{max_retries})")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"[{self.scenario_name}] MCP verify attempt {attempt}/{max_retries} failed: {e}")

            if attempt < max_retries:
                time.sleep(1.0 * attempt)

        # All retries exhausted
        self._kill_server_process()
        raise RuntimeError(
            f"MCP connection failed after {max_retries} attempts for '{self.scenario_name}': {last_error}. "
            f"Temp dir: {self.temp_dir}"
        )

    # ------------------------------------------------------------------
    # Process management
    # ------------------------------------------------------------------

    def _kill_server_process(self):
        """Kill server process only (preserve temp dir for debugging)."""
        if self.server_process:
            try:
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                self.server_process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
                except Exception:
                    pass
            self.server_process = None

        if self.server_log_file:
            try:
                self.server_log_file.close()
            except Exception:
                pass
            self.server_log_file = None

    def _cleanup_server(self):
        """Clean up server process and temporary files."""
        self._kill_server_process()

        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except Exception:
                pass
            self.temp_dir = None

        self.mcp_manager = None
        self.server_port = None
        self.current_db_path = None

    # ------------------------------------------------------------------
    # Gym-like interface
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Tool call parsing & execution — mirrors awm/core/agent.py
    # ------------------------------------------------------------------

    def _parse_tool_calls(self, action: str) -> List[Dict[str, Any]]:
        """Parse tool calls from agent response — same as awm.core.agent.parse_tool_calls()."""
        import re
        from awm.tools import tools_robust_json_loads

        tool_calls = []
        pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        matches = re.findall(pattern, action, re.DOTALL)

        for i, match in enumerate(matches):
            data = tools_robust_json_loads(match.strip())
            if not data:
                logger.warning(f"Failed to parse tool call JSON: {match[:100]}")
                continue

            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], dict):
                    data = data[0]
                else:
                    continue

            if not isinstance(data, dict):
                continue

            name = data.get("name", "")
            arguments = data.get("arguments", {})

            # Handle mcp_tool_ prefix (same as original agent.py:120-125)
            if name.startswith("mcp_tool_"):
                arguments = {
                    "tool_name": name,
                    "arguments": arguments if arguments else {},
                }
                name = "call_tool"

            tool_calls.append({
                "id": f"call_{int(time.time() * 1000)}_{i}",
                "name": name,
                "arguments": arguments,
            })

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
        Execute a tool call — mirrors awm/core/agent.py::run_agent() tool dispatch.
        """
        name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})
        tool_call_id = tool_call.get("id", "")

        if name == "list_tools":
            try:
                tools = self._run_async(self.mcp_manager.list_tools())
                self.available_tools = tools
                # Use awm.core.agent.format_tools_for_response for formatting
                from awm.core.agent import format_tools_for_response
                formatted_tools = format_tools_for_response(tools)
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
        Parse call_tool arguments — identical to awm.core.agent.parse_call_tool_arguments().
        """
        from awm.tools import tools_robust_json_loads

        if isinstance(arguments, str):
            arguments = tools_robust_json_loads(arguments)
        if not isinstance(arguments, dict):
            return "", {}

        tool_name = arguments.get("tool_name", "")
        inner_args = arguments.get("arguments", {})

        if tool_name.startswith("mcp_tool_"):
            tool_name = tool_name[len("mcp_tool_"):]

        if isinstance(inner_args, str):
            parsed = tools_robust_json_loads(inner_args) if inner_args.strip() else {}
            if isinstance(parsed, dict):
                inner_args = parsed
            else:
                inner_args = {}

        if not isinstance(inner_args, dict):
            inner_args = {}

        return tool_name, inner_args

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

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
        """
        # Pop base env_args (from trainer's env_args)
        reward_fn = env_args.pop("reward_fn", None)
        server_host = env_args.pop("server_host", "127.0.0.1")
        server_start_timeout = env_args.pop("server_start_timeout", 60.0)

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
                if "original_data" in extra_info and "db_schema" not in extra_info:
                    try:
                        original = json.loads(extra_info["original_data"])
                        if isinstance(original, dict):
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

        task_description = _get("task", "prompt") or ""
        scenario_name = _get("scenario") or "unknown"
        env_code = _get("env_code") or ""

        db_schema = _get("db_schema", "schema")
        if isinstance(db_schema, str):
            try:
                db_schema = json.loads(db_schema)
            except (json.JSONDecodeError, TypeError):
                pass

        db_sample = _get("db_sample", "sample_data")
        if isinstance(db_sample, str):
            try:
                db_sample = json.loads(db_sample)
            except (json.JSONDecodeError, TypeError):
                pass

        verifier_code = _get("verifier_code") or ""
        max_steps = _get("max_steps", "task_max_steps") or 30

        return AWMEnvironment(
            scenario_name=scenario_name,
            task_description=task_description,
            env_code=env_code,
            db_path=None,
            db_schema=db_schema,
            db_sample=db_sample,
            verifier_code=verifier_code,
            max_steps=int(max_steps),
            reward_fn=reward_fn,
            server_host=server_host,
            server_start_timeout=server_start_timeout,
        )
