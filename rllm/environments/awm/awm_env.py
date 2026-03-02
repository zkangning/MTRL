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
process).  RLLM uses ThreadPoolExecutor (all envs share one process).  Each
MCP operation (list_tools / call_tool) uses asyncio.run() to create a fresh
event loop — the mcp_agent library requires this; persistent background loops
cause the MCP initialization handshake to fail silently.
A _ThreadSafeMCPExecutor (subclass of awm.core.agent.MCPToolExecutor) avoids
the thread-unsafe isolated_mcp_env() that the parent class uses.
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
from awm.core.agent import MCPToolExecutor, format_tools_for_response
from awm.tools import (
    normalize_scenario_name as normalize_awm_name,
    get_random_available_port,
    tools_jsonl_save,
    isolated_mcp_env,
)

logger = logging.getLogger(__name__)

class _ThreadSafeMCPExecutor(MCPToolExecutor):
    """
    Thread-safe MCPToolExecutor for use in ThreadPoolExecutor.

    Two problems with the native MCPToolExecutor.__init__:

    1. It calls isolated_mcp_env() which modifies os.environ globally —
       unsafe when multiple threads share the same process.
    2. mcp_agent.config.Settings inherits from pydantic_settings.BaseSettings,
       which auto-reads os.environ.  Training env vars like ENV, DATABASE_PATH
       collide with Settings fields and cause JSON parse errors.

    Solution: use isolated_mcp_env() with a process-wide lock, and construct
    mcp-agent Settings inside the isolated context for each request. This keeps
    behavior aligned with awm.tools.check_mcp_server() while remaining
    thread-safe for RLLM's ThreadPoolExecutor.

    CRITICAL: The mcp_agent Agent must be created INSIDE the MCPApp.run()
    async context — Agent resolves server configurations from the app
    context at creation time.  Creating Agent outside app.run() causes
    silent MCP handshake failures (only SSE GET, no POST initialize).
    list_tools() and call_tool() are overridden to follow the same pattern
    as awm.tools.check_mcp_server() which creates Agent inside app.run().
    """

    # Keep server alias consistent with awm.tools.check_mcp_server()
    _MCP_SERVER_NAME = "mcp_tool"
    _settings_lock = threading.Lock()

    def __init__(self, mcp_url: str, timeout: float = 60.0):
        self.mcp_url = mcp_url
        self.timeout = timeout
        self._tools: list[dict] = []
        # Legacy implementation (kept as comments per request):
        # from mcp_agent.config import (
        #     Settings, MCPSettings, MCPServerSettings, LoggerSettings,
        # )
        # with self._settings_lock:
        #     with isolated_mcp_env():
        #         self._settings = Settings(
        #             execution_engine="asyncio",
        #             logger=LoggerSettings(
        #                 type="none",
        #                 transports=["none"],
        #                 progress_display=False,
        #                 level="error",
        #             ),
        #             mcp=MCPSettings(
        #                 servers={
        #                     self._MCP_SERVER_NAME: MCPServerSettings(
        #                         transport="streamable_http",
        #                         url=self.mcp_url,
        #                     ),
        #                 }
        #             ),
        #         )

    def _build_settings(self):
        from mcp_agent.config import (
            Settings, MCPSettings, MCPServerSettings, LoggerSettings,
        )
        return Settings(
            execution_engine="asyncio",
            logger=LoggerSettings(
                type="none",
                transports=["none"],
                progress_display=False,
                level="error",
            ),
            mcp=MCPSettings(
                servers={
                    self._MCP_SERVER_NAME: MCPServerSettings(
                        transport="streamable_http",
                        url=self.mcp_url,
                    ),
                }
            ),
        )

    async def list_tools(self) -> list[dict]:
        from mcp_agent.app import MCPApp
        from mcp_agent.agents.agent import Agent

        # Legacy implementation (kept as comments per request):
        # app = MCPApp(name="awm_agent", settings=self._settings)
        # with contextlib.redirect_stderr(io.StringIO()):
        #     async with app.run():
        #         agent = Agent(
        #             name="executor",
        #             server_names=[self._MCP_SERVER_NAME],
        #         )
        #         async with agent:
        #             result = await asyncio.wait_for(
        #                 agent.list_tools(), timeout=self.timeout
        #             )
        #             self._tools = []
        #             for t in result.tools:
        #                 tool_info = {
        #                     "name": t.name,
        #                     "description": t.description or "",
        #                     "inputSchema": t.inputSchema or {},
        #                 }
        #                 self._tools.append(tool_info)
        #             return self._tools

        # Match awm.tools.check_mcp_server() lifecycle:
        # isolate env -> build settings -> app.run() -> create Agent -> list_tools.
        with self._settings_lock:
            with isolated_mcp_env():
                app = MCPApp(name="awm_agent", settings=self._build_settings())
                with contextlib.redirect_stderr(io.StringIO()):
                    async with app.run():
                        agent = Agent(
                            name="executor",
                            server_names=[self._MCP_SERVER_NAME],
                        )
                        async with agent:
                            result = await asyncio.wait_for(
                                agent.list_tools(), timeout=self.timeout
                            )
                            self._tools = []
                            for t in result.tools:
                                tool_info = {
                                    "name": t.name,
                                    "description": t.description or "",
                                    "inputSchema": t.inputSchema or {},
                                }
                                self._tools.append(tool_info)
                            return self._tools

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        from mcp_agent.app import MCPApp
        from mcp_agent.agents.agent import Agent

        # Legacy implementation (kept as comments per request):
        # app = MCPApp(name="awm_agent", settings=self._settings)
        # with contextlib.redirect_stderr(io.StringIO()):
        #     async with app.run():
        #         agent = Agent(
        #             name="executor",
        #             server_names=[self._MCP_SERVER_NAME],
        #         )
        #         async with agent:
        #             result = await asyncio.wait_for(
        #                 agent.call_tool(tool_name, arguments),
        #                 timeout=self.timeout,
        #             )
        #             parts = []
        #             for c in result.content:
        #                 if hasattr(c, "text"):
        #                     parts.append(c.text)
        #                 else:
        #                     parts.append(str(c))
        #             text = "\n".join(parts)
        #             if result.isError:
        #                 return f"Error: {text}"
        #             return text

        with self._settings_lock:
            with isolated_mcp_env():
                app = MCPApp(name="awm_agent", settings=self._build_settings())
                with contextlib.redirect_stderr(io.StringIO()):
                    async with app.run():
                        agent = Agent(
                            name="executor",
                            server_names=[self._MCP_SERVER_NAME],
                        )
                        async with agent:
                            result = await asyncio.wait_for(
                                agent.call_tool(tool_name, arguments),
                                timeout=self.timeout,
                            )
                            parts = []
                            for c in result.content:
                                if hasattr(c, "text"):
                                    parts.append(c.text)
                                else:
                                    parts.append(str(c))
                            text = "\n".join(parts)
                            if result.isError:
                                return f"Error: {text}"
                            return text


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

    # Process-level port management: prevents duplicate port allocation
    # across concurrent threads in ThreadPoolExecutor.
    _port_lock = threading.Lock()
    _active_ports: set = set()

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
        server_start_timeout: float = 120.0,  # Increased from 60s for large scenarios with many DB tables
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
        self._mcp_executor: Optional[_ThreadSafeMCPExecutor] = None
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
        
        # System resource info (check for OOM or resource limits)
        diag.append(f"\n[SYSTEM RESOURCE INFO]")
        try:
            import resource
            rusage = resource.getrusage(resource.RUSAGE_CHILDREN)
            diag.append(f"  Child processes max RSS: {rusage.ru_maxrss / 1024:.1f} MB")
        except Exception as e:
            diag.append(f"  Resource usage error: {e}")
        
        # Check dmesg for recent OOM kills (may require permissions)
        if self.server_process:
            pid = self.server_process.pid
            diag.append(f"\n[OOM/KILL CHECK]")
            try:
                import subprocess as sp
                # Check dmesg for recent OOM or kill events mentioning our process
                result = sp.run(
                    ['dmesg', '--time-format=reltime', '-T'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    # Look for OOM or killed messages in the last few lines
                    lines = result.stdout.strip().split('\n')[-50:]  # Last 50 lines
                    oom_lines = [l for l in lines if 'oom' in l.lower() or 'killed' in l.lower() or str(pid) in l]
                    if oom_lines:
                        diag.append(f"  Recent OOM/kill events found:")
                        for line in oom_lines[-5:]:  # Show last 5 relevant lines
                            diag.append(f"    {line[:200]}")
                    else:
                        diag.append(f"  No recent OOM/kill events found in dmesg")
                else:
                    diag.append(f"  dmesg check failed (may need root)")
            except Exception as e:
                diag.append(f"  Could not check dmesg: {e}")
        
        # Check cgroup memory limits (for containerized environments)
        diag.append(f"\n[CGROUP LIMITS]")
        try:
            cgroup_paths = [
                '/sys/fs/cgroup/memory/memory.limit_in_bytes',
                '/sys/fs/cgroup/memory.max',
                '/proc/self/cgroup'
            ]
            for path in cgroup_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        content = f.read().strip()[:500]
                        diag.append(f"  {path}: {content}")
        except Exception as e:
            diag.append(f"  Could not read cgroup info: {e}")
        
        diag.append(f"\n{'=' * 60}")
        
        return "\n".join(diag)

    def _wait_for_server_ready(self, port: int, timeout: float) -> bool:
        """
        Wait for the FastAPI + MCP server to be ready.

        Unlike awm/tools.py::wait_for_server() which uses asyncio.run() (incompatible
        with ThreadPoolExecutor), this uses only stdlib urllib for TCP/HTTP checks
        and a dedicated MCP verification step — no event-loop-bound locks, fully
        thread-safe.

        Check flow:
          1. TCP connect (fast, low-level)
          2. HTTP GET /awm_health (dedicated health endpoint injected by server.py)
          3. If /awm_health fails (e.g. due to global middleware in generated code),
             fall back to TCP-only readiness

        AWM-generated code sometimes includes global middlewares or lifespan handlers
        that cause standard endpoints (/docs, /openapi.json) to return 503 even after
        uvicorn reports "Application startup complete". The /awm_health endpoint is
        injected after all generated routes to work around this, but if a global
        middleware intercepts all requests, we fall back to TCP-only.
        """
        import socket
        import urllib.request
        import urllib.error

        start_time = time.time()
        tcp_ready = False
        http_attempts = 0
        last_http_error = None
        last_http_status = None
        http_non_200_count = 0
        
        logger.info(f"[{self.scenario_name}] Waiting for server readiness on port {port} (timeout={timeout}s)...")

        health_endpoint = "/awm_health"

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
                        logger.info(f"[{self.scenario_name}] TCP port {port} is open after {elapsed:.1f}s, checking HTTP health...")
                except (socket.timeout, ConnectionRefusedError, OSError) as e:
                    if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                        logger.debug(f"[{self.scenario_name}] Still waiting for TCP port {port} ({elapsed:.0f}s elapsed): {e}")
                    time.sleep(0.3)
                    continue

            # HTTP health check on dedicated endpoint
            http_attempts += 1
            
            try:
                url = f"http://{self.server_host}:{port}{health_endpoint}"
                req = urllib.request.Request(url, method='GET')
                with urllib.request.urlopen(req, timeout=5) as resp:
                    last_http_status = resp.status
                    if resp.status == 200:
                        logger.info(
                            f"[{self.scenario_name}] Server HTTP ready on port {port} "
                            f"after {elapsed:.1f}s ({http_attempts} HTTP attempts)"
                        )
                        return True
            except urllib.error.HTTPError as e:
                last_http_status = e.code
                last_http_error = f"HTTP Error {e.code}: {e.reason}"
                http_non_200_count += 1
                
                if http_non_200_count == 1 or http_non_200_count % 20 == 0:
                    logger.info(
                        f"[{self.scenario_name}] Health check returned {e.code} on {health_endpoint} "
                        f"({http_non_200_count} total non-200, {elapsed:.1f}s elapsed)"
                    )
                
                # If we're getting consistent non-200 responses after the server has been
                # running for a while, the generated code likely has a global middleware
                # blocking all HTTP responses. Fall back to TCP-only readiness check.
                # The MCP verification in _start_server Step 7 will do the real validation.
                if http_non_200_count >= 10 and elapsed >= 15.0:
                    logger.warning(
                        f"[{self.scenario_name}] Health endpoint consistently returning {e.code} "
                        f"({http_non_200_count} times over {elapsed:.1f}s). "
                        f"Generated code likely has a global middleware. "
                        f"Proceeding with TCP-only readiness — MCP verification will follow."
                    )
                    return True
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                last_http_error = str(e)
                last_http_status = None

            # Log progress every 20 attempts
            if http_attempts % 20 == 0:
                logger.debug(
                    f"[{self.scenario_name}] HTTP check progress: {http_attempts} attempts, "
                    f"{elapsed:.0f}s elapsed, last status: {last_http_status}"
                )
            
            time.sleep(0.5)

        # Timeout - collect comprehensive diagnostic info
        elapsed = time.time() - start_time
        log_content = self._read_server_log()
        diag_info = self._get_diagnostic_info()
        
        proc_status = "running" if (self.server_process and self.server_process.poll() is None) else "exited"
        if self.server_process and self.server_process.poll() is not None:
            proc_status = f"exited (code={self.server_process.returncode})"
        
        # Provide specific advice based on the failure mode
        advice = ""
        if http_non_200_count > 0:
            advice = (
                f"\n\n[ADVICE] Health endpoint returned non-200 responses ({http_non_200_count} times).\n"
                f"  Consider increasing server_start_timeout (current: {timeout}s).\n"
                f"  For large scenarios with many database tables, try timeout=180 or timeout=300."
            )
        elif tcp_ready and http_attempts > 0:
            advice = (
                "\n\n[ADVICE] TCP port is open but health endpoint is not responding.\n"
                "  Check the server log above for FastAPI/uvicorn startup errors.\n"
                "  Common causes: import errors, database connection issues, MCP initialization failures."
            )
        elif not tcp_ready:
            advice = (
                "\n\n[ADVICE] TCP port never became reachable.\n"
                "  Check if another process is using the port or if the server crashed silently.\n"
                "  Check the server log above for early startup errors."
            )
        
        logger.error(
            f"[{self.scenario_name}] SERVER STARTUP TIMEOUT after {elapsed:.1f}s\n"
            f"  - Port: {port}\n"
            f"  - Process status: {proc_status}\n"
            f"  - TCP ready: {tcp_ready}\n"
            f"  - HTTP attempts: {http_attempts}\n"
            f"  - Non-200 count: {http_non_200_count}\n"
            f"  - Last HTTP status: {last_http_status}\n"
            f"  - Last HTTP error: {last_http_error}"
            f"{advice}\n"
            f"{diag_info}\n"
            f"--- SERVER LOG (server.log) ---\n{log_content}\n--- END SERVER LOG ---"
        )
        return False

    def _allocate_port(self) -> int:
        """Allocate a unique port not used by any other AWMEnvironment instance in this process."""
        with AWMEnvironment._port_lock:
            max_attempts = 50
            for _ in range(max_attempts):
                port = get_random_available_port()
                if port not in AWMEnvironment._active_ports:
                    AWMEnvironment._active_ports.add(port)
                    logger.info(
                        f"[{self.scenario_name}] Allocated port {port} "
                        f"({len(AWMEnvironment._active_ports)} active)"
                    )
                    return port
            raise RuntimeError(
                f"Failed to allocate unique port after {max_attempts} attempts "
                f"({len(AWMEnvironment._active_ports)} ports active)"
            )

    def _release_port(self):
        """Release the allocated port back to the available pool."""
        if self.server_port is not None:
            with AWMEnvironment._port_lock:
                AWMEnvironment._active_ports.discard(self.server_port)
            logger.debug(f"[{self.scenario_name}] Released port {self.server_port}")

    def _start_server(self):
        """
        Start AWM MCP server — follows awm/core/env.py::test_run_specific_env():

        1. Create temp dir
        2. Prepare database (copy or create from schema)
        3. Write env_config as jsonl (so awm.core.server can read it)
        4. Launch `python -m awm.core.server` as subprocess
        5. Sleep 3s then check if process crashed (same as original)
        6. Wait for server readiness (TCP + HTTP /awm_health, with middleware fallback)
        7. Verify MCP connectivity via _ThreadSafeMCPExecutor.list_tools()
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

        # ── Steps 3-6: Server launch with retry ──
        # Retry loop handles server startup failures (crashes, port conflicts).
        # MCP verification (step 7) is separate — MCP client issues should NOT
        # cause server restarts since the server itself is healthy.
        temp_server_path = os.path.join(self.temp_dir, "temp_server.py")
        max_launch_retries = 3
        last_launch_error = None

        for launch_attempt in range(1, max_launch_retries + 1):
            try:
                # ── Step 3: Allocate port (thread-safe, unique across instances) ──
                self.server_port = self._allocate_port()

                # ── Step 4: Launch subprocess ──
                logger.info(
                    f"[{self.scenario_name}] Starting AWM server on port {self.server_port} "
                    f"(attempt {launch_attempt}/{max_launch_retries})..."
                )

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

                # ── Step 5: Initial crash check ──
                code_size_kb = len(self.env_code) / 1024
                initial_wait = min(3.0 + (code_size_kb / 50), 10.0)
                logger.info(f"[{self.scenario_name}] Waiting {initial_wait:.1f}s for initial startup (code: {code_size_kb:.0f}KB)...")
                time.sleep(initial_wait)

                if self.server_process.poll() is not None:
                    rc = self.server_process.returncode
                    log_content = self._read_server_log()
                    diag_info = self._get_diagnostic_info()
                    logger.error(
                        f"[{self.scenario_name}] SERVER CRASHED ON STARTUP (exit code {rc})\n"
                        f"{diag_info}\n"
                        f"--- SERVER LOG (server.log) ---\n{log_content}\n--- END SERVER LOG ---"
                    )
                    self._kill_server_process(reason="crashed_on_startup")
                    raise RuntimeError(
                        f"AWM server crashed on startup for '{self.scenario_name}' (exit code {rc}). "
                        f"Temp dir preserved for debugging: {self.temp_dir}. "
                        f"Check server.log in temp dir for details."
                    )

                # ── Step 6: Wait for server readiness (TCP + HTTP health check) ──
                if not self._wait_for_server_ready(self.server_port, self.server_start_timeout):
                    self._kill_server_process(reason="server_readiness_timeout")
                    raise RuntimeError(
                        f"AWM server not ready within {self.server_start_timeout}s "
                        f"for scenario '{self.scenario_name}'. Temp dir: {self.temp_dir}. "
                        f"Check the logs above for detailed diagnostic information."
                    )

                break  # Server is up and healthy!

            except RuntimeError as e:
                last_launch_error = e
                self._kill_server_process(reason=f"launch_retry_{launch_attempt}")
                self._release_port()
                if launch_attempt < max_launch_retries:
                    logger.warning(
                        f"[{self.scenario_name}] Server launch attempt {launch_attempt}/{max_launch_retries} "
                        f"failed: {e}. Retrying with new port..."
                    )
                    continue
                raise

        # ── Step 7: MCP verification (separate from server launch) ──
        # MCP issues are client-side — do NOT kill a healthy server for them.
        # Re-create the executor between retries to avoid stale mcp_agent state.
        mcp_url = f"http://{self.server_host}:{self.server_port}/mcp"
        self._mcp_executor = _ThreadSafeMCPExecutor(mcp_url)

        max_mcp_retries = 5
        last_mcp_error = None
        for mcp_attempt in range(1, max_mcp_retries + 1):
            try:
                tools = self._run_async(self._mcp_executor.list_tools())
                if tools:
                    logger.info(
                        f"[{self.scenario_name}] MCP verified: {len(tools)} tools on port {self.server_port}"
                    )
                    return
                else:
                    logger.warning(
                        f"[{self.scenario_name}] MCP tools list is empty — "
                        f"proceeding anyway (FastApiMCP may not discover tools for this scenario)"
                    )
                    return
            except Exception as e:
                last_mcp_error = str(e)
                logger.warning(f"[{self.scenario_name}] MCP verify attempt {mcp_attempt}/{max_mcp_retries} failed: {e}")

            if mcp_attempt < max_mcp_retries:
                time.sleep(2.0 * mcp_attempt)
                try:
                    self._mcp_executor = _ThreadSafeMCPExecutor(mcp_url)
                except Exception:
                    pass

        logger.warning(
            f"[{self.scenario_name}] MCP verification failed after {max_mcp_retries} attempts: {last_mcp_error}. "
            f"Server is healthy — proceeding anyway. Agent tool calls may fail at runtime."
        )

    # ------------------------------------------------------------------
    # Process management
    # ------------------------------------------------------------------

    def _kill_server_process(self, reason: str = "unspecified"):
        """
        Kill server process only (preserve temp dir for debugging).
        
        Args:
            reason: Why the process is being killed (for logging)
        """
        if self.server_process:
            pid = self.server_process.pid
            poll_result = self.server_process.poll()
            
            if poll_result is not None:
                logger.info(
                    f"[{self.scenario_name}] Server process (pid={pid}) already exited "
                    f"(code={poll_result}), reason for kill call: {reason}"
                )
            else:
                logger.info(
                    f"[{self.scenario_name}] Killing server process (pid={pid}), reason: {reason}"
                )
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                    self.server_process.wait(timeout=5)
                    logger.info(f"[{self.scenario_name}] Server process terminated gracefully")
                except Exception as e:
                    logger.warning(f"[{self.scenario_name}] SIGTERM failed ({e}), sending SIGKILL...")
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                        logger.info(f"[{self.scenario_name}] Server process killed forcefully")
                    except Exception as e2:
                        logger.error(f"[{self.scenario_name}] Failed to kill server: {e2}")
            
            self.server_process = None

        if self.server_log_file:
            try:
                self.server_log_file.close()
            except Exception:
                pass
            self.server_log_file = None

    def _cleanup_server(self):
        """Clean up server process and temporary files."""
        self._kill_server_process(reason="cleanup_called")

        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except Exception:
                pass
            self.temp_dir = None

        self._mcp_executor = None
        self._release_port()
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

    # ------------------------------------------------------------------
    # Async execution — each MCP call uses asyncio.run() with a fresh
    # event loop.  The mcp_agent library does not work correctly on
    # persistent background event loops (only SSE GET is sent; the
    # initialization POSTs never fire).  This is safe because each
    # list_tools / call_tool call creates entirely new async contexts
    # (async with self._app.run(), async with self._agent).
    # ------------------------------------------------------------------

    def _run_async(self, coro, timeout: float = 120.0):  # noqa: ARG002
        """Run *coro* in a fresh event loop via asyncio.run().

        Called from ThreadPoolExecutor worker threads (no existing event loop),
        so asyncio.run() works directly.  The individual coroutines (list_tools,
        call_tool) already contain their own asyncio.wait_for timeouts.
        """
        return asyncio.run(coro)

    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool call — mirrors awm/core/agent.py::run_agent() tool dispatch.
        """
        name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})
        tool_call_id = tool_call.get("id", "")

        if name == "list_tools":
            try:
                tools = self._run_async(self._mcp_executor.list_tools())
                self.available_tools = tools
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
                result = self._run_async(self._mcp_executor.call_tool(tool_name, tool_args))
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
        server_start_timeout = env_args.pop("server_start_timeout", 120.0)  # Increased from 60s for large scenarios

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
