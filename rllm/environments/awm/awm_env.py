"""
AWM Environment for RLLM

This environment wraps AWM-generated virtual environments (FastAPI + MCP Server)
for use in RLLM's agentic RL training pipeline.

Server lifecycle follows awm/core/env.py::test_run_specific_env():
  1. Write env_config to temp jsonl
  2. Copy / create database to temp dir
  3. Launch `python -m awm.core.server` as subprocess
  4. Wait for server readiness via HTTP + MCP tool verification

MCP client architecture:
  _DirectMCPExecutor uses the low-level `mcp` SDK (mcp.client.streamable_http +
  mcp.client.session.ClientSession) directly, bypassing the mcp_agent library
  entirely. This eliminates the _global_context process-level singleton race
  that caused asyncio.run() deadlocks when multiple threads concurrently ran
  MCPApp.initialize() / cleanup().

  Each executor runs its own background daemon thread with a dedicated asyncio
  event loop and a persistent MCP session. Requests are dispatched via a
  thread-safe queue. No process-global locks, no _session_lock, no MCPApp.
"""

import asyncio
import copy
import json
import logging
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Tuple

from rllm.environments.base.base_env import BaseEnv
from rllm.rewards.reward_types import RewardOutput

# AWM core imports
from awm.core.db import create_sqlite_database
from awm.core.agent import format_tools_for_response
from awm.tools import (
    normalize_scenario_name as normalize_awm_name,
    get_random_available_port,
    tools_jsonl_save,
)

logger = logging.getLogger(__name__)

class _DirectMCPExecutor:
    """
    MCP tool executor using the low-level `mcp` SDK directly.

    Completely bypasses the `mcp_agent` library (MCPApp, Agent, Settings,
    _global_context) to eliminate the asyncio.run() deadlock caused by
    concurrent MCPApp.initialize()/cleanup() racing on the process-level
    _global_context singleton.

    Architecture:
      - A background daemon thread runs a dedicated asyncio event loop
      - The event loop maintains a persistent streamable_http connection
        and ClientSession (no per-call initialize/cleanup cycle)
      - Caller threads submit requests via a thread-safe queue and block
        on a per-request response queue
      - No process-global state, no _session_lock, no MCPApp

    This design mirrors MCPConnectionManager in rllm/environments/tools/mcp_env.py
    (which uses stdio transport) but adapted for streamable_http transport.
    """

    def __init__(self, mcp_url: str, timeout: float = 60.0):
        self.mcp_url = mcp_url
        self.timeout = timeout
        self._tools: list[dict] = []
        self._tools_cached = False
        self._request_queue: queue.Queue = queue.Queue()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._ready_event = threading.Event()
        self._init_error: Optional[str] = None

        self._start_worker()

    def _start_worker(self):
        """Start background worker thread with dedicated event loop."""
        self._running = True
        self._ready_event.clear()
        self._init_error = None
        self._worker_thread = threading.Thread(
            target=self._run_worker, daemon=True, name="mcp-direct-worker"
        )
        self._worker_thread.start()

        if not self._ready_event.wait(timeout=self.timeout + 10):
            self._running = False
            if self._init_error:
                raise RuntimeError(f"MCP session init failed: {self._init_error}")
            raise RuntimeError(
                f"MCP session init timed out after {self.timeout + 10}s for {self.mcp_url}"
            )
        if self._init_error:
            raise RuntimeError(f"MCP session init failed: {self._init_error}")

    def _run_worker(self):
        """Background thread: run asyncio event loop with persistent MCP session."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_main())
        except Exception as e:
            logger.error(f"[MCP-Direct] Worker crashed: {e}", exc_info=True)
            self._init_error = str(e)
            self._ready_event.set()
        finally:
            try:
                tasks = asyncio.all_tasks(loop)
                for t in tasks:
                    t.cancel()
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()
            except Exception:
                pass
            self._running = False

    async def _async_main(self):
        """Async main loop: establish persistent MCP connection and process requests."""
        try:
            from mcp.client.streamable_http import streamable_http_client
        except ImportError:
            from mcp.client.streamable_http import streamablehttp_client as streamable_http_client
        from mcp.client.session import ClientSession

        try:
            async with AsyncExitStack() as stack:
                transport = await stack.enter_async_context(
                    streamable_http_client(self.mcp_url)
                )
                read_stream, write_stream, _ = transport
                session = await stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await session.initialize()

                self._ready_event.set()
                logger.info(f"[MCP-Direct] Session established for {self.mcp_url}")

                while self._running:
                    # Use get_nowait() + asyncio.sleep() instead of blocking
                    # get(timeout=N). The streamable_http transport runs its
                    # post_writer/SSE tasks in the same event loop via anyio;
                    # a blocking queue.get() would starve those tasks.
                    try:
                        cmd, data, resp_q = self._request_queue.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.01)
                        continue

                    if cmd == "stop":
                        break
                    elif cmd == "list_tools":
                        await self._handle_list_tools(session, resp_q)
                    elif cmd == "call_tool":
                        await self._handle_call_tool(session, data, resp_q)
                    else:
                        if resp_q:
                            resp_q.put(("error", f"Unknown command: {cmd}"))

        except Exception as e:
            logger.error(f"[MCP-Direct] Connection failed: {e}", exc_info=True)
            self._init_error = str(e)
            self._ready_event.set()
            self._drain_pending_requests(str(e))

    def _drain_pending_requests(self, error_msg: str):
        """Drain any pending requests with error responses on shutdown."""
        while True:
            try:
                cmd, data, resp_q = self._request_queue.get_nowait()
                if resp_q:
                    resp_q.put(("error", error_msg))
            except queue.Empty:
                break

    async def _handle_list_tools(self, session, resp_q: queue.Queue):
        try:
            result = await asyncio.wait_for(
                session.list_tools(), timeout=self.timeout
            )
            tools = []
            for t in result.tools:
                tools.append({
                    "name": t.name,
                    "description": t.description or "",
                    "inputSchema": t.inputSchema or {},
                })
            resp_q.put(("ok", tools))
        except Exception as e:
            resp_q.put(("error", str(e)))

    async def _handle_call_tool(self, session, data: dict, resp_q: queue.Queue):
        try:
            tool_name = data["tool_name"]
            arguments = data["arguments"]
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments),
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
                resp_q.put(("ok", f"Error: {text}"))
            else:
                resp_q.put(("ok", text))
        except Exception as e:
            resp_q.put(("error", str(e)))

    def list_tools(self) -> list[dict]:
        """Synchronous list_tools — submits to background thread, blocks for result."""
        if self._tools_cached and self._tools:
            return self._tools

        if not self._running:
            raise RuntimeError("MCP executor is not running")

        resp_q: queue.Queue = queue.Queue()
        self._request_queue.put(("list_tools", None, resp_q))

        try:
            status, payload = resp_q.get(timeout=self.timeout + 10)
        except queue.Empty:
            raise TimeoutError(
                f"list_tools timed out after {self.timeout + 10}s"
            )

        if status == "error":
            raise RuntimeError(f"list_tools failed: {payload}")

        self._tools = payload
        self._tools_cached = True
        return self._tools

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Synchronous call_tool — submits to background thread, blocks for result."""
        if not self._running:
            raise RuntimeError("MCP executor is not running")

        resp_q: queue.Queue = queue.Queue()
        self._request_queue.put((
            "call_tool",
            {"tool_name": tool_name, "arguments": arguments},
            resp_q,
        ))

        try:
            status, payload = resp_q.get(timeout=self.timeout + 10)
        except queue.Empty:
            raise TimeoutError(
                f"call_tool({tool_name}) timed out after {self.timeout + 10}s"
            )

        if status == "error":
            raise RuntimeError(f"call_tool({tool_name}) failed: {payload}")

        return payload

    def stop(self):
        """Gracefully shut down the background worker."""
        self._running = False
        try:
            self._request_queue.put(("stop", None, None))
        except Exception:
            pass
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)

    @property
    def is_alive(self) -> bool:
        return self._running and self._worker_thread is not None and self._worker_thread.is_alive()


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
        database_dir: Optional[str] = None,
        max_steps: int = 30,
        task_max_prompt_length: Optional[int] = None,
        task_max_response_length: Optional[int] = None,
        reward_fn=None,
        server_host: str = "127.0.0.1",
        server_start_timeout: float = 120.0,
        prestart_server: bool = False,
        tool_call_timeout: float = 30.0,
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
        self.database_dir = database_dir
        self.max_steps = max_steps
        self.task_max_prompt_length = task_max_prompt_length
        self.task_max_response_length = task_max_response_length
        self.reward_fn = reward_fn
        self.server_host = server_host
        self.server_start_timeout = server_start_timeout
        self.prestart_server = prestart_server
        self.tool_call_timeout = tool_call_timeout

        # Server management
        self.server_port: Optional[int] = None
        self.server_process: Optional[subprocess.Popen] = None
        self.server_log_file = None
        self.server_log_path: Optional[str] = None
        self.temp_dir: Optional[str] = None
        self._mcp_executor: Optional[_DirectMCPExecutor] = None
        self.current_db_path: Optional[str] = None
        self.initial_db_path: Optional[str] = None

        # State tracking
        self.current_step = 0
        self.history: List[Dict[str, Any]] = []
        self.done = False
        self.available_tools: List[Dict] = []
        self._is_prestarted = False
        self._mcp_dead = False

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

    def _resolve_source_db_path(self, scenario_norm: str) -> Optional[str]:
        """Resolve initial DB source path, prioritizing explicit path then database_dir."""
        if self.db_path and os.path.exists(self.db_path):
            return self.db_path

        if self.database_dir:
            candidate = os.path.join(self.database_dir, f"{scenario_norm}.db")
            if os.path.exists(candidate):
                return candidate

        return None

    @staticmethod
    def _normalize_db_sample_examples(db_sample: Any) -> dict[str, list[str]]:
        """
        Normalize db_sample into table_name -> list[SQL] format.

        Supported formats:
        1) {"users": ["INSERT ...", ...], ...}
        2) {"tables": [{"table_name": "users", "insert_statements": [...]}, ...]}
        """
        if not isinstance(db_sample, dict):
            return {}

        table_examples: dict[str, list[str]] = {}

        # Format 1: direct map
        direct_keys = [k for k, v in db_sample.items() if isinstance(k, str) and isinstance(v, list)]
        if direct_keys and "tables" not in db_sample:
            for table_name in direct_keys:
                values = [str(sql) for sql in db_sample.get(table_name, []) if isinstance(sql, str)]
                if values:
                    table_examples[table_name] = values
            return table_examples

        # Format 2: nested tables list
        tables = db_sample.get("tables", [])
        if isinstance(tables, list):
            for table in tables:
                if not isinstance(table, dict):
                    continue
                table_name = table.get("table_name") or table.get("name")
                if not isinstance(table_name, str) or not table_name:
                    continue

                statements = table.get("insert_statements")
                if not isinstance(statements, list):
                    statements = table.get("examples")
                if not isinstance(statements, list):
                    continue

                values = [str(sql) for sql in statements if isinstance(sql, str)]
                if values:
                    table_examples[table_name] = values

        return table_examples

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
        7. Verify MCP connectivity via _DirectMCPExecutor.list_tools()
        """
        scenario_norm = normalize_awm_name(self.scenario_name)
        self.temp_dir = tempfile.mkdtemp(prefix=f"awm_env_{scenario_norm}_")

        # ── Step 1: Prepare database ──
        source_db_path = self._resolve_source_db_path(scenario_norm)
        if source_db_path:
            self.current_db_path = os.path.join(self.temp_dir, f"{scenario_norm}.db")
            shutil.copyfile(source_db_path, self.current_db_path)
            os.chmod(self.current_db_path, 0o644)
            logger.info(f"[{self.scenario_name}] Copied existing database from {source_db_path}")
        elif self.db_schema:
            logger.info(f"[{self.scenario_name}] Creating database from schema...")
            full_schema = copy.deepcopy(self.db_schema)
            table_examples = self._normalize_db_sample_examples(self.db_sample)
            if table_examples:
                for table in full_schema.get("tables", []):
                    table_name = table.get("name")
                    if table_name and table_name in table_examples:
                        table["examples"] = table_examples[table_name]

            db_path, successful, failed, errors = create_sqlite_database(
                self.scenario_name, full_schema, self.temp_dir
            )
            self.current_db_path = db_path
            logger.info(
                f"[{self.scenario_name}] Database built from schema: "
                f"tables_ok={successful}, tables_failed={failed}"
            )
            if failed > 0:
                logger.warning(f"[{self.scenario_name}] Database creation had {failed} failures: {errors}")
        else:
            raise ValueError("Either db_path or db_schema must be provided for AWMEnvironment")

        # Snapshot initial DB for verifier initial/final comparison.
        self.initial_db_path = os.path.join(self.temp_dir, f"{scenario_norm}.initial.db")
        shutil.copyfile(self.current_db_path, self.initial_db_path)
        os.chmod(self.initial_db_path, 0o644)

        # ── Step 2: Write env_config as jsonl (same format as awm/core/env.py) ──
        env_config = {
            "scenario": self.scenario_name,
            "db_path": self.current_db_path,
            "full_code": self.env_code,
        }
        temp_env_json = os.path.join(self.temp_dir, "env_config.jsonl")
        tools_jsonl_save([env_config], temp_env_json)

        # ── Steps 3-7: Server launch + MCP verification with retry ──
        # Retry loop handles startup failures (crashes/port/readiness) AND
        # transient MCP empty-tool states. We require non-empty MCP tools
        # before declaring the environment ready.
        temp_server_path = os.path.join(self.temp_dir, "temp_server.py")
        max_launch_retries = 3
        max_mcp_retries = 5

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

                # ── Step 7: MCP verification (must discover non-empty tools) ──
                #
                # _DirectMCPExecutor uses the low-level mcp SDK with a persistent
                # background connection. No MCPApp, no _global_context, no
                # process-level locks needed.
                mcp_url = f"http://{self.server_host}:{self.server_port}/mcp"
                last_mcp_error = None

                for mcp_attempt in range(1, max_mcp_retries + 1):
                    try:
                        self._mcp_executor = _DirectMCPExecutor(
                            mcp_url, timeout=self.tool_call_timeout
                        )
                        tools = self._mcp_executor.list_tools()
                        if tools:
                            logger.info(
                                f"[{self.scenario_name}] MCP verified: {len(tools)} tools on port {self.server_port}"
                            )
                            return

                        last_mcp_error = "empty_tools_list"
                        logger.warning(
                            f"[{self.scenario_name}] MCP returned empty tools list on attempt "
                            f"{mcp_attempt}/{max_mcp_retries}; retrying..."
                        )
                    except Exception as e:
                        last_mcp_error = str(e)
                        logger.warning(
                            f"[{self.scenario_name}] MCP verify attempt {mcp_attempt}/{max_mcp_retries} failed: {e}"
                        )

                    if self._mcp_executor:
                        self._mcp_executor.stop()
                        self._mcp_executor = None

                    if mcp_attempt < max_mcp_retries:
                        time.sleep(2.0 * mcp_attempt)

                raise RuntimeError(
                    f"MCP verification failed after {max_mcp_retries} attempts "
                    f"(last_error={last_mcp_error})"
                )

            except (RuntimeError, TimeoutError) as e:
                self._kill_server_process(reason=f"launch_retry_{launch_attempt}")
                self._release_port()
                if launch_attempt < max_launch_retries:
                    logger.warning(
                        f"[{self.scenario_name}] Server launch attempt {launch_attempt}/{max_launch_retries} "
                        f"failed: {e}. Retrying with new port..."
                    )
                    continue
                raise RuntimeError(str(e)) from e

    def prestart(self):
        """
        Optional server warm-up hook for training.

        This does not change default behavior. It only takes effect when caller
        explicitly enables prestart_server in env args.
        """
        if not self.prestart_server:
            return
        if self.server_process and self.server_process.poll() is None:
            self._is_prestarted = True
            return
        self._cleanup_server()
        self._start_server()
        self._is_prestarted = True

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
        if self._mcp_executor:
            try:
                self._mcp_executor.stop()
            except Exception:
                pass
            self._mcp_executor = None

        self._kill_server_process(reason="cleanup_called")

        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except Exception:
                pass
            self.temp_dir = None

        self._release_port()
        self.server_port = None
        self.current_db_path = None
        self.initial_db_path = None

    # ------------------------------------------------------------------
    # Gym-like interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not (self.prestart_server and self._is_prestarted and self.server_process and self.server_process.poll() is None):
            self._cleanup_server()

        self.current_step = 0
        self.history = []
        self.done = False
        self.available_tools = []
        self._mcp_dead = False

        if not (self.prestart_server and self._is_prestarted and self.server_process and self.server_process.poll() is None):
            self._start_server()
        self._is_prestarted = False

        observation = {
            "system_prompt": self.AWM_SYSTEM_PROMPT,
            "task": self.task_description,
            "scenario": self.scenario_name,
        }

        info = {
            "scenario": self.scenario_name,
            "task": self.task_description,
            "task_type": "awm",
            "max_steps": self.max_steps,
            "task_max_steps": self.max_steps,
        }
        if self.task_max_prompt_length is not None:
            info["task_max_prompt_length"] = int(self.task_max_prompt_length)
        if self.task_max_response_length is not None:
            info["task_max_response_length"] = int(self.task_max_response_length)

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

        if self._mcp_dead:
            self.done = True
            reward = 0.0
            observation = {"tool_results": results, "step": self.current_step}
            info = {
                "step": self.current_step, "action": action,
                "tool_calls": tool_calls, "results": results,
                "termination_reason": "mcp_deadlock",
            }
            self.history.append(info)
            return observation, reward, self.done, info

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

    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool call — mirrors awm/core/agent.py::run_agent() tool dispatch.

        _DirectMCPExecutor.list_tools() / call_tool() are synchronous calls that
        submit to the executor's background thread and block on its response queue.
        No asyncio.run(), no daemon threads, no _session_lock — the persistent
        session handles everything.

        If the background worker dies (connection lost, etc.), _mcp_dead is set
        and subsequent calls fail fast, terminating the episode.
        """
        name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})
        tool_call_id = tool_call.get("id", "")

        if self._mcp_dead:
            self.done = True
            logger.warning(
                f"[{self.scenario_name}] MCP executor is dead. "
                f"Terminating episode. tool={name}"
            )
            return {
                "tool": name,
                "tool_call_id": tool_call_id,
                "result": (
                    "Error: MCP connection is broken. This episode is terminated."
                ),
                "success": False,
            }

        if self._mcp_executor and not self._mcp_executor.is_alive:
            self._mcp_dead = True
            self.done = True
            logger.warning(
                f"[{self.scenario_name}] MCP executor worker thread died. "
                f"Terminating episode. tool={name}"
            )
            return {
                "tool": name,
                "tool_call_id": tool_call_id,
                "result": "Error: MCP connection lost. This episode is terminated.",
                "success": False,
            }

        if name == "list_tools":
            try:
                if self.available_tools:
                    tools = self.available_tools
                else:
                    tools = self._mcp_executor.list_tools()
                self.available_tools = tools
                formatted_tools = format_tools_for_response(tools)
                return {
                    "tool": "list_tools",
                    "tool_call_id": tool_call_id,
                    "result": formatted_tools,
                    "success": True
                }
            except TimeoutError:
                self._mcp_dead = True
                logger.warning(
                    f"[{self.scenario_name}] list_tools timed out after {self.tool_call_timeout}s"
                )
                return {
                    "tool": "list_tools",
                    "tool_call_id": tool_call_id,
                    "result": f"Error: list_tools timed out after {self.tool_call_timeout}s. The server may be overloaded.",
                    "success": False
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
                result = self._mcp_executor.call_tool(tool_name, tool_args)
                return {
                    "tool": tool_name,
                    "tool_call_id": tool_call_id,
                    "arguments": tool_args,
                    "result": result,
                    "success": not result.startswith("Error:")
                }
            except TimeoutError:
                self._mcp_dead = True
                logger.warning(
                    f"[{self.scenario_name}] call_tool({tool_name}) timed out after "
                    f"{self.tool_call_timeout}s"
                )
                return {
                    "tool": tool_name,
                    "tool_call_id": tool_call_id,
                    "arguments": tool_args,
                    "result": (
                        f"Error: Tool call '{tool_name}' timed out after "
                        f"{self.tool_call_timeout}s. The operation may be too slow or the server is unresponsive."
                    ),
                    "success": False
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
                "initial_db_path": self.initial_db_path or self.current_db_path or self.db_path,
                "final_db_path": self.current_db_path or self.db_path,
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
        server_start_timeout = env_args.pop("server_start_timeout", 120.0)
        tool_call_timeout = env_args.pop("tool_call_timeout", 30.0)

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
        db_path = _get("db_path")
        database_dir = _get("database_dir")
        if not database_dir:
            database_dir = os.environ.get("AWM_DATABASE_DIR")
        max_steps = _get("max_steps", "task_max_steps") or 30
        task_max_prompt_length = _get("task_max_prompt_length")
        task_max_response_length = _get("task_max_response_length")
        prestart_server = bool(_get("prestart_server") or False)

        return AWMEnvironment(
            scenario_name=scenario_name,
            task_description=task_description,
            env_code=env_code,
            db_path=db_path,
            db_schema=db_schema,
            db_sample=db_sample,
            verifier_code=verifier_code,
            database_dir=database_dir,
            max_steps=int(max_steps),
            task_max_prompt_length=int(task_max_prompt_length) if task_max_prompt_length else None,
            task_max_response_length=int(task_max_response_length) if task_max_response_length else None,
            reward_fn=reward_fn,
            server_host=server_host,
            server_start_timeout=server_start_timeout,
            prestart_server=prestart_server,
            tool_call_timeout=float(tool_call_timeout),
        )
