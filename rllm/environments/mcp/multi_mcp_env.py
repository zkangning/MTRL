import asyncio
import json
import logging
import queue
import threading
import warnings
import os
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from rllm.environments.base.base_env import BaseEnv
from rllm.rewards.reward_fn import RewardFunction, zero_reward
from rllm.tools.mcp_tool import MCPTool

logger = logging.getLogger(__name__)


class MultiMCPConnectionManager:
    """
    Manages connections to MULTIPLE MCP servers in a dedicated thread.
    Aggregates tools from all servers into a single tool_map.
    """

    def __init__(self, server_configs: Dict[str, Dict[str, Any]]):
        self.server_configs = server_configs
        self.request_queue: queue.Queue[tuple[str, Any, queue.Queue[tuple[str, Any]] | None]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.sessions: Dict[str, ClientSession] = {}
        self.tool_map: Dict[str, MCPTool] = {}
        self.running = False
        self.exit_stack: AsyncExitStack | None = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.worker_thread = threading.Thread(target=self._run_worker, daemon=True)
        self.worker_thread.start()

        # Wait for initialization
        response_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.request_queue.put(("init", None, response_queue))
        result = response_queue.get(timeout=60)
        if result[0] == "error":
            raise Exception(f"Failed to initialize Multi-MCP connection: {result[1]}")

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.request_queue.put(("stop", None, None))
        if self.worker_thread:
            self.worker_thread.join(timeout=10)

    def execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> dict[str, str]:
        if not self.running:
            raise Exception("Connection manager not running")
        response_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.request_queue.put(("execute", tool_calls, response_queue))
        result = response_queue.get(timeout=120)
        if result[0] == "error":
            raise Exception(f"Tool execution failed: {result[1]}")
        return result[1]

    def _run_worker(self):
        """Worker thread that runs the asyncio event loop."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            # 关键修改：只运行一个主循环 Task，清理逻辑包含在其中
            self.loop.run_until_complete(self._worker_loop())
        finally:
            self.loop.close()

    async def _worker_loop(self):
        """Main worker loop that processes requests."""
        # 初始化 Stack，确保整个生命周期都在这一个 Task 内
        self.exit_stack = AsyncExitStack()
        
        try:
            while self.running:
                try:
                    # 使用较短的 timeout 以便能响应 stop 信号
                    request = self.request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                command, data, response_queue = request

                if command == "init":
                    try:
                        # 注意：这里不再创建新的 Stack，而是使用当前 Task 上下文中的 Stack
                        await self._initialize_connections_in_stack()
                        if response_queue:
                            response_queue.put(("success", self.tool_map))
                    except Exception as e:
                        logger.exception("Error initializing connections")
                        if response_queue:
                            response_queue.put(("error", str(e)))

                elif command == "execute":
                    try:
                        result = await self._execute_tools(data)
                        if response_queue:
                            response_queue.put(("success", result))
                    except Exception as e:
                        logger.exception("Error executing tools")
                        if response_queue:
                            response_queue.put(("error", str(e)))

                elif command == "stop":
                    break
        finally:
            # 关键修改：清理逻辑在同一个 Task 的 finally 块中执行
            await self._cleanup()

    async def _initialize_connections_in_stack(self):
        """Initialize connections pushing contexts into the existing self.exit_stack."""
        # 此时 self.exit_stack 已经在 _worker_loop 中被创建
        self.tool_map = {}
        self.sessions = {}

        print(f"\n[MultiMCP] Initializing {len(self.server_configs)} servers...")

        for name, config in self.server_configs.items():
            try:
                server_params = StdioServerParameters(
                    command=config["command"],
                    args=config.get("args", []),
                    env=config.get("env", None)
                )

                # 将 context manager 推入现有的 stack
                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                stdio, write = stdio_transport
                session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
                await session.initialize()

                self.sessions[name] = session
                
                response = await session.list_tools()
                print(f"[MultiMCP] Connected to '{name}'. Tools: {[t.name for t in response.tools]}")

                for tool in response.tools:
                    if tool.name in self.tool_map:
                        logger.warning(f"Tool name collision: '{tool.name}' exists in multiple servers. Overwriting with version from '{name}'.")
                    
                    mcp_tool = MCPTool(
                        session=session, 
                        tool_name=tool.name, 
                        tool_description=tool.description, 
                        tool_schema=tool.inputSchema
                    )
                    self.tool_map[tool.name] = mcp_tool
                    
            except Exception as e:
                print(f"[MultiMCP] Failed to connect to server '{name}': {e}")
                raise e

    async def _execute_tools(self, tool_calls: list[dict[str, Any]]) -> dict[str, str]:
        tool_outputs: dict[str, str] = {}
        
        async def execute_single(t_call):
            t_name = t_call["function"]["name"]
            t_args_str = t_call["function"]["arguments"]
            t_id = t_call["id"]
            
            if t_name not in self.tool_map:
                return t_id, f"Error: Tool {t_name} not found"
            
            try:
                t_args = json.loads(t_args_str) if isinstance(t_args_str, str) else t_args_str
                tool_instance = self.tool_map[t_name]
                result = await tool_instance.async_forward(**t_args)
                return t_id, result.to_string()
            except Exception as exc:
                return t_id, f"Error executing {t_name}: {exc}"

        tasks = [execute_single(tc) for tc in tool_calls]
        results = await asyncio.gather(*tasks)
        
        for tid, output in results:
            tool_outputs[tid] = output

        return tool_outputs

    async def _cleanup(self) -> None:
        """Clean up all connections."""
        print("[MultiMCP] Cleaning up connections...")
        if self.exit_stack:
            await self.exit_stack.aclose()

class MultiMCPEnvironment(BaseEnv):
    """
    An environment that integrates MULTIPLE MCP servers.
    """

    _connection_manager: MultiMCPConnectionManager | None = None
    _manager_lock = threading.Lock()
    _active_configs_hash: str = ""

    def __init__(
        self, 
        task: dict[str, Any] | None = None, 
        server_configs: Dict[str, Dict[str, Any]] | None = None,
        reward_fn: RewardFunction | None = None, 
        max_steps: int = 10
    ):
        self.step_count = 0
        self.max_steps = max_steps
        self.task = task
        self.reward_fn = reward_fn or zero_reward
        
        # Initialize Shared Connection Manager
        # We use a simple hash of keys to check if configs changed (basic check)
        config_hash = str(sorted(server_configs.keys())) if server_configs else ""

        with MultiMCPEnvironment._manager_lock:
            # If manager doesn't exist OR configs changed completely, (re)start
            # Note: In a real production env, restarting global manager for every env param change is expensive.
            # We assume server_configs are consistent across one batch of execution.
            if (MultiMCPEnvironment._connection_manager is None) and server_configs:
                MultiMCPEnvironment._connection_manager = MultiMCPConnectionManager(server_configs)
                MultiMCPEnvironment._connection_manager.start()
                MultiMCPEnvironment._active_configs_hash = config_hash

    def reset(self):
        self.step_count = 0
        return self.task if self.task is not None else {}, {}

    def step(self, action: Any):
        if isinstance(action, dict):
            action = [action]
        self.step_count += 1
        
        reward = 0.0
        done = self.step_count >= self.max_steps or isinstance(action, str)
        
        # Check for finish tool
        if isinstance(action, list):
            for tool_call in action:
                if tool_call.get("function", {}).get("name") == "finish":
                    done = True
                    break

        if done:
            return self._handle_termination(action)

        # Execute tools via Multi-Manager
        try:
            if MultiMCPEnvironment._connection_manager:
                tool_outputs = MultiMCPEnvironment._connection_manager.execute_tool_calls(action)
                next_obs = {"tool_outputs": tool_outputs}
            else:
                next_obs = {"tool_outputs": {"error": "Manager not initialized"}}
        except Exception as e:
            print(f"Tool execution error: {e}")
            next_obs = {"tool_outputs": {}}

        return next_obs, reward, done, {"response": action}

    def _handle_termination(self, action):
        llm_response = ""
        if isinstance(action, str):
            llm_response = action
        elif isinstance(action, list):
            finish_action = next((tc for tc in action if tc.get("function", {}).get("name") == "finish"), None)
            if finish_action:
                args = finish_action.get("function", {}).get("arguments", {})
                if isinstance(args, str): args = json.loads(args)
                llm_response = str(args.get("response", "")) if isinstance(args, dict) else str(args)
            else:
                llm_response = str(action)

        if self.reward_fn and self.task:
            reward_out = self.reward_fn(task_info=self.task, action=llm_response)
            return {}, reward_out.reward, True, {"response": action, "metadata": reward_out.metadata}
        return {}, 0.0, True, {"response": action}

    def close(self):
        pass

    @staticmethod
    def cleanup_global_resources():
        with MultiMCPEnvironment._manager_lock:
            if MultiMCPEnvironment._connection_manager:
                MultiMCPEnvironment._connection_manager.stop()
                MultiMCPEnvironment._connection_manager = None

    @staticmethod
    def from_dict(env_args: dict[str, Any]) -> "MultiMCPEnvironment":
        server_configs = env_args.get("server_configs")
        reward_fn = env_args.get("reward_fn")
        max_steps = env_args.get("max_steps", 10)
        # Task is injected by the engine into env_args usually, or handled separately
        return MultiMCPEnvironment(task=env_args, server_configs=server_configs, reward_fn=reward_fn, max_steps=max_steps)

