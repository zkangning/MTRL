import asyncio
import json
import queue
import threading
import os
import hashlib
import logging
import atexit
import time
from contextlib import AsyncExitStack
from typing import Any, Dict, Optional, Tuple, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from rllm.environments.base.base_env import BaseEnv
from rllm.rewards.reward_fn import RewardFunction, zero_reward
from rllm.tools.mcp_tool import MCPTool

logger = logging.getLogger(__name__)

# --- 全局单例管理 (适配 Ray/Multiprocessing) ---
# 每个系统进程(OS Process)只维护一个 Manager，避免启动成百上千个 Node 进程
_PROCESS_GLOBAL_MANAGER = {}
_PROCESS_LOCK = threading.Lock()

def get_process_manager(mcp_cmd, mcp_args, mcp_env, cache_dir, allowed_tools):
    """
    获取当前物理进程的 MCP Manager。
    如果在 Ray Worker 中运行，这确保每个 Worker 进程只启动一个 Node.js 子进程。
    """
    pid = os.getpid()
    with _PROCESS_LOCK:
        # 检查现有 Manager 是否健康
        if pid in _PROCESS_GLOBAL_MANAGER:
            existing_manager = _PROCESS_GLOBAL_MANAGER[pid]
            
            # 健康检查：running 标志 + worker thread 存活
            if existing_manager.running and existing_manager.worker_thread and existing_manager.worker_thread.is_alive():
                return existing_manager
            else:
                logger.warning(f"[MCP] Existing Manager for PID={pid} is unhealthy, recreating...")
                try:
                    existing_manager.stop()
                except Exception as e:
                    logger.debug(f"[MCP] Error stopping old manager: {e}")
                del _PROCESS_GLOBAL_MANAGER[pid]
        
        # 创建新 Manager
        logger.info(f"[MCP] Initializing new Manager for Process ID: {pid}")
        manager = MCPConnectionManager(
            mcp_cmd, mcp_args, mcp_env, cache_dir, allowed_tools
        )
        try:
            manager.start()
            _PROCESS_GLOBAL_MANAGER[pid] = manager
            
            # 注册进程退出时的清理函数
            def cleanup():
                if pid in _PROCESS_GLOBAL_MANAGER:
                    logger.info(f"[MCP] Stopping Manager for Process ID: {pid}")
                    try:
                        _PROCESS_GLOBAL_MANAGER[pid].stop()
                    except:
                        pass
                    del _PROCESS_GLOBAL_MANAGER[pid]
            atexit.register(cleanup)
        except Exception as e:
            logger.error(f"[MCP] Failed to start manager for PID {pid}: {e}")
            raise e
        
        return _PROCESS_GLOBAL_MANAGER[pid]


class SplitToolCache:
    """本地文件缓存，减少重复的昂贵搜索请求"""
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        self.file_map = {
            "search": "mcp_search_cache.json",
            "markdown": "mcp_markdown_cache.json",
            "html": "mcp_html_cache.json",
            "general": "mcp_general_cache.json"
        }
        self.data: Dict[str, Dict[str, str]] = {k: {} for k in self.file_map}
        self.locks = {k: threading.Lock() for k in self.file_map}
        self._load_all()

    def _load_all(self):
        for category, filename in self.file_map.items():
            path = os.path.join(self.cache_dir, filename)
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        self.data[category] = json.load(f)
                except Exception:
                    self.data[category] = {}

    def _is_valid_response(self, tool_name: str, result: str) -> bool:
        """
        核心校验逻辑：判断工具返回的结果是否值得缓存。
        """
        if not result:
            return False
        
        if "execution failed" in result or result.strip().startswith("Error:"):
            logger.info(f"[Cache] Detected generic execution failure for {tool_name}, skipping cache.")
            return False
            
        # 1. 针对搜索引擎的过滤逻辑
        if tool_name == "search_engine":
            try:
                # 尝试解析 JSON
                data = json.loads(result)
                
                # 如果不是字典，可能发生了错误，不缓存
                if not isinstance(data, dict):
                    return False

                # 检查 organic 字段
                if "organic" in data:
                    organic_results = data["organic"]
                    # 如果 organic 存在但为空列表，视为无效结果，不缓存
                    if isinstance(organic_results, list) and len(organic_results) == 0:
                        logger.info(f"[Cache] Skipping empty search result (organic=[])")
                        return False
                
                # 如果返回的是空的 JSON 对象 {}，也不缓存
                if not data:
                    return False
                    
            except json.JSONDecodeError:
                # 如果无法解析为 JSON，可能是纯文本报错信息（如 "Error:..."），建议不缓存以便重试
                logger.warning(f"[Cache] Search result is not valid JSON, skipping cache.")
                return False

        # 2. 针对爬虫的过滤逻辑 (可选优化)
        elif tool_name == "scrape_as_markdown":
            # 如果结果太短或包含特定错误关键词，视为失败
            if len(result.strip()) < 5 or result.strip().startswith("Error:"):
                return False

            error_keywords = [
                "HTTP 400", "HTTP 401", "HTTP 403", "HTTP 404", "HTTP 429", 
                "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504",
                "Proxy request failed", 
                "unknown_proxy_error"]
            
            for keyword in error_keywords:
                if keyword in result:
                    logger.info(f"[Cache] Scrape result contains error keyword '{keyword}', skipping cache.")
                    return False

                    
        return True

    def _save_category(self, category: str):
        filename = self.file_map[category]
        path = os.path.join(self.cache_dir, filename)
        with self.locks[category]:
            try:
                temp_path = path + ".tmp"
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(self.data[category], f, indent=2, ensure_ascii=False)
                os.replace(temp_path, path)
            except Exception as e:
                logger.error(f"[Cache] Failed to save {filename}: {e}")

    def _get_category_and_key(self, tool_name: str, args: Dict[str, Any]) -> Tuple[str, str]:
        if tool_name == "search_engine":
            # 简化 Key 生成，忽略一些无关参数
            query = args.get("query", "")
            return "search", query
        elif tool_name == "scrape_as_markdown":
            return "markdown", args.get("url", "")
        else:
            key_content = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
            return "general", hashlib.sha256(key_content.encode('utf-8')).hexdigest()

    def get(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        category, key = self._get_category_and_key(tool_name, args)
        if not key: return None
        return self.data[category].get(key)

    def put(self, tool_name: str, args: Dict[str, Any], result: str):
        if not self._is_valid_response(tool_name, result):
            return

        category, key = self._get_category_and_key(tool_name, args)
        if not key: return
        
        with self.locks[category]:
            if key not in self.data[category]:
                self.data[category][key] = result
                
        # 只有在确实写入了新数据且有效时才保存
        self._save_category(category)


class MCPConnectionManager:
    """
    核心连接管理器：
    1. 运行在一个独立的 Daemon Thread 中。
    2. 维护一个 asyncio loop 处理与 Node.js MCP Server 的通信。
    3. 支持真正的并发 (Batch) 工具调用。
    """
    def __init__(
        self,
        mcp_server_command: str,
        mcp_server_args: list[str] | None,
        mcp_server_env: dict[str, str] | None,
        cache_dir: str | None,
        allowed_tools: list[str] | None
    ):
        self.mcp_server_command = mcp_server_command
        self.mcp_server_args = mcp_server_args or []
        self.mcp_server_env = mcp_server_env
        self.cache = SplitToolCache(cache_dir) if cache_dir else None
        
        # 处理工具白名单
        self.allowed_tools = set(allowed_tools) if allowed_tools else None
        if self.allowed_tools:
            if "search_engine_batch" in self.allowed_tools:
                self.allowed_tools.add("search_engine")
            if "scrape_batch" in self.allowed_tools:
                self.allowed_tools.add("scrape_as_markdown")

        self.request_queue = queue.Queue()
        self.running = False
        self.worker_thread = None
        self.tool_map = {}
        
        # 内部超时时间 (秒)。这必须小于 AgentExecutionEngine 的 trajectory_timeout
        # 如果 Bright Data 秒都没返回，我们就当做失败，让 LLM 重试或换个 Query
        self.INTERNAL_TOOL_TIMEOUT = 120.0  # 120修复1: 增加超时时间，适应 Bright Data 响应速度
        
        # 修复6: 追踪异步任务，用于优雅关闭（恢复并发能力）
        self.pending_tasks = set()

    def start(self):
        if self.running: return
        self.running = True
        self.worker_thread = threading.Thread(target=self._run_worker, daemon=True)
        self.worker_thread.start()

        # 同步等待初始化完成，确保工具列表已加载
        resp_q = queue.Queue()
        self.request_queue.put(("init", None, resp_q))
        try:
            status, payload = resp_q.get(timeout=45) # 初始化给较长时间
            if status == "error":
                self.stop()
                raise RuntimeError(f"MCP Init failed: {payload}")
            logger.info(f"MCP Started successfully. Loaded tools: {list(self.tool_map.keys())}")
        except queue.Empty:
            self.stop()
            raise RuntimeError("MCP Init timed out. Node.js process failed to respond.")

    def stop(self):
        self.running = False
        self.request_queue.put(("stop", None, None))
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2)

    def execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> dict[str, str]:
        """主线程调用此方法，阻塞等待后台线程的结果"""
        # 检查 Worker Thread 是否存活，如果死亡则尝试重启
        if not self.running or not self.worker_thread.is_alive():
            logger.warning(f"[MCP] Worker thread is dead (PID={os.getpid()}). Attempting restart...")
            self.running = False
            
            # 清理旧线程
            try:
                if self.worker_thread:
                    self.worker_thread.join(timeout=1)
            except Exception as e:
                logger.debug(f"[MCP] Error joining old thread: {e}")
            
            # 重新启动
            try:
                self.start()
                logger.info(f"[MCP] Successfully restarted Manager for PID={os.getpid()}")
            except Exception as e:
                logger.error(f"[MCP] Failed to restart Manager: {e}")
                return {tc['id']: f"Error: MCP Manager restart failed: {e}" for tc in tool_calls}

        resp_q = queue.Queue()
        self.request_queue.put(("execute", tool_calls, resp_q))
        
        try:
            # 等待时间 = 内部超时 + 缓冲。如果这里触发 Empty，说明 Worker 彻底死锁了。
            status, payload = resp_q.get(timeout=self.INTERNAL_TOOL_TIMEOUT + 10)
            
            if status == "error":
                # 返回错误信息给 LLM，而不是抛出异常导致程序崩溃
                logger.error(f"MCP Execution Error (caught): {payload}")
                return {tc['id']: f"Error executing tool: {payload}" for tc in tool_calls}
            
            return payload
            
        except queue.Empty:
            logger.error("MCP Execution CRITICAL TIMEOUT (Queue Empty). Worker likely hung.")
            # 修复2: 不立即关闭 Manager，避免连锁崩溃
            # self.running = False  # 注释掉这行，让 Worker 继续运行
            return {tc['id']: "Error: Tool execution timed out internally (System limit)." for tc in tool_calls}

    def _run_worker(self):
        """后台线程：运行 Asyncio Event Loop"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._async_main_loop())
        except Exception as e:
            logger.error(f"MCP Worker Crashed: {e}")
        finally:
            try:
                tasks = asyncio.all_tasks(loop)
                for t in tasks: t.cancel()
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()
            except Exception:
                pass

    async def _async_main_loop(self):
        """异步主循环"""
        # 1. 建立 MCP 连接
        server_params = StdioServerParameters(
            command=self.mcp_server_command,
            args=self.mcp_server_args,
            env=self.mcp_server_env
        )
        
        async with AsyncExitStack() as stack:
            try:
                stdio_transport = await stack.enter_async_context(stdio_client(server_params))
                stdio, write = stdio_transport
                session = await stack.enter_async_context(ClientSession(stdio, write))
                await session.initialize()
                
                # 获取工具列表
                tool_list = await session.list_tools()
                self._build_tool_map(session, tool_list)
                
            except Exception as e:
                # 初始化失败，通知等待的线程
                logger.error(f"[MCP] Failed to initialize session: {e}")
                try:
                    cmd, _, q = self.request_queue.get_nowait()
                    if cmd == "init" and q: q.put(("error", str(e)))
                except queue.Empty:
                    pass
                return

            # 2. 循环处理请求
            while self.running:
                try:
                    # 使用轮询方式获取 Queue，配合 sleep 让出 CPU
                    try:
                        cmd, data, resp_q = self.request_queue.get(timeout=0.05)
                    except queue.Empty:
                        await asyncio.sleep(0.01)
                        continue

                    if cmd == "init":
                        if resp_q: resp_q.put(("success", None))
                    
                    elif cmd == "stop":
                        logger.info("[MCP] Received stop command, exiting loop")
                        break
                    
                    elif cmd == "execute":
                        # 修复7: 恢复并发能力，但追踪任务避免崩溃
                        task = asyncio.create_task(self._handle_execution(data, resp_q))
                        self.pending_tasks.add(task)
                        # 任务完成后自动从集合中移除
                        task.add_done_callback(self.pending_tasks.discard)
                
                except Exception as e:
                    # 捕获异常但不退出循环，避免 Session 被意外关闭
                    logger.error(f"[MCP] Error in main loop (continuing): {e}", exc_info=True)
                    await asyncio.sleep(0.1)
            
            # 修复8: 优雅关闭 - 等待所有待处理任务完成
            if self.pending_tasks:
                logger.info(f"[MCP] Waiting for {len(self.pending_tasks)} pending tasks to complete...")
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self.pending_tasks, return_exceptions=True),
                        timeout=5.0  # 最多等待 5 秒
                    )
                    logger.info("[MCP] All pending tasks completed")
                except asyncio.TimeoutError:
                    logger.warning("[MCP] Some tasks did not complete in time, cancelling...")
                    for task in self.pending_tasks:
                        if not task.done():
                            task.cancel()

    def _build_tool_map(self, session, tool_list):
        self.tool_map = {}
        for tool in tool_list.tools:
            name_norm = tool.name.replace("-", "_")
            is_allowed = True
            if self.allowed_tools:
                if tool.name not in self.allowed_tools and name_norm not in self.allowed_tools:
                    is_allowed = False
            
            if is_allowed:
                t = MCPTool(session, tool.name, tool.description, tool.inputSchema)
                self.tool_map[tool.name] = t
                self.tool_map[name_norm] = t

    async def _handle_execution(self, tool_calls, resp_q):
        """执行单个 Step 的所有工具调用，包含超时保护"""
        try:
            # 这里的 wait_for 是防止 Node.js 卡死
            result = await asyncio.wait_for(
                self._execute_batch(tool_calls),
                timeout=self.INTERNAL_TOOL_TIMEOUT
            )
            if resp_q: resp_q.put(("success", result))
        except asyncio.TimeoutError:
            # 修复4: 更新错误消息，反映实际超时时间
            logger.warning(f"[MCP] Tool execution timeout after {self.INTERNAL_TOOL_TIMEOUT}s, but Session remains open")
            if resp_q: resp_q.put(("error", f"Timeout waiting for tool response ({self.INTERNAL_TOOL_TIMEOUT}s limit)"))
        except Exception as e:
            logger.error(f"[MCP] Tool execution error: {e}")
            if resp_q: resp_q.put(("error", str(e)))

    async def _execute_batch(self, tool_calls):
        """并发执行所有工具调用"""
        tasks = []
        for tc in tool_calls:
            tasks.append(self._process_single_tool(tc))
        
        # return_exceptions=True 确保一个工具失败不会导致整个 Batch 崩溃
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = {}
        for i, res in enumerate(results):
            tool_id = tool_calls[i]['id']
            if isinstance(res, Exception):
                output[tool_id] = f"Error: {str(res)}"
            else:
                output[tool_id] = res
        return output

    async def _process_single_tool(self, tool_call):
        name = tool_call["function"]["name"]
        raw_args = tool_call["function"]["arguments"]
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        
        # 处理 Batch 工具的内部逻辑
        if name == "search_engine_batch":
            queries = args.get("queries", [])
            if not queries: return "[]"
            # 内部再次并发
            sub_tasks = [self._call_tool_impl("search_engine", q) for q in queries]
            res = await asyncio.gather(*sub_tasks, return_exceptions=True)
            return json.dumps([str(r) if isinstance(r, Exception) else json.loads(r) if isinstance(r, str) and (r.startswith('{') or r.startswith('[')) else r for r in res])
        
        elif name == "scrape_batch":
            urls = args.get("urls", [])
            if not urls: return "[]"
            sub_tasks = [self._call_tool_impl("scrape_as_markdown", {"url": u}) for u in urls]
            res = await asyncio.gather(*sub_tasks, return_exceptions=True)
            return json.dumps([str(r) if isinstance(r, Exception) else r for r in res])
            
        else:
            return await self._call_tool_impl(name, args)

    async def _call_tool_impl(self, name, args):
        # 1. 查缓存
        if self.cache:
            hit = self.cache.get(name, args)
            if hit: return hit
        
        # 2. 调用 MCP
        if name not in self.tool_map:
            return f"Error: Tool {name} not found or not allowed."
            
        tool = self.tool_map[name]
        # MCPTool.async_forward 是异步的
        res_obj = await tool.async_forward(**args)
        res_str = res_obj.to_string()
        
        # 3. 写缓存
        if self.cache:
            self.cache.put(name, args, res_str)
            
        return res_str


class MCPEnvironment(BaseEnv):
    def __init__(
        self, 
        task: dict[str, Any] | None = None, 
        mcp_server_command: str | None = None, 
        mcp_server_args: list[str] | None = None, 
        mcp_server_env: dict[str, str] | None = None, 
        reward_fn: RewardFunction | None = None, 
        max_steps: int = 10,
        cache_dir: str | None = "./mcp_cache",
        allowed_tools: list[str] | None = None
    ):
        self.step_count = 0
        self.max_steps = max_steps
        self.task = task
        self.reward_fn = reward_fn or zero_reward
        
        # [关键修改] 使用进程级 Factory 获取单例 Manager
        # 这避免了在同一个进程内重复启动 MCP Server (Node.js)
        self.manager = None
        if mcp_server_command:
            try:
                self.manager = get_process_manager(
                    mcp_server_command, 
                    mcp_server_args, 
                    mcp_server_env, 
                    cache_dir, 
                    allowed_tools
                )
            except Exception as e:
                logger.error(f"Failed to get process manager: {e}")

    def reset(self):
        self.step_count = 0
        return self.task if self.task is not None else {}, {}

    def step(self, action: Any):
        if isinstance(action, dict): action = [action]
        self.step_count += 1
        
        # 1. 检查是否结束
        done = self.step_count >= self.max_steps
        llm_response = str(action)
        is_finish = False
        
        if isinstance(action, list):
            for tc in action:
                if tc.get("function", {}).get("name") == "finish":
                    is_finish = True
                    args = tc.get("function", {}).get("arguments", {})
                    if isinstance(args, str): 
                        try: args = json.loads(args)
                        except: pass
                    llm_response = args.get("response", "") if isinstance(args, dict) else str(args)
                    break
        
        if is_finish or isinstance(action, str):
            done = True
            reward_val = 0.0
            metadata = {}
            if self.reward_fn and self.task:
                r_out = self.reward_fn(task_info=self.task, action=llm_response)
                reward_val = r_out.reward
                metadata = r_out.metadata
            return {}, reward_val, done, {"response": llm_response, "metadata": metadata}

        # 2. 执行工具
        tool_outputs = {}
        if self.manager and isinstance(action, list):
            # 过滤掉 finish
            real_calls = [tc for tc in action if tc.get("function", {}).get("name") != "finish"]
            if real_calls:
                try:
                    tool_outputs = self.manager.execute_tool_calls(real_calls)
                except Exception as e:
                    # 捕获所有未预料到的异常，防止 Crash
                    logger.error(f"Env Step Unexpected Error: {e}")
                    tool_outputs = {tc['id']: f"System Error: {str(e)}" for tc in real_calls}

        # 3. 构造 Observation
        next_obs = {"tool_outputs": tool_outputs}
        return next_obs, 0.0, done, {"response": action, "metadata": {}}

    def close(self):
        # 不再关闭 self.manager，因为它是进程级共享的
        # 清理由 atexit 负责
        pass

    @staticmethod
    def from_dict(env_args: dict[str, Any]) -> "MCPEnvironment":
        return MCPEnvironment(
            task=env_args,
            mcp_server_command=env_args.pop("mcp_server_command", None),
            mcp_server_args=env_args.pop("mcp_server_args", None),
            mcp_server_env=env_args.pop("mcp_server_env", None),
            reward_fn=env_args.pop("reward_fn", None),
            max_steps=env_args.pop("max_steps", 10),
            cache_dir=env_args.pop("cache_dir", "./mcp_cache"),
            allowed_tools=env_args.pop("allowed_tools", None)
        )






