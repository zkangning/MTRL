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
                data = json.loads(result)
                
                if not isinstance(data, dict):
                    return False

                if "organic" in data:
                    organic_results = data["organic"]
                    if isinstance(organic_results, list) and len(organic_results) == 0:
                        logger.info(f"[Cache] Skipping empty search result (organic=[])")
                        return False
                
                if not data:
                    return False
                    
            except json.JSONDecodeError:
                logger.warning(f"[Cache] Search result is not valid JSON, skipping cache.")
                return False

        # 2. 针对爬虫的过滤逻辑
        elif tool_name == "scrape_as_markdown":
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
                
        self._save_category(category)


class MCPConnectionManager:
    """
    核心连接管理器：
    1. 运行在一个独立的 Daemon Thread 中。
    2. 维护一个 asyncio loop 处理与 Node.js MCP Server 的通信。
    3. 支持真正的并发 (Batch) 工具调用。
    
    【修复版本】：
    - 追踪所有异步任务，优雅关闭
    - 增加并发限制，防止资源耗尽
    - 改进错误恢复机制
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
        
        # 【修复1】：增加工具超时时间，适应 Bright Data 的响应速度
        self.INTERNAL_TOOL_TIMEOUT = 120.0  # 从 60 秒增加到 120 秒
        
        # 【修复2】：追踪异步任务，用于优雅关闭
        self.pending_tasks = set()
        
        # 【修复3】：限制并发工具调用数量，防止资源耗尽
        self.max_concurrent_calls = 10

    def start(self):
        if self.running: return
        self.running = True
        self.worker_thread = threading.Thread(target=self._run_worker, daemon=True)
        self.worker_thread.start()

        # 同步等待初始化完成
        resp_q = queue.Queue()
        self.request_queue.put(("init", None, resp_q))
        try:
            status, payload = resp_q.get(timeout=45)
            if status == "error":
                self.stop()
                raise RuntimeError(f"MCP Init failed: {payload}")
            logger.info(f"MCP Started successfully. Loaded tools: {list(self.tool_map.keys())}")
        except queue.Empty:
            self.stop()
            raise RuntimeError("MCP Init timed out. Node.js process failed to respond.")

    def stop(self):
        """【修复4】：优雅关闭，等待任务完成"""
        logger.info("[MCP] Stopping Manager...")
        self.running = False
        self.request_queue.put(("stop", None, None))
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)  # 增加等待时间

    def execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> dict[str, str]:
        """【修复5】：增强健康检查和错误恢复"""
        # 增强的健康检查
        max_retries = 3
        for retry in range(max_retries):
            if not self.running or not self.worker_thread or not self.worker_thread.is_alive():
                logger.warning(f"[MCP] Worker unhealthy (retry {retry+1}/{max_retries}), restarting...")
                
                # 清理
                self.running = False
                if self.worker_thread:
                    try:
                        self.worker_thread.join(timeout=2)
                    except:
                        pass
                
                # 重启
                try:
                    self.start()
                    logger.info(f"[MCP] Restart successful")
                    break
                except Exception as e:
                    logger.error(f"[MCP] Restart failed: {e}")
                    if retry == max_retries - 1:
                        return {tc['id']: f"Error: MCP Manager restart failed after {max_retries} attempts" for tc in tool_calls}
                    time.sleep(1)
                    continue
            else:
                break

        resp_q = queue.Queue()
        self.request_queue.put(("execute", tool_calls, resp_q))
        
        try:
            status, payload = resp_q.get(timeout=self.INTERNAL_TOOL_TIMEOUT + 10)
            
            if status == "error":
                logger.error(f"MCP Execution Error (caught): {payload}")
                return {tc['id']: f"Error executing tool: {payload}" for tc in tool_calls}
            
            return payload
            
        except queue.Empty:
            logger.error("MCP Execution CRITICAL TIMEOUT (Queue Empty). Worker likely hung.")
            # 【修复6】：不立即关闭 Manager，而是返回错误让上层处理
            # 避免一个超时导致整个 Manager 崩溃
            return {tc['id']: "Error: Tool execution timed out internally (System limit)." for tc in tool_calls}

    def _run_worker(self):
        """后台线程：运行 Asyncio Event Loop"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._async_main_loop())
        except Exception as e:
            logger.error(f"MCP Worker Crashed: {e}", exc_info=True)
        finally:
            try:
                # 取消所有待处理任务
                tasks = asyncio.all_tasks(loop)
                for t in tasks: 
                    t.cancel()
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()
            except Exception:
                pass

    async def _async_main_loop(self):
        """【修复7】：异步主循环 - 追踪任务并优雅关闭"""
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
                logger.error(f"[MCP] Failed to initialize session: {e}")
                try:
                    cmd, _, q = self.request_queue.get_nowait()
                    if cmd == "init" and q: q.put(("error", str(e)))
                except queue.Empty:
                    pass
                return

            # 创建信号量限制并发
            semaphore = asyncio.Semaphore(self.max_concurrent_calls)

            # 主循环
            while self.running:
                try:
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
                        # 【修复8】：创建任务并追踪
                        task = asyncio.create_task(
                            self._handle_execution_with_semaphore(data, resp_q, semaphore)
                        )
                        self.pending_tasks.add(task)
                        # 清理已完成的任务
                        task.add_done_callback(self.pending_tasks.discard)
                
                except Exception as e:
                    logger.error(f"[MCP] Error in main loop (continuing): {e}", exc_info=True)
                    await asyncio.sleep(0.1)
            
            # 【修复9】：优雅关闭 - 等待所有待处理任务完成
            if self.pending_tasks:
                logger.info(f"[MCP] Waiting for {len(self.pending_tasks)} pending tasks to complete...")
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self.pending_tasks, return_exceptions=True),
                        timeout=5.0
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

    async def _handle_execution_with_semaphore(self, tool_calls, resp_q, semaphore):
        """【修复10】：带信号量的执行处理，限制并发"""
        async with semaphore:
            await self._handle_execution(tool_calls, resp_q)

    async def _handle_execution(self, tool_calls, resp_q):
        """执行单个 Step 的所有工具调用，包含超时保护"""
        try:
            result = await asyncio.wait_for(
                self._execute_batch(tool_calls), 
                timeout=self.INTERNAL_TOOL_TIMEOUT
            )
            if resp_q: resp_q.put(("success", result))
        except asyncio.TimeoutError:
            logger.warning(f"[MCP] Tool execution timeout after {self.INTERNAL_TOOL_TIMEOUT}s")
            if resp_q: resp_q.put(("error", f"Timeout waiting for tool response ({self.INTERNAL_TOOL_TIMEOUT}s limit)"))
        except Exception as e:
            logger.error(f"[MCP] Tool execution error: {e}", exc_info=True)
            if resp_q: resp_q.put(("error", str(e)))

    async def _execute_batch(self, tool_calls):
        """并发执行所有工具调用"""
        tasks = []
        for tc in tool_calls:
            tasks.append(self._process_single_tool(tc))
        
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
        
        # 处理 Batch 工具
        if name == "search_engine_batch":
            queries = args.get("queries", [])
            if not queries: return "[]"
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
        
        # 使用进程级 Factory 获取单例 Manager
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
            real_calls = [tc for tc in action if tc.get("function", {}).get("name") != "finish"]
            if real_calls:
                try:
                    tool_outputs = self.manager.execute_tool_calls(real_calls)
                except Exception as e:
                    logger.error(f"Env Step Unexpected Error: {e}", exc_info=True)
                    tool_outputs = {tc['id']: f"System Error: {str(e)}" for tc in real_calls}

        # 3. 构造 Observation
        next_obs = {"tool_outputs": tool_outputs}
        return next_obs, 0.0, done, {"response": action, "metadata": {}}

    def close(self):
        # 不关闭 Manager，因为它是进程级共享的
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
