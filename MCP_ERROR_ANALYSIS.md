# MCP 超时和崩溃问题深度分析报告

## 问题概述

在运行 Search 任务时，系统频繁出现以下错误：

1. **ClosedResourceError**: `anyio.streams.memory.ClosedResourceError`
2. **EPIPE Error**: `Error: write EPIPE` (Node.js 端)
3. **MCP Execution CRITICAL TIMEOUT**: Worker 线程挂起
4. **Worker Crashed**: `unhandled errors in a TaskGroup`

## 错误堆栈分析

### 核心错误链

```
1. Python 端 (mcp_tool.py:98)
   └─> session.call_tool() 
       └─> session.send_request()
           └─> write_stream.send()
               └─> ClosedResourceError ❌

2. Node.js 端
   └─> StdioServerTransport.send()
       └─> process.stdout.write()
           └─> Error: write EPIPE ❌
```

### 问题根源

**EPIPE (Broken Pipe)** 错误表明：
- Python 端的 MCP Client 已经关闭了与 Node.js Server 的 stdio 连接
- 但 Node.js Server 仍在尝试向已关闭的管道写入数据
- 这是一个**竞态条件 (Race Condition)** 问题

## 深层原因分析

### 1. **并发执行导致的资源竞争**

在 `mcp_env.py` 中：

```python
# Line 356-374: 主循环处理请求
while self.running:
    try:
        cmd, data, resp_q = self.request_queue.get(timeout=0.05)
    except queue.Empty:
        await asyncio.sleep(0.01)
        continue

    if cmd == "execute":
        # 并发执行：将任务抛给 asyncio handle，不阻塞循环
        asyncio.create_task(self._handle_execution(data, resp_q))
```

**问题**：
- 使用 `asyncio.create_task()` 创建了**火忘式 (Fire-and-Forget)** 任务
- 这些任务没有被追踪或等待
- 当多个 Search 请求并发时，可能导致：
  - 任务堆积
  - Session 资源耗尽
  - 某些任务在 Session 关闭后仍在执行

### 2. **超时配置不匹配**

当前超时层级：

```
Level 1: trajectory_timeout (AgentExecutionEngine)
  └─> 默认: 1e9 秒 (几乎无限)
  
Level 2: env.step() timeout (agent_execution_engine.py:270)
  └─> timeout = trajectory_timeout - total_time
  
Level 3: INTERNAL_TOOL_TIMEOUT (MCPConnectionManager)
  └─> 60 秒 (硬编码)
  
Level 4: execute_tool_calls() timeout (mcp_env.py:292)
  └─> INTERNAL_TOOL_TIMEOUT + 10 = 70 秒
```

**问题**：
- `trajectory_timeout=null` 导致 Level 1 几乎无限大
- Level 2 的 `timeout=(trajectory_timeout - total_time)` 可能非常大
- 但 Level 3/4 只有 60-70 秒
- **不匹配导致**：外层等待时间远大于内层，造成资源泄漏

### 3. **Session 生命周期管理缺陷**

在 `_async_main_loop()` 中：

```python
async with AsyncExitStack() as stack:
    stdio_transport = await stack.enter_async_context(stdio_client(server_params))
    stdio, write = stdio_transport
    session = await stack.enter_async_context(ClientSession(stdio, write))
    
    # 循环处理请求
    while self.running:
        # ...
        asyncio.create_task(self._handle_execution(data, resp_q))
```

**问题**：
- 当 `self.running = False` 时，`while` 循环退出
- `AsyncExitStack` 立即清理资源，关闭 Session
- 但之前创建的 `asyncio.create_task()` 可能还在运行
- 这些任务尝试使用已关闭的 Session → **ClosedResourceError**

### 4. **错误恢复机制触发连锁反应**

在 `execute_tool_calls()` 中：

```python
except queue.Empty:
    logger.error("MCP Execution CRITICAL TIMEOUT (Queue Empty). Worker likely hung.")
    self.running = False  # ⚠️ 标记为需要重启
    return {tc['id']: "Error: Tool execution timed out..." for tc in tool_calls}
```

**连锁反应**：
1. 某个工具调用超时 → `queue.Empty`
2. 设置 `self.running = False`
3. Worker 线程的主循环退出
4. Session 被关闭
5. 其他正在执行的任务遇到 `ClosedResourceError`
6. Node.js 端收到 `EPIPE` 错误并崩溃
7. 大量错误日志输出

### 5. **Ray 多进程环境下的资源共享问题**

```python
# Line 27-72: 进程级单例管理
_PROCESS_GLOBAL_MANAGER = {}

def get_process_manager(...):
    pid = os.getpid()
    with _PROCESS_LOCK:
        if pid in _PROCESS_GLOBAL_MANAGER:
            existing_manager = _PROCESS_GLOBAL_MANAGER[pid]
            # 健康检查
            if existing_manager.running and existing_manager.worker_thread.is_alive():
                return existing_manager
```

**问题**：
- 在 Ray 环境中，多个 TaskRunner 进程并发运行
- 每个进程有自己的 MCP Manager 和 Node.js 子进程
- 当某个进程的 Manager 崩溃时，会触发重启
- 但重启过程中可能有其他任务正在使用该 Manager
- 导致竞态条件

## 具体触发场景

基于你的配置：

```bash
data.train_batch_size=128
actor_rollout_ref.rollout.n=8  # 每个 prompt 生成 8 个响应
rllm.agent.max_steps=10
```

**并发度计算**：
- 128 个任务 × 8 个采样 = 1024 个轨迹
- 每个轨迹最多 10 步
- 每步可能调用多个 Search 工具

**触发条件**：
1. 某个 Search 请求因为网络问题耗时超过 60 秒
2. `INTERNAL_TOOL_TIMEOUT` 触发
3. `self.running = False`
4. Session 关闭
5. 其他正在执行的 Search 请求遇到 `ClosedResourceError`
6. 级联失败

## 解决方案

### 方案 1: 修复任务追踪和优雅关闭 (推荐)

**核心思路**：追踪所有创建的异步任务，在关闭前等待它们完成。

```python
# 在 MCPConnectionManager.__init__ 中添加
self.pending_tasks = set()

# 修改 _async_main_loop
async def _async_main_loop(self):
    server_params = StdioServerParameters(...)
    
    async with AsyncExitStack() as stack:
        try:
            stdio_transport = await stack.enter_async_context(stdio_client(server_params))
            stdio, write = stdio_transport
            session = await stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize()
            
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
                    # 创建任务并追踪
                    task = asyncio.create_task(self._handle_execution(data, resp_q))
                    self.pending_tasks.add(task)
                    # 清理已完成的任务
                    task.add_done_callback(self.pending_tasks.discard)
            
            except Exception as e:
                logger.error(f"[MCP] Error in main loop (continuing): {e}", exc_info=True)
                await asyncio.sleep(0.1)
        
        # 优雅关闭：等待所有待处理任务完成
        if self.pending_tasks:
            logger.info(f"[MCP] Waiting for {len(self.pending_tasks)} pending tasks to complete...")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.pending_tasks, return_exceptions=True),
                    timeout=5.0  # 最多等待 5 秒
                )
            except asyncio.TimeoutError:
                logger.warning("[MCP] Some tasks did not complete in time, cancelling...")
                for task in self.pending_tasks:
                    task.cancel()
```

### 方案 2: 增加超时配置和错误隔离

**修改 1**: 在训练脚本中设置合理的 `trajectory_timeout`

```bash
# train_composite_all.sh
rllm.agent.trajectory_timeout=300  # 5 分钟
```

**修改 2**: 调整 `INTERNAL_TOOL_TIMEOUT`

```python
# mcp_env.py Line 238
# 根据实际网络情况调整，Bright Data 可能需要更长时间
self.INTERNAL_TOOL_TIMEOUT = 120.0  # 从 60 秒增加到 120 秒
```

**修改 3**: 在 `execute_tool_calls` 中避免立即设置 `running=False`

```python
def execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> dict[str, str]:
    # 检查 Worker Thread 是否存活
    if not self.running or not self.worker_thread.is_alive():
        logger.warning(f"[MCP] Worker thread is dead (PID={os.getpid()}). Attempting restart...")
        self.running = False
        
        try:
            if self.worker_thread:
                self.worker_thread.join(timeout=1)
        except Exception as e:
            logger.debug(f"[MCP] Error joining old thread: {e}")
        
        try:
            self.start()
            logger.info(f"[MCP] Successfully restarted Manager for PID={os.getpid()}")
        except Exception as e:
            logger.error(f"[MCP] Failed to restart Manager: {e}")
            return {tc['id']: f"Error: MCP Manager restart failed: {e}" for tc in tool_calls}

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
        # ⚠️ 不要立即设置 running=False，而是返回错误让上层处理
        # self.running = False  # 注释掉这行
        return {tc['id']: "Error: Tool execution timed out internally (System limit)." for tc in tool_calls}
```

### 方案 3: 使用信号量限制并发度

```python
class MCPConnectionManager:
    def __init__(self, ...):
        # ... 现有代码 ...
        
        # 限制同时执行的工具调用数量
        self.max_concurrent_calls = 10
        self.semaphore = None  # 在 async 上下文中初始化
    
    async def _async_main_loop(self):
        # ... 初始化代码 ...
        
        # 创建信号量
        self.semaphore = asyncio.Semaphore(self.max_concurrent_calls)
        
        # ... 主循环 ...
    
    async def _handle_execution(self, tool_calls, resp_q):
        """执行单个 Step 的所有工具调用，包含超时保护和并发限制"""
        async with self.semaphore:  # 限制并发
            try:
                result = await asyncio.wait_for(
                    self._execute_batch(tool_calls), 
                    timeout=self.INTERNAL_TOOL_TIMEOUT
                )
                if resp_q: resp_q.put(("success", result))
            except asyncio.TimeoutError:
                if resp_q: resp_q.put(("error", "Timeout waiting for tool response"))
            except Exception as e:
                if resp_q: resp_q.put(("error", str(e)))
```

### 方案 4: 增强健康检查和自动恢复

```python
def execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> dict[str, str]:
    """主线程调用此方法，阻塞等待后台线程的结果"""
    
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
                time.sleep(1)  # 等待后重试
                continue
        else:
            break
    
    # 执行工具调用
    resp_q = queue.Queue()
    self.request_queue.put(("execute", tool_calls, resp_q))
    
    try:
        status, payload = resp_q.get(timeout=self.INTERNAL_TOOL_TIMEOUT + 10)
        
        if status == "error":
            logger.error(f"MCP Execution Error: {payload}")
            return {tc['id']: f"Error executing tool: {payload}" for tc in tool_calls}
        
        return payload
        
    except queue.Empty:
        logger.error("MCP Execution CRITICAL TIMEOUT")
        # 返回错误而不是崩溃
        return {tc['id']: "Error: Tool execution timed out" for tc in tool_calls}
```

## 推荐实施步骤

### 第一阶段：快速修复（立即实施）

1. **调整超时配置**
   ```bash
   # 在 train_composite_all.sh 中添加
   rllm.agent.trajectory_timeout=300
   ```

2. **增加工具超时时间**
   ```python
   # mcp_env.py Line 238
   self.INTERNAL_TOOL_TIMEOUT = 120.0
   ```

3. **移除立即关闭逻辑**
   ```python
   # mcp_env.py Line 304
   # 注释掉: self.running = False
   ```

### 第二阶段：结构性改进（1-2天）

1. 实施**方案 1**：任务追踪和优雅关闭
2. 实施**方案 3**：并发限制
3. 增强日志记录，便于调试

### 第三阶段：长期优化（可选）

1. 考虑使用连接池而非单例模式
2. 实现更智能的重试策略
3. 添加监控指标（成功率、延迟等）

## 验证方法

修复后，运行以下测试：

```bash
# 1. 小规模测试
data.train_batch_size=16
actor_rollout_ref.rollout.n=2

# 2. 中等规模测试
data.train_batch_size=64
actor_rollout_ref.rollout.n=4

# 3. 完整规模测试
data.train_batch_size=128
actor_rollout_ref.rollout.n=8
```

观察指标：
- 错误日志数量
- Worker 重启次数
- 任务完成率
- 平均延迟

## 总结

这个问题的根本原因是：
1. **并发任务管理不当**：火忘式任务 + 资源提前释放
2. **超时配置不匹配**：外层无限大，内层 60 秒
3. **错误恢复机制过激**：一个超时导致整个 Manager 关闭

建议优先实施**第一阶段**的快速修复，然后逐步实施结构性改进。
