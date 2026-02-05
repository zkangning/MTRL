#!/usr/bin/env python3
"""
Local Retrieval Tool for dense search using a local retrieval server.

This tool connects to a locally running dense retrieval server and performs
dense retrieval using E5 embeddings on the indexed Wikipedia corpus.

Usage:
    1. Download data: python examples/search/download_search_data.py --data_dir ./search_data
    2. Merge index: cd search_data/prebuilt_indices && cat part_aa part_ab > e5_Flat.index
    3. Launch server: bash examples/search/retrieval/launch_server.sh ./search_data/prebuilt_indices 8000
    4. Set env: export RETRIEVAL_SERVER_URL="http://127.0.0.1:8000"
    
    For multi-GPU setup:
    3. Launch servers: bash examples/search/retrieval/launch_multi_gpu.sh ./search_data/prebuilt_indices 8000 4
    4. Set env: export RETRIEVAL_SERVER_URL="http://127.0.0.1:8000,http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003"

Cache Feature:
    The tool supports caching of retrieval results to avoid redundant queries.
    - Set cache_dir parameter or LOCAL_SEARCH_CACHE_DIR env var to enable caching
    - Cache uses SQLite with WAL mode for high-concurrency multi-process access
    - O(1) query, O(1) write - no file rewriting needed
    - Two-level cache: memory (L1) + SQLite (L2)
    
    Migration from JSON:
    If you have an existing local_search_cache.json file, you can import it:
        cache = LocalSearchCache("./local_search_cache")
        cache.import_from_json("./local_search_cache/local_search_cache.json")
"""

import json
import logging
import os
import random
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

import httpx

from rllm.tools.tool_base import Tool, ToolOutput

logger = logging.getLogger(__name__)


class LocalSearchCache:
    """
    基于 SQLite 的本地检索缓存，用于减少重复的检索请求，加速 local_search 任务。
    
    高性能、多进程安全的实现：
    - 使用 SQLite WAL 模式，支持并发读写（读不阻塞写，写不阻塞读）
    - O(1) 查询复杂度，基于索引快速定位
    - 增量写入，不需要重写整个文件
    - 内存一级缓存 + 磁盘二级缓存的多级缓存策略
    
    相比 JSON 实现的优势：
    - 写入：从 O(N) 降低到 O(1)
    - 读取：从 O(N) 降低到 O(log N)
    - 并发：从串行化锁降低到 WAL 模式的高并发
    """
    
    # SQLite 连接超时（秒），应对多进程写入时的短暂锁定
    DB_TIMEOUT = 30.0
    
    # 批量写入的阈值，超过这个数量时使用事务批量提交
    BATCH_COMMIT_THRESHOLD = 100
    
    def __init__(self, cache_dir: str, cache_filename: str = "local_search_cache.db"):
        """
        初始化缓存。
        
        Args:
            cache_dir: 缓存目录路径
            cache_filename: 缓存数据库文件名（默认 .db 后缀）
        """
        self.cache_dir = os.path.abspath(cache_dir)
        self.db_path = os.path.join(self.cache_dir, cache_filename)
        
        # 确保缓存目录存在
        self._ensure_cache_dir()
        
        # 线程锁（保护单进程内的多线程访问内存缓存）
        self._thread_lock = threading.Lock()
        
        # 内存一级缓存（热数据）
        self._local_cache: Dict[str, str] = {}
        
        # 统计信息
        self._stats = {
            "memory_hits": 0,
            "db_hits": 0,
            "misses": 0,
            "writes": 0,
        }
        
        # 初始化数据库
        self._init_db()
        
        # 预热：加载部分热数据到内存（可选）
        self._warmup_cache()
        
        logger.info(f"[LocalSearchCache] SQLite cache initialized at {self.db_path}, "
                    f"preloaded {len(self._local_cache)} entries")
    
    def _ensure_cache_dir(self):
        """确保缓存目录存在"""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        获取数据库连接。
        每次调用创建新连接，确保多线程安全。
        """
        conn = sqlite3.connect(self.db_path, timeout=self.DB_TIMEOUT)
        # 开启 WAL 模式和优化配置
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")  # 兼顾性能与安全
        conn.execute("PRAGMA cache_size=10000;")     # 增大页面缓存
        conn.execute("PRAGMA temp_store=MEMORY;")    # 临时表存内存
        return conn
    
    def _init_db(self):
        """初始化 SQLite 表结构"""
        try:
            with self._get_connection() as conn:
                # 创建缓存表
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS search_cache (
                        query_key TEXT PRIMARY KEY,
                        query_raw TEXT,
                        result TEXT,
                        result_length INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        access_count INTEGER DEFAULT 0
                    )
                """)
                
                # 创建索引（PRIMARY KEY 已有索引，这里为访问频次创建索引用于热数据预热）
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_access_count
                    ON search_cache(access_count DESC)
                """)
                
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error(f"[LocalSearchCache] DB initialization failed: {e}")
            raise
    
    def _warmup_cache(self, limit: int = 1000):
        """
        预热缓存：加载访问频次最高的条目到内存。
        
        Args:
            limit: 预热条目数量上限
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT query_key, result FROM search_cache "
                    "ORDER BY access_count DESC LIMIT ?",
                    (limit,)
                )
                for row in cursor:
                    self._local_cache[row[0]] = row[1]
        except sqlite3.Error as e:
            logger.warning(f"[LocalSearchCache] Cache warmup failed: {e}")
    
    def _normalize_query(self, query: str) -> str:
        """
        规范化 query 作为缓存 key。
        
        Args:
            query: 原始查询字符串
            
        Returns:
            规范化后的查询字符串
        """
        normalized = query.strip().lower()
        normalized = ' '.join(normalized.split())
        return normalized
    
    def _is_valid_result(self, result: str) -> bool:
        """
        判断检索结果是否值得缓存。
        
        Args:
            result: 格式化后的检索结果
            
        Returns:
            是否应该缓存
        """
        if not result:
            return False
        
        # 错误结果不缓存
        if "execution failed" in result or result.strip().startswith("Error:"):
            return False
        
        # 空结果不缓存
        if result in ("No relevant documents found.", "No relevant documents found for the query."):
            return False
        
        # 结果太短不缓存（可能是错误）
        if len(result.strip()) < 20:
            return False
        
        return True
    
    def get(self, query: str) -> Optional[str]:
        """
        获取缓存的检索结果。
        采用两级缓存策略：内存 -> SQLite。
        
        Args:
            query: 查询字符串
            
        Returns:
            缓存的结果，如果不存在则返回 None
        """
        key = self._normalize_query(query)
        if not key:
            return None
        
        # Level 1: 检查内存缓存（O(1)）
        with self._thread_lock:
            if key in self._local_cache:
                self._stats["memory_hits"] += 1
                return self._local_cache[key]
        
        # Level 2: 查询 SQLite（O(log N)，只查这一条，不加载整个文件）
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT result FROM search_cache WHERE query_key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if row:
                    result = row[0]
                    
                    # 更新访问计数（异步更新，不阻塞返回）
                    try:
                        conn.execute(
                            "UPDATE search_cache SET access_count = access_count + 1 "
                            "WHERE query_key = ?",
                            (key,)
                        )
                        conn.commit()
                    except sqlite3.Error:
                        pass  # 更新失败不影响主逻辑
                    
                    # 回填内存缓存
                    with self._thread_lock:
                        self._local_cache[key] = result
                    
                    self._stats["db_hits"] += 1
                    return result
                    
        except sqlite3.Error as e:
            logger.warning(f"[LocalSearchCache] DB read error: {e}")
        
        self._stats["misses"] += 1
        return None
    
    def put(self, query: str, result: str):
        """
        缓存检索结果（增量写入，多进程安全）。
        
        使用 INSERT OR IGNORE 避免并发写入冲突。
        
        Args:
            query: 查询字符串
            result: 格式化后的检索结果
        """
        if not self._is_valid_result(result):
            return
        
        key = self._normalize_query(query)
        if not key:
            return
        
        # 检查内存缓存是否已存在
        with self._thread_lock:
            if key in self._local_cache:
                return  # 已存在，不重复写入
            # 更新内存缓存
            self._local_cache[key] = result
        
        # 写入 SQLite（INSERT OR IGNORE 避免冲突）
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO search_cache "
                    "(query_key, query_raw, result, result_length) VALUES (?, ?, ?, ?)",
                    (key, query, result, len(result))
                )
                conn.commit()
                self._stats["writes"] += 1
                
        except sqlite3.Error as e:
            logger.error(f"[LocalSearchCache] DB write error: {e}")
    
    def put_batch(self, entries: Dict[str, str]):
        """
        批量缓存检索结果（使用事务，大幅减少 IO）。
        
        Args:
            entries: {query: result} 字典
        """
        valid_entries: List[Tuple[str, str, str, int]] = []
        
        for query, result in entries.items():
            if not self._is_valid_result(result):
                continue
            key = self._normalize_query(query)
            if not key:
                continue
            
            # 检查内存缓存是否已存在
            with self._thread_lock:
                if key in self._local_cache:
                    continue
                self._local_cache[key] = result
            
            valid_entries.append((key, query, result, len(result)))
        
        if not valid_entries:
            return
        
        # 批量写入 SQLite（单个事务）
        try:
            with self._get_connection() as conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO search_cache "
                    "(query_key, query_raw, result, result_length) VALUES (?, ?, ?, ?)",
                    valid_entries
                )
                conn.commit()
                self._stats["writes"] += len(valid_entries)
                logger.debug(f"[LocalSearchCache] Batch inserted {len(valid_entries)} entries")
                
        except sqlite3.Error as e:
            logger.error(f"[LocalSearchCache] DB batch write error: {e}")
    
    def size(self) -> int:
        """返回缓存总条目数"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM search_cache")
                return cursor.fetchone()[0]
        except sqlite3.Error:
            return len(self._local_cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息。
        
        Returns:
            包含命中率等统计的字典
        """
        total_requests = (self._stats["memory_hits"] +
                          self._stats["db_hits"] +
                          self._stats["misses"])
        
        hit_rate = 0.0
        if total_requests > 0:
            hit_rate = (self._stats["memory_hits"] + self._stats["db_hits"]) / total_requests
        
        return {
            "db_path": self.db_path,
            "total_entries": self.size(),
            "memory_cache_size": len(self._local_cache),
            "memory_hits": self._stats["memory_hits"],
            "db_hits": self._stats["db_hits"],
            "misses": self._stats["misses"],
            "writes": self._stats["writes"],
            "hit_rate": f"{hit_rate:.2%}",
        }
    
    def refresh(self):
        """刷新内存缓存（重新预热）"""
        with self._thread_lock:
            self._local_cache.clear()
        self._warmup_cache()
        logger.info(f"[LocalSearchCache] Cache refreshed, {len(self._local_cache)} entries loaded")
    
    def clear(self):
        """清空所有缓存（谨慎使用）"""
        with self._thread_lock:
            self._local_cache.clear()
        
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM search_cache")
                conn.execute("VACUUM")  # 回收空间
                conn.commit()
            logger.info("[LocalSearchCache] Cache cleared")
        except sqlite3.Error as e:
            logger.error(f"[LocalSearchCache] Failed to clear cache: {e}")
    
    def optimize(self):
        """优化数据库（定期调用，减少碎片）"""
        try:
            with self._get_connection() as conn:
                conn.execute("PRAGMA optimize;")
                conn.execute("VACUUM;")
                conn.commit()
            logger.info("[LocalSearchCache] Database optimized")
        except sqlite3.Error as e:
            logger.warning(f"[LocalSearchCache] Optimization failed: {e}")
    
    def export_to_json(self, output_path: str):
        """
        导出缓存到 JSON 文件（用于备份或迁移）。
        
        Args:
            output_path: 输出文件路径
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT query_key, result FROM search_cache")
                data = {row[0]: row[1] for row in cursor}
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[LocalSearchCache] Exported {len(data)} entries to {output_path}")
        except Exception as e:
            logger.error(f"[LocalSearchCache] Export failed: {e}")
    
    def import_from_json(self, input_path: str):
        """
        从 JSON 文件导入缓存（用于迁移旧数据）。
        
        Args:
            input_path: 输入文件路径
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                logger.error("[LocalSearchCache] Invalid JSON format for import")
                return
            
            # 批量导入
            entries = []
            for key, result in data.items():
                if self._is_valid_result(result):
                    entries.append((key, key, result, len(result)))
            
            with self._get_connection() as conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO search_cache "
                    "(query_key, query_raw, result, result_length) VALUES (?, ?, ?, ?)",
                    entries
                )
                conn.commit()
            
            logger.info(f"[LocalSearchCache] Imported {len(entries)} entries from {input_path}")
            
            # 刷新内存缓存
            self.refresh()
            
        except Exception as e:
            logger.error(f"[LocalSearchCache] Import failed: {e}")


class LocalRetrievalTool(Tool):
    """
    A tool for dense search using the local retrieval server.

    This tool connects to a locally running dense retrieval server (launched via retrieval_launch.sh)
    and performs dense retrieval using E5 embeddings on the indexed Wikipedia corpus.
    
    Supports multiple server URLs for load balancing across multi-GPU deployments.
    
    Features:
    - Load balancing across multiple retrieval servers
    - Automatic failover on server errors
    - Result caching to avoid redundant queries
    """

    NAME = "local_search"
    DESCRIPTION = "Search for information using a dense retrieval server with Wikipedia corpus"

    def __init__(
        self,
        name: str = NAME,
        description: str = DESCRIPTION,
        server_url: str | list[str] | None = None,
        timeout: float = 30.0,
        max_results: int = 10,
        cache_dir: str | None = None,
        enable_cache: bool = True,
    ):
        """
        Initialize the Local Retrieval Tool.

        Args:
            name: Tool name
            description: Tool description
            server_url: URL(s) of the local retrieval server. Can be:
                - Single URL string: "http://127.0.0.1:8000"
                - Comma-separated URLs: "http://127.0.0.1:8000,http://127.0.0.1:8001"
                - List of URLs: ["http://127.0.0.1:8000", "http://127.0.0.1:8001"]
                - None: checks RETRIEVAL_SERVER_URL env var
            timeout: Request timeout in seconds
            max_results: Maximum number of results to return
            cache_dir: Directory for caching retrieval results. If None, checks 
                       LOCAL_SEARCH_CACHE_DIR env var, defaults to "./local_search_cache"
            enable_cache: Whether to enable caching (default: True)
        """
        # Use environment variable if server_url not provided
        if server_url is None:
            server_url = os.environ.get("RETRIEVAL_SERVER_URL", "http://127.0.0.1:8000")

        # Parse server URLs - support comma-separated list for multi-GPU
        if isinstance(server_url, str):
            self.server_urls = [url.strip().rstrip("/") for url in server_url.split(",")]
        else:
            self.server_urls = [url.rstrip("/") for url in server_url]
        
        # For backward compatibility
        self.server_url = self.server_urls[0]
        
        self.timeout = timeout
        self.max_results = max_results
        self.client = httpx.Client(timeout=timeout)
        
        # Track server health for smart load balancing
        self._server_failures: dict[str, int] = {url: 0 for url in self.server_urls}
        self._max_failures = 3  # Mark server as unhealthy after this many consecutive failures

        # 初始化缓存
        self.enable_cache = enable_cache
        self.cache: Optional[LocalSearchCache] = None
        
        if enable_cache:
            if cache_dir is None:
                cache_dir = os.environ.get("LOCAL_SEARCH_CACHE_DIR", "./local_search_cache")
            self.cache = LocalSearchCache(cache_dir)

        super().__init__(name=name, description=description)

        # Test server connection
        self._test_connection()
        
        if len(self.server_urls) > 1:
            logger.info(f"LocalRetrievalTool initialized with {len(self.server_urls)} servers for load balancing")

    def _test_connection(self):
        """Test connection to the retrieval server(s)."""
        healthy_servers = []
        for url in self.server_urls:
            try:
                response = self.client.get(f"{url}/health")
                if response.status_code == 200:
                    logger.debug(f"Successfully connected to retrieval server at {url}")
                    healthy_servers.append(url)
                else:
                    logger.warning(f"Retrieval server {url} returned status code {response.status_code}")
            except Exception as e:
                logger.debug(f"Could not connect to retrieval server {url}: {e}")
        
        if not healthy_servers:
            logger.warning("No healthy retrieval servers found during initialization")
        elif len(healthy_servers) < len(self.server_urls):
            logger.warning(f"Only {len(healthy_servers)}/{len(self.server_urls)} servers are healthy")

    def _get_server_url(self) -> str:
        """Get a server URL using weighted random selection based on health."""
        # Filter to healthy servers (fewer than max_failures consecutive failures)
        healthy_urls = [url for url, failures in self._server_failures.items() if failures < self._max_failures]
        
        if not healthy_urls:
            # All servers marked unhealthy, reset and try all
            logger.warning("All servers marked unhealthy, resetting failure counts")
            self._server_failures = {url: 0 for url in self.server_urls}
            healthy_urls = self.server_urls
        
        # Random selection for load balancing
        return random.choice(healthy_urls)

    def _mark_server_success(self, url: str):
        """Mark a server as successful, resetting its failure count."""
        self._server_failures[url] = 0

    def _mark_server_failure(self, url: str):
        """Mark a server failure, incrementing its failure count."""
        self._server_failures[url] = self._server_failures.get(url, 0) + 1

    @property
    def json(self):
        """Return tool JSON schema for LLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query to retrieve relevant documents"},
                        "top_k": {"type": "integer", "description": f"Number of results to return (default: {self.max_results})", "minimum": 1, "maximum": 50},
                    },
                    "required": ["query"],
                },
            },
        }

    def _format_search_results(self, results: list[dict[str, Any]]) -> str:
        """Format search results for LLM consumption."""
        if not results:
            return "No relevant documents found."

        formatted_results = []
        for i, result in enumerate(results[: self.max_results], 1):
            # Extract key information
            doc_id = result.get("id", f"doc_{i}")
            content = result.get("content", "")  # Fixed: use "content" not "contents"
            score = result.get("score", 0.0)

            # Truncate content if too long (keep first 300 characters)
            if len(content) > 300:
                content = content[:300] + "..."

            formatted_result = f"[Document {i}] (ID: {doc_id}, Score: {score:.3f})\n{content}\n"
            formatted_results.append(formatted_result)

        return "\n".join(formatted_results)

    def forward(self, query: str, top_k: int | None = None, _retry_count: int = 0, _tried_servers: set | None = None) -> ToolOutput:
        """
        Execute a search query using the dense retrieval server.

        Args:
            query: Search query
            top_k: Number of results to return
            _retry_count: Internal retry counter (do not set manually)
            _tried_servers: Set of servers already tried in this request (do not set manually)

        Returns:
            ToolOutput: Search results or error message
        """
        max_retries = 3
        retry_delay = 0.5  # seconds
        
        if _tried_servers is None:
            _tried_servers = set()
        
        # 【缓存检查】尝试从缓存获取结果
        if self.cache and _retry_count == 0:  # 只在首次请求时检查缓存
            cached_result = self.cache.get(query)
            if cached_result:
                logger.debug(f"[LocalRetrievalTool] Cache HIT for query: {query[:50]}...")
                return ToolOutput(
                    name=self.name, 
                    output=cached_result, 
                    metadata={"query": query, "cache_hit": True, "retriever_type": "dense"}
                )
        
        # Select a server (load balancing)
        server_url = self._get_server_url()
        
        try:
            # Use provided parameters or defaults
            top_k = top_k or self.max_results

            # Prepare request payload
            payload = {
                "query": query,
                "top_k": min(top_k, 50),  # Cap at 50 results
            }

            # Make request to retrieval server
            response = self.client.post(f"{server_url}/retrieve", json=payload)

            if not response.is_success:
                # 标记服务器失败
                self._mark_server_failure(server_url)
                _tried_servers.add(server_url)
                
                # 检查是否是可重试的错误 (503 - GPU 显存问题)
                if response.status_code == 503 and _retry_count < max_retries:
                    try:
                        error_data = response.json()
                        if error_data.get("retry"):
                            import time
                            time.sleep(retry_delay * (2 ** _retry_count))  # 指数退避
                            # 尝试其他服务器
                            return self.forward(query, top_k, _retry_count + 1, _tried_servers)
                    except Exception:
                        pass
                
                # 如果还有未尝试的服务器，尝试下一个
                untried_servers = set(self.server_urls) - _tried_servers
                if untried_servers and _retry_count < max_retries:
                    return self.forward(query, top_k, _retry_count + 1, _tried_servers)
                
                error_msg = f"Retrieval server error: {response.status_code}"
                if response.content:
                    try:
                        error_data = response.json()
                        error_msg += f" - {error_data.get('error', 'Unknown error')}"
                    except Exception:
                        error_msg += f" - {response.text}"

                return ToolOutput(name=self.name, error=error_msg)

            # 成功，重置服务器失败计数
            self._mark_server_success(server_url)
            
            # Parse response
            response_data = response.json()
            results = response_data.get("results", [])

            if not results:
                return ToolOutput(name=self.name, output="No relevant documents found for the query.")

            # Format results
            formatted_output = self._format_search_results(results)

            # 【缓存写入】保存结果到缓存
            if self.cache:
                self.cache.put(query, formatted_output)

            # Create metadata for potential downstream use
            metadata = {
                "query": query, 
                "num_results": len(results), 
                "retriever_type": "dense", 
                "server_url": server_url,
                "cache_hit": False
            }

            return ToolOutput(name=self.name, output=formatted_output, metadata=metadata)

        except httpx.TimeoutException:
            self._mark_server_failure(server_url)
            _tried_servers.add(server_url)
            # 尝试其他服务器
            untried_servers = set(self.server_urls) - _tried_servers
            if untried_servers and _retry_count < max_retries:
                return self.forward(query, top_k, _retry_count + 1, _tried_servers)
            return ToolOutput(name=self.name, error=f"Request timeout after {self.timeout} seconds. Please check if the retrieval server is running.")
        except httpx.ConnectError:
            self._mark_server_failure(server_url)
            _tried_servers.add(server_url)
            # 尝试其他服务器
            untried_servers = set(self.server_urls) - _tried_servers
            if untried_servers and _retry_count < max_retries:
                return self.forward(query, top_k, _retry_count + 1, _tried_servers)
            return ToolOutput(name=self.name, error=f"Could not connect to retrieval servers. Please ensure the servers are running.")
        except Exception as e:
            return ToolOutput(name=self.name, error=f"Unexpected error: {str(e)}")

    def get_cache_stats(self) -> dict:
        """
        获取缓存统计信息。
        
        Returns:
            包含缓存统计的字典
        """
        if self.cache:
            stats = self.cache.get_stats()
            stats["enabled"] = True
            return stats
        return {"enabled": False}

    def __del__(self):
        """Clean up HTTP client."""
        try:
            if hasattr(self, "client"):
                self.client.close()
        except Exception:
            pass


# Convenience function for tool registry
def create_local_retrieval_tool(
    server_url: str | list[str] = "http://127.0.0.1:8000",
    max_results: int = 10,
    cache_dir: str | None = None,
    enable_cache: bool = True,
) -> LocalRetrievalTool:
    """
    Create a LocalRetrievalTool instance with specified configuration.

    Args:
        server_url: URL(s) of the dense retrieval server. Can be:
            - Single URL: "http://127.0.0.1:8000"
            - Comma-separated: "http://127.0.0.1:8000,http://127.0.0.1:8001"
            - List: ["http://127.0.0.1:8000", "http://127.0.0.1:8001"]
        max_results: Maximum number of results to return
        cache_dir: Directory for caching results (default: ./local_search_cache)
        enable_cache: Whether to enable caching (default: True)

    Returns:
        LocalRetrievalTool instance
    """
    return LocalRetrievalTool(
        server_url=server_url, 
        max_results=max_results,
        cache_dir=cache_dir,
        enable_cache=enable_cache,
    )
