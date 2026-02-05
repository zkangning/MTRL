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
    - Cache stores query -> formatted results mapping in JSON format
    - Implementation follows SplitToolCache pattern from mcp_env.py
"""

import json
import logging
import os
import random
import threading
from typing import Any, Dict, Optional

import httpx

from rllm.tools.tool_base import Tool, ToolOutput

logger = logging.getLogger(__name__)


class LocalSearchCache:
    """
    本地检索缓存，用于减少重复的检索请求，加速 local_search 任务。
    
    与 SplitToolCache (mcp_env.py) 保持完全一致的实现：
    - 使用 threading.Lock 保证线程安全
    - 使用临时文件 + os.replace 保证原子写入
    - 自动持久化到 JSON 文件
    
    缓存格式：
    {
        "query1": "formatted_result1",
        "query2": "formatted_result2",
        ...
    }
    """
    
    def __init__(self, cache_dir: str, cache_filename: str = "local_search_cache.json"):
        """
        初始化缓存。
        
        Args:
            cache_dir: 缓存目录路径
            cache_filename: 缓存文件名
        """
        self.cache_dir = cache_dir
        self.cache_filename = cache_filename
        self.cache_path = os.path.join(cache_dir, cache_filename)
        
        # 创建缓存目录
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        
        # 线程锁（与 SplitToolCache 一致）
        self._lock = threading.Lock()
        
        # 内存缓存
        self._data: Dict[str, str] = {}
        
        # 加载已有缓存
        self._load()
        
        logger.info(f"[LocalSearchCache] Initialized with {len(self._data)} cached queries at {self.cache_path}")
    
    def _load(self):
        """从文件加载缓存数据"""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}
    
    def _save(self):
        """
        持久化缓存到文件。
        与 SplitToolCache._save_category() 完全一致。
        """
        with self._lock:
            try:
                temp_path = self.cache_path + ".tmp"
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(self._data, f, indent=2, ensure_ascii=False)
                os.replace(temp_path, self.cache_path)
            except Exception as e:
                logger.error(f"[LocalSearchCache] Failed to save cache: {e}")
    
    def _normalize_query(self, query: str) -> str:
        """
        规范化 query 作为缓存 key。
        
        Args:
            query: 原始查询字符串
            
        Returns:
            规范化后的查询字符串
        """
        # 去除首尾空白，转小写，压缩连续空格
        normalized = query.strip().lower()
        normalized = ' '.join(normalized.split())
        return normalized
    
    def _is_valid_result(self, result: str) -> bool:
        """
        判断检索结果是否值得缓存。
        与 SplitToolCache._is_valid_response() 保持类似逻辑。
        
        Args:
            result: 格式化后的检索结果
            
        Returns:
            是否应该缓存
        """
        if not result:
            return False
        
        # 错误结果不缓存（与 SplitToolCache 一致）
        if "execution failed" in result or result.strip().startswith("Error:"):
            logger.info(f"[LocalSearchCache] Detected error in result, skipping cache.")
            return False
        
        # 空结果不缓存
        if result == "No relevant documents found." or result == "No relevant documents found for the query.":
            return False
        
        # 结果太短不缓存（可能是错误，与 scrape_as_markdown 检查类似）
        if len(result.strip()) < 20:
            return False
        
        return True
    
    def get(self, query: str) -> Optional[str]:
        """
        获取缓存的检索结果。
        与 SplitToolCache.get() 一致。
        
        Args:
            query: 查询字符串
            
        Returns:
            缓存的结果，如果不存在则返回 None
        """
        key = self._normalize_query(query)
        if not key:
            return None
        return self._data.get(key)
    
    def put(self, query: str, result: str):
        """
        缓存检索结果。
        与 SplitToolCache.put() 完全一致。
        
        Args:
            query: 查询字符串
            result: 格式化后的检索结果
        """
        if not self._is_valid_result(result):
            return
        
        key = self._normalize_query(query)
        if not key:
            return
        
        with self._lock:
            if key not in self._data:
                self._data[key] = result
        
        # 只有在确实写入了新数据且有效时才保存
        self._save()
    
    def size(self) -> int:
        """返回缓存条目数"""
        return len(self._data)
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._data = {}
        self._save()
        logger.info("[LocalSearchCache] Cache cleared")


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
            return {
                "enabled": True,
                "cache_dir": self.cache.cache_dir,
                "cache_size": self.cache.size(),
            }
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
