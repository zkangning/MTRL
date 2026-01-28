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
"""

import logging
import os
import random
from typing import Any

import httpx

from rllm.tools.tool_base import Tool, ToolOutput

logger = logging.getLogger(__name__)


class LocalRetrievalTool(Tool):
    """
    A tool for dense search using the local retrieval server.

    This tool connects to a locally running dense retrieval server (launched via retrieval_launch.sh)
    and performs dense retrieval using E5 embeddings on the indexed Wikipedia corpus.
    
    Supports multiple server URLs for load balancing across multi-GPU deployments.
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

            # Create metadata for potential downstream use
            metadata = {"query": query, "num_results": len(results), "retriever_type": "dense", "server_url": server_url}

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
) -> LocalRetrievalTool:
    """
    Create a LocalRetrievalTool instance with specified configuration.

    Args:
        server_url: URL(s) of the dense retrieval server. Can be:
            - Single URL: "http://127.0.0.1:8000"
            - Comma-separated: "http://127.0.0.1:8000,http://127.0.0.1:8001"
            - List: ["http://127.0.0.1:8000", "http://127.0.0.1:8001"]
        max_results: Maximum number of results to return

    Returns:
        LocalRetrievalTool instance
    """
    return LocalRetrievalTool(server_url=server_url, max_results=max_results)
