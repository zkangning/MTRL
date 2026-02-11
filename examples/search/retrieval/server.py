#!/usr/bin/env python3
"""
Dense-only retrieval server for Search training.
Provides E5 embeddings + FAISS dense indexing.

Usage:
    # 单 GPU 模式
    python server.py --data_dir ./search_data/prebuilt_indices --port 8000
    
    # 指定 GPU
    python server.py --data_dir ./search_data/prebuilt_indices --port 8000 --gpu_id 0
    
    # 多 GPU 模式（启动多个进程，每个进程使用不同的 GPU 和端口）
    python server.py --data_dir ./search_data/prebuilt_indices --port 8000 --gpu_id 0 &
    python server.py --data_dir ./search_data/prebuilt_indices --port 8001 --gpu_id 1 &
    # 然后使用 Nginx 或 HAProxy 做负载均衡
"""

import argparse
import gc
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import faiss
import torch
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer


class _HealthCheckFilter(logging.Filter):
    """过滤 /health 的 200 日志，避免频繁刷屏。"""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        # 只过滤 GET /health 的成功日志（200）
        if "GET /health" in msg and "200" in msg:
            return False
        return True


class LocalRetriever:
    """Dense-only retrieval system using FAISS with GPU memory management."""

    def __init__(
        self,
        data_dir: str,
        max_concurrent_gpu: int = 4,
        cleanup_interval: int = 100,
        gpu_id: int | None = None,
        queue_timeout: float = 120.0,
    ):
        """
        Initialize the retriever.
        
        Args:
            data_dir: Directory containing corpus and index files
            max_concurrent_gpu: Maximum concurrent GPU inference requests (default: 4)
            cleanup_interval: Number of requests between forced memory cleanup
            gpu_id: Specific GPU to use (None = use CUDA_VISIBLE_DEVICES or default)
            queue_timeout: Timeout in seconds for waiting in GPU queue (default: 120s)
        """
        self.data_dir = Path(data_dir)
        self.corpus = []
        self.dense_index = None
        self.gpu_id = gpu_id
        self.queue_timeout = queue_timeout
        
        # 设置使用的 GPU
        if gpu_id is not None:
            self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        
        # GPU 并发控制 - 允许更多并发以提高吞吐量
        # 80GB 显存的 GPU 可以安全地处理 4-8 个并发的 E5 编码请求
        self._gpu_semaphore = threading.Semaphore(max_concurrent_gpu)
        self._max_concurrent = max_concurrent_gpu
        
        # 显存管理
        self._request_count = 0
        self._cleanup_interval = cleanup_interval
        self._count_lock = threading.Lock()
        
        # 初始化模型并移动到指定设备
        self.encoder = SentenceTransformer("intfloat/e5-base-v2", device=self.device)
        self.encoder.eval()
        
        # 冻结模型参数，减少显存占用
        for param in self.encoder.parameters():
            param.requires_grad = False

        self._load_data()
        
        # 初始化后清理一次显存
        if torch.cuda.is_available():
            if gpu_id is not None:
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
        gc.collect()

    def _load_data(self):
        """Load corpus and dense index from data directory."""
        print(f"Loading data from {self.data_dir}")

        # Load corpus
        corpus_file = self.data_dir / "../wikipedia/wiki-18.jsonl"
        with open(corpus_file) as f:
            self.corpus = [json.loads(line) for line in f]
        print(f"Loaded corpus with {len(self.corpus)} documents")

        # Load dense index
        dense_index_file = self.data_dir / "e5_Flat.index"
        self.dense_index = faiss.read_index(str(dense_index_file))
        print(f"Loaded dense index with {self.dense_index.ntotal} vectors")

    def _maybe_cleanup(self, force: bool = False):
        """定期清理 CUDA 缓存以防止显存累积。"""
        with self._count_lock:
            self._request_count += 1
            should_cleanup = force or (self._request_count >= self._cleanup_interval)
            if should_cleanup:
                self._request_count = 0
        
        if should_cleanup:
            torch.cuda.empty_cache()
            gc.collect()

    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        """
        Dense retrieval using FAISS.
        
        使用信号量限制并发 GPU 推理，防止显存溢出。
        """
        # 使用信号量限制并发 GPU 访问，使用更长的超时时间
        acquired = self._gpu_semaphore.acquire(timeout=self.queue_timeout)
        if not acquired:
            raise TimeoutError(f"GPU inference queue timeout after {self.queue_timeout}s (max_concurrent={self._max_concurrent})")
        
        try:
            # 使用 no_grad 和 inference_mode 避免保留计算图
            with torch.no_grad(), torch.inference_mode():
                query_vector = self.encoder.encode(
                    [f"query: {query}"],
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=1,  # 明确指定 batch_size
                ).astype("float32")
            
            # FAISS 搜索（CPU 操作，不占用 GPU 显存）
            scores, indices = self.dense_index.search(query_vector, k)

            results = []
            for score, idx in zip(scores[0], indices[0], strict=False):
                if idx >= 0 and idx < len(self.corpus):
                    results.append({"content": self.corpus[idx], "score": float(score)})
            
            return results
        finally:
            # 确保释放信号量
            self._gpu_semaphore.release()
            # 每次请求后检查是否需要清理
            self._maybe_cleanup()


# Flask app
app = Flask(__name__)
retriever = None


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    # 添加 GPU 显存信息
    gpu_info = {}
    if torch.cuda.is_available():
        device_id = retriever.gpu_id if retriever.gpu_id is not None else 0
        gpu_info = {
            "gpu_id": device_id,
            "gpu_memory_allocated_mb": torch.cuda.memory_allocated(device_id) / 1024 / 1024,
            "gpu_memory_reserved_mb": torch.cuda.memory_reserved(device_id) / 1024 / 1024,
        }
    
    return jsonify({
        "status": "healthy",
        "corpus_size": len(retriever.corpus),
        "index_type": "dense_only",
        "index_loaded": retriever.dense_index is not None,
        "device": retriever.device,
        **gpu_info,
    })


@app.route("/retrieve", methods=["POST"])
def retrieve():
    """Main retrieval endpoint."""
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Missing 'query' in request"}), 400

        query = data["query"]
        k = data.get("top_k", data.get("k", 10))

        results = retriever.search(query=query, k=k)

        formatted_results = [
            {"id": f"doc_{i}", "content": result["content"], "score": result["score"]}
            for i, result in enumerate(results, 1)
        ]

        return jsonify({
            "query": query,
            "method": "dense",
            "results": formatted_results,
            "num_results": len(formatted_results),
        })

    except TimeoutError as e:
        # GPU 队列超时
        print(f"[WARN] GPU queue timeout: {e}")
        return jsonify({"error": "Server busy, please retry", "retry": True}), 503

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # 处理 CUDA OOM 和相关错误
        error_str = str(e)
        if "CUDA" in error_str or "cublas" in error_str.lower() or "out of memory" in error_str.lower():
            print(f"[ERROR] CUDA memory error: {e}")
            # 强制清理显存
            torch.cuda.empty_cache()
            gc.collect()
            retriever._maybe_cleanup(force=True)
            return jsonify({"error": "GPU memory exhausted, please retry", "retry": True}), 503
        raise

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[ERROR] Retrieve failed: {error_msg}")
        print(traceback.format_exc())
        return jsonify({"error": error_msg}), 500


@app.route("/clear_cache", methods=["POST"])
def clear_cache():
    """手动清理 GPU 缓存的端点。"""
    if torch.cuda.is_available():
        device_id = retriever.gpu_id if retriever.gpu_id is not None else 0
        with torch.cuda.device(device_id):
            torch.cuda.empty_cache()
    gc.collect()
    
    gpu_info = {}
    if torch.cuda.is_available():
        device_id = retriever.gpu_id if retriever.gpu_id is not None else 0
        gpu_info = {
            "gpu_id": device_id,
            "gpu_memory_allocated_mb": torch.cuda.memory_allocated(device_id) / 1024 / 1024,
            "gpu_memory_reserved_mb": torch.cuda.memory_reserved(device_id) / 1024 / 1024,
        }
    
    return jsonify({"status": "cache_cleared", **gpu_info})


def main():
    parser = argparse.ArgumentParser(description="Dense-only retrieval server")
    parser.add_argument("--data_dir", default="./search_data/prebuilt_indices", help="Directory containing corpus and dense index")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--gpu_id", type=int, default=None, help="GPU device ID to use (default: use CUDA_VISIBLE_DEVICES or 0)")
    parser.add_argument("--max_concurrent_gpu", type=int, default=4, help="Maximum concurrent GPU inference requests (default: 4, for 80GB GPU)")
    parser.add_argument("--cleanup_interval", type=int, default=100, help="Number of requests between memory cleanup")
    parser.add_argument("--queue_timeout", type=float, default=120.0, help="Timeout in seconds for waiting in GPU queue (default: 120)")

    args = parser.parse_args()

    start_time = time.time()
    # Initialize retriever
    global retriever
    try:
        retriever = LocalRetriever(
            args.data_dir,
            max_concurrent_gpu=args.max_concurrent_gpu,
            cleanup_interval=args.cleanup_interval,
            gpu_id=args.gpu_id,
            queue_timeout=args.queue_timeout,
        )
        print(f"Dense retrieval server initialized with {len(retriever.corpus)} documents")
        print(f"Max concurrent GPU requests: {args.max_concurrent_gpu}")
        print(f"Using device: {retriever.device}")
    except Exception as e:
        import traceback
        print(f"Failed to initialize retriever: {e}")
        traceback.print_exc()
        return

    # Start server
    print(f"Took {time.time() - start_time:.2f} seconds to start the server")
    print(f"Starting dense retrieval server on {args.host}:{args.port}")
    
    # 抑制 /health 200 的日志输出，避免刷屏
    logging.getLogger("werkzeug").addFilter(_HealthCheckFilter())

    # 使用 threaded=True 但通过信号量控制 GPU 并发
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
