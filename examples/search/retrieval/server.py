#!/usr/bin/env python3
"""
Dense-only retrieval server for Search training.
Provides E5 embeddings + FAISS dense indexing.

Usage:
    python server.py --data_dir ./search_data/prebuilt_indices --port 8000
"""

import argparse
import gc
import json
import threading
import time
from pathlib import Path
from typing import Any

import faiss
import torch
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer


class LocalRetriever:
    """Dense-only retrieval system using FAISS."""

    def __init__(self, data_dir: str, cleanup_interval: int = 1000):
        self.data_dir = Path(data_dir)
        self.corpus = []
        self.dense_index = None
        self.encoder = SentenceTransformer("intfloat/e5-base-v2")
        self.encoder.eval()  # 设置为评估模式
        
        # 显存管理
        self._request_count = 0
        self._cleanup_interval = cleanup_interval
        self._lock = threading.Lock()

        self._load_data()

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

    def _maybe_cleanup(self):
        """定期清理 CUDA 缓存以防止显存累积。"""
        with self._lock:
            self._request_count += 1
            should_cleanup = self._request_count >= self._cleanup_interval
            if should_cleanup:
                self._request_count = 0
        
        if should_cleanup:
            torch.cuda.empty_cache()
            gc.collect()

    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        """Dense retrieval using FAISS."""
        try:
            # 使用 no_grad 避免保留计算图，防止显存泄漏
            with torch.no_grad():
                query_vector = self.encoder.encode(
                    [f"query: {query}"],
                    convert_to_numpy=True,
                    show_progress_bar=False
                ).astype("float32")
            
            scores, indices = self.dense_index.search(query_vector, k)

            results = []
            for score, idx in zip(scores[0], indices[0], strict=False):
                # FAISS returns -1 for invalid indices (not enough results)
                if idx >= 0 and idx < len(self.corpus):
                    results.append({"content": self.corpus[idx], "score": float(score)})
            return results
        finally:
            # 每次请求后检查是否需要清理
            self._maybe_cleanup()


# Flask app
app = Flask(__name__)
retriever = None


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "corpus_size": len(retriever.corpus), "index_type": "dense_only", "index_loaded": retriever.dense_index is not None})


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

        formatted_results = [{"id": f"doc_{i}", "content": result["content"], "score": result["score"]} for i, result in enumerate(results, 1)]

        return jsonify({"query": query, "method": "dense", "results": formatted_results, "num_results": len(formatted_results)})

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # 处理 CUDA OOM 和相关错误
        error_str = str(e)
        if "CUDA" in error_str or "cublas" in error_str.lower() or "out of memory" in error_str.lower():
            print(f"[ERROR] CUDA memory error: {e}")
            # 强制清理显存
            torch.cuda.empty_cache()
            gc.collect()
            return jsonify({"error": "GPU memory exhausted, please retry", "retry": True}), 503
        raise

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[ERROR] Retrieve failed: {error_msg}")
        print(traceback.format_exc())
        return jsonify({"error": error_msg}), 500


def main():
    parser = argparse.ArgumentParser(description="Dense-only retrieval server")
    parser.add_argument("--data_dir", default="./search_data/prebuilt_indices", help="Directory containing corpus and dense index")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    start_time = time.time()
    # Initialize retriever
    global retriever
    try:
        retriever = LocalRetriever(args.data_dir)
        print(f"Dense retrieval server initialized with {len(retriever.corpus)} documents")
    except Exception as e:
        print(f"Failed to initialize retriever: {e}")
        return

    # Start server
    print(f"Took {time.time() - start_time} seconds to start the server")
    print(f"Starting dense retrieval server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
