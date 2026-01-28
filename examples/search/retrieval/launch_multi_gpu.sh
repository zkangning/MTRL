#!/bin/bash
# Multi-GPU retrieval server launcher with load balancing
#
# Usage:
#   ./launch_multi_gpu.sh <data_dir> <base_port> <num_gpus>
#
# Example:
#   ./launch_multi_gpu.sh ./search_data/prebuilt_indices 8000 4
#
# This will start 4 server instances on ports 8000, 8001, 8002, 8003
# using GPU 0, 1, 2, 3 respectively.

set -e

DATA_DIR=${1:-"./search_data/prebuilt_indices"}
BASE_PORT=${2:-8000}
NUM_GPUS=${3:-$(nvidia-smi -L | wc -l)}
HOST=${4:-"0.0.0.0"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_SCRIPT="${SCRIPT_DIR}/server.py"

echo "=========================================="
echo "Multi-GPU Retrieval Server Launcher"
echo "=========================================="
echo "Data directory: ${DATA_DIR}"
echo "Base port: ${BASE_PORT}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Host: ${HOST}"
echo "=========================================="

# 创建日志目录
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# 存储所有进程的 PID
PIDS=()

# 清理函数
cleanup() {
    echo ""
    echo "Shutting down all servers..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping server (PID: $pid)"
            kill "$pid" 2>/dev/null || true
        fi
    done
    echo "All servers stopped."
    exit 0
}

# 捕获退出信号
trap cleanup SIGINT SIGTERM

# 启动每个 GPU 上的服务器
for ((i=0; i<NUM_GPUS; i++)); do
    PORT=$((BASE_PORT + i))
    LOG_FILE="${LOG_DIR}/server_gpu${i}_port${PORT}.log"
    
    echo "Starting server on GPU ${i}, port ${PORT}..."
    
    python "${SERVER_SCRIPT}" \
        --data_dir "${DATA_DIR}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --gpu_id "${i}" \
        --max_concurrent_gpu 4 \
        --cleanup_interval 50 \
        > "${LOG_FILE}" 2>&1 &
    
    PIDS+=($!)
    echo "  PID: ${PIDS[-1]}, Log: ${LOG_FILE}"
done

echo ""
echo "=========================================="
echo "All servers started!"
echo "=========================================="
echo ""
echo "Server endpoints:"
for ((i=0; i<NUM_GPUS; i++)); do
    PORT=$((BASE_PORT + i))
    echo "  - http://${HOST}:${PORT} (GPU ${i})"
done
echo ""
echo "For load balancing, configure your client to use multiple endpoints,"
echo "or set up Nginx/HAProxy with the config below."
echo ""
echo "Press Ctrl+C to stop all servers."
echo ""

# 生成 Nginx 配置示例
NGINX_CONF="${SCRIPT_DIR}/nginx_lb.conf.example"
cat > "${NGINX_CONF}" << EOF
# Nginx load balancer configuration for multi-GPU retrieval servers
# Save this to /etc/nginx/conf.d/retrieval_lb.conf and restart nginx

upstream retrieval_backend {
    least_conn;  # 使用最少连接策略
EOF

for ((i=0; i<NUM_GPUS; i++)); do
    PORT=$((BASE_PORT + i))
    echo "    server 127.0.0.1:${PORT} weight=1;" >> "${NGINX_CONF}"
done

cat >> "${NGINX_CONF}" << EOF
}

server {
    listen ${BASE_PORT}0;  # 负载均衡端口，如 80000
    
    location / {
        proxy_pass http://retrieval_backend;
        proxy_connect_timeout 30s;
        proxy_read_timeout 60s;
        proxy_next_upstream error timeout http_503;
        proxy_next_upstream_tries 3;
    }
    
    location /health {
        proxy_pass http://retrieval_backend;
        proxy_connect_timeout 5s;
    }
}
EOF

echo "Nginx config example saved to: ${NGINX_CONF}"
echo ""

# 等待所有进程
wait
