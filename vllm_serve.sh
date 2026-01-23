CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
    --model agentica-org/DeepScaleR-1.5B-Preview \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16 > ./log/vllm_serve_DeepScaleR-1.5B-Preview.log &

CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
    --model /mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/base_models/Qwen3-8B \
    --host 0.0.0.0 \
    --port 30000 > ./log/vllm_serve_Qwen3_8B.log &

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup vllm serve /mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/base_models/Qwen3-32B \
    --gpu-memory-utilization 0.9 \
    --served-model-name Qwen3-32B \
    --trust_remote_code \
    --port 8803 \
    --tensor-parallel-size 4 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 > ./log/vllm_serve_Qwen3_32B.log &

CUDA_VISIBLE_DEVICES=0 nohup vllm serve /mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/base_models/Qwen3-8B \
    --gpu-memory-utilization 0.95 \
    --served-model-name Qwen3-8B \
    --trust_remote_code \
    --port 8801 \
    --enable-auto-tool-choice \
    --enable-prefix-caching \
    --tool-call-parser hermes \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 > ./log/difficult_test/vllm_serve_Qwen3_8B_part1.log &

CUDA_VISIBLE_DEVICES=1 nohup vllm serve /mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/base_models/Qwen3-8B \
    --gpu-memory-utilization 0.95 \
    --served-model-name Qwen3-8B \
    --trust_remote_code \
    --port 8802 \
    --enable-auto-tool-choice \
    --enable-prefix-caching \
    --tool-call-parser hermes \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 > ./log/difficult_test/vllm_serve_Qwen3_8B_part2.log &

CUDA_VISIBLE_DEVICES=2 nohup vllm serve /mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/base_models/Qwen3-8B \
    --gpu-memory-utilization 0.95 \
    --served-model-name Qwen3-8B \
    --trust_remote_code \
    --port 8803 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 > ./log/difficult_test/vllm_serve_Qwen3_8B_part3.log &
