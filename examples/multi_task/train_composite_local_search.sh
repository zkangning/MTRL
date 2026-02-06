#!/bin/bash
set -x

# ================= 环境变量设置 =================
# 根据实际情况修改 GPU 编号
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# vLLM 加速配置
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# 防止 BFCL Server 响应慢导致 vLLM 推理超时
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export BRIGHT_DATA_API_TOKEN="da9e7e42-730d-4fb7-8357-b3dafcd7cc93"

# ================= Local Search 多 GPU 配置 =================
# 多服务器负载均衡模式（推荐）- 逗号分隔多个服务器地址
# 每个服务器运行在不同的 GPU 上，客户端会自动进行负载均衡和故障转移
export RETRIEVAL_SERVER_URL="http://10.217.69.161:8000,http://10.217.69.161:8001,http://10.217.69.161:8002,http://10.217.69.161:8003,http://10.217.69.161:8004,http://10.217.69.161:8005,http://10.217.69.161:8006" # local_server_node 1
# export RETRIEVAL_SERVER_URL="http://10.217.69.175:8000,http://10.217.69.175:8001,http://10.217.69.175:8002,http://10.217.69.175:8003,http://10.217.69.175:8004,http://10.217.69.175:8005,http://10.217.69.175:8006" # local_server_node 2


# Local Search 缓存配置（可选）
# 缓存相同 query 的检索结果，避免重复请求，加速训练
# 使用 SQLite 数据库存储，支持高并发多进程读写（WAL 模式）
# 缓存数据库保存在 LOCAL_SEARCH_CACHE_DIR 目录下的 local_search_cache.db
export LOCAL_SEARCH_CACHE_DIR="/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/local_search_cache/single_local_search"


# Web Shop 任务配置
# 使用 1000 产品的小数据集和合成 goals（synthetic goals）
export WEBSHOP_DATA_DIR=/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/rllm/data/datasets/webshop
export WEBSHOP_INDEX_DIR=/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/rllm/data/datasets/webshop/search_engine
export WEBSHOP_USE_FULL_DATA=false  # 使用 1000 产品集（items_shuffle_1000.json, items_ins_v2_1000.json）

# 获取 RLLM 路径
RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")

# ================= 模型路径 =================
# 这里请替换为你实际的 Qwen3-32B 或其他 Base Model 路径
MODEL_PATH="/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/base_models/Qwen3-8B"


# ================= Webshop 数据配置 =================
# 使用 1000 产品的小数据集和合成 goals（synthetic goals）
# 合成 goals 基于产品属性和选项组合自动生成
#
# 数据划分：
#   - 测试集: goal_idx 范围 [0, 500) - 固定使用前 500 个目标
#   - 训练集: goal_idx 范围 [500, total_goals) - 数量由 webshop_num 参数决定
#
# 训练数据量通过 +data.webshop_num 参数控制

# ================= 任务级别配置说明 =================
# 【新功能】不同任务类型可以配置不同的参数：
#   - max_prompt_length: 最大 prompt 长度
#   - max_response_length: 最大 response 长度  
#   - max_steps: 最大交互步数
#
# 配置方式：通过 +task_configs.{task_type}.{param}={value} 格式
#
# 默认值（在 rllm/config/task_config.py 中定义）：
#   math:        prompt=4096,  response=16384, steps=5
#   code:        prompt=2048,  response=20480, steps=1
#   search:      prompt=1024,  response=6144,  steps=4
#   local_search: prompt=1024, response=6144,  steps=4
#   tool_call:   prompt=6400,  response=4096,  steps=1
#   webshop:     prompt=1024,  response=15360, steps=15
#   bfcl:        prompt=4096,  response=4096,  steps=5
#
# 【重要】全局 data.max_prompt_length 和 data.max_response_length 仍然需要设置：
# - 作为 padding 的全局上限
# - 应设置为所有任务中最大值的上界
# - 建议设为所有任务配置中最大值的 1.2-1.5 倍

# ================= 启动训练 =================
python3 -m examples.multi_task.train_composite \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=128 \
    +data.math_num=0 \
    +data.code_num=0 \
    +data.bfcl_num=0 \
    +data.search_num=0 \
    +data.tool_call_num=0 \
    +data.local_search_num=3200 \
    +data.webshop_num=0 \
    +data.webshop_path=null \
    +data.tool_call_data_path='/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/rllm/data/datasets/tool_call_data_1_28' \
    +data.dataset_name=m2_6_code0_tool0_localsearch3k_webshop0 \
    trainer.experiment_name='m2_6_code0_tool0_localsearch3k_webshop0-8b-mixed-tasks' \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=6400 \
    data.truncation='left' \
    data.filter_overlong_prompts=False \
    \
    '+task_configs.math.max_prompt_length=4096' \
    '+task_configs.math.max_response_length=16384' \
    '+task_configs.math.max_steps=5' \
    \
    '+task_configs.code.max_prompt_length=2048' \
    '+task_configs.code.max_response_length=20480' \
    '+task_configs.code.max_steps=1' \
    \
    '+task_configs.search.max_prompt_length=1024' \
    '+task_configs.search.max_response_length=6400' \
    '+task_configs.search.max_steps=4' \
    \
    '+task_configs.local_search.max_prompt_length=1024' \
    '+task_configs.local_search.max_response_length=6144' \
    '+task_configs.local_search.max_steps=4' \
    \
    '+task_configs.tool_call.max_prompt_length=6400' \
    '+task_configs.tool_call.max_response_length=6400' \
    '+task_configs.tool_call.max_steps=1' \
    \
    '+task_configs.webshop.max_prompt_length=1024' \
    '+task_configs.webshop.max_response_length=15360' \
    '+task_configs.webshop.max_steps=15' \
    \
    '+task_configs.bfcl.max_prompt_length=4096' \
    '+task_configs.bfcl.max_response_length=4096' \
    '+task_configs.bfcl.max_steps=5' \
    \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32000 \
    \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    +actor_rollout_ref.rollout.repetition_penalty=1.05 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    \
    algorithm.kl_ctrl.kl_coef=0.000 \
    rllm.mask_truncated_samples=True \
    \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-multitask-agent' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.total_epochs=10 \
    \
    rllm.agent.max_steps=15 \
    rllm.stepwise_advantage.enable=False \
    \
    rllm.length_penalty.enable=False \
    rllm.length_penalty.weight=0.1 \
    rllm.length_penalty.warmup_steps=0 \
    rllm.length_penalty.min_response_length=200 \
    \
    rllm.simple_length_penalty.enable=True \
    rllm.simple_length_penalty.baseline_len=2048 \
    rllm.simple_length_penalty.coefficient=0.1 \
    rllm.simple_length_penalty.max_penalty=0.2 \
    rllm.simple_length_penalty.warmup_steps=25
