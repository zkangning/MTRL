#!/bin/bash
# ===============================================================
# 优化版多任务训练脚本
# 
# 核心特性：
# 1. 动态任务级别配置 - 不同任务使用不同的 prompt/response 长度和 steps
# 2. 显存优化配置 - 针对 8xH20/H100 GPU 优化
# 3. 加速训练 - 合理的 ppo_max_token_len_per_gpu 设置
# ===============================================================
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600

# 模型路径
MODEL_PATH="/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/base_models/Qwen3-8B"

# ===============================================================
# 任务级别配置计算说明：
#
# 每种任务的 prompt + response 长度：
#   math:        2048 + 16384 = 18432
#   code:        4096 + 8192  = 12288  
#   search:      6144 + 4096  = 10240
#   tool_call:   4096 + 4096  = 8192
#   webshop:     4096 + 2048  = 6144
#
# 全局 max_prompt_length: max(2048, 4096, 6144) = 6144
# 全局 max_response_length: max(16384, 8192, 4096, 2048) = 16384
# 
# 推荐 ppo_max_token_len_per_gpu: 
#   = 2x * max(各任务 prompt+response) 
#   = 2 * 18432 = 36864
#
# 注意：如果只训练部分任务，可以进一步减小全局长度值
# ===============================================================

python3 -m examples.multi_task.train_composite \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=128 \
    \
    +data.math_num=3200 \
    +data.code_num=3200 \
    +data.search_num=0 \
    +data.tool_call_num=0 \
    +data.webshop_num=0 \
    +data.local_search_num=0 \
    +data.bfcl_num=0 \
    +data.dataset_name=math_code_optimized \
    trainer.experiment_name='math_code_optimized-8b' \
    \
    data.val_batch_size=256 \
    data.max_prompt_length=6144 \
    data.max_response_length=16384 \
    data.truncation='left' \
    data.filter_overlong_prompts=False \
    \
    '+task_configs.math.max_prompt_length=2048' \
    '+task_configs.math.max_response_length=16384' \
    '+task_configs.math.max_steps=10' \
    \
    '+task_configs.code.max_prompt_length=4096' \
    '+task_configs.code.max_response_length=8192' \
    '+task_configs.code.max_steps=1' \
    \
    '+task_configs.search.max_prompt_length=6144' \
    '+task_configs.search.max_response_length=4096' \
    '+task_configs.search.max_steps=15' \
    \
    '+task_configs.tool_call.max_prompt_length=4096' \
    '+task_configs.tool_call.max_response_length=4096' \
    '+task_configs.tool_call.max_steps=8' \
    \
    '+task_configs.webshop.max_prompt_length=4096' \
    '+task_configs.webshop.max_response_length=2048' \
    '+task_configs.webshop.max_steps=15' \
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
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=36864 \
    \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
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
    +actor_rollout_ref.rollout.repetition_penalty=1.05 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    \
    algorithm.kl_ctrl.kl_coef=0.0 \
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
    rllm.simple_length_penalty.enable=True \
    rllm.simple_length_penalty.baseline_len=2048 \
    rllm.simple_length_penalty.coefficient=0.1 \
    rllm.simple_length_penalty.max_penalty=0.2 \
    rllm.simple_length_penalty.warmup_steps=25
