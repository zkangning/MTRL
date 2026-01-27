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

# 获取 RLLM 路径
RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")

# ================= 模型路径 =================
# 这里请替换为你实际的 Qwen3-32B 或其他 Base Model 路径
MODEL_PATH="/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/base_models/Qwen3-8B"

# ================= 启动训练 =================
python3 -m examples.multi_task.train_composite \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=128 \
    +data.math_num=0 \
    +data.code_num=0 \
    +data.bfcl_num=0 \
    +data.search_num=12800 \
    +data.tool_call_num=0 \
    +data.tool_call_data_path='/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/rllm/data/datasets/tool_call_data' \
    +data.dataset_name=math0_code0_tool0_search1w \
    trainer.experiment_name='math0_code0_tool0_search1w-8b-mixed-tasks' \
    data.val_batch_size=512 \
    data.max_prompt_length=6400 \
    data.max_response_length=20480 \
    data.truncation='left' \
    data.filter_overlong_prompts=False \
    \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=26880 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.repetition_penalty=1.05 \
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
    trainer.save_freq=100 \
    trainer.test_freq=25 \
    trainer.total_epochs=5 \
    \
    rllm.agent.max_steps=10 \
    rllm.stepwise_advantage.enable=False
