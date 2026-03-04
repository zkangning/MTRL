set -x

# ================= 环境变量设置 =================
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export BASE_MODEL_PATH="/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/base_models/Qwen3-4B"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

# AWM HuggingFace 数据集路径
export AWM_DATA_DIR="/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/awm_data"

# 预检查数据目录（precheck_scenarios.py 的输出，包含已过滤的 train.parquet + test.parquet）
# 设置此项后，训练将跳过数据加载和 DB 预检查，直接使用已过滤的数据
export AWM_PREFILTERED_DIR="/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/awm_data/awm_precheck/s1000_t10_pure_code"

# ================= 任务级别配置说明 =================
# AWM 任务默认值（在 rllm/config/task_config.py 中定义）：
#   awm: prompt=2048, response=15360, steps=30
#
# 可通过 +task_configs.awm.xxx=yyy 覆盖
# AWM 数据参数通过 +data.xxx 传入 train_awm.py

# ================= 启动训练 =================
python3 -m examples.awm.train_awm \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.val_batch_size=500 \
    data.max_prompt_length=2048 \
    data.max_response_length=15360 \
    \
    +data.dataset_path="$AWM_DATA_DIR" \
    +data.prefiltered_dir="$AWM_PREFILTERED_DIR" \
    +data.train_scenarios=1000 \
    +data.test_scenarios=200 \
    +data.tasks_per_scenario=10 \
    +data.verification_mode=pure_code \
    +data.precheck_db=False \
    \
    '+task_configs.awm.max_prompt_length=2048' \
    '+task_configs.awm.max_response_length=15360' \
    '+task_configs.awm.max_steps=20' \
    \
    actor_rollout_ref.model.path=$BASE_MODEL_PATH \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-agent' \
    trainer.experiment_name='4b-awm' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=4 \
    \
    rllm.agent.max_steps=30 \
    +rllm.agent.max_parallel_agents=64 \
    +rllm.env.server_start_timeout=120.0 \
    +rllm.env.tool_call_timeout=30.0 \
    +rllm.env.prestart_server=True \
    +rllm.env.prestart_workers=64 \
    +rllm.agent.engine_args.keep_executor_alive=True \
    rllm.stepwise_advantage.enable=False
