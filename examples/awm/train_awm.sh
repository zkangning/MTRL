#!/bin/bash
# AWM (Agentic World Model) Training Script for RLLM
# 
# This script trains an agent on AWM-generated virtual environments.
# AWM provides 1000 diverse scenarios with simulated APIs and databases.
#
# Usage:
#   bash train_awm.sh                    # Run with default settings
#   bash train_awm.sh --debug            # Quick debug mode with fewer scenarios
#
# Requirements:
#   - HuggingFace dataset: Snowflake/AgenticWorldModel
#   - AWM dependencies: mcp, mcp-agent, fastapi, uvicorn, sqlalchemy

set -x

# ============================================================
# GPU and Environment Configuration
# ============================================================
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export BASE_MODEL_PATH=/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/ckpt/Qwen3-4B
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

# AWM-specific environment variables
export AWM_DATA_DIR="Snowflake/AgenticWorldModel"

# ============================================================
# AWM-specific Training Parameters (默认值)
# ============================================================
# 数据配置
TRAIN_SCENARIOS=100          # 训练场景数量 (AWM共有1000个场景)
TEST_SCENARIOS=20            # 测试场景数量
TASKS_PER_SCENARIO=10        # 每个场景的任务数 (1-10)
VERIFICATION_MODE="pure_code" # 验证模式: pure_code 或 sql

# AWM环境配置
MAX_STEPS=30                 # 每个episode最大步数
SERVER_START_TIMEOUT=60.0    # MCP服务器启动超时时间(秒)
SERVER_HOST="127.0.0.1"      # MCP服务器主机

# 模型和训练配置
TRAIN_BATCH_SIZE=32
VAL_BATCH_SIZE=500
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=8192

# PPO/GRPO配置
LEARNING_RATE=1e-6
PPO_MINI_BATCH_SIZE=32
PPO_MAX_TOKEN_LEN_PER_GPU=24000
CLIP_RATIO_HIGH=0.28
KL_COEF=0.001

# Rollout配置
TEMPERATURE=0.8
GPU_MEMORY_UTILIZATION=0.6
N_SAMPLES=8                  # 每个prompt采样数

# 训练控制
TOTAL_EPOCHS=4
SAVE_FREQ=50
TEST_FREQ=20
N_GPUS=8

# 日志配置
PROJECT_NAME='rllm-agent'
EXPERIMENT_NAME='4b-awm'

# ============================================================
# Parse Command Line Arguments
# ============================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            # Debug mode: fewer scenarios for quick testing
            TRAIN_SCENARIOS=10
            TEST_SCENARIOS=5
            TASKS_PER_SCENARIO=2
            TOTAL_EPOCHS=1
            SAVE_FREQ=2
            TEST_FREQ=1
            shift
            ;;
        --train-scenarios)
            TRAIN_SCENARIOS="$2"
            shift 2
            ;;
        --test-scenarios)
            TEST_SCENARIOS="$2"
            shift 2
            ;;
        --tasks-per-scenario)
            TASKS_PER_SCENARIO="$2"
            shift 2
            ;;
        --verification-mode)
            VERIFICATION_MODE="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --batch-size)
            TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --model-path)
            BASE_MODEL_PATH="$2"
            shift 2
            ;;
        --epochs)
            TOTAL_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================
# Validate AWM-specific Parameters
# ============================================================
if [[ "$TASKS_PER_SCENARIO" -lt 1 || "$TASKS_PER_SCENARIO" -gt 10 ]]; then
    echo "Error: tasks_per_scenario must be between 1 and 10"
    exit 1
fi

if [[ "$VERIFICATION_MODE" != "pure_code" && "$VERIFICATION_MODE" != "sql" ]]; then
    echo "Error: verification_mode must be 'pure_code' or 'sql'"
    exit 1
fi

echo "========================================"
echo "AWM Training Configuration"
echo "========================================"
echo "Train scenarios: $TRAIN_SCENARIOS"
echo "Test scenarios: $TEST_SCENARIOS"
echo "Tasks per scenario: $TASKS_PER_SCENARIO"
echo "Verification mode: $VERIFICATION_MODE"
echo "Max steps: $MAX_STEPS"
echo "Server timeout: $SERVER_START_TIMEOUT"
echo "Batch size: $TRAIN_BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $TOTAL_EPOCHS"
echo "========================================"

# Find the directory where rllm package is located
RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))" 2>/dev/null || echo "")

if [[ -z "$RLLM_DIR" ]]; then
    echo "Warning: Could not find rllm package. Make sure rllm is installed."
    RLLM_DIR="."
fi

# ============================================================
# Run Training
# ============================================================
python3 -m examples.awm.train_awm \
    algorithm.adv_estimator=grpo \
    data.dataset_path="$AWM_DATA_DIR" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.train_scenarios=$TRAIN_SCENARIOS \
    data.test_scenarios=$TEST_SCENARIOS \
    data.tasks_per_scenario=$TASKS_PER_SCENARIO \
    data.verification_mode=$VERIFICATION_MODE \
    actor_rollout_ref.model.path=$BASE_MODEL_PATH \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.n=$N_SAMPLES \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.default_hdfs_dir=null \
    rllm.agent.max_steps=$MAX_STEPS \
    rllm.agent.server_start_timeout=$SERVER_START_TIMEOUT \
    rllm.agent.server_host="$SERVER_HOST" \
    rllm.agent.parser_name=qwen \
    rllm.stepwise_advantage.enable=False \
    trainer.total_epochs=$TOTAL_EPOCHS
