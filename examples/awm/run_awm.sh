#!/bin/bash
# AWM (Agentic World Model) Training Script
# 
# This script launches training on AWM-generated virtual environments.
# 
# Usage:
#   bash run_awm.sh                    # Run with default config
#   bash run_awm.sh --config my_config # Run with custom config
#
# Requirements:
#   - HuggingFace dataset: Snowflake/AgenticWorldModel
#   - Python 3.9+
#   - Sufficient disk space for temporary databases

set -e  # Exit on error

# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default parameters
CONFIG_NAME="agent_ppo_trainer_awm"
OVERRIDES=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --train-scenarios)
            OVERRIDES="${OVERRIDES} data.train_scenarios=$2"
            shift 2
            ;;
        --test-scenarios)
            OVERRIDES="${OVERRIDES} data.test_scenarios=$2"
            shift 2
            ;;
        --tasks-per-scenario)
            OVERRIDES="${OVERRIDES} data.tasks_per_scenario=$2"
            shift 2
            ;;
        --model-path)
            OVERRIDES="${OVERRIDES} actor_rollout_ref.model.path=$2"
            shift 2
            ;;
        --batch-size)
            OVERRIDES="${OVERRIDES} data.train_batch_size=$2"
            shift 2
            ;;
        --max-steps)
            OVERRIDES="${OVERRIDES} rllm.agent.max_steps=$2"
            shift 2
            ;;
        --epochs)
            OVERRIDES="${OVERRIDES} trainer.total_epochs=$2"
            shift 2
            ;;
        --lr)
            OVERRIDES="${OVERRIDES} actor_rollout_ref.actor.optim.lr=$2"
            shift 2
            ;;
        --debug)
            OVERRIDES="${OVERRIDES} trainer.total_epochs=1 data.train_scenarios=5 data.test_scenarios=2"
            shift
            ;;
        --help)
            echo "AWM Training Script"
            echo ""
            echo "Usage: bash run_awm.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config NAME               Config file name (default: agent_ppo_trainer_awm)"
            echo "  --train-scenarios N         Number of training scenarios"
            echo "  --test-scenarios N          Number of test scenarios"
            echo "  --tasks-per-scenario N      Tasks per scenario (1-10)"
            echo "  --model-path PATH           Path to base model"
            echo "  --batch-size N              Training batch size"
            echo "  --max-steps N               Max steps per episode"
            echo "  --epochs N                  Number of training epochs"
            echo "  --lr FLOAT                  Learning rate"
            echo "  --debug                     Quick debug mode (1 epoch, few scenarios)"
            echo "  --help                      Show this help message"
            echo ""
            echo "Examples:"
            echo "  bash run_awm.sh --debug"
            echo "  bash run_awm.sh --train-scenarios 200 --batch-size 256"
            echo "  bash run_awm.sh --model-path /path/to/custom/model"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================
# Environment Setup
# ============================================================
echo "=============================================="
echo "AWM Training Launch"
echo "=============================================="
echo "Config: ${CONFIG_NAME}"
echo "Project Root: ${PROJECT_ROOT}"
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import rllm" 2>/dev/null || {
    echo "Error: rllm package not found. Please install it first:"
    echo "  cd ${PROJECT_ROOT} && pip install -e ."
    exit 1
}

# Check for AWM dependencies
python -c "import mcp, mcp_agent" 2>/dev/null || {
    echo "Warning: mcp or mcp_agent not found. Installing..."
    pip install mcp mcp-agent
}

# ============================================================
# Create necessary directories
# ============================================================
mkdir -p "${PROJECT_ROOT}/outputs/awm_databases"
mkdir -p "${PROJECT_ROOT}/logs/rllm-awm"
mkdir -p "${PROJECT_ROOT}/checkpoints/rllm-awm"

# ============================================================
# Launch Training
# ============================================================
cd "${PROJECT_ROOT}"

echo "Launching training..."
echo "Overrides: ${OVERRIDES}"
echo ""

python "${SCRIPT_DIR}/train_awm.py" \
    --config-name="${CONFIG_NAME}" \
    ${OVERRIDES}

echo ""
echo "=============================================="
echo "Training completed!"
echo "=============================================="