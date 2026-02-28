# AWM (Agentic World Model) Training for RLLM

This directory contains the implementation for training agents on AWM-generated virtual environments within the RLLM framework.

## Overview

[AWM (Agentic World Model)](https://github.com/Snowflake-Labs/agent-world-model) is a framework for generating diverse virtual environments with simulated APIs and databases. Each environment consists of:

- **Scenario**: A realistic business scenario (e.g., e-commerce, booking system)
- **Database**: SQLite database with schema and sample data
- **API**: FastAPI backend exposing business logic
- **MCP Server**: Model Context Protocol server for tool-based interaction
- **Tasks**: User tasks to complete within the environment
- **Verifier**: Automated verification code for reward computation

## Architecture

The integration consists of the following components:

### 1. Environment (`rllm/environments/awm/`)

- **`awm_env.py`**: [`AWMEnvironment`](rllm/environments/awm/awm_env.py) class
  - Manages MCP server lifecycle
  - Handles multi-turn tool-based interactions
  - Integrates with RLLM's `BaseEnv` interface

- **`awm_reward.py`**: Reward functions
  - [`AWMMCPPureCodeRewardFn`](rllm/environments/awm/awm_reward.py): Pure code-based verification
  - [`AWMMCPRewardFn`](rllm/environments/awm/awm_reward.py): LLM-augmented verification

### 2. Data Loading (`rllm/data/utils.py`)

- [`load_awm_dataset()`](rllm/data/utils.py): Loads AWM data from HuggingFace
- Supports filtering by scenarios and tasks
- Handles both training and test splits

### 3. Agent Prompts (`rllm/agents/awm_prompts.py`)

- [`AWM_SYSTEM_PROMPT`](rllm/agents/awm_prompts.py): System prompt for AWM tasks
- Tool formatters for MCP interaction

### 4. Training Script (`examples/awm/`)

- **`train_awm.py`**: Main training script
- **`run_awm.sh`**: Convenience launcher script
- **`configs/`**: Hydra configuration files

## Quick Start

### 1. Installation

Ensure you have the required dependencies:

```bash
# Install RLLM (if not already installed)
pip install -e .

# Install AWM dependencies
pip install mcp mcp-agent fastapi uvicorn sqlalchemy
```

### 2. Download AWM Dataset

The dataset will be automatically downloaded from HuggingFace when running training:

```python
from datasets import load_dataset

# Verify dataset access
dataset = load_dataset("Snowflake/AgenticWorldModel", data_files="gen_scenario.jsonl")
print(f"Available scenarios: {len(dataset['train'])}")
```

### 3. Run Training

#### Basic Training

```bash
cd examples/awm
bash run_awm.sh
```

#### Debug Mode (Quick Test)

```bash
bash run_awm.sh --debug
```

#### Custom Configuration

```bash
bash run_awm.sh \
    --train-scenarios 200 \
    --test-scenarios 50 \
    --tasks-per-scenario 5 \
    --batch-size 256 \
    --model-path Qwen/Qwen2.5-7B-Instruct
```

#### With Custom Config File

```bash
python train_awm.py --config-name=my_custom_config
```

## Configuration

### Data Configuration

```yaml
data:
  dataset_path: Snowflake/AgenticWorldModel
  train_scenarios: 100          # Number of scenarios for training
  test_scenarios: 20            # Number of scenarios for testing
  tasks_per_scenario: 10        # Tasks per scenario (1-10)
  verification_mode: pure_code  # pure_code or sql
```

### Agent Configuration

```yaml
rllm:
  agent:
    name: awm_agent
    max_steps: 30               # Maximum interaction steps
    trajectory_timeout: 120     # Timeout per episode
    parser_name: qwen           # Parser for tool calls
```

### Environment Configuration

```yaml
rllm:
  env:
    name: awm_env
    env_args:
      server_start_timeout: 30.0
      server_host: 127.0.0.1
```

## Dataset Structure

The AWM dataset on HuggingFace contains the following files:

| File | Entries | Description |
|------|---------|-------------|
| `gen_scenario.jsonl` | 1,000 | Synthesized scenario descriptions |
| `gen_tasks.jsonl` | 1,000 | 10 user tasks per scenario |
| `gen_db.jsonl` | 1,000 | Database schema definitions |
| `gen_sample.jsonl` | 1,000 | Sample data for initial database state |
| `gen_spec.jsonl` | 1,000 | API specifications |
| `gen_envs.jsonl` | 1,000 | MCP environment code (FastAPI + MCP server) |
| `gen_verifier.jsonl` | 10,000 | Verification code for LLM-as-Judge |
| `gen_verifier.pure_code.jsonl` | 10,000 | Verification code for pure code-based Judge |

## How It Works

### Episode Flow

1. **Reset**: 
   - Load scenario data and task description
   - Start MCP server with environment code
   - Initialize database with sample data
   - Return initial observation with system prompt

2. **Step**:
   - Agent generates response with tool calls
   - Parse `<tool_call>` tags from response
   - Execute tools via MCP server
   - Return observation with tool results

3. **Reward**:
   - When episode terminates, run verification code
   - Check if task is completed
   - Return binary reward (1.0 for success, 0.0 for failure)

### Tool Calling Format

Agents use the following format for tool calls:

```xml
<think>
I need to search for products in the e-commerce platform.
</think>

<tool_call>
{"name": "call_tool", "arguments": {"tool_name": "search_products", "arguments": {"query": "laptop"}}}
</tool_call>
```

## Advanced Usage

### Custom Reward Function

```python
from rllm.environments.awm import AWMMCPPureCodeRewardFn

class CustomAWMRewardFn(AWMMCPPureCodeRewardFn):
    def __call__(self, task_info, action):
        # Custom reward logic
        base_reward = super().__call__(task_info, action)
        
        # Add custom shaping
        if base_reward.reward > 0:
            # Bonus for shorter successful episodes
            num_steps = len(task_info.get("history", []))
            shaping = max(0, 1.0 - num_steps / 30)
            base_reward.reward += shaping * 0.1
        
        return base_reward
```

### Custom Environment Wrapper

```python
from rllm.environments.awm import AWMEnvironment

class CustomAWMEnv(AWMEnvironment):
    def reset(self, **kwargs):
        # Custom reset logic
        obs, info = super().reset(**kwargs)
        
        # Add custom observation processing
        obs['custom_field'] = 'custom_value'
        
        return obs, info
```

### Loading Specific Scenarios

```python
from rllm.data.utils import load_awm_dataset

# Load only e-commerce scenarios
train_data = load_awm_dataset(
    dataset_path="Snowflake/AgenticWorldModel",
    split="train",
    num_scenarios=50,
    tasks_per_scenario=5,
    verification_mode="pure_code"
)
```

## Troubleshooting

### MCP Server Timeout

If you encounter MCP server startup timeouts:

1. Increase timeout in config:
   ```yaml
   rllm:
     env:
       env_args:
         server_start_timeout: 60.0
   ```

2. Check system resources (CPU, memory)

3. Reduce parallel workers:
   ```yaml
   actor_rollout_ref:
     rollout:
       agent:
         num_workers: 2
   ```

### Database Permission Errors

Ensure the database directory has proper permissions:

```bash
mkdir -p outputs/awm_databases
chmod 755 outputs/awm_databases
```

### Port Conflicts

If you see port binding errors:

1. Check for processes using ports 8000-9000
2. Restart training (ports are randomly assigned)

## Performance Tips

1. **Reduce Server Overhead**: Use fewer `num_workers` if server startup is slow
2. **Cache Databases**: Reuse databases across episodes when possible
3. **Parallel Scenarios**: Different scenarios can run in parallel safely
4. **Verification Mode**: Use `pure_code` mode for faster rewards (no LLM calls)

## Citation

If you use AWM in your research, please cite:

```bibtex
@software{awm_2024,
  title = {Agentic World Model (AWM)},
  author = {Snowflake Labs},
  year = {2024},
  url = {https://github.com/Snowflake-Labs/agent-world-model}
}
```

## License

This integration follows the same license as the RLLM framework.