"""
AWM (Agentic World Model) Training Script for RLLM

This script trains an agent on AWM-generated virtual environments.
AWM provides 1000 diverse scenarios with simulated APIs and databases.

Usage:
    python3 -m examples.awm.train_awm

Requirements:
    - HuggingFace dataset: Snowflake/AgenticWorldModel
    - AWM dependencies: mcp, mcp-agent, fastapi, uvicorn, sqlalchemy
"""

import hydra
import logging
import os
import random
import numpy as np

from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.rewards.reward_fn import awm_reward_fn
from rllm.environments.awm import AWMEnvironment
from rllm.agents.awm_agent import AWMAgent
from rllm.agents.awm_prompts import AWM_SYSTEM_PROMPT
from rllm.data.utils import load_awm_dataset, set_task_config_manager
from rllm.config.task_config import TaskConfigManager


# ============================================================
# Global Random Seed
# ============================================================
GLOBAL_SEED = 42


def set_all_random_seeds(seed: int = GLOBAL_SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


set_all_random_seeds(GLOBAL_SEED)
logger = logging.getLogger(__name__)


def prepare_awm_dataset(
    dataset_path: str = "Snowflake/AgenticWorldModel",
    train_scenarios: int = 100,
    test_scenarios: int = 20,
    tasks_per_scenario: int = 10,
    verification_mode: str = "pure_code",
) -> str:
    """
    Prepare AWM dataset for training and testing.

    Args:
        dataset_path: HuggingFace dataset path
        train_scenarios: Number of scenarios for training
        test_scenarios: Number of scenarios for testing
        tasks_per_scenario: Tasks to load per scenario (1-10)
        verification_mode: "pure_code" or "sql"

    Returns:
        Dataset name for registry
    """
    dataset_name = "awm_dataset"

    logger.info(">>> Preparing AWM Dataset...")
    logger.info(f"  Dataset: {dataset_path}")
    logger.info(f"  Train scenarios: {train_scenarios}, Test scenarios: {test_scenarios}")
    logger.info(f"  Tasks per scenario: {tasks_per_scenario}")
    logger.info(f"  Verification mode: {verification_mode}")

    # Load training data
    train_data = load_awm_dataset(
        dataset_path=dataset_path,
        split="train",
        num_scenarios=train_scenarios,
        tasks_per_scenario=tasks_per_scenario,
        verification_mode=verification_mode,
    )

    # Load test data
    test_data = load_awm_dataset(
        dataset_path=dataset_path,
        split="test",
        num_scenarios=test_scenarios,
        tasks_per_scenario=tasks_per_scenario,
        verification_mode=verification_mode,
    )

    # Register datasets
    DatasetRegistry.register_dataset(dataset_name, train_data, split="train")
    DatasetRegistry.register_dataset(dataset_name, test_data, split="test")

    logger.info(f">>> Dataset prepared: {len(train_data)} train, {len(test_data)} test samples")
    return dataset_name


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    """Main training function."""

    # ============================================================
    # Task Config Manager (支持 +task_configs.awm.xxx=yyy 覆盖)
    # ============================================================
    task_configs = config.get("task_configs", {})
    custom_task_configs = {}
    if task_configs:
        for task_type, params in task_configs.items():
            if isinstance(params, dict):
                custom_task_configs[task_type] = dict(params)

    task_config_manager = TaskConfigManager(custom_task_configs)
    set_task_config_manager(task_config_manager)
    logger.info("\n" + task_config_manager.summary())

    # ============================================================
    # AWM Data Configuration (通过 +data.xxx 覆盖)
    # ============================================================
    dataset_path = config.data.get("dataset_path", "Snowflake/AgenticWorldModel")
    train_scenarios = config.data.get("train_scenarios", 100)
    test_scenarios = config.data.get("test_scenarios", 20)
    tasks_per_scenario = config.data.get("tasks_per_scenario", 10)
    verification_mode = config.data.get("verification_mode", "pure_code")

    logger.info(">>> AWM Training Configuration:")
    logger.info(f"  Dataset: {dataset_path}")
    logger.info(f"  Train scenarios: {train_scenarios}")
    logger.info(f"  Test scenarios: {test_scenarios}")
    logger.info(f"  Tasks per scenario: {tasks_per_scenario}")
    logger.info(f"  Verification mode: {verification_mode}")

    # ============================================================
    # Prepare Dataset
    # ============================================================
    dataset_name = prepare_awm_dataset(
        dataset_path=dataset_path,
        train_scenarios=train_scenarios,
        test_scenarios=test_scenarios,
        tasks_per_scenario=tasks_per_scenario,
        verification_mode=verification_mode,
    )

    train_dataset = DatasetRegistry.load_dataset(dataset_name, "train")
    test_dataset = DatasetRegistry.load_dataset(dataset_name, "test")

    # ============================================================
    # Agent & Environment Arguments
    # ============================================================
    agent_args = {
        "system_prompt": AWM_SYSTEM_PROMPT,
        "parser_name": "qwen",
        "max_steps": config.rllm.agent.get("max_steps", 30),
    }

    env_args = {
        "reward_fn": awm_reward_fn,
        "server_host": "127.0.0.1",
        "server_start_timeout": 30.0,
    }

    # ============================================================
    # Initialize Trainer & Start Training
    # ============================================================
    trainer = AgentTrainer(
        agent_class=AWMAgent,
        env_class=AWMEnvironment,
        agent_args=agent_args,
        env_args=env_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )

    logger.info(">>> Starting AWM Training...")
    trainer.train()


if __name__ == "__main__":
    main()
