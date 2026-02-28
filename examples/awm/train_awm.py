"""
AWM (Agentic World Model) Training Script for RLLM

This script trains an agent on AWM-generated virtual environments.
AWM provides 1000 diverse scenarios with simulated APIs and databases.

Usage:
    python train_awm.py
    python train_awm.py --config-name=agent_ppo_trainer_awm

Requirements:
    - HuggingFace dataset: Snowflake/AgenticWorldModel
    - AWM dependencies: mcp, mcp-agent, fastapi, uvicorn
"""

import hydra
import logging
import os
import random
import sys
from typing import List, Dict, Any
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.rewards.reward_fn import awm_reward_fn
from rllm.environments.awm import AWMEnvironment
from rllm.agents.awm_agent import AWMAgent
from rllm.agents.awm_prompts import AWM_SYSTEM_PROMPT
from rllm.data.utils import load_awm_dataset


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
    verification_mode: str = "pure_code"
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
    logger.info(f">>> Dataset: {dataset_path}")
    logger.info(f">>> Train scenarios: {train_scenarios}, Test scenarios: {test_scenarios}")
    logger.info(f">>> Tasks per scenario: {tasks_per_scenario}")
    logger.info(f">>> Verification mode: {verification_mode}")
    
    # Load training data
    train_data = load_awm_dataset(
        dataset_path=dataset_path,
        split="train",
        num_scenarios=train_scenarios,
        tasks_per_scenario=tasks_per_scenario,
        verification_mode=verification_mode
    )
    
    # Load test data
    test_data = load_awm_dataset(
        dataset_path=dataset_path,
        split="test",
        num_scenarios=test_scenarios,
        tasks_per_scenario=tasks_per_scenario,
        verification_mode=verification_mode
    )
    
    # Register datasets
    DatasetRegistry.register_dataset(dataset_name, train_data, split="train")
    DatasetRegistry.register_dataset(dataset_name, test_data, split="test")
    
    logger.info(f">>> Dataset prepared: {len(train_data)} train, {len(test_data)} test samples")
    
    return dataset_name


@hydra.main(config_path="../../rllm/trainer/config", config_name="agent_ppo_trainer", version_base="1.1")
def main(config):
    """Main training function."""
    
    # ============================================================
    # Configuration
    # ============================================================
    dataset_path = config.data.get("dataset_path", "Snowflake/AgenticWorldModel")
    train_scenarios = config.data.get("train_scenarios", 100)
    test_scenarios = config.data.get("test_scenarios", 20)
    tasks_per_scenario = config.data.get("tasks_per_scenario", 10)
    verification_mode = config.data.get("verification_mode", "pure_code")
    
    # Environment configuration
    max_steps = config.rllm.agent.get("max_steps", 30)
    server_start_timeout = config.rllm.agent.get("server_start_timeout", 30.0)
    
    logger.info(">>> AWM Training Configuration:")
    logger.info(f"  Dataset: {dataset_path}")
    logger.info(f"  Train scenarios: {train_scenarios}")
    logger.info(f"  Test scenarios: {test_scenarios}")
    logger.info(f"  Tasks per scenario: {tasks_per_scenario}")
    logger.info(f"  Verification mode: {verification_mode}")
    logger.info(f"  Max steps per episode: {max_steps}")
    
    # ============================================================
    # Prepare Dataset
    # ============================================================
    dataset_name = prepare_awm_dataset(
        dataset_path=dataset_path,
        train_scenarios=train_scenarios,
        test_scenarios=test_scenarios,
        tasks_per_scenario=tasks_per_scenario,
        verification_mode=verification_mode
    )
    
    train_dataset = DatasetRegistry.load_dataset(dataset_name, "train")
    test_dataset = DatasetRegistry.load_dataset(dataset_name, "test")
    
    # ============================================================
    # Environment Arguments
    # ============================================================
    # Note: AWMEnvironment requires dynamic initialization per task
    # The actual env_code, db_path, etc. are provided in task's extra_info
    env_args = {
        "reward_fn": awm_reward_fn,
        "max_steps": max_steps,
        "server_start_timeout": server_start_timeout,
        "server_host": "127.0.0.1",
    }
    
    # ============================================================
    # Agent Arguments
    # ============================================================
    agent_args = {
        "system_prompt": AWM_SYSTEM_PROMPT,
        "parser_name": config.rllm.agent.get("parser_name", "qwen"),
        "max_steps": max_steps,
    }
    
    # ============================================================
    # Initialize Trainer
    # ============================================================
    logger.info(">>> Initializing AWM Trainer...")
    
    trainer = AgentTrainer(
        agent_class=AWMAgent,
        env_class=AWMEnvironment,
        agent_args=agent_args,
        env_args=env_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    
    # ============================================================
    # Start Training
    # ============================================================
    logger.info(">>> Starting AWM Training...")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        # Cleanup any remaining environments
        logger.info(">>> Cleaning up...")


if __name__ == "__main__":
    main()