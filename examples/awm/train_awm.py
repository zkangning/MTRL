"""
AWM (Agentic World Model) Training Script for RLLM

This script trains an agent on AWM-generated virtual environments.
AWM provides 1000 diverse scenarios with simulated APIs and databases.

【独立管道】数据流说明：
  load_awm_dataset() 直接生成 verl 兼容的 dict 列表
  -> save_awm_parquet() 保存为 parquet 文件（绕过 DatasetRegistry / apply_verl_postprocessing）
  -> AgentTrainer 通过 config.data.train_files / val_files 直接读取
  -> RLHFDataset.__getitem__() 将 extra_info dict 存入 non_tensor_batch
  -> init_envs_and_agents() 展开 extra_info 传给 AWMEnvironment.from_dict()

Usage:
    python3 -m examples.awm.train_awm

Requirements:
    - HuggingFace dataset: Snowflake/AgenticWorldModel
    - AWM dependencies: mcp, mcp-agent, fastapi, uvicorn, sqlalchemy
"""

import hydra
import logging
import os
import json
import random
import numpy as np
import datasets as hf_datasets

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


def save_awm_parquet(data: list[dict], output_path: str) -> str:
    """
    将 load_awm_dataset() 返回的数据直接保存为 verl 兼容的 parquet 文件。
    
    【关键】绕过 DatasetRegistry.apply_verl_postprocessing()，避免数据被二次包装。
    
    verl 的 RLHFDataset 需要 parquet 包含：
      - prompt: List[Dict] (chat messages)
      - extra_info: Dict (包含 index 和 AWM 特定字段)
      - data_source: str
    
    Args:
        data: load_awm_dataset() 返回的 dict 列表
        output_path: 输出 parquet 文件路径
        
    Returns:
        保存的文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 直接使用 HuggingFace Datasets 保存，它能正确处理嵌套 dict/list 结构
    hf_dataset = hf_datasets.Dataset.from_list(data)
    hf_dataset.to_parquet(output_path)
    
    logger.info(f"Saved {len(data)} records to {output_path}")
    return output_path


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
    # Prepare Dataset — 独立管道，直接生成 verl parquet
    # ============================================================
    output_dir = os.path.join(
        config.data.get("output_dir", "/tmp/awm_data"),
        f"s{train_scenarios}_t{tasks_per_scenario}_{verification_mode}"
    )

    train_parquet_path = os.path.join(output_dir, "train.parquet")
    val_parquet_path = os.path.join(output_dir, "val.parquet")

    # 加载训练数据
    logger.info(">>> Loading AWM training data...")
    train_data = load_awm_dataset(
        dataset_path=dataset_path,
        split="train",
        num_scenarios=train_scenarios,
        tasks_per_scenario=tasks_per_scenario,
        verification_mode=verification_mode,
    )
    save_awm_parquet(train_data, train_parquet_path)

    # 加载验证数据
    logger.info(">>> Loading AWM validation data...")
    val_data = load_awm_dataset(
        dataset_path=dataset_path,
        split="test",
        num_scenarios=test_scenarios,
        tasks_per_scenario=tasks_per_scenario,
        verification_mode=verification_mode,
    )
    save_awm_parquet(val_data, val_parquet_path)

    logger.info(f">>> Dataset prepared: {len(train_data)} train, {len(val_data)} val samples")
    logger.info(f"  Train parquet: {train_parquet_path}")
    logger.info(f"  Val parquet:   {val_parquet_path}")

    # ============================================================
    # 直接设置 verl 数据路径（绕过 DatasetRegistry）
    # ============================================================
    config.data.train_files = train_parquet_path
    config.data.val_files = val_parquet_path

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
        # 不传 train_dataset / val_dataset，已直接设置 config.data.train_files
    )

    logger.info(">>> Starting AWM Training...")
    trainer.train()


if __name__ == "__main__":
    main()
