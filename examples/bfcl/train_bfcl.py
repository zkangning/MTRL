import hydra
import logging
import requests
import json
from typing import List, Dict, Any

# RLLM 核心组件
from rllm.data.dataset import DatasetRegistry, Dataset
from rllm.trainer.agent_trainer import AgentTrainer

# 你自定义的 BFCL 组件
from rllm.environments.tools.bfcl_env_v2 import BFCLEnvironment
from rllm.agents.bfcl_agent import BFCLReadyAgent

logger = logging.getLogger(__name__)

def fetch_and_register_bfcl_data(base_url: str, dataset_name: str = "bfcl_data"):
    """
    从 BFCL Server 获取所有可用的 task_id，并注册到 DatasetRegistry 中。
    这样 Verl 训练时就会遍历这些 task_id。
    """
    # 检查是否已经存在，如果存在且不强制刷新，可以直接加载
    # 但为了保证 task 列表最新，这里每次运行前检查一下
    if DatasetRegistry.dataset_exists(dataset_name, "train"):
        logger.info(f"Dataset {dataset_name} already exists. Skipping fetch.")
        return

    logger.info(f"Fetching BFCL task profiles from {base_url}...")
    try:
        # 构造请求获取训练集任务列表
        # 注意：这里假设你的 Env Server 有 /get_env_profile 接口
        resp = requests.post(f"{base_url}/get_env_profile", json={
            "env_type": "bfcl", 
            "params": {"split": "train"}
        }, timeout=30)
        
        data = resp.json()
        # 兼容不同的返回格式 (list of strings 或 list of dicts)
        if isinstance(data, dict):
            task_list = data.get("data", []) or data.get("task_ids", [])
        else:
            task_list = data
            
        if not task_list:
            raise ValueError("Received empty task list from BFCL Server.")

        # 构造 Dataset 需要的格式
        # 原始数据放入 extra_info，verl 会将其传递给环境
        # 这里我们将 task_id 包装成 dict
        dataset_records = []
        for t_id in task_list:
            # 如果 server 返回的是字符串 ID
            if isinstance(t_id, str):
                dataset_records.append({"task_id": t_id})
            elif isinstance(t_id, dict):
                dataset_records.append(t_id)

        # 注册训练集
        logger.info(f"Registering {len(dataset_records)} tasks to dataset '{dataset_name}' (train)...")
        DatasetRegistry.register_dataset(dataset_name, dataset_records, split="train")
        
        # 为了避免报错，注册一个小的验证集 (可以是训练集的子集)
        val_size = min(len(dataset_records), 32)
        DatasetRegistry.register_dataset(dataset_name, dataset_records[:val_size], split="test")

    except Exception as e:
        logger.error(f"Failed to fetch/register BFCL dataset: {e}")
        logger.warning("Falling back to dummy dataset for debugging purposes.")
        # Fallback: 注册一个 Dummy 数据，防止程序直接崩溃，方便调试环境连通性
        dummy_data = [{"task_id": "multi_turn_base_1"}]
        DatasetRegistry.register_dataset(dataset_name, dummy_data, split="train")
        DatasetRegistry.register_dataset(dataset_name, dummy_data, split="test")

@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    # 1. 动态注册数据集
    # 从 config 或默认值获取 base_url
    # 注意：这里假设 env_args.base_url 在 config 中，或者我们硬编码
    base_url = "http://localhost:8801" 
    dataset_name = "bfcl_data"
    
    fetch_and_register_bfcl_data(base_url, dataset_name)

    # 2. 加载数据集
    train_dataset = DatasetRegistry.load_dataset(dataset_name, "train")
    test_dataset = DatasetRegistry.load_dataset(dataset_name, "test")
    
    logger.info(f"Train dataset size: {len(train_dataset)}")

    # 3. 配置 Environment 参数
    # 注意：verl 在运行时，会从 dataset 中提取 extra_info (包含 task_id)
    # 但是 AgentTrainer 的 TaskRunner 需要支持将 task_id 传入 env.reset()
    # 即使不支持，BFCLEnvironment 也会随机采样，保证训练进行
    env_args = {
        "base_url": base_url,
        "env_type": "bfcl",
        "max_steps": config.rllm.agent.max_steps,
    }

    # 4. 配置 Agent 参数
    agent_args = {
        "parser_name": "qwen",
        # 初始 System Prompt，后续会被 Env 返回的 Prompt 覆盖
        "system_prompt": "You are a helpful assistant.", 
    }

    # 5. 初始化并运行 Trainer
    trainer = AgentTrainer(
        agent_class=BFCLReadyAgent,
        env_class=BFCLEnvironment,
        agent_args=agent_args,
        env_args=env_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
