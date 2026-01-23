import asyncio
import argparse
import logging
import json
import random
import requests
from typing import List, Dict, Any

# RLLM 基础组件
from transformers import AutoTokenizer
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.data.dataset import DatasetRegistry
from rllm.utils import colorful_print
from rllm.utils.compute_pass_at_k import save_clean_trajectories

# 引入 Reward Functions
from rllm.rewards.reward_fn import math_reward_fn, code_reward_fn

# 引入您的 Composite 组件 (请根据实际文件路径调整 import)
# 假设路径为 rllm/agents/composite_agent.py 和 rllm/environments/composite_env.py
from rllm.agents.composite_agent import CompositeAgent
from rllm.environments.composite.composite_env import CompositeEnvironment

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. 数据加载辅助函数 (复用训练逻辑) ---

def fetch_bfcl_tasks(base_url: str, split: str = "train") -> List[Dict[str, Any]]:
    """从 BFCL Server 获取真实任务列表"""
    try:
        url = f"{base_url}/get_env_profile"
        logger.info(f"Fetching BFCL tasks from {url}...")
        resp = requests.post(url, json={
            "env_type": "bfcl", "params": {"split": split}
        }, timeout=10)
        data = resp.json()
        raw_list = data.get("data", []) if isinstance(data, dict) else data
        
        tasks = []
        for item in raw_list:
            # BFCL API 返回的可能是 task_id 字符串，也可能是包含 task_id 的字典
            t_id = item if isinstance(item, str) else item.get("task_id")
            if t_id:
                # 注意：BFCL Environment 需要 task_id 来初始化，且通常不需要 prompt (server 会给)
                # 但为了本地打印预览，我们这里尽量看看能不能获取到 question，或者只存 task_id
                tasks.append({"task_id": t_id, "task_type": "bfcl"})
        return tasks
    except Exception as e:
        logger.error(f"Failed to fetch BFCL tasks: {e}")
        return []

def load_local_dataset(dataset_name: str, split: str, tag: str) -> List[Dict]:
    """加载本地/注册的数据集"""
    try:
        # 假设数据集已经被注册，或者 rllm 能自动处理 huggingface 数据集
        logger.info(f"Loading {dataset_name} ({split})...")
        raw_data = DatasetRegistry.load_dataset(dataset_name, split)
        tagged_data = []
        for item in raw_data:
            d = dict(item) if not isinstance(item, dict) else item.copy()
            d["task_type"] = tag
            tagged_data.append(d)
        return tagged_data
    except Exception as e:
        logger.error(f"Failed to load {dataset_name}: {e}")
        return []

def prepare_test_samples(bfcl_url: str, num_per_type: int = 2) -> List[Dict]:
    """
    构造测试样本集：从真实数据源中各抽取 N 个样本
    """
    colorful_print(">>> 正在从数据集加载样本...", "cyan")
    
    # 1. BFCL
    bfcl_all = fetch_bfcl_tasks(bfcl_url, "test") # 优先用 test 集
    if not bfcl_all: bfcl_all = fetch_bfcl_tasks(bfcl_url, "train")
    bfcl_samples = bfcl_all[:num_per_type] if bfcl_all else []

    # 2. Math (DeepScaler / GSM8K)
    math_all = load_local_dataset("deepscaler_math", "train", "math")
    math_samples = math_all[:num_per_type] if math_all else []

    # 3. Code (DeepCoder / MBPP)
    code_all = load_local_dataset("deepcoder", "test", "code")
    code_samples = code_all[:num_per_type] if code_all else []

    combined = bfcl_samples + math_samples + code_samples
    return combined

# --- 2. 打印信息辅助函数 ---

def inspect_dataset(tasks: List[Dict]):
    """打印数据集详细信息"""
    colorful_print(f"\n{'='*20} DATASET INSPECTION {'='*20}", "yellow")
    colorful_print(f"Total Samples Loaded: {len(tasks)}", "yellow")
    
    # 统计分布
    counts = {}
    for t in tasks:
        counts[t['task_type']] = counts.get(t['task_type'], 0) + 1
    colorful_print(f"Task Distribution: {counts}", "yellow")

    print("\n--- Sample Details ---")
    for i, t in enumerate(tasks):
        t_type = t.get("task_type", "UNKNOWN")
        print(f"\nSample #{i+1} [{t_type}]")
        
        if t_type == "bfcl":
            import pdb; pdb.set_trace()
            print(f"  Task ID: {t.get('task_id')}")
            # BFCL 的 prompt 通常在 Env reset 后才由 Server 返回，这里可能只有 ID
            print(f"  (Prompt will be fetched from BFCL Server upon reset)")
            
        elif t_type == "math":
            import pdb; pdb.set_trace()
            # q = t.get("question", "N/A")
            q = t.get("problem", "N/A")
            a = t.get("ground_truth", "N/A")
            print(f"  Question: {q[:100]}..." if len(q) > 100 else f"  Question: {q}")
            print(f"  GT Answer: {a}")
            
        elif t_type == "code":
            import pdb; pdb.set_trace()
            desc = t.get("problem", "N/A")
            tests = t.get("ground_truth", "N/A")
            print(f"  Description: {desc[:100]}..." if len(desc) > 100 else f"  Description: {desc}")
            print(f"  Num Tests: {len(tests)}")
            
    colorful_print(f"{'='*60}\n", "yellow")

# --- 3. 主流程 ---

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bfcl_url", type=str, default="http://localhost:8801", help="BFCL Env URL")
    parser.add_argument("--model_url", type=str, default="http://localhost:8803/v1", help="Model API URL")
    parser.add_argument("--model_name", type=str, default="Qwen3-32B", help="Model Name")
    parser.add_argument("--samples_per_type", type=int, default=2, help="每个类别抽样数量")
    args = parser.parse_args()

    # 1. 准备并打印数据
    test_tasks = prepare_test_samples(args.bfcl_url, num_per_type=args.samples_per_type)
    
    if not test_tasks:
        colorful_print("Error: No tasks loaded. Please check dataset paths and BFCL server.", "red")
        return

    # 打印详细信息（满足您的需求）
    inspect_dataset(test_tasks)

    # 2. 构造环境配置
    composite_env_args = {
        "bfcl_args": {
            "base_url": args.bfcl_url,
            "env_type": "bfcl",
            "max_steps": 20, # 测试时步数可以少一点
        },
        "math_args": {
            "reward_fn": math_reward_fn
        },
        "code_args": {
            "reward_fn": code_reward_fn
        }
    }

    # 3. 构造 Agent 配置
    # 注意：这里需要根据您模型的特性调整 System Prompt
    composite_agent_args = {
        "bfcl_agent_args": {
            "parser_name": "qwen", # 或您的自定义 parser
            "system_prompt": "You are a helpful assistant with access to tools. Please use the tools provided to answer the user's request.",
        },
        "math_agent_args": {
            # "parser_name": "qwen",
            "accumulate_thinking": True, # 开启 CoT
            # "system_prompt": "You are a math expert. Please think step-by-step to solve the problem. Wrap your thinking process in <think>...</think> tags.",
        },
        "code_agent_args": {
            # "parser_name": "qwen",
            "accumulate_thinking": True,
            # "system_prompt": "You are a proficient Python developer. Please write Python code to solve the problem.",
        }
    }

    # 4. 初始化 Tokenizer 和 Engine
    model_name = "/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/base_models/Qwen3-32B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    rollout_args = {
        "base_url": args.model_url,
        "model": args.model_name,
        "temperature": 0.0,
        "max_tokens": 2048,
    }

    engine = AgentExecutionEngine(
        engine_name="openai",
        n_parallel_agents=1, # 串行执行方便观察，大规模测试可调大
        max_steps=10,
        agent_class=CompositeAgent,
        env_class=CompositeEnvironment,
        agent_args=composite_agent_args,
        env_args=composite_env_args,
        rollout_engine_args=rollout_args,
        tokenizer=tokenizer,
        max_prompt_length=30720,
        max_response_length=30720
    )

    # 5. 执行任务
    colorful_print(">>> Starting Execution Engine...", "green")
    trajectories = await engine.execute_tasks(test_tasks)

    # 6. 保存并简单展示结果
    save_path = "./trajectories/composite_test_trajectories.json"
    save_clean_trajectories(trajectories, save_path=save_path)
    colorful_print(f"\nExecution finished. Trajectories saved to {save_path}", "green")

    # 简单的 Pass/Fail 统计
    total_reward = sum([sum(s.reward for s in t.steps) for t in trajectories])
    colorful_print(f"Total Cumulative Reward: {total_reward:.2f}", "cyan")

if __name__ == "__main__":
    asyncio.run(main())
