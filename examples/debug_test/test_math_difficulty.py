import asyncio
import os
import json
import logging
import random
import uuid
import argparse
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any

from datasets import load_dataset
from transformers import AutoTokenizer

# --- RLLM 核心组件 ---
from rllm.agents.composite_agent import CompositeAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.composite.composite_env import CompositeEnvironment
from rllm.rewards.reward_fn import math_reward_fn, code_reward_fn, search_reward_fn
from rllm.agents.system_prompts import MATH_SYSTEM_PROMPT


random.seed(42)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================================================================
# 0. JSON Encoder 修复 (解决 int64/float32 报错)
# =========================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# =========================================================================
# 1. 工具函数 
# =========================================================================

def create_standard_sample(prompt: str, response: str, task_type: str, raw_data: Dict) -> Dict[str, str]:
    """
    强制统一数据格式，防止 PyArrow Schema 报错。
    """
    safe_prompt = str(prompt) if prompt is not None else ""
    safe_response = str(response) if response is not None else ""
    
    return {
        "prompt": safe_prompt,      
        "response": safe_response,  
        "task_type": str(task_type),
        "data_source": str(task_type),
        "extra_info": json.dumps(raw_data, ensure_ascii=False, cls=NumpyEncoder),   
    }

# =========================================================================
# 2. 数据集加载逻辑 
# =========================================================================

DATASET_CONFIGS = {
    "DeepScaleR": {
        "path": "agentica-org/DeepScaleR-Preview-Dataset",
        "split": "train",
        "mapping": {"prompt": ["problem", "question"], "response": ["answer"]}
    },
    "DAPO": {
        "path": "open-r1/DAPO-Math-17k-Processed",
        "split": "train",
        "mapping": {"prompt": ["prompt", "problem"], "response": ["solution"]}
    },
    "DeepMath": {
        "path": "zwhe99/DeepMath-103K",
        "split": "train",
        "mapping": {"prompt": ["problem", "question"], "response": ["final_answer", "solution", "answer"]}
    }
}

def load_eval_dataset_unified(dataset_key: str, max_samples: int = None) -> List[Dict]:
    config = DATASET_CONFIGS.get(dataset_key)
    if not config:
        logger.error(f"Unknown dataset key: {dataset_key}")
        return []

    logger.info(f"Loading {dataset_key} from {config['path']}...")
    try:
        ds = load_dataset(config['path'], split=config['split'])
    except Exception as e:
        logger.error(f"Failed to load {dataset_key}: {e}")
        return []

    raw_list = list(ds)

    if max_samples and max_samples > 0 and len(raw_list) > max_samples:
        random.shuffle(raw_list) # 可选：如果不需要随机抽样可注释
        raw_list = raw_list[:max_samples]
        logger.info(f"Sampled top {max_samples} items.")

    processed_data = []
    mapping = config["mapping"]

    for item in raw_list:
        d = dict(item)
        
        prompt_text = ""
        for key in mapping["prompt"]:
            if key in d and d[key]:
                prompt_text = d[key]
                break
        
        response_text = ""
        for key in mapping["response"]:
            if key in d and d[key]:
                response_text = str(d[key])
                break
        
        if not prompt_text or not response_text:
            continue

        clean_sample = create_standard_sample(
            prompt=prompt_text,
            response=response_text,
            task_type="math",
            raw_data=d
        )
        
        original_id = d.get("id") or d.get("uuid") or str(uuid.uuid4())
        clean_sample["uid"] = str(original_id)
        clean_sample["dataset_source"] = dataset_key

        processed_data.append(clean_sample)

    logger.info(f"Loaded {len(processed_data)} samples for {dataset_key}.")
    return processed_data

# =========================================================================
# 3. 结果聚合与保存
# =========================================================================

def aggregate_and_save_results(results, dataset_name: str, n_rollouts: int, output_dir: str = "./difficulty_eval"):
    os.makedirs(output_dir, exist_ok=True)
    
    grouped_data = defaultdict(list)
    for traj in results:
        uid = traj.task.get("uid")
        grouped_data[uid].append(traj)

    final_records = []
    logger.info(f"Aggregating results... (Total unique tasks: {len(grouped_data)})")

    for uid, trajs in grouped_data.items():
        if not trajs: continue
        
        base_task = trajs[0].task
        # 收集 Reward
        rewards = [t.reward for t in trajs]
        
        # 计算统计指标 (注意：使用 np.mean 后可能是 numpy 类型)
        mean_score = np.mean(rewards)
        pass_at_1 = 1.0 if any(r >= 1.0 for r in rewards) else 0.0
        
        rollout_details = []
        for t in trajs:
            final_step_response = "No Steps"
            if t.steps:
                final_step_response = t.steps[-1].model_response
            
            rollout_details.append({
                "reward": float(t.reward), # 强制转 float
                "final_response": final_step_response
            })

        record = {
            "uid": uid,
            "dataset": dataset_name,
            "question": base_task.get("prompt"),
            "ground_truth": base_task.get("response"),
            "stats": {
                "n_rollouts": int(len(rewards)),
                "mean_score": float(mean_score), # 强制转 float
                "is_solved": bool(pass_at_1)     # 强制转 bool
            },
            "rollouts": rollout_details
        }
        final_records.append(record)

    output_path = os.path.join(output_dir, f"difficulty_{dataset_name}_N{n_rollouts}.json")
    
    # 使用自定义 Encoder 进行保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_records, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
        
    logger.info(f"Saved aggregated difficulty report to: {output_path}")

# =========================================================================
# 4. 主程序 (CLI 支持)
# =========================================================================

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    parser = argparse.ArgumentParser(description="Evaluate Dataset Difficulty with RLLM")
    
    # 核心参数
    parser.add_argument("--dataset", type=str, required=True, choices=["DeepScaleR", "DAPO", "DeepMath"], help="Target dataset name")
    parser.add_argument("--port", type=int, required=True, help="API Server Port (e.g., 8801)")
    
    # 可选参数
    parser.add_argument("--model_name", type=str, default="Qwen3-8B", help="Model name for API")
    parser.add_argument("--model_path", type=str, default="/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/base_models/Qwen3-8B", help="Local path for tokenizer")
    parser.add_argument("--n_rollouts", type=int, default=4, help="Number of rollouts per sample")
    parser.add_argument("--debug_samples", type=int, default=12800, help="Number of samples to run (set -1 for full dataset)")
    parser.add_argument("--n_parallel", type=int, default=256, help="Number of parallel agents")

    args = parser.parse_args()

    # 构建 API URL
    API_BASE_URL = f"http://localhost:{args.port}/v1"
    
    # 处理 max_samples
    MAX_SAMPLES = None if args.debug_samples <= 0 else args.debug_samples

    logger.info(f"Starting Evaluation for {args.dataset} on Port {args.port}")
    logger.info(f"API: {API_BASE_URL} | Model: {args.model_name}")
    logger.info(f"Rollouts: {args.n_rollouts} | Samples Limit: {MAX_SAMPLES}")

    # 1. 初始化 Tokenizer & Engine~
    logger.info("Initializing Engine...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    agent_args = {
        "math_agent_args": {
            "tools": ["python"], 
            "parser_name": "qwen", 
            "system_prompt": MATH_SYSTEM_PROMPT
        },
        "code_agent_args": {"accumulate_thinking": True},
        "bfcl_agent_args": {"parser_name": "qwen"},
        "search_agent_args": {"tool_map": {}}
    }
    
    env_args = {
        "math_args": {"tools": ["python"], "reward_fn": math_reward_fn},
        "code_args": {"reward_fn": code_reward_fn},
        "bfcl_args": {"base_url": "http://localhost", "env_type": "bfcl", "max_steps": 10},
        "search_args": {"allowed_tools": [], "reward_fn": search_reward_fn}
    }

    engine = AgentExecutionEngine(
        agent_class=CompositeAgent,
        agent_args=agent_args,
        env_class=CompositeEnvironment,
        env_args=env_args,
        engine_name="openai",
        rollout_engine_args={"base_url": API_BASE_URL, "api_key": "None", "model_name": args.model_name},
        tokenizer=tokenizer,
        sampling_params={
            "temperature": 1.0, 
            "top_p": 0.95, 
            "model": args.model_name, 
            "max_tokens": 8192
        },
        max_prompt_length=4096,
        max_response_length=16384,
        n_parallel_agents=args.n_parallel
    )

    # 2. 执行单一数据集
    ds_name = args.dataset
    
    # A. 加载数据
    tasks = load_eval_dataset_unified(ds_name, max_samples=MAX_SAMPLES)
    
    if tasks:
        # B. 任务扩充
        logger.info(f"Expanding {len(tasks)} samples by {args.n_rollouts}x rollouts...")
        expanded_tasks = []
        for task in tasks:
            for _ in range(args.n_rollouts):
                expanded_tasks.append(task.copy())
        
        logger.info(f"Total tasks sent to engine: {len(expanded_tasks)}")
        
        # C. 执行
        try:
            results = asyncio.run(engine.execute_tasks(expanded_tasks))
            # D. 保存
            aggregate_and_save_results(results, ds_name, args.n_rollouts)
            logger.info(f"Finished {ds_name} successfully.")
        except Exception as e:
            logger.error(f"Error during execution: {e}", exc_info=True)
    else:
        logger.warning(f"No tasks loaded for {ds_name}.")