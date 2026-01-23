import asyncio
import os
import json
import logging
from typing import List, Dict
import random

from transformers import AutoTokenizer

# RLLM 核心组件
from rllm.agents.composite_agent import CompositeAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.composite.composite_env import CompositeEnvironment
from rllm.rewards.reward_fn import math_reward_fn, code_reward_fn
from rllm.utils import compute_pass_at_k
from rllm.agents.system_prompts import MATH_SYSTEM_PROMPT

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_dataset_for_composite(dataset_items, force_task_type="math") -> List[Dict]:
    """
    [数据适配逻辑]
    1. 修复 Ground Truth 字段映射 (response/extra_info -> answer)。
    2. [关键] 添加 'task_type' 字段，以便 CompositeAgent 知道将其分发给 math 子系统。
    """
    fixed_tasks = []
    logger.info("Checking and fixing dataset fields for Composite execution...")
    
    for i, item in enumerate(dataset_items):
        task = dict(item) if not isinstance(item, dict) else item.copy()
        
        # --- 1. 修复 Ground Truth ---
        ground_truth = None
        if "extra_info" in task and task["extra_info"]:
            try:
                raw_data = json.loads(task["extra_info"]) if isinstance(task["extra_info"], str) else task["extra_info"]
                if "answer" in raw_data: ground_truth = raw_data["answer"]
                elif "solution" in raw_data: ground_truth = raw_data["solution"]
            except Exception:
                pass

        if not ground_truth and "response" in task:
            ground_truth = task["response"]
        
        # 兼容 math_reward_fn
        if ground_truth:
            task["answer"] = str(ground_truth)
        else:
            task["answer"] = "GROUND_TRUTH_MISSING"

        # --- 2. [关键] 设置 Task Type ---
        # CompositeAgent 根据这个字段决定调用哪个子 Agent (bfcl, math, code)
        if "task_type" not in task or not task["task_type"]:
            task["task_type"] = force_task_type
        
        # 确保 extra_info 是 JSON 字符串 (有些环境要求)
        if "extra_info" in task and not isinstance(task["extra_info"], str):
             task["extra_info"] = json.dumps(task["extra_info"], ensure_ascii=False)

        fixed_tasks.append(task)
        
    logger.info(f"Processed {len(fixed_tasks)} tasks. Task type set to: {force_task_type}")
    return fixed_tasks


def save_detailed_trajectories(results, output_path: str = "debug_trajectories.json"):
    """
    保存详细轨迹用于分析
    """
    logger.info(f"Saving detailed trajectories to {output_path}...")
    export_data = []
    
    for traj in results:
        task_data = dict(traj.task) if not isinstance(traj.task, dict) else traj.task
        
        question = task_data.get("question") or task_data.get("prompt") or task_data.get("input")
        ground_truth = task_info.get("ground_truth") or task_info.get("response") or task_info.get("answer")

        steps_details = []
        if traj.steps:
            for step_idx, step in enumerate(traj.steps):
                steps_details.append({
                    "step_index": step_idx,
                    "model_response": step.model_response,
                    "action": str(step.action) if step.action else None,
                    "observation": str(step.observation) if step.observation else None,
                })
        
        record = {
            "uid": traj.uid,
            "reward": traj.reward,
            "task_type": task_data.get("task_type", "unknown"), # 记录任务类型
            "question": question,
            "ground_truth_expected": str(ground_truth),
            "trajectory_steps": steps_details
        }
        export_data.append(record)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=4, ensure_ascii=False)
        print(f"\n[Saved] Detailed debug file is at: {os.path.abspath(output_path)}")
    except Exception as e:
        logger.error(f"Failed to save trajectories: {e}")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # --- 配置区域 ---
    n_parallel_agents = 1
    model_name = "Qwen3-32B" 
    model_path = "/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/base_models/Qwen3-32B"
    dataset_name = "math1000_code1000_bfcl100"
    api_base_url = "http://localhost:8803/v1"
    
    # 调试样本数
    debug_subset_size = 10

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # --- 1. 构造 CompositeAgent 参数 ---
    # CompositeAgent 会根据 task_type 分发到对应的子 Agent
    # 这里我们主要配置 math_agent_args
    agent_args = {
        "math_agent_args": {
            "tools": ["python"], 
            "parser_name": "qwen", 
            "system_prompt": MATH_SYSTEM_PROMPT
        },
        # 如果需要跑代码或BFCL，可以在这里添加 code_agent_args 或 bfcl_agent_args
        # 对于本脚本，留空即可
        "code_agent_args": {"accumulate_thinking": True}, 
        "bfcl_agent_args": {"parser_name": "qwen"}
    }
    
    # --- 2. 构造 CompositeEnvironment 参数 ---
    env_args = {
        "math_args": {
            "tools": ["python"],
            "reward_fn": math_reward_fn,
        },
        "code_args": {"reward_fn": code_reward_fn},
        "bfcl_args": {"base_url": "http://localhost:8888", "env_type": "bfcl", "max_steps": 20}
    }

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name, "max_tokens": 32768}

    # --- 3. 初始化引擎 ---
    # 注意 agent_class 和 env_class 变为 Composite 版本
    engine = AgentExecutionEngine(
        agent_class=CompositeAgent,
        agent_args=agent_args,
        env_class=CompositeEnvironment,
        env_args=env_args,
        engine_name="openai",
        rollout_engine_args={"base_url": api_base_url, "api_key": "None", "model_name": model_name},
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_response_length=32768,
        max_prompt_length=32768,
        n_parallel_agents=n_parallel_agents,
    )

    # --- 4. 加载数据 ---
    logger.info(f"Loading dataset from Registry: {dataset_name} (split: test)")
    raw_dataset = DatasetRegistry.load_dataset(dataset_name, "test")
    
    if not raw_dataset:
        raise ValueError(f"Dataset {dataset_name} not found or empty!")

    # --- 5. 预处理数据 (注入 task_type="math") ---
    # 这是 CompositeAgent 能正常工作的关键
    tasks = raw_dataset.get_data()
    
    random.shuffle(tasks)
    
    # --- 6. 执行调试 ---
    logger.info(f"Running Composite evaluation on first {debug_subset_size} samples...")
    
    results = asyncio.run(engine.execute_tasks(tasks[:debug_subset_size]))
    
    # --- 7. 保存与分析 ---
    save_detailed_trajectories(results, "./trajectories/composite_debug_trajectories.json")
    metrics = compute_pass_at_k(results)
    print(f"\n>>> Metrics on subset: {metrics}")

    # 打印失败案例分析
    print("\n" + "="*60)
    print("COMPOSITE DEBUG ANALYSIS")
    print("="*60)

    # for i, traj in enumerate(results):
    #     score = traj.reward
    #     if score < 1.0:
    #         task_data = traj.task if isinstance(traj.task, dict) else dict(traj.task)
    #         prompt = task_data.get("prompt", "") or task_data.get("question")
    #         gt = task_data.get("answer", "Unknown")
    #         task_type = task_data.get("task_type", "Unknown")

    #         model_response_text = ""
    #         tool_observation = ""
    #         if traj.steps:
    #             last_step = traj.steps[-1]
    #             model_response_text = last_step.model_response
    #             if last_step.observation:
    #                 tool_observation = str(last_step.observation)
    #         else:
    #             model_response_text = "No steps executed."

    #         print(f"\n[Case #{i} | Reward: {score} | Type: {task_type}]")
    #         print(f"UID                 : {traj.uid}")
    #         print(f"Prompt (Top 100)    : {str(prompt)[:100]}...")
    #         print(f"Expect Ground Truth : {gt}")
    #         print("-" * 20 + " Model Output " + "-" * 20)
    #         print(f"{model_response_text[-300:]}") 
    #         if tool_observation:
    #             print("-" * 20 + " Tool Obs " + "-" * 20)
    #             print(f"{tool_observation[:200]}...")
    #         print("-" * 60)
