import asyncio
import os
import json
import logging
import random
from typing import List, Dict

from transformers import AutoTokenizer

# RLLM 核心组件
from rllm.agents.composite_agent import CompositeAgent
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.composite.composite_env import CompositeEnvironment
# 引入 Reward Functions
from rllm.rewards.reward_fn import (
    tool_call_reward_fn, math_reward_fn, code_reward_fn, 
    search_reward_fn, webshop_reward_fn
)
from rllm.data.utils import create_standard_sample, load_webshop_data
# 引入 System Prompts
from rllm.agents.system_prompts import MATH_SYSTEM_PROMPT, SEARCH_SYSTEM_PROMPT
from rllm.agents.webshop_agent import WEBSHOP_SYSTEM_PROMPT

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- 轨迹保存函数 ---
def save_detailed_trajectories(results, output_path: str = "debug_trajectories.json"):
    """
    保存详细轨迹用于分析
    """
    logger.info(f"Saving detailed trajectories to {output_path}...")
    export_data = []
    
    for traj in results:
        task_data = dict(traj.task) if not isinstance(traj.task, dict) else traj.task
        
        # Webshop 任务的 prompt 是占位符，实际指令在 instruction 中
        question = task_data.get("prompt") or task_data.get("question") or task_data.get("input")
        goal_idx = task_data.get("goal_idx", "unknown")
        ground_truth = task_data.get("ground_truth") or task_data.get("response") or task_data.get("answer") or task_data.get("output")

        steps_details = []
        instruction_from_env = ""
        if traj.steps:
            for step_idx, step in enumerate(traj.steps):
                # 尝试从 step info 中获取 instruction
                if step.info and "instruction" in step.info:
                    instruction_from_env = step.info.get("instruction", "")
                
                steps_details.append({
                    "step_index": step_idx,
                    "model_response": step.model_response,
                    "action": str(step.action) if step.action else None,
                    "observation": str(step.observation) if step.observation else None,  # 截断长观察
                    "task_score": step.info.get("task_score") if step.info else None,
                    "parsed_action": step.info.get("parsed_action") if step.info else None,
                })
        
        record = {
            "uid": traj.uid,
            "reward": traj.reward,
            "task_type": task_data.get("task_type", "webshop"),
            "goal_idx": goal_idx,
            "instruction": instruction_from_env,
            "question": question,
            "ground_truth_expected": str(ground_truth) if ground_truth else "N/A (webshop task)",
            "trajectory_steps": steps_details,
            "num_steps": len(steps_details),
        }
        export_data.append(record)

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=4, ensure_ascii=False)
        print(f"\n[Saved] Detailed debug file is at: {os.path.abspath(output_path)}")
    except Exception as e:
        logger.error(f"Failed to save trajectories: {e}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # --- 配置区域 ---
    debug_task_type = "webshop"
    
    n_parallel_agents = 1  # Webshop 环境不是线程安全的，建议使用 1
    model_name = "Qwen3-8B" 
    model_path = "/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/base_models/Qwen3-8B"
    
    api_base_url = "http://localhost:8803/v1"
    debug_subset_size = 30  # 调试样本数
    
    # Webshop 环境配置
    webshop_max_steps = 15  # 每个 episode 最大步数
    webshop_path = None  # 如果 webshop 已在 PYTHONPATH 中，可以设为 None

    logger.info(f"Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Failed to load tokenizer from {model_path}: {e}")
        raise e

    # --- 1. 构造 CompositeAgent 参数 ---
    agent_args = {
        "math_agent_args": {
            "tools": ["python"], 
            "parser_name": "qwen", 
            "system_prompt": MATH_SYSTEM_PROMPT
        },
        "code_agent_args": {"accumulate_thinking": True}, 
        "bfcl_agent_args": {"parser_name": "qwen"},
        "tool_call_agent_args": {
            "parser_name": "qwen"
        },
        "search_agent_args": {
            "parser_name": "qwen",
            "system_prompt": SEARCH_SYSTEM_PROMPT,
            "tool_map": {}
        },
        # Webshop Agent 配置
        "webshop_agent_args": {
            "system_prompt": WEBSHOP_SYSTEM_PROMPT
        }
    }
    
    # --- 2. 构造 CompositeEnvironment 参数 ---
    env_args = {
        "math_args": {
            "tools": ["python"],
            "reward_fn": math_reward_fn,
        },
        "code_args": {"reward_fn": code_reward_fn},
        "bfcl_args": {"base_url": "http://localhost:8888", "env_type": "bfcl", "max_steps": 20},
        "tool_call_args": {
            "reward_fn": tool_call_reward_fn
        },
        "search_args": {
            "mcp_server_command": "npx",
            "mcp_server_args": ["-y", "@brightdata/mcp"],
            "mcp_server_env": {},
            "reward_fn": search_reward_fn,
            "cache_dir": "./search_cache_data",
            "allowed_tools": []
        },
        # Webshop Environment 配置
        "webshop_args": {
            "max_steps": webshop_max_steps,
            "observation_mode": "text",
            "webshop_path": webshop_path,
            "reward_fn": webshop_reward_fn,
            # 使用 1000 个商品的小数据集（需要先构建 indexes_1k）
            # 如果使用完整数据集，设为 None
            "num_products": 1000,
        }
    }

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name, "max_tokens": 32768}

    # --- 3. 初始化引擎 ---
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
        max_steps=webshop_max_steps,  # 传递给 Engine，控制最大交互步数
    )

    # --- 4. 加载 Webshop 数据 ---
    logger.info(f"Loading Webshop Dataset (subset: {debug_subset_size})...")
    tasks = load_webshop_data("test", debug_subset_size)
    
    if not tasks:
        raise ValueError("No Webshop tasks loaded! Please check if webshop environment is properly set up.")

    random.shuffle(tasks)
    
    logger.info(f"Loaded {len(tasks)} Webshop tasks.")
    for i, task in enumerate(tasks[:3]):
        logger.info(f"  Task {i}: goal_idx={task.get('goal_idx')}, task_type={task.get('task_type')}")
    
    # --- 5. 执行调试 ---
    logger.info(f"Running {debug_task_type} evaluation on {len(tasks)} samples...")
    
    results = asyncio.run(engine.execute_tasks(tasks))
    
    # --- 6. 保存与分析 ---
    output_file = f"./trajectories/{debug_task_type}_debug_trajectories.json"
    save_detailed_trajectories(results, output_file)
    
    # 统计指标
    success_count = sum(1 for r in results if r.reward >= 1.0)
    partial_count = sum(1 for r in results if 0 < r.reward < 1.0)
    fail_count = sum(1 for r in results if r.reward == 0.0)
    
    print(f"\n>>> WEBSHOP Metrics:")
    print(f"    Perfect (reward=1.0): {success_count}/{len(results)} ({success_count/len(results):.2%})")
    print(f"    Partial (0<reward<1): {partial_count}/{len(results)} ({partial_count/len(results):.2%})")
    print(f"    Failed  (reward=0.0): {fail_count}/{len(results)} ({fail_count/len(results):.2%})")

    # 打印案例分析
    print("\n" + "="*60)
    print("WEBSHOP DEBUG ANALYSIS")
    print("="*60)

    for i, traj in enumerate(results):
        score = traj.reward
        # 打印前3个或者所有失败的
        if i < 3 or score < 1.0:
            task_data = traj.task if isinstance(traj.task, dict) else dict(traj.task)
            goal_idx = task_data.get("goal_idx", "unknown")
            task_type = task_data.get("task_type", "webshop")
            
            # 获取 instruction（从轨迹步骤中提取）
            instruction = ""
            task_score = 0.0
            num_steps = 0
            if traj.steps:
                num_steps = len(traj.steps)
                for step in traj.steps:
                    if step.info:
                        if "instruction" in step.info and not instruction:
                            instruction = step.info.get("instruction", "")
                        if "task_score" in step.info:
                            task_score = step.info.get("task_score", 0.0)

            # 获取最后的模型响应
            model_response_text = ""
            last_action = ""
            if traj.steps:
                last_step = traj.steps[-1]
                model_response_text = last_step.model_response[:300] if last_step.model_response else ""
                if last_step.info:
                    last_action = last_step.info.get("parsed_action", "")

            print(f"\n[Case #{i} | Reward: {score} | Task Score: {task_score:.3f} | Steps: {num_steps}]")
            print(f"Goal Index          : {goal_idx}")
            print(f"Instruction         : {instruction[:200]}..." if len(instruction) > 200 else f"Instruction         : {instruction}")
            print(f"Last Action         : {last_action}")
            print("-" * 20 + " Last Model Output " + "-" * 20)
            print(f"{model_response_text.replace(chr(10), ' ')}...")
            print("-" * 60)
