import asyncio
import os
import json
import logging
import pandas as pd
import random
from typing import List, Dict

from transformers import AutoTokenizer

# RLLM 核心组件
from rllm.agents.composite_agent import CompositeAgent
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.composite.composite_env import CompositeEnvironment
# 引入 Tool Call 和 Search 相关的 Reward Function
from rllm.rewards.reward_fn import tool_call_reward_fn, math_reward_fn, code_reward_fn, search_reward_fn
from rllm.data.utils import create_standard_sample
# 引入 System Prompts
from rllm.agents.system_prompts import MATH_SYSTEM_PROMPT, SEARCH_SYSTEM_PROMPT
# 引入数据加载函数
from rllm.data.utils import load_tool_call_dataset, load_search_data

# [新增] 引入 MCP 组件 (用于 Search/Browsing)
try:
    from rllm.environments.tools.mcp_env import MCPConnectionManager
except ImportError:
    MCPConnectionManager = None

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- 2. 轨迹保存函数 (复用 test_composite_rollout.py 中的逻辑) ---
def save_detailed_trajectories(results, output_path: str = "debug_trajectories.json"):
    """
    保存详细轨迹用于分析
    """
    logger.info(f"Saving detailed trajectories to {output_path}...")
    export_data = []
    
    for traj in results:
        task_data = dict(traj.task) if not isinstance(traj.task, dict) else traj.task
        
        question = task_data.get("question") or task_data.get("prompt") or task_data.get("input")
        # 尝试获取 instruction 以便调试时查看工具定义
        instruction = task_data.get("instruction", "")
        ground_truth = task_data.get("ground_truth") or task_data.get("response") or task_data.get("answer") or task_data.get("output")

        import pdb; pdb.set_trace()
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
            "instruction": instruction, # 额外保存 Instruction
            "ground_truth_expected": str(ground_truth),
            "trajectory_steps": steps_details
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
    # 控制当前测试的任务类型: "tool_call" 或 "search"
    debug_task_type = "tool_call"  
    
    n_parallel_agents = 1
    model_name = "Qwen3-8B" 
    model_path = "/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/base_models/Qwen3-8B"
    
    # Tool Call 数据路径
    tool_call_data_path = "/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/grpotool/dataset/10_9_mix" 
    
    api_base_url = "http://localhost:8803/v1"
    debug_subset_size = 5 # 调试样本数

    # --- [新增] Search / MCP 环境配置 ---
    # 确保环境变量中有 BRIGHT_DATA_API_TOKEN，或者在这里硬编码用于测试
    bright_data_token = os.getenv("BRIGHT_DATA_API_TOKEN") or "da9e7e42-730d-4fb7-8357-b3dafcd7cc93"
    mcp_tool_map = {}
    
    mcp_server_command = "npx"
    mcp_server_args = ["-y", "@brightdata/mcp"]
    mcp_server_env = {
        "API_TOKEN": bright_data_token or "",
        "GROUPS": "advanced_scraping",
        "PATH": os.environ.get("PATH", ""),
        "PRO_MODE": "true",
        "WEB_UNLOCKER_ZONE": "web_unlocker_zkn" # 根据实际情况调整 Zone
    }
    search_cache_dir = "./search_cache_data"
    os.makedirs(search_cache_dir, exist_ok=True)
    allowed_mcp_tools = ["search_engine", "scrape_as_markdown", "search_engine_batch", "scrape_batch"]

    logger.info(f"Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Failed to load tokenizer from {model_path}: {e}")
        raise e

    # --- [新增] 预取 Search Tools 定义 ---
    # 即使不跑 Search 任务，如果 CompositeAgent 配置了 search_agent_args，最好也准备好 tool_map
    # 或者仅当 debug_task_type == "search" 时才去获取
    if MCPConnectionManager is not None and (debug_task_type == "search" or bright_data_token):
        logger.info("Initializing MCP Connection to fetch Search tools...")
        try:
            temp_manager = MCPConnectionManager(
                mcp_server_command, 
                mcp_server_args, 
                mcp_server_env,
                search_cache_dir,
                allowed_tools=allowed_mcp_tools
            )
            temp_manager.start()
            mcp_tool_map = temp_manager.tool_map
            temp_manager.stop()
            logger.info(f"✅ Fetched {len(mcp_tool_map)} tools from Bright Data MCP.")
        except Exception as e:
            logger.error(f"❌ Failed to fetch MCP tools: {e}")
            if debug_task_type == "search":
                logger.warning("Running search task without valid tool definitions might fail.")

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
        
        # [新增] Search Agent 配置
        "search_agent_args": {
            "parser_name": "qwen",
            "system_prompt": SEARCH_SYSTEM_PROMPT,
            "tool_map": mcp_tool_map  # 传入预获取的工具 Schema
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
        
        # [新增] Search Environment 配置
        "search_args": {
            "mcp_server_command": mcp_server_command,
            "mcp_server_args": mcp_server_args,
            "mcp_server_env": mcp_server_env,
            "reward_fn": search_reward_fn,
            "cache_dir": search_cache_dir,
            "allowed_tools": allowed_mcp_tools
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
    )

    # --- 4. 加载数据 (根据 debug_task_type 选择) ---
    tasks = []
    
    if debug_task_type == "tool_call":
        logger.info(f"Loading Tool Call Dataset from {tool_call_data_path}...")
        tasks = load_tool_call_dataset(tool_call_data_path, split="test", num_samples=debug_subset_size)
        
    elif debug_task_type == "search":
        logger.info(f"Loading Search Dataset (subset: {debug_subset_size})...")
        # load_search_data 内部应该处理了 task_type="search" 的设置
        # 如果没有，可能需要在 load_search_data 源码中确认，或者手动补充
        tasks = load_search_data("test", debug_subset_size)
        
    else:
        logger.error(f"Unknown debug_task_type: {debug_task_type}")

    if not tasks:
        raise ValueError(f"No tasks loaded for type {debug_task_type}!")

    random.shuffle(tasks)
    
    # --- 5. 执行调试 ---
    logger.info(f"Running {debug_task_type} evaluation on {len(tasks)} samples...")
    
    results = asyncio.run(engine.execute_tasks(tasks))
    
    # --- 6. 保存与分析 ---
    output_file = f"./trajectories/{debug_task_type}_debug_trajectories.json"
    save_detailed_trajectories(results, output_file)
    
    # 简单的 Pass Rate 统计
    success_count = sum(1 for r in results if r.reward >= 1.0)
    print(f"\n>>> {debug_task_type.upper()} Metrics: Accuracy = {success_count}/{len(results)} ({success_count/len(results):.2%})")

    # 打印失败/成功案例分析
    print("\n" + "="*60)
    print(f"{debug_task_type.upper()} DEBUG ANALYSIS")
    print("="*60)

    for i, traj in enumerate(results):
        score = traj.reward
        # 只打印前3个或者 reward < 1 的
        if i < 3 or score < 1.0:
            task_data = traj.task if isinstance(traj.task, dict) else dict(traj.task)
            prompt = task_data.get("prompt", "") or task_data.get("input") or task_data.get("question")
            gt = task_data.get("output", "") or task_data.get("response") or task_data.get("answer") or "Unknown"
            task_type = task_data.get("task_type", "Unknown")

            model_response_text = ""
            if traj.steps:
                last_step = traj.steps[-1]
                model_response_text = last_step.model_response
            else:
                model_response_text = "No steps executed."

            print(f"\n[Case #{i} | Reward: {score} | Type: {task_type}]")
            print(f"UID                 : {traj.uid}")
            print(f"Prompt              : {str(prompt)}...")
            print(f"Expect Ground Truth : {str(gt)}...")
            print("-" * 20 + " Model Output " + "-" * 20)
            print(f"{model_response_text.replace(chr(10), ' ')}") 
            print("-" * 60)
