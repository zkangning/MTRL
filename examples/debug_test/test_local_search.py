"""
Local Search 任务测试脚本

测试基于本地 Dense Retrieval 的搜索任务，使用 HotpotQA 数据集。

使用前请确保:
1. 下载数据: python examples/search/download_search_data.py --data_dir ./search_data
2. 启动检索服务器: bash examples/search/retrieval/launch_server.sh ./search_data/prebuilt_indices 8000
3. 启动 vLLM/SGLang 模型服务
4. 设置环境变量: export RETRIEVAL_SERVER_URL="http://127.0.0.1:8000"
"""

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
# 引入 Reward Function
from rllm.rewards.reward_fn import search_reward_fn, math_reward_fn, code_reward_fn, tool_call_reward_fn
# 引入 System Prompts
from rllm.agents.system_prompts import MATH_SYSTEM_PROMPT, SEARCH_SYSTEM_PROMPT, LOCAL_SEARCH_SYSTEM_PROMPT
# 引入数据加载函数
from rllm.data.utils import load_local_search_data

# 引入 LocalRetrievalTool (从 rllm.tools 导入)
from rllm.tools import LocalRetrievalTool

# [新增] 引入 MCP 组件 (用于 Search/Browsing)
try:
    from rllm.environments.tools.mcp_env import MCPConnectionManager
except ImportError:
    MCPConnectionManager = None

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_detailed_trajectories(results, output_path: str = "local_search_trajectories.json"):
    """
    保存详细轨迹用于分析
    """
    logger.info(f"Saving detailed trajectories to {output_path}...")
    export_data = []
    
    for traj in results:
        task_data = dict(traj.task) if not isinstance(traj.task, dict) else traj.task
        
        question = task_data.get("question") or task_data.get("prompt") or task_data.get("input")
        ground_truth = task_data.get("ground_truth") or task_data.get("response") or task_data.get("answer") or task_data.get("output")

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
            "task_type": task_data.get("task_type", "unknown"),
            "question": question,
            "ground_truth_expected": str(ground_truth),
            "trajectory_steps": steps_details
        }
        export_data.append(record)

    try:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=4, ensure_ascii=False)
        print(f"\n[Saved] Detailed debug file is at: {os.path.abspath(output_path)}")
    except Exception as e:
        logger.error(f"Failed to save trajectories: {e}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # --- 配置区域 ---
    n_parallel_agents = 1
    model_name = "Qwen3-8B" 
    model_path = "/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/base_models/Qwen3-8B"
    
    api_base_url = "http://localhost:8803/v1"
    debug_subset_size = 10  # 调试样本数
    
    # Local Search 配置
    retrieval_server_url = os.getenv("RETRIEVAL_SERVER_URL", "http://127.0.0.1:8000")
    logger.info(f"Using Retrieval Server: {retrieval_server_url}")

    # --- MCP Search 配置 (与 test_tool_call.py 保持一致) ---
    bright_data_token = os.getenv("BRIGHT_DATA_API_TOKEN") or "da9e7e42-730d-4fb7-8357-b3dafcd7cc93"
    mcp_tool_map = {}
    
    mcp_server_command = "npx"
    mcp_server_args = ["-y", "@brightdata/mcp"]
    mcp_server_env = {
        "API_TOKEN": bright_data_token or "",
        "GROUPS": "advanced_scraping",
        "PATH": os.environ.get("PATH", ""),
        "PRO_MODE": "true",
        "WEB_UNLOCKER_ZONE": "web_unlocker_zkn"
    }
    search_cache_dir = "./search_cache_data"
    os.makedirs(search_cache_dir, exist_ok=True)
    allowed_mcp_tools = ["search_engine", "scrape_as_markdown", "search_engine_batch", "scrape_batch"]

    # --- 加载 Tokenizer ---
    logger.info(f"Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Failed to load tokenizer from {model_path}: {e}")
        raise e

    # --- 初始化 Local Search Tool ---
    logger.info("Initializing Local Search Tool...")
    local_search_tool_map = {"local_search": LocalRetrievalTool}
    logger.info(f"✅ Local Search Tool initialized (Server: {retrieval_server_url})")

    # --- 预取 MCP Search Tools 定义 (可选，用于完整配置) ---
    if MCPConnectionManager is not None and bright_data_token:
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
            logger.warning(f"⚠️ Failed to fetch MCP tools (non-critical for local_search): {e}")

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
            "tool_map": mcp_tool_map  # MCP Search 工具定义
        },
        # Local Search Agent 配置
        "local_search_agent_args": {
            "parser_name": "qwen",
            "system_prompt": LOCAL_SEARCH_SYSTEM_PROMPT,
            "tool_map": local_search_tool_map
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
        # MCP Search Environment 配置 (完整配置)
        "search_args": {
            "mcp_server_command": mcp_server_command,
            "mcp_server_args": mcp_server_args,
            "mcp_server_env": mcp_server_env,
            "reward_fn": search_reward_fn,
            "cache_dir": search_cache_dir,
            "allowed_tools": allowed_mcp_tools
        },
        # Local Search Environment 配置
        "local_search_args": {
            "tool_map": local_search_tool_map,
            "reward_fn": search_reward_fn,
            "max_steps": 20,
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

    # --- 4. 加载 Local Search 数据 ---
    logger.info(f"Loading Local Search Dataset (subset: {debug_subset_size})...")
    tasks = load_local_search_data("test", debug_subset_size)
    
    if not tasks:
        raise ValueError("No tasks loaded for local_search!")

    random.shuffle(tasks)
    
    # --- 5. 执行调试 ---
    logger.info(f"Running local_search evaluation on {len(tasks)} samples...")
    
    results = asyncio.run(engine.execute_tasks(tasks))
    
    # --- 6. 保存与分析 ---
    output_file = "./trajectories/local_search_debug_trajectories.json"
    save_detailed_trajectories(results, output_file)
    
    # 简单的 Pass Rate 统计
    success_count = sum(1 for r in results if r.reward >= 1.0)
    print(f"\n>>> LOCAL_SEARCH Metrics: Accuracy = {success_count}/{len(results)} ({success_count/len(results):.2%})")

    # 打印失败/成功案例分析
    print("\n" + "="*60)
    print("LOCAL_SEARCH DEBUG ANALYSIS")
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
            print(f"Prompt              : {str(prompt)[:200]}...")
            print(f"Expect Ground Truth : {str(gt)[:200]}...")
            print("-" * 20 + " Model Output " + "-" * 20)
            print(f"{model_response_text[:500].replace(chr(10), ' ')}...") 
            print("-" * 60)
