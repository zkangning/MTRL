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
from rllm.rewards.reward_fn import math_reward_fn, code_reward_fn, search_reward_fn, tool_call_reward_fn
from rllm.data.utils import create_standard_sample, fetch_bfcl_tasks
# 引入 System Prompts
from rllm.agents.system_prompts import MATH_SYSTEM_PROMPT, SEARCH_SYSTEM_PROMPT

# [新增] 引入 MCP 组件 (用于 Search/Browsing)
try:
    from rllm.environments.tools.mcp_env import MCPConnectionManager
except ImportError:
    MCPConnectionManager = None

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 轨迹保存函数 ---
def save_detailed_trajectories(results, output_path: str = "debug_bfcl_trajectories.json"):
    """
    保存详细轨迹用于分析
    """
    logger.info(f"Saving detailed trajectories to {output_path}...")
    export_data = []
    
    for traj in results:
        task_data = dict(traj.task) if not isinstance(traj.task, dict) else traj.task
        
        # 提取任务信息
        task_id = task_data.get("task_id", "unknown")
        extra_info = task_data.get("extra_info", {})
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except:
                extra_info = {}
        
        steps_details = []
        if traj.steps:
            for step_idx, step in enumerate(traj.steps):
                # 提取思维链(如果有)
                thought = getattr(step, "thought", "")
                raw_response = getattr(step, "raw_model_response", step.model_response)
                
                steps_details.append({
                    "step_index": step_idx,
                    "thought": thought,
                    "model_response": step.model_response,
                    "raw_model_response": raw_response,
                    "action": str(step.action) if step.action else None,
                    "observation": str(step.observation) if step.observation else None,
                })
        
        record = {
            "uid": traj.uid,
            "reward": traj.reward,
            "task_type": task_data.get("task_type", "bfcl"),
            "task_id": task_id,
            "extra_info": extra_info,
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
    n_parallel_agents = 1  # 调试时使用单线程便于观察
    model_name = "Qwen3-8B"
    model_path = "/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/base_models/Qwen3-8B"
    
    api_base_url = "http://localhost:8803/v1"
    debug_subset_size = 3  # 调试样本数，可以调小便于快速迭代
    
    # 是否使用交互式调试模式
    interactive_mode = False
    
    # 是否使用手动构造的调试样本（而非从数据集加载）
    use_manual_samples = False

    # --- Search / MCP 环境配置 ---
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

    logger.info(f"Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Failed to load tokenizer from {model_path}: {e}")
        # 尝试使用默认的 Qwen tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B", trust_remote_code=True)
            logger.info("Fallback to Qwen/Qwen2-7B tokenizer")
        except:
            raise e

    # --- 预取 Search Tools 定义 ---
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
            logger.error(f"❌ Failed to fetch MCP tools: {e}")

    # --- 1. 构造 CompositeAgent 参数 ---
    # Math Agent 使用 ToolAgent，支持 Python 工具调用
    agent_args = {
        "math_agent_args": {
            "tools": ["python"],  # 启用 Python 工具
            "parser_name": "qwen",  # 使用 Qwen 的工具调用解析器
            "system_prompt": MATH_SYSTEM_PROMPT
        },
        "code_agent_args": {"accumulate_thinking": True},
        "bfcl_agent_args": {"parser_name": "qwen"},
        "tool_call_agent_args": {"parser_name": "qwen"},
        # Search Agent 配置
        "search_agent_args": {
            "parser_name": "qwen",
            "system_prompt": SEARCH_SYSTEM_PROMPT,
            "tool_map": mcp_tool_map  # 传入预获取的工具 Schema
        }
    }
    
    # --- 2. 构造 CompositeEnvironment 参数 ---
    env_args = {
        "math_args": {
            "tools": ["python"],  # 环境也需要配置 Python 工具
            "reward_fn": math_reward_fn,
            "max_steps": 10,  # Math 任务通常不需要太多步骤
        },
        "code_args": {"reward_fn": code_reward_fn},
        "bfcl_args": {"base_url": "http://localhost:8888", "env_type": "bfcl", "max_steps": 20},
        "tool_call_args": {
            "reward_fn": tool_call_reward_fn
        },
        # Search Environment 配置
        "search_args": {
            "mcp_server_command": mcp_server_command,
            "mcp_server_args": mcp_server_args,
            "mcp_server_env": mcp_server_env,
            "reward_fn": search_reward_fn,
            "cache_dir": search_cache_dir,
            "allowed_tools": allowed_mcp_tools
        }
    }

    sampling_params = {
        "temperature": 0.6, 
        "top_p": 0.95, 
        "model": model_name, 
        "max_tokens": 32768
    }

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
        max_steps=10,  # 最大交互步数
    )

    bfcl_base_url = 'http://localhost:8888'
    split = 'train'
    # --- 4. 加载 BFCL 数据 ---
    logger.info(f"Loading BFCL Dataset from {bfcl_base_url} (split={split})...")
    tasks = fetch_bfcl_tasks(base_url=bfcl_base_url, split=split)
    
    if not tasks:
        raise ValueError(f"No BFCL tasks loaded from {bfcl_base_url}!")
    
    # 随机采样指定数量的任务
    if debug_subset_size > 0 and debug_subset_size < len(tasks):
        random.shuffle(tasks)
        tasks = tasks[:debug_subset_size]
    
    logger.info(f"Loaded {len(tasks)} BFCL tasks for testing.")
    
    # --- 5. 执行调试 ---
    logger.info(f"Running BFCL evaluation on {len(tasks)} samples...")
    
    results = asyncio.run(engine.execute_tasks(tasks))
    
    # --- 6. 保存与分析 ---
    output_file = "./trajectories/bfcl_debug_trajectories.json"
    save_detailed_trajectories(results, output_file)
    
    # 简单的 Pass Rate 统计
    success_count = sum(1 for r in results if r.reward >= 1.0)
    print(f"\n>>> BFCL Metrics: Accuracy = {success_count}/{len(results)} ({success_count/len(results):.2%})")

    # 打印失败/成功案例分析
    print("\n" + "="*60)
    print("BFCL DEBUG ANALYSIS")
    print("="*60)

    for i, traj in enumerate(results):
        score = traj.reward
        # 只打印前3个或者 reward < 1 的
        if i < 3 or score < 1.0:
            task_data = traj.task if isinstance(traj.task, dict) else dict(traj.task)
            task_id = task_data.get("task_id", "unknown")
            
            # 提取 extra_info
            extra_info = task_data.get("extra_info", {})
            if isinstance(extra_info, str):
                try:
                    extra_info = json.loads(extra_info)
                except:
                    extra_info = {}
            
            task_id_from_extra = extra_info.get("task_id", task_id)

            model_response_text = ""
            thought_text = ""
            if traj.steps:
                last_step = traj.steps[-1]
                model_response_text = last_step.model_response
                thought_text = getattr(last_step, "thought", "")
            else:
                model_response_text = "No steps executed."

            print(f"\n[Case #{i} | Reward: {score} | Task ID: {task_id_from_extra}]")
            print(f"UID                 : {traj.uid}")
            print(f"Task ID             : {task_id_from_extra}")
            print(f"Steps Count         : {len(traj.steps)}")
            
            if thought_text:
                print("-" * 20 + " Thought " + "-" * 20)
                print(f"{thought_text[:200]}...")
            
            print("-" * 20 + " Model Output " + "-" * 20)
            print(f"{model_response_text[:300]}...") 
            print("-" * 60)
