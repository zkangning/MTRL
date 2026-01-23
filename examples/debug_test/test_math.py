"""
test_math.py - 用于调试和观察 Math Agent 在交互过程中的 Prompt 拼接和调用逻辑

基于 test_tool_call.py 的实现，专门针对 Math 任务进行调试。
可以观察：
1. System Prompt 的构建
2. 用户问题的拼接
3. 工具调用（Python）的解析和执行
4. 多轮对话的消息累积
5. Reward 计算逻辑
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
from rllm.rewards.reward_fn import math_reward_fn, code_reward_fn, search_reward_fn, tool_call_reward_fn
from rllm.data.utils import create_standard_sample
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


def create_math_sample(question: str, answer: str) -> Dict:
    """
    创建一个标准的 Math 样本
    """
    raw_data = {
        "question": question,
        "answer": answer,
        "task_type": "math",
        "source": "debug_manual"
    }
    return create_standard_sample(
        prompt=question,
        response=answer,
        task_type="math",
        raw_data=raw_data
    )


def get_debug_math_samples() -> List[Dict]:
    """
    返回一些用于调试的 Math 样本
    包含不同难度和类型的数学问题
    """
    samples = [
        # 简单算术
        create_math_sample(
            question="What is 123 + 456?",
            answer="579"
        ),
        # 需要 Python 计算的问题
        create_math_sample(
            question="Calculate the sum of all prime numbers less than 100.",
            answer="1060"
        ),
        # 代数问题
        create_math_sample(
            question="Solve for x: 2x + 5 = 17",
            answer="6"
        ),
        # 几何问题
        create_math_sample(
            question="A circle has a radius of 7. What is its area? Express your answer in terms of pi.",
            answer="49\\pi"
        ),
        # 组合数学
        create_math_sample(
            question="How many ways can you arrange the letters in the word 'MATH'?",
            answer="24"
        ),
    ]
    return samples


def save_detailed_trajectories(results, output_path: str = "math_debug_trajectories.json"):
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
                step_detail = {
                    "step_index": step_idx,
                    "observation": str(step.observation) if step.observation else None,
                    "model_response": step.model_response,
                    "action": str(step.action) if step.action else None,
                    "reward": step.reward if hasattr(step, 'reward') else None,
                    "done": step.done if hasattr(step, 'done') else None,
                }
                
                # 如果有 chat_completions，也保存下来用于分析 Prompt 拼接
                if hasattr(step, 'chat_completions') and step.chat_completions:
                    step_detail["chat_completions"] = step.chat_completions
                    
                steps_details.append(step_detail)
        
        record = {
            "uid": traj.uid,
            "reward": traj.reward,
            "task_type": task_data.get("task_type", "math"),
            "question": question,
            "ground_truth_expected": str(ground_truth),
            "num_steps": len(traj.steps) if traj.steps else 0,
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


def print_prompt_analysis(results):
    """
    打印 Prompt 拼接分析，帮助理解消息构建逻辑
    """
    print("\n" + "=" * 80)
    print("PROMPT CONSTRUCTION ANALYSIS")
    print("=" * 80)
    
    for i, traj in enumerate(results):
        task_data = traj.task if isinstance(traj.task, dict) else dict(traj.task)
        print(f"\n{'─' * 80}")
        print(f"[Case #{i}] Reward: {traj.reward}")
        print(f"{'─' * 80}")
        
        if not traj.steps:
            print("  No steps recorded.")
            continue
            
        for step_idx, step in enumerate(traj.steps):
            print(f"\n  ┌─ Step {step_idx} ─────────────────────────────────────────")
            
            # 打印 chat_completions 来观察 Prompt 拼接
            if hasattr(step, 'chat_completions') and step.chat_completions:
                print(f"  │ Chat Completions ({len(step.chat_completions)} messages):")
                for msg_idx, msg in enumerate(step.chat_completions):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    # 截断过长的内容
                    if len(content) > 500:
                        content = content[:500] + "... [TRUNCATED]"
                    print(f"  │   [{msg_idx}] {role.upper()}:")
                    # 缩进内容
                    for line in content.split('\n')[:10]:  # 只显示前10行
                        print(f"  │       {line}")
                    if content.count('\n') > 10:
                        print(f"  │       ... [MORE LINES]")
            
            # 打印 Observation
            if step.observation:
                obs_str = str(step.observation)
                if len(obs_str) > 300:
                    obs_str = obs_str[:300] + "... [TRUNCATED]"
                print(f"  │")
                print(f"  │ Observation: {obs_str}")
            
            # 打印 Model Response
            if step.model_response:
                resp_str = step.model_response
                if len(resp_str) > 500:
                    resp_str = resp_str[:500] + "... [TRUNCATED]"
                print(f"  │")
                print(f"  │ Model Response:")
                for line in resp_str.split('\n')[:15]:
                    print(f"  │   {line}")
                if resp_str.count('\n') > 15:
                    print(f"  │   ... [MORE LINES]")
            
            # 打印 Action
            if step.action:
                action_str = str(step.action)
                if len(action_str) > 200:
                    action_str = action_str[:200] + "... [TRUNCATED]"
                print(f"  │")
                print(f"  │ Action: {action_str}")
            
            print(f"  │")
            print(f"  │ Reward: {step.reward if hasattr(step, 'reward') else 'N/A'}")
            print(f"  │ Done: {step.done if hasattr(step, 'done') else 'N/A'}")
            print(f"  └{'─' * 60}")


async def run_interactive_debug(engine, tasks):
    """
    交互式调试模式：逐个执行任务，允许在每个任务后暂停观察
    """
    print("\n" + "=" * 80)
    print("INTERACTIVE DEBUG MODE")
    print("=" * 80)
    
    results = []
    for i, task in enumerate(tasks):
        print(f"\n>>> Processing Task {i+1}/{len(tasks)}")
        print(f"    Question: {task.get('prompt', 'N/A')[:100]}...")
        
        # 执行单个任务
        task_results = await engine.execute_tasks([task])
        results.extend(task_results)
        
        # 打印当前任务的结果
        if task_results:
            traj = task_results[0]
            print(f"    Reward: {traj.reward}")
            print(f"    Steps: {len(traj.steps) if traj.steps else 0}")
            
            # 可以在这里设置断点进行调试
            # import pdb; pdb.set_trace()
    
    return results


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

    # --- 4. 加载数据 ---
    if use_manual_samples:
        logger.info("Using manually constructed debug samples...")
        tasks = get_debug_math_samples()[:debug_subset_size]
    else:
        # 从 HuggingFace 加载数据
        logger.info("Loading Math Dataset from HuggingFace...")
        from rllm.data.utils import load_dapo_math_dataset, load_deepmath_dataset
        try:
            tasks = load_dapo_math_dataset(debug_subset_size)
        except Exception as e:
            logger.warning(f"Failed to load DAPO dataset: {e}, trying DeepMath...")
            tasks = load_deepmath_dataset(debug_subset_size)

    if not tasks:
        raise ValueError("No tasks loaded!")

    logger.info(f"Loaded {len(tasks)} Math tasks for debugging")
    
    # 打印第一个任务的结构，帮助理解数据格式
    print("\n" + "=" * 80)
    print("SAMPLE TASK STRUCTURE")
    print("=" * 80)
    if tasks:
        sample_task = tasks[0]
        print(json.dumps(sample_task, indent=2, ensure_ascii=False)[:2000])
    
    # --- 5. 执行调试 ---
    logger.info(f"Running Math evaluation on {len(tasks)} samples...")
    
    
    results = asyncio.run(engine.execute_tasks(tasks))
    
    # --- 6. 分析 Prompt 拼接逻辑 ---
    print_prompt_analysis(results)
    
    # --- 7. 保存轨迹 ---
    output_file = "./trajectories/math_debug_trajectories.json"
    save_detailed_trajectories(results, output_file)
    
    # --- 8. 统计结果 ---
    success_count = sum(1 for r in results if r.reward >= 1.0)
    print(f"\n>>> MATH Metrics: Accuracy = {success_count}/{len(results)} ({success_count/len(results):.2%})")

    # 打印详细的失败/成功案例分析
    print("\n" + "=" * 80)
    print("MATH DEBUG ANALYSIS - SUMMARY")
    print("=" * 80)

    for i, traj in enumerate(results):
        score = traj.reward
        task_data = traj.task if isinstance(traj.task, dict) else dict(traj.task)
        
        # 解析 extra_info 获取原始数据
        extra_info = task_data.get("extra_info", "{}")
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except:
                extra_info = {}
        
        prompt = task_data.get("prompt", "") or extra_info.get("question", "")
        gt = task_data.get("response", "") or extra_info.get("answer", "") or "Unknown"

        # 获取模型最终回答
        model_response_text = ""
        if traj.steps:
            last_step = traj.steps[-1]
            model_response_text = last_step.model_response if last_step.model_response else "No response"
        else:
            model_response_text = "No steps executed."

        status = "✓ PASS" if score >= 1.0 else "✗ FAIL"
        print(f"\n[Case #{i} | {status} | Reward: {score}]")
        print(f"Question: {str(prompt)[:200]}...")
        print(f"Expected Answer: {str(gt)}")
        print(f"Num Steps: {len(traj.steps) if traj.steps else 0}")
        print("-" * 40)
        # 只打印模型回答的最后部分（通常包含 \boxed{} 答案）
        if len(model_response_text) > 500:
            print(f"Model Response (last 500 chars): ...{model_response_text[-500:]}")
        else:
            print(f"Model Response: {model_response_text}")
        print("-" * 80)
