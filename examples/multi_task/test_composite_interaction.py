import asyncio
import argparse
import logging
import json
import uuid
from transformers import AutoTokenizer
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.utils import colorful_print

# 1. 导入基础组件
from rllm.rewards.reward_fn import math_reward_fn

# 2. 导入刚刚构建的 Composite 组件
# 假设你已经按照上一步将它们保存到了对应的路径
from rllm.environments.composite.composite_env import CompositeEnvironment
from rllm.agents.composite_agent import CompositeAgent
from rllm.utils.compute_pass_at_k import save_clean_trajectories
# from rllm.agents.system_prompts import TOOL_SYSTEM_PROMPT

# 设置日志
logging.basicConfig(level=logging.WARNING)

def parse_step_display(step, task_type):
    """根据任务类型解析显示内容"""
    output = []
    
    # 显示 Model Thinking (Math 任务特有)
    if hasattr(step, "thought") and step.thought:
        output.append(f"  [Thinking]: {step.thought[:100]}... (len={len(step.thought)})")
    
    # 显示 Action / Response
    action = step.action
    if isinstance(action, list): # 通常是 BFCL 的 Tool Call
        tools = []
        for call in action:
            if isinstance(call, dict) and 'function' in call:
                func = call['function']
                tools.append(f"{func.get('name')}({json.dumps(func.get('arguments'))})")
        output.append(f"  [Action-Tools]: {' | '.join(tools)}")
    else: # String Action (Math 答案 或 BFCL 对话)
        output.append(f"  [Action-Text]: {str(action)[:100]}")
        
    return "\n".join(output)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bfcl_url", type=str, default="http://localhost:8801", help="BFCL Env URL")
    parser.add_argument("--model_url", type=str, default="http://localhost:8803/v1", help="Model API URL")
    parser.add_argument("--model_name", type=str, default="Qwen3-32B", help="Model Name for API")
    args = parser.parse_args()

    print("构造混合任务列表")
    # 1. 手动构造混合任务列表
    # 我们故意混合两种完全不同的任务来测试路由
    mixed_tasks = [
        # --- 任务 1: BFCL 任务 (多轮，需要工具调用) ---
        # {
        #     "task_type": "bfcl",
        #     "task_id": "multi_turn_base_1", # 确保这个 ID 在你的 BFCL Server 中存在
        # },
        # --- 任务 2: Math 任务 (单轮，需要 CoT) ---
        {
            "task_type": "math",
            "task_id": "math_base_0",
            "question": "If x + 5 = 12, what is the value of 2x?",
            "ground_truth": ["14"]
        }
    ]

    colorful_print(f"Loaded {len(mixed_tasks)} mixed tasks for testing.", "cyan")

    # 2. 准备配置参数 (Nested Config)
    
    # 2.1 BFCL 配置
    bfcl_env_args = {
        "base_url": args.bfcl_url,
        "env_type": "bfcl",
        "max_steps": 10,
    }
    bfcl_agent_args = {
        "parser_name": "qwen",
        "system_prompt": "You are a helpful assistant.", 
    }

    # 2.2 Math 配置
    math_env_args = {
        "reward_fn": math_reward_fn
    }
    math_agent_args = {
        "accumulate_thinking": False # 开启 CoT 收集
    }

    # 2.3 组合配置
    composite_env_args = {
        "bfcl_args": bfcl_env_args,
        "math_args": math_env_args
    }
    composite_agent_args = {
        "bfcl_agent_args": bfcl_agent_args,
        "math_agent_args": math_agent_args
    }

    # 3. 初始化 Tokenizer (仅用于长度计算)
    model_name = "/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/base_models/Qwen3-32B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 4. 初始化引擎
    # 注意：这里传入的是 CompositeAgent 和 CompositeEnvironment
    rollout_args = {
        "base_url": args.model_url,
        "model": args.model_name,
        "temperature": 0.0,
        "top_p": 0.95,
        "api_key": "empty"
    }

    engine = AgentExecutionEngine(
        engine_name="openai",
        n_parallel_agents=1, # 同时运行两个不同的任务
        max_steps=10,
        agent_class=CompositeAgent,
        env_class=CompositeEnvironment,
        agent_args=composite_agent_args,
        env_args=composite_env_args,
        rollout_engine_args=rollout_args,
        tokenizer=tokenizer,
        max_prompt_length=4096,
        max_response_length=4096
    )

    # 5. 执行任务
    colorful_print(">>> Starting Execution Engine...", "yellow")
    trajectories = await engine.execute_tasks(mixed_tasks)

    # 6. 验证结果 (Verification)
    colorful_print(f"\n{'='*20} Result Analysis {'='*20}", "green")


    save_clean_trajectories(trajectories, save_path="./trajectories/multi_task_reward_test.json")
    # for i, traj in enumerate(trajectories):
    #     # 从 info 中获取 task_type，这是验证路由是否成功的关键
    #     # 注意：traj.steps[0].info 可能包含 reset 时的 info
    #     # 或者 traj.info (取决于 RLLM 版本实现)
        
    #     # 尝试获取任务类型
    #     task_type = "Unknown"
    #     if traj.steps:
    #         # 这里的 info 是 env.step 返回的，我们在 CompositeEnv.step 中注入了 task_type
    #         task_type = traj.steps[0].info.get("task_type", "Unknown")
        
    #     print(f"\nTask #{i+1} | Type: [{task_type.upper()}] | Steps: {len(traj.steps)}")
        
    #     total_reward = 0
    #     for step_idx, step in enumerate(traj.steps):
    #         total_reward += step.reward
            
    #         print(f"  Step {step_idx+1}:")
            
    #         # 打印 Model 输出 (Action / Thought)
    #         print(parse_step_display(step, task_type))
            
    #         # 打印 Reward 和 Observation
    #         print(f"    -> Reward: {step.reward}")
            
    #         # 验证不同任务的 Observation 特征
    #         obs = step.observation
    #         if isinstance(obs, dict):
    #             if "tool_outputs" in obs:
    #                 print(f"    -> Obs: Tool Output (Keys: {list(obs['tool_outputs'].keys())})")
    #             elif "question" in obs:
    #                 print(f"    -> Obs: User Question: {obs['question']}")
            
    #     print(f"  Total Reward: {total_reward}")
        
    #     # --- 自动断言检查 ---
    #     if task_type == "bfcl":
    #         if any(isinstance(s.action, list) for s in traj.steps):
    #             colorful_print("  [PASS] BFCL task successfully generated list-type Actions (Tool Calls).", "green")
    #         else:
    #             colorful_print("  [WARN] BFCL task did not generate tool calls (might be simple chat or failure).", "yellow")
                
        # elif task_type == "math":
        #     has_thought = any(hasattr(s, "thought") and s.thought for s in traj.steps)
        #     if has_thought:
        #         colorful_print("  [PASS] Math task successfully captured <think> content.", "green")
        #     else:
        #         colorful_print("  [WARN] Math task did not capture thought (check model capability).", "yellow")

if __name__ == "__main__":
    asyncio.run(main())
