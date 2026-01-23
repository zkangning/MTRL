import asyncio
import argparse
import logging
import json
from transformers import AutoTokenizer
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.utils import colorful_print
from rllm.agents.tool_agent import ToolAgent

# 导入修改后的 Env 和 Agent
from rllm.environments.tools.bfcl_env_v2 import BFCLEnvironment
from rllm.agents.bfcl_agent import BFCLReadyAgent
from rllm.utils.compute_pass_at_k import save_clean_trajectories

# 设置日志级别，减少干扰，专注看 print 输出
logging.basicConfig(level=logging.WARNING)

def parse_action_display(action):
    """辅助函数：让 Action 显示更易读"""
    if isinstance(action, str):
        return action
    if isinstance(action, list):
        # 尝试提取函数名
        displays = []
        for call in action:
            if isinstance(call, dict) and 'function' in call:
                func = call['function']
                displays.append(f"{func.get('name')}({json.dumps(func.get('arguments'))})")
        return " | ".join(displays)
    return str(action)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, default="http://localhost:8801", help="BFCL EnvService URL")
    parser.add_argument("--model_url", type=str, default="http://localhost:8803/v1", help="vLLM API URL")
    # 建议 n_agents 设为 1 或少量，方便肉眼观察详细日志
    parser.add_argument("--n_agents", type=int, default=1, help="Number of parallel agents") 
    parser.add_argument("--max_steps", type=int, default=20, help="Max steps per episode")
    # 可以指定特定的测试 ID 来复现，例如 multi_turn_base_1 通常有多步
    parser.add_argument("--task_id", type=str, default="multi_turn_base_1", help="Specific task ID to test")
    args = parser.parse_args()

    # 模型加载 (用于 Tokenizer)
    model_name = "/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/base_models/Qwen3-32B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 1. Environment Config
    env_args = {
        "base_url": args.base_url,
        "env_type": "bfcl",
        "max_steps": args.max_steps,
        # 如果指定了 task_id，会强制测试该任务，方便调试一致性
        "task_id": args.task_id 
    }

    # 2. Agent Config
    agent_args = {
        "parser_name": "qwen", 
        "system_prompt": "Placeholder", 
    }

    # 3. Rollout / VLLM Config
    rollout_args = {
        "base_url": args.model_url,
        "model": "Qwen3-32B",
        "temperature": 0.0, # 设置为 0 方便复现
        "top_p": 0.95,
        "stop_token_ids": [151645], 
    }

    # 4. Engine Initialization
    engine = AgentExecutionEngine(
        engine_name="openai",
        n_parallel_agents=args.n_agents,
        max_steps=args.max_steps,
        agent_class=BFCLReadyAgent,
        env_class=BFCLEnvironment,
        agent_args=agent_args,
        env_args=env_args,
        rollout_engine_args=rollout_args,
        tokenizer=tokenizer,
        max_prompt_length=30720,
        max_response_length=30720
    )

    # 5. Task Execution
    # 构造任务列表
    tasks = [{"task_id": args.task_id} for _ in range(args.n_agents)]

    colorful_print(f"Starting execution for Task ID: {args.task_id}", "cyan")
    
    trajectories = await engine.execute_tasks(tasks)

    # 6. Detailed Step-wise Reward Analysis
    colorful_print(f"\n{'='*20} Step-wise Reward Analysis {'='*20}", "yellow")
    
    total_consistency_score = 0
    total_completion_score = 0
    
    for i, traj in enumerate(trajectories):
        print(f"\n>>> Trajectory #{i+1} (Instance ID: {traj.info.get('instance_id', 'N/A')})")
        
        cumulative_reward = 0.0
        
        for step_idx, step in enumerate(traj.steps):
            # 获取 Action 和 Reward
            action_display = parse_action_display(step.action)
            step_reward = step.reward
            cumulative_reward += step_reward
            
            # 获取 Info 中的细粒度分数
            # 注意：Trajectory 中的 step.info 应该包含了环境返回的 info
            step_info = step.info or {}
            consistency = step_info.get("step_consistency_score", 0.0)
            final_bonus = step_info.get("final_eval_score", 0.0)
            
            # 颜色高亮
            reward_color = "green" if step_reward > 0 else "red"
            
            print(f"  [Step {step_idx + 1}]")
            print(f"    Action      : {action_display}")
            
            # 打印关键的 Reward 构成
            reward_breakdown = f"Consistency={consistency:.2f}"
            if final_bonus > 0 or step.done:
                reward_breakdown += f" + FinalBonus={final_bonus:.2f}"
            
            colorful_print(f"    Step Reward : {step_reward:.2f} ({reward_breakdown})", reward_color)
            
            # 检查是否有工具输出
            if "tool_outputs" in step.observation:
                print(f"    Observation : Tool Output Received (Keys: {list(step.observation['tool_outputs'])})")
            elif "question" in step.observation:
                 print(f"    Observation : New User Question -> {step.observation['question']}")
            
            if step.done:
                print(f"    [Terminated] Reason: {step.info.get('error', 'Completed')}")

        # 统计
        final_info = traj.steps[-1].info if traj.steps else {}
        traj_completion = final_info.get("final_eval_score", 0.0)
        
        print(f"-"*30)
        print(f"  Total Trajectory Reward: {cumulative_reward:.2f}")
        print(f"  Final Completion Status: {'SUCCESS' if traj_completion > 0 else 'FAILURE'}")

    save_clean_trajectories(trajectories, save_path="./trajectories/bfcl_step_reward_test.json")
    colorful_print(f"\nAnalysis saved to ./trajectories/bfcl_step_reward_test.json", "cyan")

if __name__ == "__main__":
    asyncio.run(main())
