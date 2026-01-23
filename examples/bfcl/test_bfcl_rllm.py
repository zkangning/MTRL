# 这个版本添加了BFCL中的多轮问答
import asyncio
import argparse
import logging
from transformers import AutoTokenizer
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.utils import colorful_print
from rllm.agents.tool_agent import ToolAgent

# 导入上面修改后的 Env
from rllm.environments.tools.bfcl_env import BFCLEnvironment
from rllm.agents.bfcl_agent import BFCLReadyAgent
from rllm.utils.compute_pass_at_k import save_clean_trajectories

logging.basicConfig(level=logging.INFO)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, default="http://localhost:8801", help="BFCL EnvService URL")
    parser.add_argument("--model_url", type=str, default="http://localhost:8803/v1", help="vLLM API URL")
    parser.add_argument("--n_agents", type=int, default=1, help="Number of parallel agents")
    parser.add_argument("--max_steps", type=int, default=20, help="Max steps per episode")
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
        "max_steps": args.max_steps
    }

    # 2. Agent Config
    # parser_name 必须匹配你的 LLM 输出格式，Qwen 通常使用 xml 或 function call parser
    agent_args = {
        "parser_name": "qwen", 
        "system_prompt": "Placeholder", # 将被 Env 覆盖
    }

    # 3. Rollout / VLLM Config
    rollout_args = {
        "base_url": args.model_url,
        "model": "Qwen3-32B",
        "temperature": 0.0,
        "top_p": 0.95,
        "stop_token_ids": [151645], # <|im_end|>
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
    # 我们传入 dummy tasks，具体的 Task ID 获取逻辑在 BFCLEnvironment.reset() 中处理
    tasks = [{"task_id": "multi_turn_base_1"} for _ in range(args.n_agents)]

    colorful_print(f"Starting {args.n_agents} Agents...", "cyan")
    
    trajectories = await engine.execute_tasks(tasks)

    # 6. Analysis
    total_reward = sum(t.reward for t in trajectories)
    success_count = sum(1 for t in trajectories if t.reward > 0)
    
    colorful_print(f"\nExecution Finished.", "green")
    colorful_print(f"Average Reward: {total_reward / len(trajectories):.2f}", "green")
    colorful_print(f"Success Rate: {success_count / len(trajectories):.2%}", "green")

    save_clean_trajectories(trajectories, save_path="./trajectories/bfcl_base_01.json")
    
    # for i, traj in enumerate(trajectories):
    #     print(f"\n--- Trajectory {i} (Final Reward: {traj.reward}) ---")
    #     for step in traj.steps:
    #         # 打印简略 Action
    #         action_str = str(step.action)
    #         if len(action_str) > 100: action_str = action_str[:100] + "..."
            
    #         # 打印观察类型
    #         obs_type = "Tool Output" if "tool_outputs" in step.observation else "New Question"
            
    #         print(f"Step {step.step_num}: Action=[{action_str}] -> ObsType=[{obs_type}]")

if __name__ == "__main__":
    asyncio.run(main())
