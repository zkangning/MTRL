import asyncio
import os
import sys
import shutil
import json
from typing import List, Any, Dict

from transformers import AutoTokenizer
from rllm.agents.tool_agent import MCPToolAgent
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.tools.mcp_env import MCPConnectionManager, MCPEnvironment

# from rllm.types import Trajectory 

# ---------------- 1. 自定义 Environment (关键修改) ----------------

class FileSystemMCPEnv(MCPEnvironment):
    """
    FileSystem 专用环境。
    职责：
    1. 管理 sandbox_dir。
    2. 在 reset() 时，将 sandbox_dir 的路径约束注入到 Agent 的 observation 中，
       而不修改原始的 user task。
    """
    def __init__(self, sandbox_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.sandbox_dir = sandbox_dir

    def reset(self):
        # 1. 获取原始的 observation (通常就是 self.task)
        obs, info = super().reset()
        
        # 2. 注入环境上下文
        if isinstance(obs, dict) and "question" in obs:
            # 深拷贝一份，以免修改了 self.task 中的原始记录
            agent_obs = obs.copy()
            original_question = agent_obs["question"]
            
            # 构造带约束的 Prompt
            context_prompt = (
                f"{original_question}\n\n"
                f"--- ENVIRONMENT CONTEXT ---\n"
                f"Root Directory: {self.sandbox_dir}\n"
                f"Requirement: You MUST use absolute paths starting with {self.sandbox_dir}."
            )
            
            # 替换 Agent 看到的内容
            agent_obs["question"] = context_prompt
            return agent_obs, info
            
        return obs, info

    @staticmethod
    def from_dict(env_args: dict[str, Any]) -> "FileSystemMCPEnv":
        # 必须重写 from_dict，否则 Engine 会实例化父类 MCPEnvironment
        
        # 提取本类特有的参数
        sandbox_dir = env_args.get("sandbox_dir")
        if not sandbox_dir:
            raise ValueError("FileSystemMCPEnv requires 'sandbox_dir' in env_args")

        # 提取父类参数
        mcp_server_command = env_args.pop("mcp_server_command", None)
        mcp_server_args = env_args.pop("mcp_server_args", None)
        mcp_server_env = env_args.pop("mcp_server_env", None)
        reward_fn = env_args.pop("reward_fn", None)
        max_steps = env_args.pop("max_steps", 10)

        return FileSystemMCPEnv(
            task=env_args, # 注意：此时传入 task 会作为初始 task
            mcp_server_command=mcp_server_command,
            mcp_server_args=mcp_server_args,
            mcp_server_env=mcp_server_env,
            reward_fn=reward_fn,
            max_steps=max_steps,
            sandbox_dir=sandbox_dir
        )

# ---------------- 2. 增强后的轨迹保存 ----------------

def save_clean_trajectories(trajectories: List[Any], save_path: str):
    cleaned_data = []
    
    for traj in trajectories:
        # 获取原始任务信息
        task_info = traj.task if traj.task else {}
        task_id = task_info.get("id", "unknown")
        
        # 关键：保存原始 User Query，而不是被 Env 修改过的 Prompt
        user_instruction = task_info.get("instruction", "")
        user_question = task_info.get("question", "")
        
        interaction_log = []
        for step in traj.steps:
            # 提取动作
            actions = []
            if isinstance(step.action, list):
                for act in step.action:
                    if act.get("type") == "function":
                        func = act.get("function", {})
                        actions.append({"tool": func.get("name"), "args": func.get("arguments")})
            
            # 提取观察
            observations = []
            if step.observation and isinstance(step.observation, dict):
                outputs = step.observation.get("tool_outputs", {})
                for _, output in outputs.items():
                    # 截断长输出
                    observations.append(output[:500] + "..." if len(output) > 500 else output)
            
            interaction_log.append({
                "agent_response": step.model_response,
                "tool_calls": actions,
                "tool_outputs": observations
            })

        cleaned_data.append({
            "task_id": task_id,
            "user_query": user_instruction, # 保存干净的用户查询
            "final_reward": traj.reward,
            "interactions": interaction_log
        })

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

# ---------------- 3. Reward Function (保持不变) ----------------

class RewardOutput:
    def __init__(self, reward: float, metadata: Dict[str, Any] = None):
        self.reward = reward
        self.metadata = metadata or {}

def filesystem_reward_fn(task_info: Dict[str, Any], action: str) -> RewardOutput:
    sandbox_dir = task_info.get("sandbox_dir")
    ground_truth = task_info.get("ground_truth")
    
    if not ground_truth or not sandbox_dir:
        return RewardOutput(reward=1.0)

    target_file = os.path.join(sandbox_dir, ground_truth)
    if os.path.exists(target_file):
        return RewardOutput(reward=1.0, metadata={"status": "success"})
    else:
        return RewardOutput(reward=0.0, metadata={"status": "failed", "path": target_file})

# ---------------- 4. 主流程 ----------------

async def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    # === 配置 ===
    model_name = "/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/base_models/Qwen3-8B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 路径配置
    sandbox_dir = os.path.abspath("./test_environments")
    if os.path.exists(sandbox_dir):
        shutil.rmtree(sandbox_dir)
    os.makedirs(sandbox_dir, exist_ok=True)
    with open(os.path.join(sandbox_dir, "existing_file.txt"), "w") as f:
        f.write("Secret content inside sandbox.")

    print(f"Sandbox Root: {sandbox_dir}")

    mcp_server_command = "npx"
    mcp_server_args = ["-y", "@modelcontextprotocol/server-filesystem", sandbox_dir]
    mcp_server_env = os.environ.copy()

    # 获取工具 (保持不变)
    print("Fetching tools...")
    temp_manager = MCPConnectionManager(mcp_server_command, mcp_server_args, mcp_server_env)
    temp_manager.start()
    try:
        mcp_tool_map = temp_manager.tool_map
    finally:
        temp_manager.stop()

    # === 初始化 Engine ===
    # 注意：System Prompt 只负责设定角色，不再负责硬编码路径
    system_prompt = (
        "You are a helpful assistant with file system access.\n"
        "Analyze the user request and the environment context provided to perform file operations."
    )

    engine = AgentExecutionEngine(
        agent_class=MCPToolAgent,
        env_class=FileSystemMCPEnv,  # <--- 使用新类
        agent_args={
            "parser_name": "qwen", 
            "system_prompt": system_prompt, 
            "tool_map": mcp_tool_map
        },
        env_args={
            "mcp_server_command": mcp_server_command,
            "mcp_server_args": mcp_server_args,
            "mcp_server_env": mcp_server_env,
            "reward_fn": filesystem_reward_fn,
            "max_steps": 5,
            "sandbox_dir": sandbox_dir  # <--- 传给 Env
        },
        engine_name="openai",
        rollout_engine_args={"base_url": "http://localhost:30000/v1", "api_key": "EMPTY"},
        tokenizer=tokenizer,
        sampling_params={"temperature": 0.1, "max_tokens": 32768},
        n_parallel_agents=1,
        max_prompt_length=8192
    )

    # === 定义任务 (现在非常干净) ===
    tasks = [
        {
            "id": 1, 
            "instruction": "Create a file named 'hello.txt' with content 'Hello World'.",
            "question": "Create a file named 'hello.txt' with content 'Hello World'.",
            "ground_truth": "hello.txt",
            "sandbox_dir": sandbox_dir # 依然需要传给 task 以便 reward_fn 使用
        },
        {
            "id": 2, 
            "instruction": "List all files in the current directory.",
            "question": "List all files in the current directory.",
            "ground_truth": None,
            "sandbox_dir": sandbox_dir
        },
        {
            "id": 3, 
            "instruction": "Read the content of 'existing_file.txt'.",
            "question": "Read the content of 'existing_file.txt'.",
            "ground_truth": None,
            "sandbox_dir": sandbox_dir
        }
    ]

    try:
        results = await engine.execute_tasks(tasks)
        
        save_dir = "./trajectories/mcp_filesystem"
        os.makedirs(save_dir, exist_ok=True)
        save_clean_trajectories(results, os.path.join(save_dir, "trajectories.json"))
        print("Done.")

        if os.path.exists(os.path.join(sandbox_dir, "hello.txt")):
            print("VERIFICATION: SUCCESS")
        else:
            print("VERIFICATION: FAILED")

    finally:
        MCPEnvironment.cleanup_global_resources()

if __name__ == "__main__":
    asyncio.run(main())
