import asyncio
import os
import shutil
import json
import logging
from transformers import AutoTokenizer
from rllm.agents.tool_agent import MCPToolAgent
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.utils.compute_pass_at_k import save_clean_trajectories

# 导入你的环境类
from rllm.environments.mcp.multi_mcp_env import MultiMCPEnvironment, MultiMCPConnectionManager

# === 1. 修改后的配置：使用 SQLite 替代 Calculator ===
SERVER_LIST_JSON = """
{
  "date": {
    "stdio": {
      "command": "python3",
      "args": ["-m", "mcpuniverse.mcp.servers.date"]
    }
  },
  "echo": {
    "stdio": {
      "command": "python3",
      "args": ["-m", "mcpuniverse.mcp.servers.echo"]
    }
  },
"calculator": {
    "stdio": {
      "command": "python3",
      "args": [
        "-m", "mcp_server_calculator"
      ]
    }
  },
  "weather": {
    "stdio": {
      "command": "python3",
      "args": [
        "-m", "mcpuniverse.mcp.servers.weather"
      ]
    },
    "sse": {
      "command": "python3",
      "args": [
        "-m", "mcpuniverse.mcp.servers.weather",
        "--transport", "sse",
        "--port", "{{PORT}}"
      ]
    }
  }
}
"""

def parse_server_configs(json_content: str, selected_servers: list[str], vars_map: dict[str, str]) -> dict:
    """解析 JSON，替换变量，提取 stdio 配置"""
    try:
        all_configs = json.loads(json_content)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON config: {e}")
        return {}

    final_configs = {}
    
    for name in selected_servers:
        if name not in all_configs:
            print(f"Warning: Server '{name}' not found in provided JSON.")
            continue
            
        server_def = all_configs[name]
        
        # 优先使用 stdio 模式
        if "stdio" not in server_def:
            print(f"Warning: Server '{name}' does not have stdio config.")
            continue
            
        config = server_def["stdio"].copy()
        
        # 处理 args 中的变量替换
        if "args" in config:
            new_args = []
            for arg in config["args"]:
                for var_key, var_val in vars_map.items():
                    arg = arg.replace(f"{{{{{var_key}}}}}", var_val)
                new_args.append(arg)
            config["args"] = new_args
            
        # 处理 env
        env_vars = os.environ.copy()
        # 确保 npx 能找到 node
        env_vars["PATH"] = os.environ.get("PATH", "")
        
        if "env" in server_def:
            for k, v in server_def["env"].items():
                for var_key, var_val in vars_map.items():
                    v = v.replace(f"{{{{{var_key}}}}}", var_val)
                env_vars[k] = v
        
        config["env"] = env_vars
        final_configs[name] = config
        
    return final_configs

async def main():
    # === 2. 准备环境路径 ===
    sandbox_dir = os.path.abspath("./multi_mcp_sandbox")
    if os.path.exists(sandbox_dir):
        shutil.rmtree(sandbox_dir)
    os.makedirs(sandbox_dir, exist_ok=True)
    
    print(f"Sandbox Directory: {sandbox_dir}")

    # === 3. 生成 Server Configs ===
    selected_servers = ["date", "echo", "calculator", "weather"]
    
    vars_map = {
        "FILESYSTEM_DIRECTORY": sandbox_dir,
        "PORT": "0"
    }
    
    server_configs = parse_server_configs(SERVER_LIST_JSON, selected_servers, vars_map)
    print(f"Active Servers to Initialize: {list(server_configs.keys())}")

    # === 4. 预取 Tools ===
    print("\n[Step 1] Connecting to servers and fetching tools...")
    temp_manager = MultiMCPConnectionManager(server_configs)
    
    # 使用 try-except 块来捕获连接错误，避免直接崩溃
    try:
        temp_manager.start()
        tool_map = temp_manager.tool_map
        print(f"Successfully loaded {len(tool_map)} tools.")
        print(f"Tools List: {list(tool_map.keys())}")
    except Exception as e:
        print(f"\nCRITICAL ERROR during initialization: {e}")
        print("Tip: If 'sqlite' or 'filesystem' failed, ensure 'npm' and 'npx' are installed and in your PATH.")
        print("Tip: If 'date' or 'echo' failed, ensure 'mcpuniverse' is in your PYTHONPATH.")
        if hasattr(temp_manager, 'stop'):
            temp_manager.stop()
        return
    finally:
        # 停止临时 manager，后面 Env 会启动一个新的
        temp_manager.stop()

    if not tool_map:
        print("No tools loaded. Exiting.")
        return

    # === 5. 初始化 LLM 和 Engine ===
    model_name = "/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/base_models/Qwen3-8B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Warning: Could not load tokenizer from {model_name}, using gpt2. Error: {e}")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    system_prompt = (
        "You are a helpful assistant integrated with multiple tools."
        "Use the appropriate tool to solve the user's task.\n"
    )

    engine = AgentExecutionEngine(
        agent_class=MCPToolAgent,
        env_class=MultiMCPEnvironment,
        agent_args={
            "parser_name": "qwen", 
            "system_prompt": system_prompt,
            "tool_map": tool_map
        },
        env_args={
            "server_configs": server_configs,
            "max_steps": 6,
            "reward_fn": None
        },
        engine_name="openai",
        # 确保这里的 URL 和 Key 是正确的
        rollout_engine_args={"base_url": "http://localhost:8803/v1", "api_key": "EMPTY"},
        tokenizer=tokenizer,
        n_parallel_agents=1,
        max_prompt_length=4096
    )

    # === 6. 定义任务 (已更新) ===
    tasks = [
        # 验证 Date Server
        {
            "id": 1,
            "instruction": "What is the current date?",
            "question": "What is the current date?"
        },
        # 验证 Calculator
        {
            "id": 2,
            "instruction": "Calculate (123 + 456) * 2.",
            "question": "Calculate (123 + 456) * 2."
        },
        # 验证 Echo Server
        {
            "id": 3,
            "instruction": "Echo the phrase 'MultiMCP is working'.",
            "question": "Echo the phrase 'MultiMCP is working'."
        },
        # 验证 Weather Server
        {
            "id": 4,
            "instruction": "What is the weather in London today?",
            "question": "What is the weather in London today?"
        }
    ]

    # === 7. 执行 ===
    print("\n[Step 2] Executing tasks...")
    try:
        results = await engine.execute_tasks(tasks)

        save_dir = "./trajectories/multi_mcp"
        os.makedirs(save_dir, exist_ok=True)
        save_clean_trajectories(results, os.path.join(save_dir, "trajectories.json"))
        print("Done.")
        
                        
    finally:
        print("\nCleaning up resources...")
        MultiMCPEnvironment.cleanup_global_resources()

if __name__ == "__main__":
    asyncio.run(main())
