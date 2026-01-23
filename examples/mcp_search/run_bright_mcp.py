import asyncio
import os
import sys

# from prepare_hotpotqa_data import prepare_hotpotqa_data
from transformers import AutoTokenizer

from rllm.agents.system_prompts import SEARCH_SYSTEM_PROMPT
from rllm.agents.tool_agent import MCPToolAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.tools.mcp_env import MCPConnectionManager, MCPEnvironment
from rllm.rewards.reward_fn import search_reward_fn
from rllm.utils import save_trajectories
from rllm.utils.compute_pass_at_k import save_clean_trajectories

# bright data API: da9e7e42-730d-4fb7-8357-b3dafcd7cc93
async def main():
    # 1. 修改参数检查逻辑
    if len(sys.argv) < 2:
        print("Usage: python run_tool_mcp.py <bright_data_api_token>")
        print("This will run HotpotQA evaluation using Bright Data MCP server")
        sys.exit(1)

    # 2. 获取 Bright Data API Token
    api_token = sys.argv[1]
    
    # 设置 HuggingFace 相关环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # 建议：指定一个明确的缓存目录，方便观察
    CACHE_DIR = "./search_cache_data"
    
    # 确保目录存在（虽然 env 代码里也会创建，但这里创建可以提前告知用户）
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        print(f"📁 缓存目录已创建: {os.path.abspath(CACHE_DIR)}")
    else:
        print(f"📁 使用已有缓存目录: {os.path.abspath(CACHE_DIR)}")

    n_parallel_agents = 1
    model_name = "/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/base_models/Qwen3-32B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. 配置 Bright Data MCP Server
    mcp_server_command = "npx"
    # ⚠️ 注意：务必添加 "-y" 参数
    mcp_server_args = ["-y", "@brightdata/mcp"]
    
    # 对应 JSON 中的 "env"
    mcp_server_env = {
        "API_TOKEN": api_token,
        "GROUPS": "advanced_scraping",
        "WEB_UNLOCKER_ZONE": "web_unlocker_zkn",
        # ⚠️ 关键：必须继承系统的 PATH
        "PATH": os.environ.get("PATH", "")
    }

    print("正在启动 Bright Data MCP Server 以获取工具列表...")
    
    # 临时启动一次以获取工具列表
    # 注意：这里不需要传 cache_dir，因为我们只是想快速拿个定义，不需要加载缓存文件
    temp_manager = MCPConnectionManager(mcp_server_command, mcp_server_args, mcp_server_env, cache_dir="./test_cache", allowed_tools=[])
    temp_manager.start()
    try:
        mcp_tool_map = temp_manager.tool_map
        print(f"✅ 成功获取工具列表: {list(mcp_tool_map.keys())}")
    finally:
        temp_manager.stop()

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    # 4. 初始化引擎
    engine = AgentExecutionEngine(
        agent_class=MCPToolAgent,
        env_class=MCPEnvironment,
        agent_args={
            "parser_name": "qwen", 
            "system_prompt": SEARCH_SYSTEM_PROMPT, 
            "tool_map": mcp_tool_map
        },
        env_args={
            "mcp_server_command": mcp_server_command,
            "mcp_server_args": mcp_server_args,
            "mcp_server_env": mcp_server_env,
            "reward_fn": search_reward_fn,
            "cache_dir": CACHE_DIR
        },
        engine_name="openai",
        # 请确保本地 vLLM 或其他推理服务已在 30000 端口启动
        rollout_engine_args={"base_url": "http://localhost:8803/v1", "api_key": "None", "model_name": "Qwen3-32B"},
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_response_length=32768,
        max_prompt_length=32768,
        n_parallel_agents=n_parallel_agents,
    )

    test_dataset = DatasetRegistry.load_dataset("hotpotqa", "test")
    # if test_dataset is None:
    #     print("Dataset not found, preparing dataset...")
    #     _, test_dataset = prepare_hotpotqa_data()

    tasks = test_dataset.get_data()
    print(f"Running evaluation on {len(tasks)} HotpotQA tasks...")
    
    # 取两个任务来验证效果
    tasks = tasks[300:305]

    try:
        print("🚀 开始执行任务...")
        print(f"👀 你可以另外打开一个终端，执行 'ls -l {CACHE_DIR}' 观察缓存文件大小的变化")
        
        results = await engine.execute_tasks(tasks)
        save_clean_trajectories(results, "./trajectories/mcp_search.json")
        print("\n✅ 任务完成，轨迹已保存。")
        
    finally:
        # MCPEnvironment.cleanup_global_resources()
        pass


if __name__ == "__main__":
    asyncio.run(main())
