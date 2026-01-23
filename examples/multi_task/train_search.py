import hydra
import logging
import requests
import random
import os
import json
import sys
from typing import List, Dict, Any

# HF Dataset
from datasets import load_dataset

# RLLM 基础组件
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.rewards.reward_fn import math_reward_fn, code_reward_fn, search_reward_fn
from rllm.agents.composite_agent import CompositeAgent
from rllm.environments.composite.composite_env import CompositeEnvironment
from rllm.agents.system_prompts import MATH_SYSTEM_PROMPT, SEARCH_SYSTEM_PROMPT
from rllm.data.utils import create_standard_sample, fetch_bfcl_tasks, load_comprehensive_math_test, load_and_tag_dataset

# [修改] 导入修改后的 MCP 组件 (假设在 mcp_env.py 中)
from rllm.environments.tools.mcp_env import MCPConnectionManager

random.seed(42)
logger = logging.getLogger(__name__)

def load_search_data(split: str, sample_num: int) -> List[Dict]:
    """
    加载 HotpotQA 数据 (用于 Search 任务)
    HotpotQA 在 HF 上通常叫 'hotpot_qa'，配置为 'distractor'
    """
    logger.info(f"Loading HotpotQA data ({split})...")
    try:
        hf_split = "train" if split == "train" else "validation"
        
        ds = load_dataset("hotpot_qa", "distractor", split=hf_split)
        
        raw_list = list(ds)
        if sample_num > 0 and sample_num < len(raw_list):
            random.shuffle(raw_list)
            raw_list = raw_list[:sample_num]
            
        processed_data = []
        for item in raw_list:
            d = dict(item)
            prompt_text = d.get("question", "")
            response_text = d.get("answer", "")
            
            d["task_type"] = "search"
            
            clean_d = create_standard_sample(
                prompt=prompt_text,
                response=response_text,
                task_type="search",
                raw_data=d
            )
            processed_data.append(clean_d)
            
        logger.info(f"Loaded {len(processed_data)} Search tasks from HotpotQA.")
        return processed_data
    except Exception as e:
        logger.error(f"Failed to load HotpotQA: {e}")
        return []

# --- 数据准备主逻辑 ---

def prepare_quad_mixed_dataset(
    bfcl_url: str, 
    dataset_name: str, 
    math_num: int, 
    code_num: int, 
    bfcl_num: int,
    search_num: int
):
    logger.info(">>> Start preparing Quad-Mixed Dataset (BFCL + Math + Code + Search)...")
    
    bfcl_train = fetch_bfcl_tasks(bfcl_url, "train") if bfcl_num > 0 else []
    bfcl_test = fetch_bfcl_tasks(bfcl_url, "val") if bfcl_num > 0 else []
    
    math_train = load_and_tag_dataset("deepscaler_math", "train", "math")[:math_num] if math_num > 0 else []
    math_test = load_comprehensive_math_test() if math_num > 0 else []
    
    code_train = load_and_tag_dataset("deepcoder", "train", "code")[:code_num] if code_num > 0 else []
    code_test = load_and_tag_dataset("deepcoder", "test", "code") if code_num > 0 else []

    if search_num > 0:
        search_train = load_search_data("train", search_num)
        search_test = load_search_data("test", 500)
    else:
        search_train = []
        search_test = []

    mixed_train = bfcl_train + math_train + code_train + search_train
    mixed_test = bfcl_test + math_test + code_test + search_test
    
    if not mixed_train:
        logger.warning("No training data found!")
    else:
        random.shuffle(mixed_train)
    
    logger.info(f"Prepared Data Details:")
    logger.info(f"  Train: BFCL={len(bfcl_train)}, Math={len(math_train)}, Code={len(code_train)}, Search={len(search_train)} | Total={len(mixed_train)}")
    logger.info(f"  Test : BFCL={len(bfcl_test)}, Math={len(math_test)}, Code={len(code_test)}, Search={len(search_test)} | Total={len(mixed_test)}")
    
    DatasetRegistry.register_dataset(dataset_name, mixed_train, split="train")
    DatasetRegistry.register_dataset(dataset_name, mixed_test, split="test")
    
    return dataset_name

# --- 主训练函数 ---

@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    # 获取 BFCL URL
    bfcl_url = config.get("bfcl_url", os.getenv("BFCL_URL", "http://localhost:8801"))
    
    bright_data_token = os.getenv("BRIGHT_DATA_API_TOKEN")
    if not bright_data_token:
        logger.warning("⚠️ No Bright Data API Token found. Search tasks might fail if configured.")
        assert False
    
    # --- MCP 配置区域 ---
    mcp_server_command = "npx"
    mcp_server_args = ["-y", "@brightdata/mcp"]
    mcp_server_env = {
        "API_TOKEN": bright_data_token or "",
        "GROUPS": "advanced_scraping",
        "PATH": os.environ.get("PATH", ""),
        "PRO_MODE": "true",
        "TOOLS": "extract"
    }
    search_cache_dir = "./search_cache_data"
    os.makedirs(search_cache_dir, exist_ok=True)

    # [新增] 定义 Agent 允许使用的工具列表
    # 只需要搜索和抓取 Markdown，不需要 HTML 或其他无关工具
    allowed_mcp_tools = ["search_engine", "scrape_as_markdown", "search_engine_batch", "scrape_batch"]

    # [修改] 预先连接 MCP Server 以获取工具定义 (tool_map)
    # 传入 allowed_tools 确保 LLM 看到的 System Prompt 只包含这些工具
    mcp_tool_map = {}
    if config.data.search_num > 0 and bright_data_token:
        logger.info(f"Initializing MCP Connection to fetch tools (Filtered by: {allowed_mcp_tools})...")
        try:
            # 这里的 allowed_tools 会传递给 MCPConnectionManager，它会过滤工具列表
            temp_manager = MCPConnectionManager(
                mcp_server_command, 
                mcp_server_args, 
                mcp_server_env,
                allowed_tools=allowed_mcp_tools
            )
            temp_manager.start()
            mcp_tool_map = temp_manager.tool_map
            temp_manager.stop()
            logger.info(f"✅ Fetched {len(mcp_tool_map)} tools from Bright Data MCP.")
            logger.info(f"   Available tools: {list(mcp_tool_map.keys())}")
        except Exception as e:
            logger.error(f"❌ Failed to fetch MCP tools: {e}")

    # 1. 准备数据
    dataset_name = prepare_quad_mixed_dataset(
        bfcl_url,
        config.data.dataset_name, 
        config.data.math_num, 
        config.data.code_num, 
        config.data.bfcl_num,
        config.data.search_num
    )
    
    train_dataset = DatasetRegistry.load_dataset(dataset_name, "train")
    test_dataset = DatasetRegistry.load_dataset(dataset_name, "test")

    # 2. 构造 Environment 参数
    composite_env_args = {
        "bfcl_args": {
            "base_url": bfcl_url,
            "env_type": "bfcl",
            "max_steps": config.rllm.agent.max_steps,
        },
        "math_args": {
            "tools": ["python"],
            "reward_fn": math_reward_fn,
        },
        "code_args": {
            "reward_fn": code_reward_fn
        },

        # [修改] Search Environment Args 中加入 allowed_tools
        "search_args": {
            "mcp_server_command": mcp_server_command,
            "mcp_server_args": mcp_server_args,
            "mcp_server_env": mcp_server_env,
            "reward_fn": search_reward_fn,
            "cache_dir": search_cache_dir,
            "allowed_tools": allowed_mcp_tools  # <--- 关键：确保环境运行时也应用该限制
        }
    }

    # 3. 构造 Agent 参数
    composite_agent_args = {
        "bfcl_agent_args": {
            "parser_name": "qwen",
            "system_prompt": "You are a helpful assistant with access to tools. Use the provided tools to fulfill the user request.",
        },
        "math_agent_args": {
            "tools": ["python"], 
            "parser_name": "qwen", 
            "system_prompt": MATH_SYSTEM_PROMPT
        },
        "code_agent_args": {
            "accumulate_thinking": True,
        },
        # Search Agent Args
        "search_agent_args": {
            "parser_name": "qwen",
            "system_prompt": SEARCH_SYSTEM_PROMPT,
            "tool_map": mcp_tool_map  # 这里传入过滤后的 tool_map
        }
    }

    # 4. 初始化 Trainer
    trainer = AgentTrainer(
        agent_class=CompositeAgent,
        env_class=CompositeEnvironment,
        agent_args=composite_agent_args,
        env_args=composite_env_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    
    logger.info(">>> Starting Multi-Task Training (BFCL + Math + Code + Search)...")
    trainer.train()

if __name__ == "__main__":
    main()
