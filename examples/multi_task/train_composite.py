import hydra
import logging
import requests
import random
import os
import json
import sys
from typing import List, Dict, Any
import numpy as np

# HF Dataset
from datasets import load_dataset

# RLLM 基础组件
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer
# 引入所有任务的 Reward Function
from rllm.rewards.reward_fn import math_reward_fn, code_reward_fn, search_reward_fn, tool_call_reward_fn, webshop_reward_fn
from rllm.agents.composite_agent import CompositeAgent
from rllm.environments.composite.composite_env import CompositeEnvironment

from rllm.data.utils import (
    create_standard_sample,
    fetch_bfcl_tasks,
    load_comprehensive_math_test,
    load_and_tag_dataset,
    load_dapo_math_dataset,
    load_deepmath_dataset,
    load_deepmath_dataset_top_k,
    load_search_data,
    load_local_search_data,  # [新增] Local Search 数据加载
    load_tool_call_dataset,
    load_tool_call_json_dataset,
    load_webshop_data,  # [新增] Webshop 数据加载
)

# 引入 System Prompts
from rllm.agents.system_prompts import MATH_SYSTEM_PROMPT, SEARCH_SYSTEM_PROMPT, LOCAL_SEARCH_SYSTEM_PROMPT
from rllm.agents.webshop_agent import WEBSHOP_SYSTEM_PROMPT  # [新增] Webshop System Prompt


# 引入 MCP 组件 (用于 Search/Browsing)
# 假设该模块在项目中存在，如果不存在需确保路径正确
try:
    from rllm.environments.tools.mcp_env import MCPConnectionManager
except ImportError:
    MCPConnectionManager = None


# ============================================================
# 全局随机种子设置 - 确保训练和测试数据的完全可重复性
# ============================================================
GLOBAL_SEED = 42

def set_all_random_seeds(seed: int = GLOBAL_SEED):
    """
    设置所有相关库的随机种子，确保可重复性。
    包括：Python random, NumPy, PyTorch, HuggingFace datasets
    """
    # Python 内置 random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch (延迟导入，避免不必要的依赖)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况
        # 确保 CUDA 操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # 设置 Python 哈希种子
    os.environ['PYTHONHASHSEED'] = str(seed)

# 在模块加载时立即设置种子
set_all_random_seeds(GLOBAL_SEED)

logger = logging.getLogger(__name__)


import hashlib

# --- 数据准备主逻辑 (支持 4 种任务) ---
def compute_dataset_hash(dataset: List[Dict]) -> str:
    """
    计算数据集的哈希值，用于验证数据集的一致性。
    
    Args:
        dataset: 数据集列表
        
    Returns:
        数据集的 SHA256 哈希值
    """
    # 将数据集序列化为 JSON 字符串（排序键以确保一致性）
    content = json.dumps(dataset, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def prepare_composite_dataset(
    bfcl_url: str,
    dataset_name: str,
    math_num: int,
    code_num: int,
    bfcl_num: int,
    search_num: int,
    tool_call_num: int,
    tool_call_data_path: str,
    local_search_num: int = 0,  # [新增] Local Search 数量参数
    webshop_num: int = 0  # [新增] Webshop 数量参数
):
    logger.info(">>> Start preparing Composite Dataset (BFCL + Math + Code + Search + LocalSearch + Webshop)...")
    logger.info(f">>> Using Random Seed: {GLOBAL_SEED}")
    
    # 1. BFCL
    bfcl_train = fetch_bfcl_tasks(bfcl_url, "train") if bfcl_num > 0 else []
    bfcl_test = fetch_bfcl_tasks(bfcl_url, "val") if bfcl_num > 0 else []
    
    # 2. Math
    if math_num > 0:
        # 假设 deepscaler_math 已在 Registry 中，或者按需替换为 load_dataset
        # math_train = load_dapo_math_dataset(math_num)
        # math_train = load_deepmath_dataset(math_num)
        math_train = load_deepmath_dataset_top_k(math_num) # 直接取难度最高的math_num个数据
        math_test = load_comprehensive_math_test()
    else:
        math_train, math_test = [], []

    # 3. Code
    if code_num > 0:
        code_train = load_and_tag_dataset("deepcoder", "train", "code")
        if code_train and len(code_train) > code_num:
            code_train = code_train[:code_num]
        code_test = load_and_tag_dataset("deepcoder", "test", "code")
    else:
        code_train, code_test = [], []

    # 4. Search
    if search_num > 0:
        search_train = load_search_data("train", search_num)
        search_test = load_search_data("test", 500) # 这里的 500 可配置化
    else:
        search_train, search_test = [], []

    # 5. Tool Call
    if tool_call_num > 0 and tool_call_data_path:
        # 分别加载 train 和 test parquet
        tc_train = load_tool_call_json_dataset(tool_call_data_path, split="train", num_samples=tool_call_num)
        # 测试集通常不需要采样太多，这里取全部或限制数量
        tc_test = load_tool_call_json_dataset(tool_call_data_path, split="test", num_samples=2000)
    else:
        tc_train, tc_test = [], []

    # 6. Local Search [新增]
    if local_search_num > 0:
        local_search_train = load_local_search_data("train", local_search_num)
        local_search_test = load_local_search_data("test", 500)  # 测试集数量可配置化
    else:
        local_search_train, local_search_test = [], []

    # 7. Webshop [新增]
    if webshop_num > 0:
        webshop_train = load_webshop_data("train", webshop_num)
        # 测试集数量：默认 500，会根据实际可用的 goals 数量自动调整
        webshop_test = load_webshop_data("test", 500)
    else:
        webshop_train, webshop_test = [], []

    # 混合
    mixed_train = bfcl_train + math_train + code_train + search_train + tc_train + local_search_train + webshop_train
    mixed_test = bfcl_test + math_test + code_test + search_test + tc_test + local_search_test + webshop_test
    
    if not mixed_train:
        logger.error("No training data found! Please check data configuration.")
        # 这里可以选择抛出异常，或者让 Trainer 去处理空数据
    else:
        random.shuffle(mixed_train)
    
    # 统计日志
    logger.info(f"Prepared Data Details:")
    logger.info(f"  Train: BFCL={len(bfcl_train)}, Math={len(math_train)}, Code={len(code_train)}, Search={len(search_train)}, Tool_Call={len(tc_train)}, Local_Search={len(local_search_train)}, Webshop={len(webshop_train)} | Total={len(mixed_train)}")
    logger.info(f"  Test : BFCL={len(bfcl_test)}, Math={len(math_test)}, Code={len(code_test)}, Search={len(search_test)}, Tool_Call={len(tc_test)}, Local_Search={len(local_search_test)}, Webshop={len(webshop_test)} | Total={len(mixed_test)}")
    
    # 注册数据集
    DatasetRegistry.register_dataset(dataset_name, mixed_train, split="train")
    DatasetRegistry.register_dataset(dataset_name, mixed_test, split="test")
    
    return dataset_name


# --- 主训练函数 ---

@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    # 配置读取
    bfcl_url = config.get("bfcl_url", os.getenv("BFCL_URL", "http://localhost:8801"))
    
    # 读取各任务数量配置，兼容旧 config
    math_num = config.data.get("math_num", 0)
    code_num = config.data.get("code_num", 0)
    bfcl_num = config.data.get("bfcl_num", 0)
    search_num = config.data.get("search_num", 0)
    tool_call_num = config.data.get("tool_call_num", 0)
    tool_call_data_path = config.data.get("tool_call_data_path", "./data/toolace")
    local_search_num = config.data.get("local_search_num", 0)  # [新增] Local Search 数量
    webshop_num = config.data.get("webshop_num", 0)  # [新增] Webshop 数量
    webshop_path = config.data.get("webshop_path", None)  # [新增] Webshop 环境路径

    # --- Search / MCP 环境配置 ---
    mcp_tool_map = {}
    bright_data_token = os.getenv("BRIGHT_DATA_API_TOKEN")
    
    # MCP Server 设置
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
    
    # 限制 Agent 可见的工具，避免 Context 污染
    allowed_mcp_tools = ["search_engine", "scrape_as_markdown", "search_engine_batch", "scrape_batch"]

    # 如果启用了 Search 任务，尝试初始化 MCP 并获取工具定义
    if search_num > 0:
        if not bright_data_token:
            logger.error("⚠️ Search task enabled but BRIGHT_DATA_API_TOKEN not found! Search steps may fail.")
        elif MCPConnectionManager is None:
             logger.error("⚠️ MCPConnectionManager import failed. Cannot initialize MCP tools.")
        else:
            logger.info(f"Initializing MCP Connection to fetch tools (Filtered by: {allowed_mcp_tools})...")
            try:
                # 临时启动一个 Manager 来获取工具 Schema
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

    # --- Local Search 环境配置 [新增] ---
    local_search_tool_map = {}
    retrieval_server_url = os.getenv("RETRIEVAL_SERVER_URL", "http://127.0.0.1:8000")
    
    if local_search_num > 0:
        logger.info(f"Initializing Local Search Tool (Server: {retrieval_server_url})...")
        try:
            # 从 rllm.tools 导入 LocalRetrievalTool
            from rllm.tools import LocalRetrievalTool
            local_search_tool_map = {"local_search": LocalRetrievalTool}
            logger.info("✅ Local Search Tool initialized successfully.")
        except ImportError as e:
            logger.error(f"❌ Failed to import LocalRetrievalTool: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Local Search Tool: {e}")

    # 1. 准备数据
    dataset_name = prepare_composite_dataset(
        bfcl_url,
        config.data.dataset_name,
        math_num,
        code_num,
        bfcl_num,
        search_num,
        tool_call_num,
        tool_call_data_path,
        local_search_num,
        webshop_num  # [新增]
    )
    
    train_dataset = DatasetRegistry.load_dataset(dataset_name, "train")
    test_dataset = DatasetRegistry.load_dataset(dataset_name, "test")

    # 2. 构造 Environment 参数 (支持所有任务)
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
        "search_args": {
            "mcp_server_command": mcp_server_command,
            "mcp_server_args": mcp_server_args,
            "mcp_server_env": mcp_server_env,
            "reward_fn": search_reward_fn,
            "cache_dir": search_cache_dir,
            "allowed_tools": allowed_mcp_tools  # 运行时过滤
        },
        "tool_call_args": {
            "reward_fn": tool_call_reward_fn
        },
        # [新增] Local Search 环境参数
        "local_search_args": {
            "tool_map": local_search_tool_map,
            "reward_fn": search_reward_fn,  # 复用 search_reward_fn
            "max_steps": config.rllm.agent.get("max_steps", 20),
        },
        # [新增] Webshop 环境参数
        # 使用 1000 产品的小数据集和合成 goals（synthetic goals）
        "webshop_args": {
            "reward_fn": webshop_reward_fn,
            "max_steps": config.rllm.agent.get("max_steps", 15),
            "webshop_path": webshop_path,
            "observation_mode": "text",
            "num_products": 1000,  # 使用 1000 产品的小数据集
            "human_goals": False,  # 使用合成 goals（synthetic goals）
        }
    }

    # 3. 构造 Agent 参数 (支持所有任务)
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
        "search_agent_args": {
            "parser_name": "qwen",
            "system_prompt": SEARCH_SYSTEM_PROMPT,
            "tool_map": mcp_tool_map  # 传入预获取的工具 Schema
        },
        "tool_call_agent_args": {
            "parser_name": "qwen"
        },
        # [新增] Local Search Agent 参数
        "local_search_agent_args": {
            "parser_name": "qwen",
            "system_prompt": LOCAL_SEARCH_SYSTEM_PROMPT,
            "tool_map": local_search_tool_map
        },
        # [新增] Webshop Agent 参数
        "webshop_agent_args": {
            "system_prompt": WEBSHOP_SYSTEM_PROMPT,
        }
    }

    # 4. 预初始化共享环境（优化：避免每个 batch 都重新加载 Webshop 数据）
    if webshop_num > 0:
        logger.info(">>> Pre-initializing shared Webshop environment...")
        CompositeEnvironment.pre_initialize_shared_envs(composite_env_args)
        logger.info(">>> Shared Webshop environment initialized successfully.")

    # 5. 初始化 Trainer
    trainer = AgentTrainer(
        agent_class=CompositeAgent,
        env_class=CompositeEnvironment,
        agent_args=composite_agent_args,
        env_args=composite_env_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    
    logger.info(">>> Starting Composite Training...")
    try:
        trainer.train()
    finally:
        # 清理共享环境
        if webshop_num > 0:
            logger.info(">>> Cleaning up shared environments...")
            CompositeEnvironment.cleanup_shared_envs()

if __name__ == "__main__":
    main()
