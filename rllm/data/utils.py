"""Utility functions for loading and processing datasets."""

import hydra
import logging
import requests
import random
import os
import json
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

from rllm.data.dataset_types import TestDataset, TrainDataset
from rllm.data.dataset import DatasetRegistry
from rllm.system_prompts import LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE, LCB_FORMATTING_WITHOUT_STARTER_CODE, LCB_SYSTEM_MESSAGE_GENERIC


# 设置全局随机种子，确保数据加载的可重复性
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)

# 注意：torch 的种子设置在 train_composite.py 中完成，因为这里不一定导入 torch

# ============================================================
# 全局任务配置管理器（延迟初始化）
# ============================================================
_task_config_manager = None


def get_task_config_manager():
    """获取全局任务配置管理器（延迟初始化）"""
    global _task_config_manager
    if _task_config_manager is None:
        from rllm.config.task_config import TaskConfigManager
        _task_config_manager = TaskConfigManager()
    return _task_config_manager


def set_task_config_manager(manager):
    """设置全局任务配置管理器"""
    global _task_config_manager
    _task_config_manager = manager


def create_standard_sample(
    prompt: str,
    response: str,
    task_type: str,
    raw_data: Dict,
    inject_task_config: bool = True
) -> Dict[str, str]:
    """
    强制统一数据格式，防止 PyArrow Schema 报错。
    所有字段强制转为字符串，复杂对象存入 extra_info。
    
    【新增功能】自动注入任务级别的配置参数：
    - task_max_prompt_length: 该任务类型的最大 prompt 长度
    - task_max_response_length: 该任务类型的最大 response 长度
    - task_max_steps: 该任务类型的最大交互步数
    
    这些参数会被存入 extra_info，在执行引擎中动态读取使用。
    
    Args:
        prompt: 提示文本
        response: 响应文本
        task_type: 任务类型 (math, code, search, tool_call, local_search, webshop, bfcl)
        raw_data: 原始数据字典
        inject_task_config: 是否注入任务配置（默认 True）
        
    Returns:
        标准化的数据样本字典
    """
    safe_prompt = str(prompt) if prompt is not None else ""
    safe_response = str(response) if response is not None else ""
    
    # 确保 raw_data 包含 task_type
    if "task_type" not in raw_data:
        raw_data["task_type"] = task_type
    
    # 注入任务级别的配置参数
    if inject_task_config:
        try:
            manager = get_task_config_manager()
            config = manager.get_config(task_type)
            raw_data["task_max_prompt_length"] = config.max_prompt_length
            raw_data["task_max_response_length"] = config.max_response_length
            raw_data["task_max_steps"] = config.max_steps
        except Exception as e:
            # 如果配置获取失败，使用默认值
            logging.getLogger(__name__).debug(f"Failed to get task config for {task_type}: {e}")
            pass
    
    return {
        "prompt": safe_prompt,
        "response": safe_response,
        "task_type": str(task_type),
        "data_source": str(task_type),
        "extra_info": json.dumps(raw_data, ensure_ascii=False),
    }

# def load_dataset(dataset_enum: TrainDataset.Math | TrainDataset.Code | TestDataset.Math | TestDataset.Code) -> list[dict[str, Any]]:
#     """Load a dataset from a JSON file based on the dataset enum.

#     This function takes a dataset enum value and loads the corresponding JSON file
#     from the appropriate directory structure. The directory structure follows the pattern:
#     {data_dir}/{category_dir}/{dataset_name}.json
#     where:
#     - data_dir is either 'train' or 'test'
#     - category_dir is either 'math' or 'code'
#     - dataset_name is the lowercase value of the enum

#     Args:
#         dataset_enum: An enum value from either TrainDataset or TestDataset classes,
#                      specifying which dataset to load.

#     Returns:
#         List[Dict[str, Any]]: A list of dictionaries containing the dataset items.
#             Each dictionary represents one item in the dataset with its associated fields.

#     Raises:
#         ValueError: If the dataset file cannot be found or contains invalid JSON.

#     Examples:
#         >>> # Load training AIME dataset
#         >>> aime_data = load_dataset(TrainDataset.Math.AIME)
#         >>> # Load test APPS dataset
#         >>> apps_data = load_dataset(TestDataset.Code.APPS)
#     """
#     dataset_name = dataset_enum.value.lower()
#     category_dir = dataset_enum.__class__.__name__.lower()

#     # Determine if dataset is for training or testing
#     if dataset_enum.__class__ in [TrainDataset.Math, TrainDataset.Code, TrainDataset.Web]:
#         data_dir = "train"
#     else:
#         data_dir = "test"

#     # Construct file path
#     current_dir = os.path.dirname(os.path.realpath(__file__))

#     file_path = os.path.join(current_dir, data_dir, category_dir, f"{dataset_name}.json")

#     if not os.path.exists(file_path):
#         raise ValueError(f"Dataset file not found: {file_path}")

#     try:
#         with open(file_path, encoding="utf-8") as f:
#             data = json.load(f)
#         return data
#     except json.JSONDecodeError:
#         raise ValueError(f"Invalid JSON format in {file_path}") from None
#     except Exception as e:
#         raise ValueError(f"Error loading dataset: {str(e)}") from e

def fetch_live_code_bench_system_prompt(prompt: str, starter_code: str | None = None):
    # https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
    prompt = LCB_SYSTEM_MESSAGE_GENERIC + "\n\n" + prompt
    if starter_code:
        prompt += f"### Format: {LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt

def fetch_bfcl_tasks(base_url: str, split: str = "train") -> List[Dict[str, Any]]:
    try:
        url = f"{base_url}/get_env_profile"
        logger.info(f"Fetching BFCL tasks from {url} ({split})...")
        resp = requests.post(url, json={
            "env_type": "bfcl", "params": {"split": split}
        }, timeout=10)
        
        if resp.status_code != 200:
            logger.warning(f"BFCL Server returned status {resp.status_code}")
            return []

        data = resp.json()
        raw_list = data.get("data", []) if isinstance(data, dict) else data
        
        tasks = []
        for item in raw_list:
            t_id = item if isinstance(item, str) else item.get("task_id")
            if t_id:
                raw_data = {"task_id": t_id, "task_type": "bfcl", "sub_source": "bfcl"}
                task = create_standard_sample(
                    prompt="",
                    response="",
                    task_type="bfcl",
                    raw_data=raw_data
                )
                tasks.append(task)
        return tasks
    except Exception as e:
        logger.warning(f"BFCL Server error: {e}. Returning empty list.")
        return []

def load_comprehensive_math_test() -> List[Dict]:
    """
    加载高难度数学测试集：
      - AIME 2024, AIME 2025
      - 筛选后的 AMO-Bench
      - Qwen/PolyMath 英文版: High & Top 难度
      - MathArena/hmmt_feb_2025 (New)
      - MathArena/hmmt_nov_2025 (New)
    """
    logger.info(">>> Loading Hard Math Test Set (AIME + AMO + PolyMath + HMMT)...")
    combined_samples = []

    # 1. 加载 AIME 系列
    # ---------------------------------------------------------
    aime_sources = [
        ("HuggingFaceH4/aime_2024", "train", "aime2024"), 
        ("yentinglin/aime_2025", "train", "aime2025"),
    ]

    for hf_path, split, source_name in aime_sources:
        try:
            logger.info(f"Fetching {hf_path} ({split})...")
            ds = load_dataset(hf_path, split=split)
            for item in ds:
                d = dict(item)
                prompt_text = d.get("problem") or d.get("question") or d.get("prompt") or ""
                response_text = str(d.get("answer") or d.get("ground_truth") or d.get("solution") or "")
                d["answer"] = response_text
                d["sub_source"] = source_name
                d["task_type"] = "math"
                if response_text:
                    combined_samples.append(create_standard_sample(prompt_text, response_text, "math", d))
        except Exception as e:
            logger.error(f"Failed to load {hf_path}: {e}")

    # 2. 加载并筛选 AMO-Bench
    # ---------------------------------------------------------
    # amo_path = "meituan-longcat/AMO-Bench"
    # allowed_types = {"number", "set", "variable"} 
    # try:
    #     logger.info(f"Fetching {amo_path}...")
    #     try:
    #         ds_dict = load_dataset(amo_path)
    #         ds_amo = ds_dict.get('train') or ds_dict.get('test') if hasattr(ds_dict, 'keys') else ds_dict
    #     except Exception:
    #         ds_amo = load_dataset(amo_path, split="train")

    #     for item in ds_amo:
    #         d = dict(item)
    #         if str(d.get("answer_type", "")).lower().strip() not in allowed_types: continue
            
    #         prompt_text = d.get("prompt", "")
    #         response_text = str(d.get("answer", "") or "")
    #         if not response_text: continue

    #         d["answer"] = response_text
    #         d["sub_source"] = "amo_bench"
    #         d["task_type"] = "math"
    #         combined_samples.append(create_standard_sample(prompt_text, response_text, "math", d))
    # except Exception as e:
    #     logger.error(f"Failed to load {amo_path}: {e}")

    # 3. 加载 Qwen/PolyMath
    # ---------------------------------------------------------
    polymath_targets = [("Top", "en/top.parquet"), ("High", "en/high.parquet")]
    for difficulty_name, file_pattern in polymath_targets:
        try:
            logger.info(f"Fetching Qwen/PolyMath (file='{file_pattern}')...")
            ds = load_dataset("Qwen/PolyMath", data_files=file_pattern, split="train")
            for item in ds:
                d = dict(item)
                prompt_text = d.get("problem") or d.get("question") or d.get("prompt") or d.get("content") or ""
                response_text = str(d.get("answer") or "")
                if not prompt_text or not response_text: continue
                d["answer"] = response_text
                d["sub_source"] = f"polymath_en_{difficulty_name.lower()}"
                d["task_type"] = "math"
                combined_samples.append(create_standard_sample(prompt_text, response_text, "math", d))
        except Exception as e:
            logger.error(f"Failed to load PolyMath {difficulty_name}: {e}")

    # 4. [New] 加载 MathArena HMMT 数据集
    # ---------------------------------------------------------
    hmmt_sources = [
        ("MathArena/hmmt_feb_2025", "train", "hmmt_feb_2025"),
        ("MathArena/hmmt_nov_2025", "train", "hmmt_nov_2025")
    ]
    
    for hf_path, split, source_name in hmmt_sources:
        try:
            logger.info(f"Fetching {hf_path} ({split})...")
            ds = load_dataset(hf_path, split=split)
            
            count = 0
            for item in ds:
                d = dict(item)
                
                # HMMT 核心字段: problem, answer
                prompt_text = d.get("problem", "")
                response_text = str(d.get("answer", "") or "")
                
                if not prompt_text:
                    continue
                
                d["answer"] = response_text
                d["sub_source"] = source_name
                d["task_type"] = "math"
                
                combined_samples.append(
                    create_standard_sample(
                        prompt=prompt_text,
                        response=response_text,
                        task_type="math",
                        raw_data=d
                    )
                )
                count += 1
            logger.info(f"Loaded {count} samples from {hf_path}.")
            
        except Exception as e:
            logger.error(f"Failed to load {hf_path}: {e}")

    logger.info(f"Total High-Difficulty Math Test Samples Loaded: {len(combined_samples)}")
    return combined_samples

def load_and_tag_dataset(dataset_name: str, split: str, tag: str) -> List[Dict]:
    try:
        logger.info(f"Loading dataset from Registry: {dataset_name} split: {split}")
        raw_data = DatasetRegistry.load_dataset(dataset_name, split)
        tagged_data = []
        
        for item in raw_data:
            d = dict(item) if not isinstance(item, dict) else item.copy()
            
            prompt_text = ""
            if "question" in d: prompt_text = d["question"]
            elif "problem" in d: prompt_text = d["problem"]
            elif "problem_description" in d: prompt_text = d["problem_description"]
            elif "prompt" in d: prompt_text = d["prompt"]
            
            response_text = ""
            if "answer" in d: response_text = d["answer"]
            elif "ground_truth" in d: response_text = str(d["ground_truth"])
            elif "response" in d: response_text = d["response"]
            elif "solution" in d: response_text = d["solution"]

            d["task_type"] = tag
            # 添加 sub_source 字段，使用 dataset_name 作为子数据源标识
            d["sub_source"] = dataset_name
            
            clean_d = create_standard_sample(
                prompt=prompt_text,
                response=response_text,
                task_type=tag,
                raw_data=d
            )
            tagged_data.append(clean_d)
            
        logger.info(f"Loaded {len(tagged_data)} tasks from {dataset_name} ({split}).")
        return tagged_data
    except Exception as e:
        logger.warning(f"Failed to load {dataset_name} ({split}): {e}.")
        return []

def load_search_data(split: str, sample_num: int) -> List[Dict]:
    """
    加载 HotpotQA 数据 (用于 Search 任务)
    
    【数据采样策略】
    - 测试集 (split="test"): 始终取前 sample_num 条，保证固定性和可复现性
    - 训练集 (split="train"): 使用固定种子随机采样，保证跨实验一致性
    """
    if sample_num <= 0:
        return []

    logger.info(f"Loading HotpotQA data ({split})...")
    try:
        hf_split = "train" if split == "train" else "validation"
        # 加载 hotpot_qa distractor 配置
        ds = load_dataset("hotpot_qa", "distractor", split=hf_split)
        
        raw_list = list(ds)
        # 如果需要截断
        if sample_num > 0 and sample_num < len(raw_list):
            if split == "test":
                # 测试集：始终取前 sample_num 条，保证固定性
                raw_list = raw_list[:sample_num]
            else:
                # 训练集：使用固定种子随机采样，保证可复现性
                rng = random.Random(GLOBAL_SEED)
                rng.shuffle(raw_list)
                raw_list = raw_list[:sample_num]
            
        processed_data = []
        for item in raw_list:
            d = dict(item)
            prompt_text = d.get("question", "")
            response_text = d.get("answer", "")
            
            d["task_type"] = "search"
            d["sub_source"] = "hotpotqa"  # 添加 sub_source 字段
            
            clean_d = create_standard_sample(
                prompt=prompt_text,
                response=response_text,
                task_type="search",
                raw_data=d
            )
            processed_data.append(clean_d)
            
        logger.info(f"Loaded {len(processed_data)} Search tasks from HotpotQA ({split}).")
        return processed_data
    except Exception as e:
        logger.error(f"Failed to load HotpotQA: {e}")
        return []


def load_local_search_data(split: str, sample_num: int) -> List[Dict]:
    """
    加载 HotpotQA 数据 (用于 Local Search 任务)
    与 load_search_data 类似，但 task_type 为 "local_search"
    
    【数据采样策略】
    - 测试集 (split="test"): 始终取前 sample_num 条，保证固定性和可复现性
    - 训练集 (split="train"): 使用固定种子随机采样，保证跨实验一致性
    """
    if sample_num <= 0:
        return []

    logger.info(f"Loading HotpotQA data for local_search ({split})...")
    try:
        hf_split = "train" if split == "train" else "validation"
        # 加载 hotpot_qa distractor 配置
        ds = load_dataset("hotpot_qa", "distractor", split=hf_split)
        
        raw_list = list(ds)
        # 如果需要截断
        if sample_num > 0 and sample_num < len(raw_list):
            if split == "test":
                # 测试集：始终取前 sample_num 条，保证固定性
                raw_list = raw_list[:sample_num]
            else:
                # 训练集：使用固定种子随机采样，保证可复现性
                rng = random.Random(GLOBAL_SEED)
                rng.shuffle(raw_list)
                raw_list = raw_list[:sample_num]
            
        processed_data = []
        for item in raw_list:
            d = dict(item)
            prompt_text = d.get("question", "")
            response_text = d.get("answer", "")
            
            d["task_type"] = "local_search"
            d["sub_source"] = "hotpotqa"  # 添加 sub_source 字段
            
            clean_d = create_standard_sample(
                prompt=prompt_text,
                response=response_text,
                task_type="local_search",
                raw_data=d
            )
            processed_data.append(clean_d)
            
        logger.info(f"Loaded {len(processed_data)} Local Search tasks from HotpotQA ({split}).")
        return processed_data
    except Exception as e:
        logger.error(f"Failed to load HotpotQA for local_search: {e}")
        return []

def load_dapo_math_dataset(num_samples: int) -> List[Dict]:
    """
    加载 open-r1/DAPO-Math-17k-Processed 数据集作为 Math 训练集
    字段映射: prompt -> prompt, solution -> response
    """
    dataset_path = "open-r1/DAPO-Math-17k-Processed"
    logger.info(f"Loading Math Train Data from HuggingFace: {dataset_path}...")
    
    try:
        # 直接从 HF 加载
        ds = load_dataset(dataset_path, split="train")
        
        # 转换为 list
        raw_list = list(ds)
        
        # 如果指定了数量限制，使用固定种子随机采样，保证可复现性
        if num_samples > 0 and len(raw_list) > num_samples:
            rng = random.Random(GLOBAL_SEED)
            rng.shuffle(raw_list)
            raw_list = raw_list[:num_samples]
            
        processed_data = []
        for item in raw_list:
            d = dict(item)
            
            # 提取 DAPO 数据集的特定字段
            prompt_text = d.get("prompt", "")
            response_text = d.get("solution", "") # 用户指定 solution 字段
            
            # 标记任务类型和子数据源
            d["task_type"] = "math"
            d["source"] = "dapo_math_17k"
            d["sub_source"] = "dapo_math_17k"  # 添加 sub_source 字段
            
            # 使用标准构建函数，保持逻辑一致
            clean_d = create_standard_sample(
                prompt=prompt_text,
                response=response_text,
                task_type="math",
                raw_data=d
            )
            processed_data.append(clean_d)
            
        logger.info(f"Loaded {len(processed_data)} tasks from {dataset_path}.")
        return processed_data
        
    except Exception as e:
        logger.error(f"Failed to load {dataset_path}: {e}")
        return []

def load_deepmath_dataset(num_samples: int) -> List[Dict]:
    """
    加载 zwhe99/DeepMath-103K 数据集作为 Math 训练集
    逻辑优化：
    1. 仅保留 difficulty >= 7.0 的数据。
    2. 在保留的数据中，按照各难度（difficulty）的分布比例进行分层抽样。
    3. 确保最终输出数量严格等于 num_samples（如果数据足够）。
    """
    dataset_path = "zwhe99/DeepMath-103K"
    logger.info(f"Loading DeepMath Data from HuggingFace: {dataset_path}...")

    # 辅助函数：解析难度
    def parse_difficulty(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    try:
        ds = load_dataset(dataset_path, split="train")
        raw_list = list(ds)
        total_raw = len(raw_list)

        # -------------------------------------------------------
        # 1. 原始分布统计 (保留原逻辑用于日志观察)
        # -------------------------------------------------------
        diff_counter = Counter()
        for it in raw_list:
            v = parse_difficulty(it.get("difficulty", None))
            key = "UNPARSED" if v is None else v 
            diff_counter[key] += 1

        logger.info("DeepMath RAW difficulty distribution:")
        numeric_keys = sorted([k for k in diff_counter.keys() if k != "UNPARSED"])
        keys_order = numeric_keys + (["UNPARSED"] if "UNPARSED" in diff_counter else [])
        for k in keys_order:
            c = diff_counter[k]
            ratio = (c / total_raw) if total_raw > 0 else 0.0
            logger.info(f"  difficulty={k}: {c} ({ratio:.2%})")

        # -------------------------------------------------------
        # 2. 筛选：只取 difficulty >= 6.0
        # -------------------------------------------------------
        min_difficulty = 7.0
        # 使用 defaultdict 按难度分组存储
        eligible_groups = defaultdict(list)
        total_eligible = 0

        for item in raw_list:
            diff = parse_difficulty(item.get("difficulty"))
            if diff is not None and diff >= min_difficulty:
                eligible_groups[diff].append(item)
                total_eligible += 1

        logger.info(f"Filtered (>= {min_difficulty}): Found {total_eligible} samples.")

        # -------------------------------------------------------
        # 3. 分层抽样计算 (Stratified Sampling)
        # -------------------------------------------------------
        selected_raw_list = []
        
        if total_eligible == 0:
            logger.warning("No samples found with difficulty >= 6.0")
            return []
        
        if num_samples >= total_eligible:
            # 如果请求数量大于等于符合条件的数量，直接返回所有符合条件的
            logger.info(f"Requested {num_samples} samples, but only {total_eligible} eligible. Returning all.")
            for items in eligible_groups.values():
                selected_raw_list.extend(items)
        else:
            # 需要抽样
            logger.info(f"Sampling {num_samples} from {total_eligible} eligible samples proportionally...")
            
            # 3.1 计算每组的配额 (使用最大余额法/Hamilton method 处理整数舍入)
            quotas = {}
            remainders = [] # (小数部分, difficulty)
            current_sum = 0
            
            # 对难度排序以保证确定性
            sorted_diffs = sorted(eligible_groups.keys())
            
            for diff in sorted_diffs:
                group_count = len(eligible_groups[diff])
                # 理论应抽数量 (float)
                exact_quota = (group_count / total_eligible) * num_samples
                # 整数部分
                base_quota = int(exact_quota)
                
                quotas[diff] = base_quota
                current_sum += base_quota
                
                # 记录小数部分用于分配剩余名额
                remainders.append((exact_quota - base_quota, diff))
            
            # 3.2 分配因舍去小数而缺失的名额
            shortage = num_samples - current_sum
            # 按小数部分从大到小排序
            remainders.sort(key=lambda x: x[0], reverse=True)
            
            for i in range(shortage):
                diff_to_bump = remainders[i][1]
                quotas[diff_to_bump] += 1
            
            # 3.3 执行固定种子随机抽取，保证可复现性
            rng = random.Random(GLOBAL_SEED)
            for diff, count in quotas.items():
                group_items = eligible_groups[diff]
                # 使用固定种子随机采样
                sampled = rng.sample(group_items, count)
                selected_raw_list.extend(sampled)
                logger.info(f"  difficulty={diff}: Total {len(group_items)} -> Sampled {count}")

        # -------------------------------------------------------
        # 4. 标准化处理
        # -------------------------------------------------------
        # 使用固定种子打乱最终列表，避免按难度排序
        rng = random.Random(GLOBAL_SEED)
        rng.shuffle(selected_raw_list)

        processed_data = []
        for item in selected_raw_list:
            d = dict(item)
            prompt_text = d.get("question", "")
            response_text = d.get("final_answer", "")

            d["task_type"] = "math"
            d["source"] = "deepmath_103k"
            d["sub_source"] = "deepmath_103k"  # 添加 sub_source 字段

            clean_d = create_standard_sample(
                prompt=prompt_text,
                response=response_text,
                task_type="math",
                raw_data=d
            )
            processed_data.append(clean_d)

        logger.info(f"Loaded {len(processed_data)} tasks from {dataset_path} (Stratified >= 7.0).")
        return processed_data

    except Exception as e:
        logger.error(f"Failed to load {dataset_path}: {e}", exc_info=True)
        return []

def load_deepmath_dataset_top_k(num_samples: int) -> List[Dict]:
    """
    加载 zwhe99/DeepMath-103K 数据集，并筛选难度最高的 num_samples 个样本。
    逻辑：
    1. 解析所有数据的难度。
    2. 按难度从大到小排序。
    3. 截取前 num_samples 个。
    4. 标准化输出并随机打乱。
    """
    dataset_path = "zwhe99/DeepMath-103K"
    logger.info(f"Loading Top-{num_samples} DeepMath Data from HuggingFace: {dataset_path}...")

    # 辅助函数：解析难度
    def parse_difficulty(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    try:
        ds = load_dataset(dataset_path, split="train")
        raw_list = list(ds)
        total_raw = len(raw_list)

        # -------------------------------------------------------
        # 1. 原始分布统计 (保留日志观察)
        # -------------------------------------------------------
        diff_counter = Counter()
        items_with_diff = []

        for it in raw_list:
            v = parse_difficulty(it.get("difficulty"))
            if v is not None:
                diff_counter[v] += 1
                items_with_diff.append((v, it)) # 存储 (难度, 数据内容) 元组
            else:
                diff_counter["UNPARSED"] += 1

        logger.info("DeepMath RAW difficulty distribution:")
        numeric_keys = sorted([k for k in diff_counter.keys() if isinstance(k, (int, float))], reverse=True)
        for k in numeric_keys:
            c = diff_counter[k]
            ratio = (c / total_raw) if total_raw > 0 else 0.0
            logger.info(f"  difficulty={k}: {c} ({ratio:.2%})")

        # -------------------------------------------------------
        # 2. 核心逻辑：按难度从高到低排序并取 Top-K
        # -------------------------------------------------------
        # 按难度降序排序
        items_with_diff.sort(key=lambda x: x[0], reverse=True)
        
        # 截取前 num_samples 个
        selected_pairs = items_with_diff[:num_samples]
        actual_count = len(selected_pairs)
        
        if actual_count > 0:
            min_diff_in_top = selected_pairs[-1][0]
            max_diff_in_top = selected_pairs[0][0]
            logger.info(f"Selected top {actual_count} samples. Difficulty range: [{min_diff_in_top} - {max_diff_in_top}]")
        else:
            logger.warning("No valid samples with difficulty found.")
            return []

        # -------------------------------------------------------
        # 3. 标准化处理
        # -------------------------------------------------------
        selected_raw_list = [pair[1] for pair in selected_pairs]
        
        # 使用固定种子打乱，避免模型在训练时按难度顺序学习导致梯度不稳定，同时保证可复现性
        rng = random.Random(GLOBAL_SEED)
        rng.shuffle(selected_raw_list)

        processed_data = []
        for item in selected_raw_list:
            d = dict(item)
            prompt_text = d.get("question", "")
            response_text = d.get("final_answer", "")

            # 补充元数据
            d["task_type"] = "math"
            d["source"] = "deepmath_103k_top_k"
            d["sub_source"] = "deepmath_103k_top_k"  # 添加 sub_source 字段

            clean_d = create_standard_sample(
                prompt=prompt_text,
                response=response_text,
                task_type="math",
                raw_data=d
            )
            processed_data.append(clean_d)

        logger.info(f"Successfully loaded {len(processed_data)} highest difficulty samples.")
        return processed_data

    except Exception as e:
        logger.error(f"Failed to load {dataset_path}: {e}", exc_info=True)
        return []
    
def load_tool_call_dataset(data_dir: str, split: str = "train", num_samples: int = 0) -> List[Dict]:
    """
    加载本地 Tool Call Parquet 数据 (train.parquet / test.parquet)。
    修复了 TypeError: Object of type ndarray is not JSON serializable 问题。
    """
    # 1. 确定文件路径
    target_file = data_dir
    if os.path.isdir(data_dir):
        if split == 'train':
            target_file = os.path.join(data_dir, f"{split}_overall.parquet")
        elif split == 'test':
            target_file = os.path.join(data_dir, f"{split}_specific.parquet")
        else:
            # Fallback
            target_file = os.path.join(data_dir, f"{split}.parquet")
            
    if not os.path.exists(target_file):
        logger.warning(f"Tool Call dataset file not found: {target_file}")
        return []

    logger.info(f"Loading Tool Call data from {target_file}...")
    
    try:
        # 2. 读取 Parquet
        df = pd.read_parquet(target_file)
        
        # 3. 截断与打乱
        if num_samples > 0 and num_samples < len(df):
            df = df.sample(n=num_samples, random_state=42)
        
        # 转为 list of dict
        raw_list = df.to_dict('records')
        
        # --- 定义一个辅助函数用于递归转换 NumPy 类型 ---
        def convert_numpy_to_python(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic): # 处理 numpy scalar (如 np.float32, np.int64)
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_python(i) for i in obj]
            return obj

        processed_data = []
        for item in raw_list:
            # item 中可能包含 numpy 对象，先进行清洗，或者在最后构建 raw_data 时清洗
            # 这里我们在解析前尽量不做额外操作，先尝试提取
            
            raw_extra = item.get("extra_info", {})
            if isinstance(raw_extra, str):
                try:
                    raw_extra = json.loads(raw_extra)
                except:
                    pass
            
            # 提取字段
            prompt_text = ""
            if isinstance(raw_extra, dict):
                prompt_text = raw_extra.get("input", "")
                response_text = raw_extra.get("output", "")
                system_instruction = raw_extra.get("instruction", "")
            else:
                response_text = ""
                system_instruction = ""

            # 兼容逻辑
            if not prompt_text and "prompt" in item:
                 prompt_text = item["prompt"]
            
            if not prompt_text or not response_text:
                continue

            # 4. 构建数据字典
            d = dict(item)
            d["input"] = prompt_text
            d["output"] = response_text
            d["instruction"] = system_instruction
            
            # [关键修复]：在传递给 create_standard_sample 之前，将所有 numpy 类型转为 python 原生类型
            # 这样 json.dumps 就不会报错了
            d_clean = convert_numpy_to_python(d)

            clean_d = create_standard_sample(
                prompt=prompt_text,
                response=response_text,
                task_type="tool_call",
                raw_data=d_clean  # 传入清洗后的字典
            )
            processed_data.append(clean_d)
            
        logger.info(f"Loaded {len(processed_data)} Tool Call tasks from {split}.")
        return processed_data

    except Exception as e:
        logger.error(f"Failed to load Tool Call data: {e}")
        import traceback
        traceback.print_exc()
        return []


def load_awm_dataset(
    dataset_path: str = "Snowflake/AgenticWorldModel",
    split: str = "train",
    num_scenarios: int = 0,
    tasks_per_scenario: int = 10,
    verification_mode: str = "pure_code"  # "pure_code" or "sql"
) -> List[Dict]:
    """
    加载 AWM (Agentic World Model) 数据集。
    
    支持两种数据源：
    1. 本地目录（如 "awm_data"）— 直接从 JSONL 文件加载
    2. HuggingFace 数据集路径（如 "Snowflake/AgenticWorldModel"）
    
    AWM 数据集包含以下文件:
    - gen_scenario.jsonl: 场景描述 (1000 scenarios)
      格式: {"name": "scenario_name", "description": "..."}
    - gen_tasks.jsonl: 用户任务 (每个 scenario 一条记录，包含 10 个 tasks)
      格式: {"scenario": "scenario_name", "tasks": ["task1", "task2", ...]}
    - gen_db.jsonl: 数据库 schema
      格式: {"scenario": "scenario_name", "db_schema": {...}, "db_path": "..."}
    - gen_sample.jsonl: 初始数据库数据
      格式: {"scenario": "scenario_name", "sample_data": {"tables": [...]}}
    - gen_spec.jsonl: API 规范
      格式: {"scenario": "scenario_name", "api_spec": {...}}
    - gen_envs.jsonl: FastAPI + MCP Server 代码
      格式: {"scenario": "scenario_name", "full_code": "..."}
    - gen_verifier.jsonl: 验证代码 (code-augmented LLM-as-Judge)
      格式: {"scenario": "...", "task_idx": N, "task": "...", "verification": {"code": "..."}}
    - gen_verifier.pure_code.jsonl: 纯代码验证
      格式: {"scenario": "...", "task_idx": N, "task": "...", "verification": {"code": "..."}}
    
    Args:
        dataset_path: 本地数据目录路径或 HuggingFace 数据集路径
        split: "train" 或 "test"（用于 train/test 场景划分）
        num_scenarios: 要加载的场景数量 (0 表示全部)
        tasks_per_scenario: 每个场景加载的任务数量 (1-10)
        verification_mode: "pure_code" 或 "sql"
        
    Returns:
        标准化的数据样本列表
    """
    logger.info(f"Loading AWM dataset from {dataset_path} ({split})...")
    logger.info(f"  num_scenarios={num_scenarios}, tasks_per_scenario={tasks_per_scenario}")
    logger.info(f"  verification_mode={verification_mode}")
    
    try:
        # 判断是本地目录还是 HuggingFace 路径
        is_local = os.path.isdir(dataset_path)
        
        def _load_jsonl(filename: str) -> List[Dict]:
            """加载单个 JSONL 文件，支持本地和 HuggingFace 两种方式。"""
            if is_local:
                filepath = os.path.join(dataset_path, filename)
                if not os.path.exists(filepath):
                    logger.warning(f"File not found: {filepath}")
                    return []
                data = []
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))
                return data
            else:
                from datasets import load_dataset as hf_load_dataset
                ds = hf_load_dataset(dataset_path, data_files=filename, split="train")
                return list(ds)
        
        # 加载所有必要的文件
        scenarios = _load_jsonl("gen_scenario.jsonl")
        tasks_raw = _load_jsonl("gen_tasks.jsonl")
        dbs = _load_jsonl("gen_db.jsonl")
        samples = _load_jsonl("gen_sample.jsonl")
        specs = _load_jsonl("gen_spec.jsonl")
        envs = _load_jsonl("gen_envs.jsonl")
        
        # 根据 verification_mode 选择验证文件
        if verification_mode == "pure_code":
            verifier_file = "gen_verifier.pure_code.jsonl"
        else:
            verifier_file = "gen_verifier.jsonl"
        verifiers = _load_jsonl(verifier_file)
        
        # 构建查找字典
        # gen_scenario.jsonl 使用 "name" 字段作为场景名称
        scenario_map = {normalize_awm_scenario_name(s["name"]): s for s in scenarios}
        
        # gen_tasks.jsonl: 每条记录格式为 {"scenario": "xxx", "tasks": ["task1", "task2", ...]}
        # 需要展开为按 scenario 分组的任务字符串列表
        tasks_map = {}  # scenario_key -> List[str]
        for t in tasks_raw:
            key = normalize_awm_scenario_name(t["scenario"])
            task_list = t.get("tasks", [])
            if isinstance(task_list, list):
                if key not in tasks_map:
                    tasks_map[key] = []
                tasks_map[key].extend(task_list)
            else:
                # 兼容可能的单任务格式
                if key not in tasks_map:
                    tasks_map[key] = []
                tasks_map[key].append(str(task_list))
        
        dbs_map = {normalize_awm_scenario_name(d["scenario"]): d for d in dbs}
        samples_map = {normalize_awm_scenario_name(s["scenario"]): s for s in samples}
        specs_map = {normalize_awm_scenario_name(s["scenario"]): s for s in specs}
        envs_map = {normalize_awm_scenario_name(e["scenario"]): e for e in envs}
        
        # 验证代码按 scenario + task 索引
        verifiers_map = {}
        for v in verifiers:
            key = (normalize_awm_scenario_name(v["scenario"]), v["task"])
            verifiers_map[key] = v
        
        # 限制场景数量
        if num_scenarios > 0 and num_scenarios < len(scenarios):
            rng = random.Random(GLOBAL_SEED)
            selected_scenarios = rng.sample(scenarios, num_scenarios)
        else:
            selected_scenarios = scenarios
        
        processed_data = []
        
        for scenario_data in selected_scenarios:
            # gen_scenario.jsonl 使用 "name" 字段
            scenario_raw_name = scenario_data["name"]
            scenario_name = normalize_awm_scenario_name(scenario_raw_name)
            
            # 获取该场景的任务列表（字符串列表）
            scenario_tasks = tasks_map.get(scenario_name, [])
            db_data = dbs_map.get(scenario_name)
            sample_data = samples_map.get(scenario_name)
            spec_data = specs_map.get(scenario_name)
            env_data = envs_map.get(scenario_name)
            
            if not all([db_data, sample_data, spec_data, env_data]):
                logger.warning(f"Missing data for scenario {scenario_raw_name}, skipping...")
                continue
            
            if not scenario_tasks:
                logger.warning(f"No tasks found for scenario {scenario_raw_name}, skipping...")
                continue
            
            # 限制任务数量
            if tasks_per_scenario > 0 and tasks_per_scenario < len(scenario_tasks):
                rng = random.Random(GLOBAL_SEED)
                selected_tasks = rng.sample(scenario_tasks, tasks_per_scenario)
            else:
                selected_tasks = scenario_tasks
            
            for task_description in selected_tasks:
                # task_description 现在就是任务字符串
                
                # 获取验证代码
                verifier_key = (scenario_name, task_description)
                verifier_data = verifiers_map.get(verifier_key, {})
                verifier_code = verifier_data.get("verification", {}).get("code", "")
                
                # 构建环境数据
                env_code = env_data.get("full_code", "")
                db_schema = db_data.get("db_schema", {})
                db_sample = sample_data.get("sample_data", {})
                
                # 构建原始数据字典
                raw_data = {
                    "scenario": scenario_raw_name,
                    "scenario_description": scenario_data.get("description", ""),
                    "task": task_description,
                    "env_code": env_code,
                    "db_schema": db_schema,
                    "db_sample": db_sample,
                    "schema": db_schema,
                    "sample_data": db_sample,
                    "api_spec": spec_data.get("api_spec", {}),
                    "verifier_code": verifier_code,
                    "verification_mode": verification_mode,
                    "task_type": "awm",
                    "sub_source": "awm",
                    "max_steps": 30,  # 默认最大步数
                }
                
                # 注意: db_path 需要在环境初始化时动态创建
                # 这里只存储 schema 和 sample 信息
                
                clean_d = create_standard_sample(
                    prompt=task_description,
                    response="",  # AWM 没有标准答案
                    task_type="awm",
                    raw_data=raw_data
                )
                processed_data.append(clean_d)
        
        # 打乱数据
        rng = random.Random(GLOBAL_SEED)
        rng.shuffle(processed_data)
        
        logger.info(f"Loaded {len(processed_data)} AWM tasks from {len(selected_scenarios)} scenarios.")
        return processed_data
        
    except Exception as e:
        logger.error(f"Failed to load AWM dataset: {e}", exc_info=True)
        return []


def normalize_awm_scenario_name(name: str) -> str:
    """Normalize AWM scenario name for consistent cross-file lookup.
    
    所有数据文件中的 scenario 名称格式一致（如 "e_commerce_33"、"content_platform_1"），
    直接转为小写并 strip 即可。不应移除下划线，因为它是名称的组成部分。
    """
    return name.strip().lower()


def load_tool_call_json_dataset(data_dir: str, split: str = "train", num_samples: int = 0) -> List[Dict]:
    """
    加载本地 Tool Call JSON 数据。
    数据源格式为 List[Dict]，每个 Dict 包含: "instruction", "input", "output"。
    """
    # 1. 确定文件路径
    # 如果传入的是目录，自动拼接文件名；如果传入的是文件路径，直接使用
    target_file = data_dir
    if os.path.isdir(data_dir):
        # 假设文件名格式为 train.json 或 test.json
        # 如果你的文件名不同（例如 selected_12800.json），请在这里调整逻辑
        target_file = os.path.join(data_dir, f"{split}.json")
            
    if not os.path.exists(target_file):
        logger.warning(f"Tool Call JSON dataset file not found: {target_file}")
        return []

    logger.info(f"Loading Tool Call data from {target_file}...")
    
    try:
        # 2. 读取 JSON
        with open(target_file, 'r', encoding='utf-8') as f:
            raw_list = json.load(f)
        
        if not isinstance(raw_list, list):
            logger.error(f"JSON file content expects a list of dicts, but got {type(raw_list)}")
            return []

        # 3. 截断与采样 - 使用独立的固定种子 Random 实例，保证可复现性
        if num_samples > 0 and num_samples <= len(raw_list):
            # 使用固定种子的独立 Random 实例进行无放回抽样
            rng = random.Random(GLOBAL_SEED)
            raw_list = rng.sample(raw_list, num_samples)
            
        processed_data = []
        for item in raw_list:
            # 提取字段
            # 根据你的描述，JSON 直接包含这三个 key
            system_instruction = item.get("instruction", "")
            prompt_text = item.get("input", "")
            response_text = item.get("output", "")

            # 简单校验
            if not prompt_text or not response_text:
                continue

            # 4. 构建数据字典
            item["instruction"] = system_instruction
            item["input"] = prompt_text
            item["output"] = response_text

            # 调用标准化函数 (假设该函数在上下文中可用)
            clean_d = create_standard_sample(
                prompt=prompt_text,
                response=response_text,
                task_type="tool_call",
                raw_data=item  # 传入完整的 dict 作为 raw_data
            )
            processed_data.append(clean_d)
            
        logger.info(f"Loaded {len(processed_data)} Tool Call tasks from {target_file}.")
        return processed_data

    except Exception as e:
        logger.error(f"Failed to load Tool Call JSON data: {e}")
        import traceback
        traceback.print_exc()
        return []


def load_webshop_data(split: str, num_samples: int, total_goals: int = None) -> List[Dict]:
    """
    加载 Webshop 数据 (用于 Webshop 购物任务)
    
    【重要说明】
    Webshop 任务的数据与其他任务不同：
    1. 产品数据和目标指令存储在 webshop 环境的 data/ 目录下
    2. 目标(goals)是在环境初始化时从产品数据动态生成的
    3. 这里只需要生成 goal_idx 索引，实际的 instruction 会在环境 reset 时获取
    
    【数据来源】
    使用 1000 产品的小数据集：
    - items_shuffle_1000.json: 产品信息 (1000个产品)
    - items_ins_v2_1000.json: 产品属性（合成指令）
    
    使用合成 goals (synthetic goals) 而非人工标注的 goals：
    - 合成 goals 基于产品属性和选项组合自动生成
    - 1000 产品可能生成数千到数万个 goals（取决于选项组合数）
    
    【数据划分】
    - 测试集: goal_idx 范围 [0, 500) - 固定使用前 500 个目标
    - 训练集: goal_idx 范围 [500, total_goals) - 数量由参数决定
    
    【奖励范围】
    环境返回的 task_score 在 [0.0, 1.0] 范围内，与其他任务一致。
    
    Args:
        split: 数据集划分 ("train" 或 "test")
        num_samples: 需要的样本数量
        total_goals: 合成 goals 的总数量（如果为 None，则使用默认估计值）
                    对于 1000 产品，通常会有 3000-10000 个合成 goals
        
    Returns:
        List[Dict]: 标准化的数据样本列表，包含 goal_idx 用于环境 reset
    """
    if num_samples <= 0:
        return []
    
    logger.info(f"Generating Webshop task indices ({split}, num={num_samples})...")
    
    try:
        # Webshop 使用合成 goals
        # 测试集：固定使用前 500 个 goals
        # 训练集：从 500 开始到 total_goals
        if split == "test":
            goal_idx_start = 0
            goal_idx_end = 500  # 固定使用前 500 个作为测试集
        else:  # train
            goal_idx_start = 500
            # 合成 goals 的数量取决于产品数和选项组合
            # 对于 1000 产品，估计有 5000-15000 个 goals
            # 使用提供的 total_goals 或默认估计值
            if total_goals is not None and total_goals > 500:
                goal_idx_end = total_goals
            else:
                # 默认估计：假设 1000 产品平均每个产品 10 个选项组合
                goal_idx_end = 10000
        
        # 生成可用的 goal 索引列表
        available_indices = list(range(goal_idx_start, goal_idx_end))
        
        # 如果请求的数量超过可用数量
        if num_samples > len(available_indices):
            logger.warning(
                f"Requested {num_samples} samples but only {len(available_indices)} "
                f"available for {split}. Using all available."
            )
            selected_indices = available_indices
        else:
            if split == "test":
                # 测试集：始终取前 num_samples 个索引，保证固定性
                selected_indices = available_indices[:num_samples]
            else:
                # 训练集：使用固定种子随机选择，保证可复现性
                rng = random.Random(GLOBAL_SEED)
                rng.shuffle(available_indices)
                selected_indices = available_indices[:num_samples]
        
        processed_data = []
        for goal_idx in selected_indices:
            # 构建任务数据
            # goal_idx 会在 WebshopEnvironment.reset() 时传递给底层环境
            d = {
                "goal_idx": goal_idx,
                "session_idx": goal_idx,  # 兼容字段
                "task_type": "webshop",
                "sub_source": "webshop",
            }
            
            # 使用标准化函数创建样本
            # prompt 是占位符，实际的购物指令会在环境 reset 时从 webshop 获取
            clean_d = create_standard_sample(
                prompt=f"[Webshop Task #{goal_idx}]",
                response="",  # 购物任务没有预定义的 response
                task_type="webshop",
                raw_data=d
            )
            processed_data.append(clean_d)
        
        logger.info(f"Generated {len(processed_data)} Webshop task indices for {split}.")
        return processed_data
        
    except Exception as e:
        logger.error(f"Failed to generate Webshop data: {e}")
        import traceback
        traceback.print_exc()
        return []


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TestScript")


if __name__ == "__main__":
    math_dataset = load_deepmath_dataset(12800)

