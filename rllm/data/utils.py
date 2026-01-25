"""Utility functions for loading and processing datasets."""

import hydra
import logging
import requests
import random
import os
import json
from typing import List, Dict, Any
from datasets import load_dataset
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

from rllm.data.dataset_types import TestDataset, TrainDataset
from rllm.data.dataset import DatasetRegistry
from rllm.system_prompts import LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE, LCB_FORMATTING_WITHOUT_STARTER_CODE, LCB_SYSTEM_MESSAGE_GENERIC


random.seed(42)

def create_standard_sample(prompt: str, response: str, task_type: str, raw_data: Dict) -> Dict[str, str]:
    """
    强制统一数据格式，防止 PyArrow Schema 报错。
    所有字段强制转为字符串，复杂对象存入 extra_info。
    """
    safe_prompt = str(prompt) if prompt is not None else ""
    safe_response = str(response) if response is not None else ""
    
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
                raw_data = {"task_id": t_id, "task_type": "bfcl"}
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
            
        logger.info(f"Loaded {len(processed_data)} Search tasks from HotpotQA ({split}).")
        return processed_data
    except Exception as e:
        logger.error(f"Failed to load HotpotQA: {e}")
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
        
        # 如果指定了数量限制，先 Shuffle 再切片，保证数据随机性
        if num_samples > 0 and len(raw_list) > num_samples:
            random.shuffle(raw_list)
            raw_list = raw_list[:num_samples]
            
        processed_data = []
        for item in raw_list:
            d = dict(item)
            
            # 提取 DAPO 数据集的特定字段
            prompt_text = d.get("prompt", "")
            response_text = d.get("solution", "") # 用户指定 solution 字段
            
            # 标记任务类型
            d["task_type"] = "math"
            d["source"] = "dapo_math_17k"
            
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
            
            # 3.3 执行随机抽取
            for diff, count in quotas.items():
                group_items = eligible_groups[diff]
                # 随机打乱后取 count 个，或者直接用 random.sample
                sampled = random.sample(group_items, count)
                selected_raw_list.extend(sampled)
                logger.info(f"  difficulty={diff}: Total {len(group_items)} -> Sampled {count}")

        # -------------------------------------------------------
        # 4. 标准化处理
        # -------------------------------------------------------
        # 打乱最终列表，避免按难度排序
        random.shuffle(selected_raw_list)

        processed_data = []
        for item in selected_raw_list:
            d = dict(item)
            prompt_text = d.get("question", "")
            response_text = d.get("final_answer", "")

            d["task_type"] = "math"
            d["source"] = "deepmath_103k"

            clean_d = create_standard_sample(
                prompt=prompt_text,
                response=response_text,
                task_type="math",
                raw_data=d
            )
            processed_data.append(clean_d)

        logger.info(f"Loaded {len(processed_data)} tasks from {dataset_path} (Stratified >= 6.0).")
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
        
        # 重点：虽然选的是最高的，但在返回前要打乱，避免模型在训练时按难度顺序学习导致梯度不稳定
        random.shuffle(selected_raw_list)

        processed_data = []
        for item in selected_raw_list:
            d = dict(item)
            prompt_text = d.get("question", "")
            response_text = d.get("final_answer", "")

            # 补充元数据
            d["task_type"] = "math"
            d["source"] = "deepmath_103k_top_k"

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

        # 3. 截断与打乱
        if num_samples > 0 and num_samples <= len(raw_list):
            # 使用 random.sample 进行无放回抽样 (相当于 shuffle + slice)
            random.seed(42)
            raw_list = random.sample(raw_list, num_samples)
            
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TestScript")


if __name__ == "__main__":
    math_dataset = load_deepmath_dataset(12800)

