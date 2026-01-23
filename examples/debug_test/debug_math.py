import asyncio
import os
import json
import logging
from typing import List, Dict
import random

from transformers import AutoTokenizer

from rllm.agents import ToolAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.utils import compute_pass_at_k
from rllm.utils.compute_pass_at_k import save_clean_trajectories
from rllm.agents.system_prompts import MATH_SYSTEM_PROMPT


# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_dataset_fields(dataset_items) -> List[Dict]:
    """
    [核心修复逻辑]
    从 DatasetRegistry 加载的数据，Ground Truth 可能存在于 'response' 字段
    或者被序列化在 'extra_info' 中。
    此函数将其映射回 'answer' 字段，以便 math_reward_fn 能识别。
    """
    fixed_tasks = []
    logger.info("Checking and fixing dataset fields...")
    
    for i, item in enumerate(dataset_items):
        # 转换为可编辑的 dict
        task = dict(item) if not isinstance(item, dict) else item.copy()
        
        ground_truth = None
        
        # 1. 优先检查 extra_info (最原始的数据通常在这里)
        if "extra_info" in task and task["extra_info"]:
            try:
                if isinstance(task["extra_info"], str):
                    raw_data = json.loads(task["extra_info"])
                else:
                    raw_data = task["extra_info"]
                
                # 尝试从原始数据中找 answer/solution
                if "answer" in raw_data:
                    ground_truth = raw_data["answer"]
                elif "solution" in raw_data:
                    ground_truth = raw_data["solution"]
            except Exception:
                pass

        # 2. 如果没找到，检查 response 字段 (训练脚本中 create_standard_sample 把答案放这里了)
        if not ground_truth and "response" in task:
            ground_truth = task["response"]

        # 3. 赋值给 'answer'，这是 math_reward_fn 的默认查找字段
        if ground_truth:
            task["answer"] = str(ground_truth)
        else:
            # 如果实在找不到，标记一下，方便 debug 打印出来
            task["answer"] = "GROUND_TRUTH_MISSING"

        fixed_tasks.append(task)
        
    logger.info(f"Processed {len(fixed_tasks)} tasks.")
    return fixed_tasks


def save_detailed_trajectories(results, output_path: str = "debug_trajectories.json"):
    """
    将完整的轨迹信息（包括每一步的思考、工具调用、工具返回）保存为 JSON 文件。
    便于详细分析模型为什么回答错误。
    """
    logger.info(f"Saving detailed trajectories to {output_path}...")
    
    export_data = []
    
    for traj in results:
        # 1. 提取题目基本信息
        # 确保 task 是字典
        task_data = dict(traj.task) if not isinstance(traj.task, dict) else traj.task
        
        # 尝试获取问题文本
        question = task_data.get("question") or task_data.get("prompt") or task_data.get("input")
        
        # 尝试获取答案
        ground_truth = task_data.get("answer")
        if not ground_truth and "response" in task_data:
             ground_truth = task_data["response"]

        # 2. 提取每一步的详细交互 (Thought -> Action -> Observation)
        steps_details = []
        if traj.steps:
            for step_idx, step in enumerate(traj.steps):
                step_info = {
                    "step_index": step_idx,
                    # 模型生成的文本（通常包含分析和代码块）
                    "model_response": step.model_response,
                    # 如果有工具调用，提取工具名和参数
                    "action": str(step.action) if step.action else None,
                    # 工具执行的结果（报错信息或计算结果）
                    "observation": str(step.observation) if step.observation else None,
                }
                steps_details.append(step_info)
        
        # 3. 组装单个样本的完整记录
        record = {
            "uid": traj.uid,
            "reward": traj.reward,  # 最终得分
            # "finished": traj.finished, # 是否正常结束
            "question": question,
            "ground_truth_expected": str(ground_truth),
            "trajectory_steps": steps_details
        }
        export_data.append(record)

    # 4. 写入文件 (使用 ensure_ascii=False 保证中文正常显示)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Successfully saved {len(export_data)} trajectories.")
        print(f"\n[Saved] Detailed debug file is at: {os.path.abspath(output_path)}")
    except Exception as e:
        logger.error(f"Failed to save trajectories: {e}")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # 配置
    n_parallel_agents = 1
    model_name = "Qwen3-32B" # 请确认这是你的实际模型路径
    model_path = "/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/base_models/Qwen3-32B"
    dataset_name = "math2560_code0_bfcl0" # 你指定的已存在的数据集
    api_base_url = "http://localhost:8803/v1"

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    agent_args = {
        "tools": ["python"], 
        "parser_name": "qwen", 
        "system_prompt": MATH_SYSTEM_PROMPT
    }
    
    env_args = {
        "tools": ["python"],
        "reward_fn": math_reward_fn,
    }

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name, "max_tokens": 32768}

    engine = AgentExecutionEngine(
        agent_class=ToolAgent,
        agent_args=agent_args,
        env_class=ToolEnvironment,
        env_args=env_args,
        engine_name="openai",
        rollout_engine_args={"base_url": api_base_url, "api_key": "None", "model_name": model_name},
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_response_length=32768,
        max_prompt_length=32768,
        n_parallel_agents=n_parallel_agents,
    )

    # 1. 直接加载本地数据集
    logger.info(f"Loading dataset from Registry: {dataset_name} (split: test)")
    raw_dataset = DatasetRegistry.load_dataset(dataset_name, "test")
    
    if not raw_dataset:
        raise ValueError(f"Dataset {dataset_name} not found or empty!")

    # 2. [关键] 修复字段映射 (Map 'response' -> 'answer')
    # tasks = fix_dataset_fields(raw_dataset)
    tasks = raw_dataset.get_data()
    random.shuffle(tasks)
    # import pdb; pdb.set_trace()
    # 3. 运行 (先只跑前 20 个样本做快速 Debug，确认无误后再跑全量)
    debug_subset_size = 2
    logger.info(f"Running evaluation on first {debug_subset_size} samples for debugging...")
    
    results = asyncio.run(engine.execute_tasks(tasks[:debug_subset_size]))
    save_detailed_trajectories(results, "./trajectories/debug_math_trajectories.json")
    metrics = compute_pass_at_k(results)
    print(f"\n>>> Metrics on subset: {metrics}")

    # 5. [核心 Debug] 打印 失败案例 的详细对比
    print("\n" + "="*60)
    print("DETAILED FAILURE ANALYSIS (Based on Trajectory Class)")
    print("="*60)

    # results 是 List[Trajectory]
    for i, traj in enumerate(results):
        # 1. 获取分数 (Class定义中属性为 reward)
        score = traj.reward
        
        # 只要不是满分，就打印出来看看原因
        if score < 1.0:
            # 2. 获取 Task 信息
            # task 通常是一个 dict，但为了安全起见做个类型检查
            task_data = traj.task if isinstance(traj.task, dict) else dict(traj.task)
            
            prompt = task_data.get("prompt", "")
            if not prompt and "question" in task_data:
                prompt = task_data["question"]
            
            # --- 尝试提取 Ground Truth (仅用于 Debug 显示) ---
            gt = task_data.get("answer", None)
            
            # 如果顶层没有 answer，尝试去 extra_info 里找 (模拟 math_reward_fn 找不到的情况)
            gt_source = "Top-level"
            if not gt:
                if "response" in task_data and task_data["response"]:
                    gt = task_data["response"]
                    gt_source = "Response-field"
                elif "extra_info" in task_data:
                    try:
                        extra = json.loads(task_data["extra_info"]) if isinstance(task_data["extra_info"], str) else task_data["extra_info"]
                        if "answer" in extra:
                            gt = extra["answer"]
                            gt_source = "Extra-info (Hidden)"
                        elif "solution" in extra:
                            gt = extra["solution"]
                            gt_source = "Extra-info (Hidden Solution)"
                    except:
                        pass
            
            if not gt:
                gt = "!!! NOT FOUND !!!"
                gt_source = "None"

            # 3. 获取模型输出 (从 Steps 中获取)
            model_response_text = ""
            tool_observation = ""
            
            if traj.steps:
                last_step = traj.steps[-1] # 获取最后一步
                
                # 模型生成的文本 (包含解题过程和 \boxed{})
                model_response_text = last_step.model_response
                
                # 如果有代码执行结果
                if last_step.observation:
                    tool_observation = str(last_step.observation)
            else:
                model_response_text = "No steps executed (Empty Trajectory)"

            # 4. 打印详情
            print(f"\n[Case #{i} | Reward: {score}]")
            print(f"Task ID             : {traj.uid}")
            print(f"Prompt (Top 100)    : {str(prompt)}...")
            print(f"Ground Truth Source : {gt_source}")
            print(f"Expect Ground Truth : {gt}")
            print("-" * 20 + " Model Output (Last Step) " + "-" * 20)
            print(f"{model_response_text[-500:]}") # 打印最后500字符，通常包含答案
            if tool_observation:
                print("-" * 20 + " Tool Observation " + "-" * 20)
                print(f"{tool_observation[:200]}...")
            print("-" * 60)
            
            # 严重错误警告：如果 Debug 逻辑在这里找到了 GT，但 Source 是 'Extra-info' 或 'Response-field'
            # 说明 math_reward_fn 可能根本没拿到这个 GT，因为它通常只看 task['answer']
            if gt_source != "Top-level" and gt_source != "None":
                print(f"!!! DIAGNOSIS: Reward function likely failed because answer is in '{gt_source}' but expected in 'answer' field.")
