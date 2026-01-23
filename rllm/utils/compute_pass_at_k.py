import os
from collections import defaultdict
import json
from typing import List, Any, Dict
import torch
import re


def split_think_and_visible(text: str) -> tuple[str, str]:
    if not isinstance(text, str):
        return "", str(text)

    match = re.match(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    
    if match:
        return match.group(1).strip("\n"), match.group(2).strip("\n")
    
    return "", text.strip()

def compute_pass_at_k(results):
    import hashlib
    import json

    # Create a map to store correct answers per problem
    problem_correct_map: defaultdict[str, int] = defaultdict(int)
    problem_total_map: defaultdict[str, int] = defaultdict(int)

    # Count correct answers for each problem
    for trajectory in results:
        task = trajectory.task

        # Generate hash of problem dict/string
        if isinstance(task, dict):
            problem_str = json.dumps(task, sort_keys=True)
        else:
            problem_str = str(task)
        problem_hash = hashlib.md5(problem_str.encode()).hexdigest()

        is_correct = 1 if trajectory.reward > 0 else 0

        problem_correct_map[problem_hash] += is_correct
        problem_total_map[problem_hash] += 1

    # Calculate pass@1 and pass@16
    total_problems = len(problem_correct_map)
    pass_at_1 = sum(problem_correct_map.values()) / sum(problem_total_map.values())
    pass_at_k = sum(1 for problem, correct in problem_correct_map.items() if correct > 0) / total_problems

    print("Total unique problems:", total_problems)
    print("Average Pass@1 Accuracy:", pass_at_1)
    print("Average Pass@k Accuracy:", pass_at_k)


def save_trajectories(results, save_dir="./trajectories", filename="trajectories.pt"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    torch.save(results, save_path)
    print(f"Trajectories saved to {save_path}")
    return save_path

def save_trajectories_json(results, save_dir="./trajectories", filename="trajectories.json"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    
    # 如果 results 中有 numpy / tensor 等，需要先转成可序列化的类型
    def default_converter(o):
        try:
            import torch
            if isinstance(o, torch.Tensor):
                return o.tolist()
        except ImportError:
            pass
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        # 其他无法直接序列化的对象按字符串保存
        return str(o)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=default_converter)

    print(f"Trajectories saved to {save_path}")
    return save_path

def save_clean_trajectories(trajectories: List[Any], save_path: str):
    cleaned_data = []
    
    for traj in trajectories:
        # 获取原始任务信息
        task_info = traj.task if traj.task else {}
        task_id = task_info.get("task_id", "unknown")
        
        # 关键：保存原始 User Query，而不是被 Env 修改过的 Prompt
        user_instruction = task_info.get("instruction", "")
        user_question = task_info.get("question", "")
        ground_truth = task_info.get("ground_truth", "")
        
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
            user_query = ""
            if step.observation and isinstance(step.observation, dict):
                user_query = step.observation.get("question", "")
                outputs = step.observation.get("tool_outputs", {})
                for _, output in outputs.items():
                    # 截断长输出
                    observations.append(output[:500] + "..." if len(output) > 500 else output)

            action = step.model_response
            if "</think>" in step.model_response:
                thought, action = split_think_and_visible(step.model_response)
            interaction_log.append({
                "user_query": user_query,
                "agent_response": step.model_response,
                "reason_content": step.thought,
                "action": action,
                "tool_calls": actions,
                "tool_outputs": observations
            })

        cleaned_data.append({
            "task_id": task_id,
            "user_query": user_instruction, # 保存干净的用户查询
            "user_question": user_question,
            "ground_truth": ground_truth,
            "final_reward": traj.reward,
            "interactions": interaction_log,
        })

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)