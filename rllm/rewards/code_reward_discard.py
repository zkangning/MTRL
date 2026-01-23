"""
This module contains the RewardCode class, which evaluates code datasets answers
and assigns rewards based on their correctness on unit tests.
"""

import ast
import json
import multiprocessing
import re
import queue  # 用于 Queue异常捕获
from typing import Any, Dict, List, Union

# 保持原有工具库引用
from rllm.rewards.code_utils.firejail_exec import code_exec_firejail as lc_code_exec
from rllm.rewards.code_utils.humanevalplus import get_num_test_cases
from rllm.rewards.code_utils.humanevalplus import run_test as humanevalplus_run_test
from rllm.rewards.code_utils.kodcode import code_exec as kod_code_exec
from rllm.rewards.code_utils.livecodebench import run_test as lcb_run_test
from rllm.rewards.code_utils.taco import run_test as taco_run_test
from rllm.rewards.reward_types import RewardConfig, RewardOutput, RewardType
from rllm.tools.code_tools.code_tool import CodeTool
from rllm.tools.code_tools.together_tool import TogetherCodeTool

# --- 全局配置 ---
# 默认快速超时时间 (秒)。对于算法题，Python 3秒通常足够，超过即判负。
FAST_TIMEOUT = 3.0 

def extract_code_from_model(model_response: str):
    """Extracts the code from a Markdown-style code block."""
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[-1].strip()

def clean_code_main_block(code: str) -> str:
    """Removes `if __name__ == "__main__"` blocks."""
    code_lines = code.split("\n")
    filtered_lines = []
    skip_block = False
    for line in code_lines:
        if line.strip().startswith(('if __name__ == "__main__"', "if __name__ == '__main__'")):
            skip_block = True
            continue
        if skip_block:
            if line.strip() and not line.startswith((" ", "\t")):
                skip_block = False
            else:
                continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines)

# --- [优化] 通用 Worker 函数，用于替代 Manager ---
def _worker_run_tests(tests, code, test_fn, debug, timeout, result_queue):
    """
    通用 Worker，结果写入 Queue 而非 Manager List。
    """
    try:
        # 运行具体的测试函数
        # 注意：具体的 test_fn (如 taco_run_test) 内部通常也会处理超时，
        # 但这里的 timeout 参数主要用于传递配置
        result = test_fn(tests, test=code, debug=debug, timeout=timeout)
        result_queue.put(result)
    except Exception as e:
        # 捕获执行器内部崩溃
        result_queue.put({"error": str(e), "internal_crash": True})

def check_correctness(tests: list[dict[str, str]] | dict[str, list[str]], code: str, test_fn, timeout_per_test: int = 3, max_tests: int = 15) -> tuple[bool, dict[str, Any]]:
    """
    [Optimized] Check if generated code passes all test cases.
    Using multiprocessing.Queue instead of Manager for lower overhead.
    """
    # 1. 筛选测试用例 (在主进程完成，开销极小)
    original_tests = tests
    if isinstance(tests, list):
        list_tests = tests
        total_tests = len(list_tests)
        if total_tests > max_tests:
            # 优先测长的 Input
            selected_indices = sorted(range(total_tests), key=lambda i: len(list_tests[i]["input"]), reverse=True)[:max_tests]
            tests = [list_tests[i] for i in selected_indices]
        num_tests = len(tests)
    else:
        dict_tests = tests
        total_tests = len(dict_tests["inputs"])
        if total_tests > max_tests:
            selected_indices = sorted(range(total_tests), key=lambda i: len(dict_tests["inputs"][i]), reverse=True)[:max_tests]
            selected_tests = {"inputs": [dict_tests["inputs"][i] for i in selected_indices], "outputs": [dict_tests["outputs"][i] for i in selected_indices]}
            tests = selected_tests
        num_tests = len(tests["inputs"])

    # 2. 启动子进程 (使用 Queue)
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_worker_run_tests, 
        args=(tests, code, test_fn, False, timeout_per_test, result_queue)
    )
    process.start()
    
    # 计算最大等待时间：如果是逐个跑测试，总时间 = 单个超时 * 测试数
    # 但为了快速失败，这里设置一个硬上限 (Hard Limit)，例如最大 15秒
    hard_limit = min(timeout_per_test * num_tests + 2, 15)
    process.join(timeout=hard_limit)

    test_results_data = []
    
    # 3. 处理超时与结果获取
    if process.is_alive():
        process.kill()
        process.join() # 清理僵尸进程
        # 超时视为全错
        test_results_data = [False] * num_tests
    else:
        if not result_queue.empty():
            raw_result = result_queue.get()
            if isinstance(raw_result, dict) and raw_result.get("internal_crash"):
                 test_results_data = [False] * num_tests # 执行器崩溃
            else:
                test_results_data = raw_result
        else:
            # 进程死掉但没写回数据 (OOM 等)
            test_results_data = [False] * num_tests

    # 4. 格式化结果
    detailed_results: dict[str, Any] = {"all_passed": False, "test_results": [], "total_tests": num_tests, "passed_tests": 0}
    
    # 防止结果长度不一致
    if len(test_results_data) < num_tests:
        test_results_data.extend([False] * (num_tests - len(test_results_data)))

    passed_results = [r is True for r in test_results_data] # 确保是 bool

    test_results_list_typed = detailed_results["test_results"]
    if isinstance(original_tests, list):
        # 注意：这里只回填被选中的 tests 的结果，为了简化逻辑，仅展示运行了的 tests
        for i, (test, result) in enumerate(zip(tests, passed_results)):
            test_results_list_typed.append({"input": test.get("input", ""), "expected": test.get("output", ""), "passed": result})
    else:
        for i, (inp, out, result) in enumerate(zip(tests["inputs"], tests["outputs"], passed_results)):
            test_results_list_typed.append({"input": inp, "expected": out, "passed": result})

    detailed_results["passed_tests"] = sum(passed_results)
    detailed_results["all_passed"] = all(passed_results) and len(passed_results) > 0

    return detailed_results["all_passed"], detailed_results

def postprocess_lcb_sample(sample):
    """Helper for LCB format."""
    sample_inputs = [s["input"] for s in sample]
    sample_outputs = [s["output"] for s in sample]

    sample_dict = {
        "inputs": sample_inputs,
        "outputs": sample_outputs,
    }

    if sample[0].get("testtype") == "functional":
        metadata = sample[0].get("metadata", {})
        fn_name = metadata.get("func_name", None)
        if fn_name:
            sample_dict["fn_name"] = fn_name

    sample_processed = {
        "input_output": json.dumps(sample_dict),
    }
    return sample_processed

# --- [优化] LCB Worker ---
def _lcb_worker(sample, generation, debug, timeout, result_queue):
    try:
        res, metadata = lcb_run_test(sample, test=generation, debug=debug, timeout=timeout)
        result_queue.put((res, metadata))
    except Exception as e:
        result_queue.put((False, {"error": str(e)}))

def lcb_check_correctness_v2(sample, generation, timeout=3.0, debug=False):
    """
    [Optimized] Check correctness for LiveCodeBench / CodeForces / etc.
    Replaced Manager with Queue and enforced strict process cleanup.
    """
    assert len(sample) >= 1, "Sample must contain at least one test case"
    sample_processed = postprocess_lcb_sample(sample)

    result_queue = multiprocessing.Queue()
    
    # 启动进程
    p = multiprocessing.Process(
        target=_lcb_worker,
        args=(sample_processed, generation, debug, timeout, result_queue),
    )
    p.start()
    
    # 计算 Hard Limit: 给 python 解释器启动留一点 buffer
    in_outs = json.loads(sample_processed["input_output"])
    num_inputs = len(in_outs["inputs"])
    
    # 这里的 timeout 是传给 lcb_run_test 的，通常它内部会针对每个 case 或总时间做控制
    # 我们设置外部 join timeout 防止死锁
    hard_limit = timeout + 2.0 
    p.join(timeout=hard_limit)

    detailed_results = {"all_passed": False, "test_results": [], "total_tests": num_inputs, "passed_tests": 0}

    # 处理结果
    raw_res = None
    metadata = {}
    
    is_timeout = False

    if p.is_alive():
        p.kill()
        p.join()
        is_timeout = True
    else:
        if not result_queue.empty():
            raw_res, metadata = result_queue.get()
        else:
            # 进程退出但无结果 (Crash)
            is_timeout = True 
            metadata = {"error": "Process crashed silently"}

    if is_timeout or raw_res is None:
        # 全错
        pass_list = [False] * num_inputs
        error_msg = metadata.get("error", "Global Timeout" if is_timeout else "Execution Error")
        for i in range(num_inputs):
            detailed_results["test_results"].append({
                "input": in_outs["inputs"][i], 
                "expected": in_outs["outputs"][i], 
                "passed": False, 
                "error": error_msg
            })
        return False, detailed_results

    # 规范化 raw_res (可能是 bool 或 list[bool])
    if isinstance(raw_res, bool):
        pass_list = [raw_res] * num_inputs # 如果只返回单 bool，假设全对或全错 (很少见)
    else:
        pass_list = raw_res

    # 填充结果
    if len(pass_list) < num_inputs:
        pass_list.extend([False] * (num_inputs - len(pass_list)))

    for i in range(num_inputs):
        detailed_results["test_results"].append({
            "input": in_outs["inputs"][i],
            "expected": in_outs["outputs"][i],
            "passed": pass_list[i] == True,
            "error": metadata.get("error"),
            "output": metadata.get("output") # 通常只有第一个错误用例的 output
        })

    detailed_results["passed_tests"] = sum(1 for r in pass_list if r == True)
    detailed_results["all_passed"] = all(r == True for r in pass_list)

    return detailed_results["all_passed"], detailed_results

def leetcode_check_correctness(tests: dict[str, str], code: str) -> tuple[bool, dict[str, Any]]:
    # LeetCode executor usually runs fast, but make sure lc_code_exec has internal timeout
    succ, output = lc_code_exec(code + "\n" + tests["functional"])
    detailed_results = {"all_passed": succ, "output": output, "test_results": [{"passed": succ, "output": output}]}
    if not succ:
        pass 
    return succ, detailed_results

def kodcode_check_correctness(test: str, code: str, timeout_per_test: int = 3) -> tuple[bool, dict[str, Any]]:
    num_tests = test.count("def test")
    code = clean_code_main_block(code)
    succ, output = kod_code_exec(code, test, timeout_per_test * num_tests)
    detailed_results = {"all_passed": succ, "output": output, "total_tests": num_tests, "test_results": [{"passed": succ, "output": output}]}
    return succ, detailed_results

def humanevalplus_check_correctness(test: str, code: str, timeout_per_test: int = 1) -> tuple[bool, dict[str, Any]]:
    code = clean_code_main_block(code)
    num_test_cases = get_num_test_cases(test)
    # Humaneval usually needs very short time
    succ, output = humanevalplus_run_test(code, test, timeout_per_test * num_test_cases)
    detailed_results = {"all_passed": succ, "output": output, "total_tests": num_test_cases, "test_results": [{"passed": succ, "output": output}]}
    return succ, detailed_results

def primeintellect_check_correctness(tests, code, use_tci=False):
    if isinstance(tests, str):
        try:
            tests = ast.literal_eval(tests)
        except (ValueError, SyntaxError) as e:
            return False, {"all_passed": False, "error": str(e)}

    assert len(tests) >= 1
    inputs = [t["input"] for t in tests]
    outputs = [t["output"] for t in tests]
    fn_name = tests[0].get("fn_name", None)
    tests_formatted = {"inputs": inputs, "outputs": outputs}
    if fn_name:
        tests_formatted["fn_name"] = fn_name

    if use_tci:
        codetool = TogetherCodeTool()
        return codetool_check_correctness(tests_formatted, code, codetool, is_taco_format=True)

    # 使用优化后的 check_correctness
    return check_correctness(tests_formatted, code, taco_run_test, timeout_per_test=FAST_TIMEOUT)

def taco_to_lcb_format(tests):
    inputs = tests.get("inputs", [])
    outputs = tests.get("outputs", [])
    n = max(len(inputs), len(outputs))
    test_cases = []
    for i in range(n):
        inp = inputs[i] if i < len(inputs) else (inputs[0] if inputs else "")
        out = outputs[i] if i < len(outputs) else (outputs[0] if outputs else "")
        out = out[0] if isinstance(out, list) else out
        test_case: dict[str, Any] = {"input": inp, "output": out, "metadata": {}}
        if "fn_name" in tests:
            test_case["testtype"] = "functional"
            test_case["metadata"]["func_name"] = tests["fn_name"]
        test_cases.append(test_case)
    return test_cases

def codetool_check_correctness(tests: Any, code: str, codetool: CodeTool, is_taco_format=True, timeout=30) -> tuple[bool, dict[str, Any]]:
    from rllm.tools.utils import call_based_test_code_wrapper, stdin_test_code_wrapper
    fn_name = None
    call_based = False
    if isinstance(tests, dict) and "fn_name" in tests:
        call_based = True
        fn_name = tests.get("fn_name", None)

    new_tests = taco_to_lcb_format(tests) if is_taco_format and not fn_name else tests

    if call_based:
        test_wrapped_code = call_based_test_code_wrapper(code, new_tests)
    else:
        test_wrapped_code = stdin_test_code_wrapper(code, new_tests)

    tool_response = codetool(code=test_wrapped_code, timeout=timeout)
    
    # 构造结果
    detailed_results = {"all_passed": not tool_response.error, "output": tool_response.output, "error": tool_response.error, "test_results": []}
    if isinstance(new_tests, list):
        detailed_results["total_tests"] = len(new_tests)
        detailed_results["test_results"] = [{"input": test.get("input", ""), "expected": test.get("output", ""), "passed": not tool_response.error} for test in new_tests]
    
    return detailed_results["all_passed"], detailed_results

# --- [核心修改] 优化后的 RewardCodeFn ---
class RewardCodeFn:
    """
    Reward function for evaluating code datasets answers.
    Optimized for RL training throughput.
    """

    def __init__(self, config: RewardConfig):
        self.config = config

    def __call__(self, task_info: dict, action: str) -> RewardOutput:
        dataset_name = task_info.get("data_source", "")
        tests = task_info.get("ground_truth", None)

        if tests is None:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False, metadata={"error": "No tests found"})

        model_code = extract_code_from_model(action)
        if model_code is None:
            # 严格模式：没有 Markdown code block 直接判负
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False, metadata={"error": "No code block"})

        # --- [Optimization 1: AST Pre-check] ---
        # 如果代码语法错误，直接返回，不启动任何子进程
        # 这在 RL 训练初期能节省 90% 的执行时间
        try:
            ast.parse(model_code)
        except SyntaxError as e:
            return RewardOutput(
                reward=self.config.format_error_reward, # 或定义专门的 syntax_error_reward
                is_correct=False, 
                metadata={"error": f"Syntax Error: {str(e)}"}
            )
        except Exception:
            pass # 忽略其他解析错误，交给 sandbox
        # ---------------------------------------

        if self.config.use_together_code_interpreter:
            codetool = TogetherCodeTool()

        is_correct = False
        test_details: dict[str, Any] = {}
        
        # --- [Optimization 2: Short Timeouts] ---
        # 显式传递 FAST_TIMEOUT (3.0s)
        try:
            if dataset_name in ["taco", "apps", "code_contests"]:
                if self.config.use_together_code_interpreter:
                    is_correct, test_details = codetool_check_correctness(tests, model_code, codetool, is_taco_format=True)
                else:
                    tests_formatted = taco_to_lcb_format(tests)
                    is_correct, test_details = lcb_check_correctness_v2(tests_formatted, model_code, timeout=FAST_TIMEOUT, debug=False)
            
            elif dataset_name == "leetcode":
                is_correct, test_details = leetcode_check_correctness(tests, model_code)
            
            elif dataset_name in ["livecodebench", "codeforces", "primeintellect"]:
                if isinstance(tests, str):
                    try:
                        tests = json.loads(tests)
                    except json.JSONDecodeError:
                        return RewardOutput(reward=self.config.unk_error_reward, is_correct=False, metadata={"error": "Invalid JSON tests"})
                is_correct, test_details = lcb_check_correctness_v2(tests, model_code, timeout=FAST_TIMEOUT, debug=False)
            
            elif dataset_name == "kodcode":
                is_correct, test_details = kodcode_check_correctness(tests, model_code, timeout_per_test=1)
            
            elif dataset_name == "humanevalplus":
                is_correct, test_details = humanevalplus_check_correctness(tests, model_code, timeout_per_test=1)
            
            elif dataset_name == "code": 
                # Generic fallback
                if isinstance(tests, list):
                    is_correct, test_details = lcb_check_correctness_v2(tests, model_code, timeout=FAST_TIMEOUT, debug=False)
            else:
                pass
        
        except Exception as e:
            print(f"[Reward Error] {dataset_name}: {e}")
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False, metadata={"error": str(e)})

        if is_correct:
            return RewardOutput(reward=self.config.correct_reward, is_correct=True, metadata=test_details)
        else:
            return RewardOutput(reward=self.config.incorrect_reward, is_correct=False, metadata=test_details)

def rllm_reward_fn_code(data_source: str, llm_solution: str, ground_truth: Union[Dict, List, str, Any], **kwargs):
    """
    Adapter function for RLHF trainer.
    """
    reward_config = RewardConfig()
    reward_fn = RewardCodeFn(reward_config)
    
    # 提取 Test Cases (逻辑保持不变，用于适配各种数据集格式)
    real_tests = None

    if isinstance(ground_truth, dict):
        if "extra_info" in ground_truth and isinstance(ground_truth["extra_info"], dict):
            extra = ground_truth["extra_info"]
            if "tests" in extra: real_tests = extra["tests"]
            elif "test" in extra: real_tests = extra["test"]
            elif "input_output" in extra: real_tests = extra["input_output"]
            elif "inputs" in extra and "outputs" in extra: real_tests = extra
            elif "inputs" in ground_truth: real_tests = ground_truth
        elif "inputs" in ground_truth or "functional" in ground_truth:
            real_tests = ground_truth
    elif isinstance(ground_truth, (list, str)):
        real_tests = ground_truth

    if real_tests is None:
        real_tests = ground_truth

    task_info = {
        "problem": None, 
        "problem_type": RewardType.CODE, 
        "data_source": data_source, 
        "ground_truth": real_tests 
    }

    reward_response = reward_fn(task_info, llm_solution)
    return reward_response
