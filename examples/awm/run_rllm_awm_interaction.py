"""
Run end-to-end interaction between AWMAgent and AWMEnvironment in RLLM.

This script uses real AWM dataset records and executes trajectories through
RLLM's AgentExecutionEngine (OpenAI-compatible backend, e.g. vLLM).
"""

import argparse
import asyncio
import json
import logging
import os
import pdb
import statistics
from typing import Any

from rllm.agents.awm_agent import AWMAgent
from rllm.agents.awm_prompts import AWM_SYSTEM_PROMPT
from rllm.data.utils import load_awm_dataset
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.awm import AWMEnvironment
from rllm.rewards.reward_fn import awm_reward_fn


logger = logging.getLogger("rllm_awm_interaction")


def _normalize_scenario_name(name: str) -> str:
    return name.strip().lower()


def _format_task_preview(task: dict[str, Any], max_len: int = 140) -> str:
    text = str(task.get("task", ""))
    return text if len(text) <= max_len else f"{text[:max_len]}..."


def _maybe_set_local_no_proxy(vllm_url: str, enable: bool):
    if not enable:
        return
    if "127.0.0.1" in vllm_url or "localhost" in vllm_url:
        no_proxy = "127.0.0.1,localhost"
        os.environ["NO_PROXY"] = no_proxy
        os.environ["no_proxy"] = no_proxy


def _pick_tasks(
    dataset_path: str,
    split: str,
    scenario: str | None,
    num_scenarios: int,
    tasks_per_scenario: int,
    max_tasks: int,
) -> list[dict]:
    # If a specific scenario is requested, load all scenarios first and then
    # filter deterministically. Otherwise random scenario sampling may exclude
    # the target scenario before filtering.
    effective_num_scenarios = 0 if scenario else num_scenarios

    records = load_awm_dataset(
        dataset_path=dataset_path,
        split=split,
        num_scenarios=effective_num_scenarios,
        tasks_per_scenario=tasks_per_scenario,
        verification_mode="pure_code",
        output_format="flat",
    )

    if scenario:
        scenario_norm = _normalize_scenario_name(scenario)
        filtered = []
        for r in records:
            scenario_name = _normalize_scenario_name(str(r.get("scenario", "")))
            if scenario_name == scenario_norm:
                filtered.append(r)
        records = filtered

    if max_tasks > 0:
        records = records[:max_tasks]
    return records


def _print_trajectory_summary(trajectory, idx: int):
    steps = trajectory.steps
    print(f"\n=== Trajectory #{idx} ===")
    print(f"reward={trajectory.reward}, steps={len(steps)}")
    for i, step in enumerate(steps, 1):
        action_preview = str(step.action)
        if len(action_preview) > 180:
            action_preview = f"{action_preview[:180]}..."
        print(f"  - step {i}: reward={step.reward}, done={step.done}")
        print(f"    action: {action_preview}")


def _save_trajectories(path: str, trajectories: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
    logger.info("Saved trajectories to %s", path)


def _print_reward_stats(trajectories: list):
    rewards = [float(t.reward) for t in trajectories]
    if not rewards:
        print("\nReward stats: no trajectories.")
        return

    mean_reward = sum(rewards) / len(rewards)
    min_reward = min(rewards)
    max_reward = max(rewards)
    std_reward = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    positive_ratio = sum(1 for r in rewards if r > 0) / len(rewards)

    print("\n=== Reward Statistics ===")
    print(f"count          : {len(rewards)}")
    print(f"mean_reward    : {mean_reward:.6f}")
    print(f"std_reward     : {std_reward:.6f}")
    print(f"min_reward     : {min_reward:.6f}")
    print(f"max_reward     : {max_reward:.6f}")
    print(f"positive_ratio : {positive_ratio:.2%}")


async def _run(args):
    _maybe_set_local_no_proxy(args.vllm_url, args.no_proxy_local)

    tasks = _pick_tasks(
        dataset_path=args.dataset_path,
        split=args.split,
        scenario=args.scenario,
        num_scenarios=args.num_scenarios,
        tasks_per_scenario=args.tasks_per_scenario,
        max_tasks=args.max_tasks,
    )
    if not tasks:
        raise RuntimeError("No matching AWM tasks found. Please check dataset_path/scenario.")

    print("Loaded tasks:")
    for i, t in enumerate(tasks, 1):
        print(f"  {i}. scenario={t.get('scenario')} | {_format_task_preview(t)}")

    tokenizer_name = args.tokenizer_name_or_path or args.model
    logger.info("Loading tokenizer: %s", tokenizer_name)
    try:
        from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tokenizer from '{tokenizer_name}'. "
            "Please provide --tokenizer_name_or_path with a valid local path or HF model id."
        ) from e

    n_parallel = min(args.n_parallel_agents, len(tasks))
    env_args = {
        "reward_fn": awm_reward_fn if args.enable_reward else None,
        "server_host": "127.0.0.1",
        "server_start_timeout": args.server_start_timeout,
    }
    agent_args = {
        "system_prompt": AWM_SYSTEM_PROMPT,
        "parser_name": "qwen",
        "max_steps": args.max_steps,
    }
    rollout_engine_args = {
        "model": args.model,
        "base_url": args.vllm_url,
        "api_key": args.api_key,
    }
    sampling_params = {
        "temperature": args.temperature,
        "max_tokens": args.max_new_tokens,
    }

    engine = AgentExecutionEngine(
        engine_name="openai",
        tokenizer=tokenizer,
        n_parallel_agents=n_parallel,
        trajectory_timeout=args.trajectory_timeout,
        max_steps=args.max_steps,
        max_workers=args.max_workers,
        agent_class=AWMAgent,
        env_class=AWMEnvironment,
        agent_args=agent_args,
        env_args=env_args,
        rollout_engine_args=rollout_engine_args,
        sampling_params=sampling_params,
    )
    # Keep interaction path consistent with examples.awm.test_single_scenario:
    # force OpenAI-compatible chat/completions instead of completions endpoint.
    if args.force_chat_completions:
        engine.rollout_engine._use_chat_completions = True

    # Break before every model request during env-agent interaction.
    # This captures each step where the engine asks the Agent/LLM for a reply.
    original_get_model_response = engine.get_model_response

    async def _debug_get_model_response(prompt, application_id, **kwargs):
        pdb.set_trace()
        return await original_get_model_response(prompt, application_id, **kwargs)

    engine.get_model_response = _debug_get_model_response

    try:
        trajectories = await engine.execute_tasks(tasks)
    finally:
        engine.shutdown()

    print(f"\nCompleted trajectories: {len(trajectories)}")
    _print_reward_stats(trajectories)
    if args.print_trajectories:
        for i, traj in enumerate(trajectories, 1):
            _print_trajectory_summary(traj, i)

    if args.save_jsonl:
        _save_trajectories(args.save_jsonl, trajectories)


def build_parser():
    parser = argparse.ArgumentParser(description="Run RLLM AWMAgent <-> AWMEnvironment interaction")
    parser.add_argument("--dataset_path", required=True, help="Local AWM dataset path")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--scenario", default=None, help="Exact scenario name (e.g. booking_marketplace_1)")
    parser.add_argument("--num_scenarios", type=int, default=1, help="How many scenarios to sample before filtering")
    parser.add_argument("--tasks_per_scenario", type=int, default=1, help="Tasks sampled per scenario")
    parser.add_argument("--max_tasks", type=int, default=1, help="Final max tasks to execute")

    parser.add_argument("--vllm_url", required=True, help="OpenAI-compatible vLLM URL, e.g. http://127.0.0.1:8803/v1")
    parser.add_argument("--model", required=True, help="Served model name in vLLM")
    parser.add_argument(
        "--tokenizer_name_or_path",
        default="",
        help="Tokenizer name/path for RLLM parser & token accounting (defaults to --model)",
    )
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"), help="API key for OpenAI-compatible endpoint")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max tokens per model response")
    parser.add_argument(
        "--force_chat_completions",
        action="store_true",
        default=True,
        help="Force /v1/chat/completions path for model calls",
    )

    parser.add_argument("--max_steps", type=int, default=10, help="Max env interaction steps")
    parser.add_argument("--trajectory_timeout", type=int, default=900, help="Per-trajectory timeout seconds")
    parser.add_argument("--server_start_timeout", type=float, default=120.0, help="AWM server start timeout")
    parser.add_argument("--n_parallel_agents", type=int, default=1, help="Parallel active agents")
    parser.add_argument("--max_workers", type=int, default=4, help="ThreadPool workers for env ops")
    parser.add_argument("--enable_reward", action="store_true", help="Enable awm_reward_fn")
    parser.add_argument("--print_trajectories", action="store_true", help="Print per-trajectory full summary")
    parser.add_argument("--no_proxy_local", action="store_true", help="Force NO_PROXY for localhost/127.0.0.1")
    parser.add_argument("--save_jsonl", default="", help="Optional output path for trajectory JSONL")
    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
