"""
AWM (Agentic World Model) Training Script for RLLM

This script trains an agent on AWM-generated virtual environments.
AWM provides 1000 diverse scenarios with simulated APIs and databases.

【独立管道】数据流说明：
  load_awm_dataset() 直接生成 verl 兼容的 dict 列表
  -> save_awm_parquet() 保存为 parquet 文件（绕过 DatasetRegistry / apply_verl_postprocessing）
  -> AgentTrainer 通过 config.data.train_files / val_files 直接读取
  -> RLHFDataset.__getitem__() 将 extra_info dict 存入 non_tensor_batch
  -> init_envs_and_agents() 展开 extra_info 传给 AWMEnvironment.from_dict()

Usage:
    python3 -m examples.awm.train_awm

Requirements:
    - HuggingFace dataset: Snowflake/AgenticWorldModel
    - AWM dependencies: mcp, mcp-agent, fastapi, uvicorn, sqlalchemy
"""

import hydra
import logging
import os
import json
import random
import copy
import tempfile
import numpy as np
import datasets as hf_datasets

from rllm.trainer.agent_trainer import AgentTrainer
from rllm.rewards.reward_fn import awm_reward_fn
from rllm.environments.awm import AWMEnvironment
from rllm.agents.awm_agent import AWMAgent
from rllm.agents.awm_prompts import AWM_SYSTEM_PROMPT
from rllm.data.utils import load_awm_dataset, set_task_config_manager
from rllm.config.task_config import TaskConfigManager
from awm.core.db import create_sqlite_database


# ============================================================
# Global Random Seed
# ============================================================
GLOBAL_SEED = 42


def set_all_random_seeds(seed: int = GLOBAL_SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


set_all_random_seeds(GLOBAL_SEED)
logger = logging.getLogger(__name__)


def save_awm_parquet(data: list[dict], output_path: str) -> str:
    """
    将 load_awm_dataset() 返回的数据直接保存为 verl 兼容的 parquet 文件。
    
    【关键】绕过 DatasetRegistry.apply_verl_postprocessing()，避免数据被二次包装。
    
    verl 的 RLHFDataset 需要 parquet 包含：
      - prompt: List[Dict] (chat messages)
      - extra_info: Dict (包含 index 和 AWM 特定字段)
      - data_source: str
    
    Args:
        data: load_awm_dataset() 返回的 dict 列表
        output_path: 输出 parquet 文件路径
        
    Returns:
        保存的文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 直接使用 HuggingFace Datasets 保存，它能正确处理嵌套 dict/list 结构
    hf_dataset = hf_datasets.Dataset.from_list(data)
    hf_dataset.to_parquet(output_path)
    
    logger.info(f"Saved {len(data)} records to {output_path}")
    return output_path


def _normalize_db_sample_examples(db_sample: object) -> dict[str, list[str]]:
    """Normalize db_sample into table_name -> list[SQL] format."""
    if not isinstance(db_sample, dict):
        return {}

    table_examples: dict[str, list[str]] = {}

    # Format 1: direct map {"users": ["INSERT ...", ...], ...}
    direct_keys = [k for k, v in db_sample.items() if isinstance(k, str) and isinstance(v, list)]
    if direct_keys and "tables" not in db_sample:
        for table_name in direct_keys:
            values = [str(sql) for sql in db_sample.get(table_name, []) if isinstance(sql, str)]
            if values:
                table_examples[table_name] = values
        return table_examples

    # Format 2: {"tables": [{"table_name": "...", "insert_statements": [...]}, ...]}
    tables = db_sample.get("tables", [])
    if isinstance(tables, list):
        for table in tables:
            if not isinstance(table, dict):
                continue
            table_name = table.get("table_name") or table.get("name")
            if not isinstance(table_name, str) or not table_name:
                continue

            statements = table.get("insert_statements")
            if not isinstance(statements, list):
                statements = table.get("examples")
            if not isinstance(statements, list):
                continue

            values = [str(sql) for sql in statements if isinstance(sql, str)]
            if values:
                table_examples[table_name] = values

    return table_examples


def _extract_extra_info(record: dict) -> dict:
    """Extract extra_info from verl record, tolerating string/dict formats."""
    extra_info = record.get("extra_info", {})
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except json.JSONDecodeError:
            return {}
    return extra_info if isinstance(extra_info, dict) else {}


def _parse_json_field(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def precheck_and_filter_awm_records(
    records: list[dict],
    split_name: str,
    max_failed_tables: int = 0,
) -> tuple[list[dict], dict]:
    """
    Pre-check SQLite DB build per scenario and filter out problematic scenarios.

    This pre-check validates whether db_schema + db_sample can be built via
    create_sqlite_database(). If failed table count exceeds max_failed_tables,
    all tasks from that scenario are removed.
    """
    scenario_records: dict[str, list[dict]] = {}
    for rec in records:
        extra_info = _extract_extra_info(rec)
        scenario = str(extra_info.get("scenario", "")).strip()
        if not scenario:
            continue
        scenario_records.setdefault(scenario, []).append(rec)

    if not scenario_records:
        logger.warning(f"[AWM DB precheck:{split_name}] No valid scenarios found in records; skip precheck.")
        return records, {"kept": 0, "dropped": 0, "errors": {}}

    passed_scenarios: set[str] = set()
    failed_reasons: dict[str, str] = {}

    for scenario, recs in scenario_records.items():
        sample_rec = recs[0]
        extra_info = _extract_extra_info(sample_rec)

        db_schema = _parse_json_field(extra_info.get("db_schema", {}))
        db_sample = _parse_json_field(extra_info.get("db_sample", {}))

        if not isinstance(db_schema, dict):
            failed_reasons[scenario] = "db_schema is not a valid dict/json"
            continue

        try:
            full_schema = copy.deepcopy(db_schema)
            table_examples = _normalize_db_sample_examples(db_sample)
            if table_examples:
                for table in full_schema.get("tables", []):
                    table_name = table.get("name")
                    if table_name and table_name in table_examples:
                        table["examples"] = table_examples[table_name]

            with tempfile.TemporaryDirectory(prefix=f"awm_precheck_{split_name}_") as tmp_dir:
                _, successful, failed, errors = create_sqlite_database(
                    scenario, full_schema, tmp_dir
                )

            if failed <= max_failed_tables:
                passed_scenarios.add(scenario)
            else:
                reason = (
                    f"failed_tables={failed}, successful_tables={successful}, "
                    f"threshold={max_failed_tables}, errors={list(errors)[:2]}"
                )
                failed_reasons[scenario] = reason
        except Exception as e:
            failed_reasons[scenario] = f"exception during precheck: {e}"

    filtered = []
    for rec in records:
        extra_info = _extract_extra_info(rec)
        scenario = str(extra_info.get("scenario", "")).strip()
        if scenario in passed_scenarios:
            filtered.append(rec)

    logger.info(
        f"[AWM DB precheck:{split_name}] scenarios_total={len(scenario_records)}, "
        f"scenarios_kept={len(passed_scenarios)}, scenarios_dropped={len(failed_reasons)}, "
        f"records_before={len(records)}, records_after={len(filtered)}"
    )

    if failed_reasons:
        preview = list(failed_reasons.items())[:5]
        logger.warning(f"[AWM DB precheck:{split_name}] dropped scenario preview: {preview}")

    return filtered, {
        "kept": len(passed_scenarios),
        "dropped": len(failed_reasons),
        "errors": failed_reasons,
    }


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    """Main training function."""

    # ============================================================
    # Task Config Manager (支持 +task_configs.awm.xxx=yyy 覆盖)
    # ============================================================
    task_configs = config.get("task_configs", {})
    custom_task_configs = {}
    if task_configs:
        for task_type, params in task_configs.items():
            if isinstance(params, dict):
                custom_task_configs[task_type] = dict(params)

    task_config_manager = TaskConfigManager(custom_task_configs)
    set_task_config_manager(task_config_manager)
    logger.info("\n" + task_config_manager.summary())

    # ============================================================
    # AWM Data Configuration (通过 +data.xxx 覆盖)
    # ============================================================
    dataset_path = config.data.get("dataset_path", "Snowflake/AgenticWorldModel")
    train_scenarios = config.data.get("train_scenarios", 100)
    test_scenarios = config.data.get("test_scenarios", 20)
    tasks_per_scenario = config.data.get("tasks_per_scenario", 10)
    verification_mode = config.data.get("verification_mode", "pure_code")
    precheck_db = bool(config.data.get("precheck_db", False))
    precheck_max_failed_tables = int(config.data.get("precheck_max_failed_tables", 0))

    logger.info(">>> AWM Training Configuration:")
    logger.info(f"  Dataset: {dataset_path}")
    logger.info(f"  Train scenarios: {train_scenarios}")
    logger.info(f"  Test scenarios: {test_scenarios}")
    logger.info(f"  Tasks per scenario: {tasks_per_scenario}")
    logger.info(f"  Verification mode: {verification_mode}")
    logger.info(f"  DB precheck enabled: {precheck_db}")
    if precheck_db:
        logger.info(f"  DB precheck max_failed_tables: {precheck_max_failed_tables}")

    # ============================================================
    # Prepare Dataset — 独立管道，直接生成 verl parquet
    # ============================================================
    output_dir = os.path.join(
        config.data.get("output_dir", "/tmp/awm_data"),
        f"s{train_scenarios}_t{tasks_per_scenario}_{verification_mode}"
    )

    train_parquet_path = os.path.join(output_dir, "train.parquet")
    val_parquet_path = os.path.join(output_dir, "val.parquet")

    # 加载训练数据
    logger.info(">>> Loading AWM training data...")
    train_data = load_awm_dataset(
        dataset_path=dataset_path,
        split="train",
        num_scenarios=train_scenarios,
        tasks_per_scenario=tasks_per_scenario,
        verification_mode=verification_mode,
    )

    # 加载验证数据
    logger.info(">>> Loading AWM validation data...")
    val_data = load_awm_dataset(
        dataset_path=dataset_path,
        split="test",
        num_scenarios=test_scenarios,
        tasks_per_scenario=tasks_per_scenario,
        verification_mode=verification_mode,
    )

    if precheck_db:
        logger.info(">>> Running database precheck for train split...")
        train_data, train_precheck_report = precheck_and_filter_awm_records(
            train_data, split_name="train", max_failed_tables=precheck_max_failed_tables
        )
        logger.info(
            f">>> Train precheck done: kept={train_precheck_report['kept']}, "
            f"dropped={train_precheck_report['dropped']}"
        )

        logger.info(">>> Running database precheck for val split...")
        val_data, val_precheck_report = precheck_and_filter_awm_records(
            val_data, split_name="val", max_failed_tables=precheck_max_failed_tables
        )
        logger.info(
            f">>> Val precheck done: kept={val_precheck_report['kept']}, "
            f"dropped={val_precheck_report['dropped']}"
        )

        if not train_data:
            raise RuntimeError(
                "All training records were filtered out by DB precheck. "
                "Try increasing +data.precheck_max_failed_tables or disabling +data.precheck_db."
            )
        if not val_data:
            logger.warning(
                "All validation records were filtered out by DB precheck. "
                "Validation metrics may be unavailable."
            )

    save_awm_parquet(train_data, train_parquet_path)
    save_awm_parquet(val_data, val_parquet_path)
    

    logger.info(f">>> Dataset prepared: {len(train_data)} train, {len(val_data)} val samples")
    logger.info(f"  Train parquet: {train_parquet_path}")
    logger.info(f"  Val parquet:   {val_parquet_path}")

    # ============================================================
    # 直接设置 verl 数据路径（绕过 DatasetRegistry）
    # ============================================================
    config.data.train_files = train_parquet_path
    config.data.val_files = val_parquet_path

    # ============================================================
    # Agent & Environment Arguments
    # ============================================================
    agent_args = {
        "system_prompt": AWM_SYSTEM_PROMPT,
        "parser_name": "qwen",
        "max_steps": config.rllm.agent.get("max_steps", 30),
    }

    env_args = {
        "reward_fn": awm_reward_fn,
        "server_host": "127.0.0.1",
        "server_start_timeout": config.rllm.env.get("server_start_timeout", 120.0),
        "tool_call_timeout": config.rllm.env.get("tool_call_timeout", 30.0),
        "task_max_prompt_length": config.data.get("max_prompt_length"),
        "task_max_response_length": config.data.get("max_response_length"),
        "prestart_server": bool(config.rllm.env.get("prestart_server", False)),
    }

    # ============================================================
    # Initialize Trainer & Start Training
    # ============================================================
    trainer = AgentTrainer(
        agent_class=AWMAgent,
        env_class=AWMEnvironment,
        agent_args=agent_args,
        env_args=env_args,
        config=config,
        # 不传 train_dataset / val_dataset，已直接设置 config.data.train_files
    )

    logger.info(">>> Starting AWM Training...")
    trainer.train()


if __name__ == "__main__":
    main()
