"""
AWM Scenario Pre-validation Script

Validates AWM scenarios BEFORE training by:
  1. Testing SQLite database construction (db_schema + db_sample)
  2. Optionally testing MCP server startup + tool discovery

Outputs:
  - A JSON file listing valid/invalid scenarios with error details
  - Optionally, pre-filtered parquet files ready for training

Usage:
    # Database-only validation (fast, no server startup):
    python -m examples.awm.precheck_scenarios \
        --dataset_path /path/to/awm_data \
        --num_scenarios 100 \
        --tasks_per_scenario 10 \
        --output_dir /tmp/awm_precheck

    # Full validation including MCP server startup:
    python -m examples.awm.precheck_scenarios \
        --dataset_path /path/to/awm_data \
        --num_scenarios 100 \
        --tasks_per_scenario 10 \
        --output_dir /tmp/awm_precheck \
        --check_server

    # Then use the filtered parquet in training:
    #   +data.precheck_db=False  (already filtered, no need to re-check)
    #   config.data.train_files=/tmp/awm_precheck/.../train.parquet
"""

import argparse
import copy
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from awm.core.db import create_sqlite_database
from awm.tools import normalize_scenario_name, get_random_available_port, tools_jsonl_save

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _normalize_db_sample_examples(db_sample: Any) -> dict[str, list[str]]:
    """Normalize db_sample into table_name -> list[SQL] format."""
    if not isinstance(db_sample, dict):
        return {}

    table_examples: dict[str, list[str]] = {}

    direct_keys = [k for k, v in db_sample.items() if isinstance(k, str) and isinstance(v, list)]
    if direct_keys and "tables" not in db_sample:
        for table_name in direct_keys:
            values = [str(sql) for sql in db_sample.get(table_name, []) if isinstance(sql, str)]
            if values:
                table_examples[table_name] = values
        return table_examples

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


def _parse_json_field(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def check_database(
    scenario_name: str,
    db_schema: dict,
    db_sample: Any,
    max_failed_tables: int = 0,
) -> Tuple[bool, str]:
    """
    Test whether the database can be built successfully.

    Returns:
        (passed, reason): True if the scenario passed, else the failure reason.
    """
    if not isinstance(db_schema, dict):
        return False, "db_schema is not a valid dict"

    try:
        full_schema = copy.deepcopy(db_schema)
        table_examples = _normalize_db_sample_examples(db_sample)
        if table_examples:
            for table in full_schema.get("tables", []):
                table_name = table.get("name")
                if table_name and table_name in table_examples:
                    table["examples"] = table_examples[table_name]

        with tempfile.TemporaryDirectory(prefix=f"awm_precheck_db_") as tmp_dir:
            _, successful, failed, errors = create_sqlite_database(
                scenario_name, full_schema, tmp_dir
            )

        if failed <= max_failed_tables:
            return True, f"ok (tables_ok={successful}, tables_failed={failed})"
        else:
            err_preview = list(errors)[:3]
            return False, (
                f"failed_tables={failed}, successful_tables={successful}, "
                f"threshold={max_failed_tables}, errors={err_preview}"
            )
    except Exception as e:
        return False, f"exception: {e}"


def check_server(
    scenario_name: str,
    env_code: str,
    db_schema: dict,
    db_sample: Any,
    server_timeout: float = 60.0,
    mcp_timeout: float = 30.0,
) -> Tuple[bool, str]:
    """
    Test whether the MCP server can start and return tools.

    This performs a full lifecycle test:
      1. Build the database
      2. Write env_config jsonl
      3. Launch the server subprocess
      4. Wait for HTTP readiness
      5. Verify MCP tools can be listed
      6. Tear down everything

    Returns:
        (passed, reason)
    """
    import socket
    import urllib.request
    import urllib.error

    scenario_norm = normalize_scenario_name(scenario_name)
    tmp_dir = tempfile.mkdtemp(prefix=f"awm_precheck_srv_{scenario_norm}_")
    server_process = None

    try:
        # Build database
        full_schema = copy.deepcopy(db_schema)
        table_examples = _normalize_db_sample_examples(db_sample)
        if table_examples:
            for table in full_schema.get("tables", []):
                tname = table.get("name")
                if tname and tname in table_examples:
                    table["examples"] = table_examples[tname]

        db_path, _, _, _ = create_sqlite_database(scenario_name, full_schema, tmp_dir)

        # Write env config
        env_config = {
            "scenario": scenario_name,
            "db_path": db_path,
            "full_code": env_code,
        }
        env_jsonl = os.path.join(tmp_dir, "env_config.jsonl")
        tools_jsonl_save([env_config], env_jsonl)

        # Allocate port
        port = get_random_available_port()
        temp_server_path = os.path.join(tmp_dir, "temp_server.py")
        log_path = os.path.join(tmp_dir, "server.log")

        # Launch server
        log_file = open(log_path, "w")
        server_process = subprocess.Popen(
            [
                sys.executable, '-m', 'awm.core.server',
                '--port', str(port),
                '--scenario', scenario_norm,
                '--db_path', db_path,
                '--temp_server_path', temp_server_path,
                '--envs_load_path', env_jsonl,
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )

        # Wait for TCP + HTTP readiness
        start = time.time()
        tcp_ready = False
        while time.time() - start < server_timeout:
            if server_process.poll() is not None:
                log_file.flush()
                with open(log_path) as f:
                    log_content = f.read()[-2000:]
                return False, f"server crashed (rc={server_process.returncode}): {log_content}"

            if not tcp_ready:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(1.0)
                        s.connect(("127.0.0.1", port))
                        tcp_ready = True
                except (socket.timeout, ConnectionRefusedError, OSError):
                    time.sleep(0.5)
                    continue

            try:
                url = f"http://127.0.0.1:{port}/awm_health"
                req = urllib.request.Request(url, method='GET')
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        break
            except (urllib.error.HTTPError, urllib.error.URLError, OSError, TimeoutError):
                pass
            time.sleep(0.5)
        else:
            return False, f"server not ready within {server_timeout}s"

        # Verify MCP tools
        try:
            from awm.tools import isolated_mcp_env
            from mcp_agent.app import MCPApp
            from mcp_agent.agents.agent import Agent
            from mcp_agent.config import (
                Settings, MCPSettings, MCPServerSettings, LoggerSettings,
            )
            import asyncio
            import contextlib
            import io

            mcp_url = f"http://127.0.0.1:{port}/mcp"
            with isolated_mcp_env():
                settings = Settings(
                    execution_engine="asyncio",
                    logger=LoggerSettings(
                        type="none", transports=["none"],
                        progress_display=False, level="error",
                    ),
                    mcp=MCPSettings(
                        servers={
                            "mcp_tool": MCPServerSettings(
                                transport="streamable_http", url=mcp_url,
                            ),
                        }
                    ),
                )

            app = MCPApp(name="precheck", settings=settings)

            async def _list():
                with contextlib.redirect_stderr(io.StringIO()):
                    async with app.run():
                        agent = Agent(name="executor", server_names=["mcp_tool"])
                        async with agent:
                            result = await asyncio.wait_for(agent.list_tools(), timeout=mcp_timeout)
                            return [t.name for t in result.tools]

            tools = asyncio.run(asyncio.wait_for(_list(), timeout=mcp_timeout + 10))
            if tools:
                return True, f"ok (tools={len(tools)})"
            else:
                return False, "MCP returned empty tools list"
        except Exception as e:
            return False, f"MCP verification failed: {e}"

    except Exception as e:
        return False, f"exception: {e}"
    finally:
        if server_process and server_process.poll() is None:
            try:
                os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
                server_process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
                except Exception:
                    pass
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def load_scenarios(dataset_path: str, num_scenarios: int) -> List[Dict]:
    """Load scenario data from AWM dataset (reuses logic from rllm.data.utils)."""
    is_local = os.path.isdir(dataset_path)

    def _load_jsonl(filename: str) -> List[Dict]:
        if is_local:
            filepath = os.path.join(dataset_path, filename)
            if not os.path.exists(filepath):
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

    scenarios = _load_jsonl("gen_scenario.jsonl")
    dbs = _load_jsonl("gen_db.jsonl")
    samples = _load_jsonl("gen_sample.jsonl")
    envs = _load_jsonl("gen_envs.jsonl")

    def _norm(name: str) -> str:
        return name.strip().lower()

    dbs_map = {_norm(d["scenario"]): d for d in dbs}
    samples_map = {_norm(s["scenario"]): s for s in samples}
    envs_map = {_norm(e["scenario"]): e for e in envs}

    import random
    if num_scenarios > 0 and num_scenarios < len(scenarios):
        rng = random.Random(42)
        scenarios = rng.sample(scenarios, num_scenarios)

    result = []
    for s in scenarios:
        name = s["name"]
        key = _norm(name)
        db_data = dbs_map.get(key)
        sample_data = samples_map.get(key)
        env_data = envs_map.get(key)
        if not all([db_data, sample_data, env_data]):
            logger.warning(f"Missing data for scenario {name}, skipping")
            continue
        result.append({
            "scenario": name,
            "db_schema": db_data.get("db_schema", {}),
            "db_sample": sample_data.get("sample_data", {}),
            "env_code": env_data.get("full_code", ""),
        })

    return result


def run_precheck(
    dataset_path: str,
    num_scenarios: int,
    max_failed_tables: int,
    do_check_server: bool,
    server_timeout: float,
    mcp_timeout: float,
    max_workers: int,
    output_dir: str,
    tasks_per_scenario: int,
    verification_mode: str,
):
    """Run the full pre-validation pipeline."""
    logger.info(f"Loading scenarios from {dataset_path}...")
    all_scenarios = load_scenarios(dataset_path, num_scenarios)
    logger.info(f"Loaded {len(all_scenarios)} scenarios")

    passed: List[str] = []
    failed_report: Dict[str, str] = {}

    # Phase 1: Database check (parallelized)
    logger.info("=" * 60)
    logger.info("Phase 1: Database construction check")
    logger.info("=" * 60)

    def _db_check_wrapper(scenario_data):
        name = scenario_data["scenario"]
        db_schema = _parse_json_field(scenario_data["db_schema"])
        db_sample = _parse_json_field(scenario_data["db_sample"])
        ok, reason = check_database(name, db_schema, db_sample, max_failed_tables)
        return name, ok, reason

    db_passed = []
    db_failed = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_db_check_wrapper, s): s for s in all_scenarios}
        for future in as_completed(futures):
            name, ok, reason = future.result()
            if ok:
                db_passed.append(name)
                logger.info(f"  [PASS] {name}: {reason}")
            else:
                db_failed.append(name)
                failed_report[name] = f"[DB] {reason}"
                logger.warning(f"  [FAIL] {name}: {reason}")

    logger.info(
        f"Phase 1 result: {len(db_passed)} passed, {len(db_failed)} failed "
        f"out of {len(all_scenarios)} total"
    )

    # Phase 2: Server check (optional, sequential to avoid port conflicts)
    if do_check_server:
        logger.info("=" * 60)
        logger.info("Phase 2: MCP server startup check")
        logger.info("=" * 60)

        scenarios_for_server = [
            s for s in all_scenarios if s["scenario"] in db_passed
        ]

        server_passed = []
        server_failed = []
        for i, scenario_data in enumerate(scenarios_for_server):
            name = scenario_data["scenario"]
            logger.info(f"  [{i+1}/{len(scenarios_for_server)}] Testing server for {name}...")
            db_schema = _parse_json_field(scenario_data["db_schema"])
            db_sample = _parse_json_field(scenario_data["db_sample"])
            env_code = scenario_data["env_code"]

            ok, reason = check_server(
                name, env_code, db_schema, db_sample,
                server_timeout=server_timeout,
                mcp_timeout=mcp_timeout,
            )
            if ok:
                server_passed.append(name)
                logger.info(f"  [PASS] {name}: {reason}")
            else:
                server_failed.append(name)
                failed_report[name] = f"[SERVER] {reason}"
                logger.warning(f"  [FAIL] {name}: {reason}")

        passed = server_passed
        logger.info(
            f"Phase 2 result: {len(server_passed)} passed, {len(server_failed)} failed "
            f"out of {len(scenarios_for_server)} tested"
        )
    else:
        passed = db_passed

    # Save report
    os.makedirs(output_dir, exist_ok=True)
    report = {
        "total_scenarios": len(all_scenarios),
        "passed_count": len(passed),
        "failed_count": len(failed_report),
        "max_failed_tables": max_failed_tables,
        "check_server": do_check_server,
        "passed_scenarios": sorted(passed),
        "failed_scenarios": failed_report,
    }
    report_path = os.path.join(output_dir, "precheck_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Report saved to {report_path}")

    # Save valid scenario list
    valid_list_path = os.path.join(output_dir, "valid_scenarios.json")
    with open(valid_list_path, "w") as f:
        json.dump(sorted(passed), f, indent=2, ensure_ascii=False)
    logger.info(f"Valid scenario list saved to {valid_list_path}")

    # Generate filtered parquet files for training
    logger.info("=" * 60)
    logger.info("Generating filtered parquet files...")
    logger.info("=" * 60)

    passed_set = set(passed)

    from rllm.data.utils import load_awm_dataset

    for split, n_scenarios in [("train", num_scenarios), ("test", max(20, num_scenarios // 5))]:
        logger.info(f"Loading {split} split (num_scenarios={n_scenarios})...")
        data = load_awm_dataset(
            dataset_path=dataset_path,
            split=split,
            num_scenarios=n_scenarios,
            tasks_per_scenario=tasks_per_scenario,
            verification_mode=verification_mode,
        )

        def _norm_name(name: str) -> str:
            return name.strip().lower()

        before = len(data)
        filtered = []
        for rec in data:
            extra_info = rec.get("extra_info", {})
            if isinstance(extra_info, str):
                try:
                    extra_info = json.loads(extra_info)
                except json.JSONDecodeError:
                    continue
            scenario = str(extra_info.get("scenario", "")).strip()
            if _norm_name(scenario) in {_norm_name(s) for s in passed_set}:
                filtered.append(rec)

        logger.info(f"  {split}: {before} -> {len(filtered)} records after filtering")

        if filtered:
            import datasets as hf_datasets
            parquet_subdir = os.path.join(
                output_dir, f"s{num_scenarios}_t{tasks_per_scenario}_{verification_mode}"
            )
            os.makedirs(parquet_subdir, exist_ok=True)
            parquet_path = os.path.join(parquet_subdir, f"{split}.parquet")
            hf_dataset = hf_datasets.Dataset.from_list(filtered)
            hf_dataset.to_parquet(parquet_path)
            logger.info(f"  Saved to {parquet_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("PRE-CHECK SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total scenarios:  {len(all_scenarios)}")
    logger.info(f"Passed:           {len(passed)}")
    logger.info(f"Failed:           {len(failed_report)}")
    logger.info(f"Pass rate:        {len(passed)/max(len(all_scenarios),1)*100:.1f}%")
    logger.info(f"Report:           {report_path}")
    logger.info(f"Valid list:       {valid_list_path}")
    if failed_report:
        logger.info(f"\nFailed scenarios (first 10):")
        for name, reason in list(failed_report.items())[:10]:
            logger.info(f"  {name}: {reason[:120]}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="AWM Scenario Pre-validation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to AWM dataset (local dir or HuggingFace path)")
    parser.add_argument("--num_scenarios", type=int, default=100,
                        help="Number of scenarios to validate")
    parser.add_argument("--tasks_per_scenario", type=int, default=10,
                        help="Tasks per scenario (for parquet generation)")
    parser.add_argument("--verification_mode", type=str, default="pure_code",
                        choices=["pure_code", "sql"],
                        help="Verification mode for parquet generation")
    parser.add_argument("--output_dir", type=str, default="/tmp/awm_precheck",
                        help="Output directory for reports and filtered data")
    parser.add_argument("--max_failed_tables", type=int, default=0,
                        help="Max allowed failed tables per scenario (0 = strict)")
    parser.add_argument("--check_server", action="store_true",
                        help="Also test MCP server startup (slower but more thorough)")
    parser.add_argument("--server_timeout", type=float, default=60.0,
                        help="Server startup timeout in seconds")
    parser.add_argument("--mcp_timeout", type=float, default=30.0,
                        help="MCP tool verification timeout in seconds")
    parser.add_argument("--max_workers", type=int, default=16,
                        help="Max parallel workers for database checks")
    args = parser.parse_args()

    run_precheck(
        dataset_path=args.dataset_path,
        num_scenarios=args.num_scenarios,
        max_failed_tables=args.max_failed_tables,
        do_check_server=args.check_server,
        server_timeout=args.server_timeout,
        mcp_timeout=args.mcp_timeout,
        max_workers=args.max_workers,
        output_dir=args.output_dir,
        tasks_per_scenario=args.tasks_per_scenario,
        verification_mode=args.verification_mode,
    )


if __name__ == "__main__":
    main()
