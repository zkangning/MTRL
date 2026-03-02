"""
AWM Single Scenario Diagnostic Test

Launches a single AWM scenario server, inspects the MCP tool list at every layer,
and optionally runs one agent trajectory — all within the rllm framework.

Usage:
    # Basic: load first scenario from local AWM dataset, diagnose MCP tools
    python -m examples.awm.test_single_scenario \
        --dataset_path /path/to/awm_data \
        --scenario news_portal_1

    # Full trajectory with a vLLM server (optional)
    python -m examples.awm.test_single_scenario \
        --dataset_path /path/to/awm_data \
        --scenario news_portal_1 \
        --run_trajectory \
        --vllm_url http://localhost:8001/v1 \
        --model Qwen3-4B
"""

import argparse
import asyncio
import contextlib
import io
import json
import logging
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("awm_diag")

from awm.core.db import create_sqlite_database
from awm.tools import (
    normalize_scenario_name,
    get_random_available_port,
    tools_jsonl_save,
    tools_jsonl_load,
)


# ─────────────────────────── Data Loading ────────────────────────────

def load_single_scenario(dataset_path: str, target_scenario: str | None = None):
    """Load a single scenario's data from the AWM dataset directory."""
    def _load_jsonl(filename):
        filepath = os.path.join(dataset_path, filename)
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return []
        data = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    scenarios = _load_jsonl("gen_scenario.jsonl")
    envs = _load_jsonl("gen_envs.jsonl")
    dbs = _load_jsonl("gen_db.jsonl")
    samples = _load_jsonl("gen_sample.jsonl")
    tasks_raw = _load_jsonl("gen_tasks.jsonl")
    verifiers = _load_jsonl("gen_verifier.pure_code.jsonl")

    envs_map = {normalize_scenario_name(e["scenario"]): e for e in envs}
    dbs_map = {normalize_scenario_name(d["scenario"]): d for d in dbs}
    samples_map = {normalize_scenario_name(s["scenario"]): s for s in samples}
    tasks_map = {}
    for t in tasks_raw:
        key = normalize_scenario_name(t["scenario"])
        tasks_map.setdefault(key, []).extend(
            t["tasks"] if isinstance(t.get("tasks"), list) else [str(t.get("tasks", ""))]
        )
    verifiers_map = {}
    for v in verifiers:
        verifiers_map[(normalize_scenario_name(v["scenario"]), v["task"])] = v

    if target_scenario:
        norm = normalize_scenario_name(target_scenario)
    else:
        norm = normalize_scenario_name(scenarios[0]["name"]) if scenarios else None

    if norm is None or norm not in envs_map:
        available = list(envs_map.keys())[:20]
        raise ValueError(
            f"Scenario '{target_scenario}' not found. Available (first 20): {available}"
        )

    env_data = envs_map[norm]
    db_data = dbs_map.get(norm, {})
    sample_data = samples_map.get(norm, {})
    scenario_tasks = tasks_map.get(norm, ["Explore the environment"])
    task_desc = scenario_tasks[0]
    verifier_data = verifiers_map.get((norm, task_desc), {})

    return {
        "scenario": env_data["scenario"],
        "env_code": env_data.get("full_code", ""),
        "db_schema": db_data.get("db_schema", {}),
        "db_sample": sample_data.get("sample_data", {}),
        "task": task_desc,
        "verifier_code": verifier_data.get("verification", {}).get("code", ""),
    }


# ───────────────────── Layer-by-layer MCP diagnosis ──────────────────

def diagnose_raw_http(host: str, port: int):
    """Layer 1: Raw HTTP probes — health, openapi, mcp."""
    base = f"http://{host}:{port}"
    endpoints = [
        ("/awm_health", "Health endpoint"),
        ("/openapi.json", "OpenAPI schema (outer app)"),
        ("/docs", "Swagger UI (outer app)"),
        ("/mcp", "MCP endpoint (POST expected, GET may 405)"),
    ]
    print("\n" + "=" * 70)
    print("LAYER 1: Raw HTTP Probes")
    print("=" * 70)
    for path, label in endpoints:
        url = base + path
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read(2000).decode("utf-8", errors="replace")
                print(f"  [{resp.status}] {label:40s} {url}")
                if path == "/openapi.json":
                    try:
                        schema = json.loads(body)
                        paths = list(schema.get("paths", {}).keys())
                        print(f"         OpenAPI paths ({len(paths)}): {paths[:15]}{'...' if len(paths) > 15 else ''}")
                    except json.JSONDecodeError:
                        print(f"         (not valid JSON)")
                elif path == "/awm_health":
                    print(f"         Body: {body[:200]}")
        except urllib.error.HTTPError as e:
            print(f"  [{e.code}] {label:40s} {url}  reason={e.reason}")
        except Exception as e:
            print(f"  [ERR]  {label:40s} {url}  {e}")


def diagnose_awm_native(host: str, port: int):
    """Layer 2: AWM native check_mcp_server (same code as awm/tools.py)."""
    from awm.tools import check_mcp_server

    print("\n" + "=" * 70)
    print("LAYER 2: AWM native check_mcp_server()")
    print("=" * 70)
    mcp_url = f"http://{host}:{port}/mcp"
    running, count, tools, err = asyncio.run(check_mcp_server(mcp_url, timeout=15))
    print(f"  Running : {running}")
    print(f"  Tools   : {count}")
    print(f"  Error   : {err}")
    if tools:
        for i, t in enumerate(tools[:10], 1):
            print(f"    {i}. {t.get('name', '?'):40s} {(t.get('description') or '')[:60]}")
        if len(tools) > 10:
            print(f"    ... and {len(tools) - 10} more")
    return tools


def diagnose_rllm_executor(host: str, port: int):
    """Layer 3: rllm's _ThreadSafeMCPExecutor (the one used during training)."""
    from rllm.environments.awm.awm_env import _ThreadSafeMCPExecutor

    print("\n" + "=" * 70)
    print("LAYER 3: rllm _ThreadSafeMCPExecutor (used during training)")
    print("=" * 70)
    mcp_url = f"http://{host}:{port}/mcp"
    executor = _ThreadSafeMCPExecutor(mcp_url)

    loop = asyncio.new_event_loop()
    import threading
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()

    try:
        future = asyncio.run_coroutine_threadsafe(executor.list_tools(), loop)
        tools = future.result(timeout=30)
        print(f"  Tools returned: {len(tools)}")
        if tools:
            for i, tool in enumerate(tools[:10], 1):
                print(f"    {i}. {tool.get('name', '?'):40s} {(tool.get('description') or '')[:60]}")
            if len(tools) > 10:
                print(f"    ... and {len(tools) - 10} more")
        else:
            print("  *** EMPTY — this is the bug that causes server kills ***")
        return tools
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        return []
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=5)
        loop.close()


def diagnose_awm_environment(scenario_data: dict):
    """Layer 4: Full AWMEnvironment.reset() — the real training path."""
    from rllm.environments.awm.awm_env import AWMEnvironment

    print("\n" + "=" * 70)
    print("LAYER 4: AWMEnvironment.reset() (full training path)")
    print("=" * 70)

    env = AWMEnvironment(
        scenario_name=scenario_data["scenario"],
        task_description=scenario_data["task"],
        env_code=scenario_data["env_code"],
        db_schema=scenario_data["db_schema"],
        db_sample=scenario_data["db_sample"],
        verifier_code=scenario_data["verifier_code"],
        max_steps=10,
        server_start_timeout=60.0,
    )

    try:
        obs, info = env.reset()
        print(f"  Reset successful!")
        print(f"  Scenario : {info.get('scenario')}")
        print(f"  Task     : {info.get('task', '')[:100]}")
        print(f"  Port     : {env.server_port}")
        print(f"  Server PID: {env.server_process.pid if env.server_process else 'N/A'}")

        # list_tools via the environment's tool execution path
        from awm.core.agent import format_tools_for_response
        print("\n  --- Calling list_tools via AWMEnvironment._execute_tool_call ---")
        result = env._execute_tool_call({
            "id": "diag_list_tools",
            "name": "list_tools",
            "arguments": {},
        })
        tools_text = result.get("result", "")
        success = result.get("success", False)
        print(f"  Success  : {success}")
        print(f"  Tools text length: {len(tools_text)} chars")
        if tools_text:
            preview = tools_text[:500]
            print(f"  Preview:\n{preview}")
        return env
    except Exception as e:
        logger.error(f"AWMEnvironment.reset() failed: {e}", exc_info=True)
        env.close()
        return None


# ──────────────────── Optional: run agent trajectory ─────────────────

def run_simple_trajectory(env, vllm_url: str, model: str, max_steps: int = 5):
    """Run a simple agent trajectory using vLLM for inference."""
    print("\n" + "=" * 70)
    print(f"TRAJECTORY: Running {max_steps} steps with {model}")
    print("=" * 70)

    from openai import OpenAI
    from rllm.agents.awm_prompts import AWM_SYSTEM_PROMPT

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), base_url=vllm_url)

    messages = [
        {"role": "system", "content": AWM_SYSTEM_PROMPT},
        {"role": "user", "content": env.task_description},
    ]

    for step in range(1, max_steps + 1):
        print(f"\n--- Step {step}/{max_steps} ---")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2048,
                temperature=0.6,
                extra_body={
                    "add_generation_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            )
            content = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"  vLLM error: {e}")
            break

        messages.append({"role": "assistant", "content": content})
        preview = content[:300] + "..." if len(content) > 300 else content
        print(f"  Model ({len(content)} chars): {preview}")

        # Parse and execute tool calls
        tool_calls = env._parse_tool_calls(content)
        if not tool_calls:
            print("  No tool calls — agent finished.")
            break

        for tc in tool_calls:
            print(f"  Executing: {tc['name']}({json.dumps(tc.get('arguments', {}), ensure_ascii=False)[:150]})")
            result = env._execute_tool_call(tc)
            print(f"  Result ({len(str(result.get('result', ''))):>5} chars, success={result.get('success')})")
            result_preview = str(result.get("result", ""))[:300]
            print(f"  {result_preview}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": str(result.get("result", "")),
            })


# ──────────────────────── Manual server start ────────────────────────

def start_server_manually(scenario_data: dict, host="127.0.0.1"):
    """Start the AWM server manually (same path as AWMEnvironment._start_server)
    but keep it running so we can diagnose interactively."""

    scenario_norm = normalize_scenario_name(scenario_data["scenario"])
    temp_dir = tempfile.mkdtemp(prefix=f"awm_diag_{scenario_norm}_")
    logger.info(f"Temp dir: {temp_dir}")

    # Prepare database
    db_schema = scenario_data["db_schema"]
    db_sample = scenario_data["db_sample"]
    if db_schema:
        full_schema = db_schema.copy() if isinstance(db_schema, dict) else json.loads(db_schema)
        if db_sample:
            sample = db_sample if isinstance(db_sample, dict) else json.loads(db_sample)
            for table in full_schema.get("tables", []):
                tname = table.get("name")
                if tname and tname in sample:
                    table["examples"] = sample[tname]
        db_path, ok, failed, errors = create_sqlite_database(
            scenario_data["scenario"], full_schema, temp_dir
        )
        if failed > 0:
            logger.warning(f"DB creation had {failed} failures: {errors}")
    else:
        raise ValueError("No db_schema provided")

    # Write env_config
    env_config = {
        "scenario": scenario_data["scenario"],
        "db_path": db_path,
        "full_code": scenario_data["env_code"],
    }
    temp_env_json = os.path.join(temp_dir, "env_config.jsonl")
    tools_jsonl_save([env_config], temp_env_json)

    port = get_random_available_port()
    temp_server_path = os.path.join(temp_dir, "temp_server.py")
    server_log_path = os.path.join(temp_dir, "server.log")
    log_file = open(server_log_path, "w")

    logger.info(f"Starting server: scenario={scenario_norm}, port={port}")

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "awm.core.server",
            "--port", str(port),
            "--scenario", scenario_norm,
            "--db_path", db_path,
            "--temp_server_path", temp_server_path,
            "--envs_load_path", temp_env_json,
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )

    logger.info(f"Server PID: {proc.pid}, waiting for startup...")

    # Wait for server readiness
    start = time.time()
    ready = False
    while time.time() - start < 60:
        if proc.poll() is not None:
            log_file.flush()
            with open(server_log_path, "r") as f:
                logger.error(f"Server crashed! Log:\n{f.read()[:3000]}")
            return None, None, temp_dir, port

        try:
            url = f"http://{host}:{port}/awm_health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            pass
        time.sleep(0.5)

    if not ready:
        logger.warning("Health check didn't return 200, checking TCP fallback...")
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                s.connect((host, port))
                ready = True
                logger.info("TCP port is open — proceeding with diagnosis")
        except Exception:
            logger.error("Server not reachable")
            return None, None, temp_dir, port

    elapsed = time.time() - start
    logger.info(f"Server ready in {elapsed:.1f}s on port {port}")
    return proc, log_file, temp_dir, port


# ──────────────────────────── Main ───────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AWM Single Scenario Diagnostic")
    parser.add_argument("--dataset_path", required=True, help="Path to AWM dataset directory")
    parser.add_argument("--scenario", default=None, help="Scenario name (default: first available)")
    parser.add_argument("--run_trajectory", action="store_true", help="Run an agent trajectory")
    parser.add_argument("--use_awm_env", action="store_true", help="Use AWMEnvironment (Layer 4) instead of manual server")
    parser.add_argument("--vllm_url", default="http://localhost:8001/v1", help="vLLM API URL")
    parser.add_argument("--model", default="Qwen3-4B", help="Model name for trajectory")
    parser.add_argument("--max_steps", type=int, default=5, help="Max trajectory steps")
    parser.add_argument("--keep_alive", action="store_true", help="Keep server running after diagnosis")
    args = parser.parse_args()

    print("=" * 70)
    print("AWM SINGLE SCENARIO DIAGNOSTIC")
    print("=" * 70)

    # Load scenario data
    scenario_data = load_single_scenario(args.dataset_path, args.scenario)
    print(f"  Scenario  : {scenario_data['scenario']}")
    print(f"  Task      : {scenario_data['task'][:120]}")
    print(f"  Code size : {len(scenario_data['env_code'])} chars")
    print(f"  DB schema : {type(scenario_data['db_schema']).__name__} ({len(json.dumps(scenario_data['db_schema']))} chars)")
    print(f"  Verifier  : {'yes' if scenario_data['verifier_code'] else 'no'} ({len(scenario_data['verifier_code'])} chars)")

    if args.use_awm_env:
        # Layer 4: Use the full AWMEnvironment (real training path)
        env = diagnose_awm_environment(scenario_data)
        if env and args.run_trajectory:
            run_simple_trajectory(env, args.vllm_url, args.model, args.max_steps)
        if env:
            if args.keep_alive:
                print(f"\n  Server running on port {env.server_port}. Press Ctrl+C to stop.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
            env.close()
        return

    # Manual server start + layer-by-layer diagnosis
    proc, log_file, temp_dir, port = start_server_manually(scenario_data)
    if proc is None:
        print("\n*** Server failed to start. Check log above. ***")
        print(f"  Temp dir preserved: {temp_dir}")
        return

    host = "127.0.0.1"
    try:
        # Layer 1: Raw HTTP
        diagnose_raw_http(host, port)

        # Layer 2: AWM native check_mcp_server
        native_tools = diagnose_awm_native(host, port)

        # Layer 3: rllm _ThreadSafeMCPExecutor
        rllm_tools = diagnose_rllm_executor(host, port)

        # Summary
        print("\n" + "=" * 70)
        print("DIAGNOSIS SUMMARY")
        print("=" * 70)
        print(f"  AWM native tools : {len(native_tools) if native_tools else 0}")
        print(f"  rllm executor    : {len(rllm_tools) if rllm_tools else 0}")

        if native_tools and not rllm_tools:
            print("\n  *** BUG: AWM native finds tools but rllm executor does NOT ***")
            print("  Root cause is likely in _ThreadSafeMCPExecutor or event loop handling.")
        elif not native_tools and not rllm_tools:
            print("\n  *** Both layers return empty. Issue is in FastApiMCP tool extraction ***")
            print("  Check the generated code's OpenAPI schema and FastApiMCP compatibility.")
        elif native_tools and rllm_tools:
            print("\n  Both layers work. MCP tool discovery is functional for this scenario.")

        if args.run_trajectory and rllm_tools:
            from rllm.environments.awm.awm_env import AWMEnvironment
            print("\n  (For trajectory, use --use_awm_env flag instead)")

        if args.keep_alive:
            print(f"\n  Server running on port {port}. Press Ctrl+C to stop.")
            print(f"  You can test manually:")
            print(f"    curl http://{host}:{port}/awm_health")
            print(f"    curl http://{host}:{port}/openapi.json | python -m json.tool | head -50")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass

    finally:
        # Cleanup
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
        if log_file:
            log_file.close()

        print(f"\n  Server log: {os.path.join(temp_dir, 'server.log')}")
        print(f"  Temp dir  : {temp_dir}")
        if not args.keep_alive:
            # Print server log tail for diagnosis
            log_path = os.path.join(temp_dir, "server.log")
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    log_content = f.read()
                if log_content:
                    print(f"\n--- SERVER LOG ({len(log_content)} chars) ---")
                    print(log_content[-3000:] if len(log_content) > 3000 else log_content)
                    print("--- END SERVER LOG ---")


if __name__ == "__main__":
    main()
