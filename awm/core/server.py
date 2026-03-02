from awm.tools import tools_jsonl_load, normalize_scenario_name
from dataclasses import dataclass
import os
import sys
import textwrap
import traceback
import time
from loguru import logger

@dataclass
class Config:
    scenario: str
    envs_load_path: str # specify a path to load generated_envs.jsonl
    db_path: str | None = None # specify a path to load the database file, xxx.db
    host: str = "127.0.0.1"
    port: int = 8001
    temp_server_path: str | None = None # specify a temp server path, the code will be written to this path


    def pre_process(self):
        self.scenario = normalize_scenario_name(self.scenario)
        assert os.path.exists(self.envs_load_path), f"Environment file {self.envs_load_path} not found"
        if self.temp_server_path is None:
            folder = os.path.dirname(self.envs_load_path)
            self.temp_server_path = os.path.join(folder, f"temp_server_{self.scenario.lower()}.py")


def format_raw_code_to_lines(raw_code: str, indent: int) -> list[str]:
    no_indent_code = textwrap.dedent(raw_code).strip()
    indent_code = textwrap.indent(no_indent_code, ' ' * indent)
    return indent_code.split("\n")

def run_server(args: Config):
    """
    Run the AWM server with detailed startup logging for debugging.
    """
    start_time = time.time()
    logger.info(f"=" * 60)
    logger.info(f"AWM Server Startup - Scenario: {args.scenario}")
    logger.info(f"=" * 60)
    logger.info(f"[STARTUP] PID: {os.getpid()}")
    logger.info(f"[STARTUP] Host: {args.host}, Port: {args.port}")
    logger.info(f"[STARTUP] Envs load path: {args.envs_load_path}")
    logger.info(f"[STARTUP] Temp server path: {args.temp_server_path}")
    logger.info(f"[STARTUP] DB path (initial): {args.db_path}")
    
    # Step 1: Load environment configuration
    logger.info(f"[STEP 1/5] Loading environment configuration...")
    try:
        envs = tools_jsonl_load(args.envs_load_path)
        logger.info(f"[STEP 1/5] Loaded {len(envs)} environment(s) from {args.envs_load_path}")
        envs = {normalize_scenario_name(e["scenario"]): e for e in envs}
        logger.info(f"[STEP 1/5] Available scenarios: {list(envs.keys())}")
        
        if args.scenario not in envs:
            logger.error(f"[STEP 1/5] FAILED: Scenario '{args.scenario}' not found in {list(envs.keys())}")
            raise KeyError(f"Scenario '{args.scenario}' not found")
        
        env = envs[args.scenario]
        logger.info(f"[STEP 1/5] SUCCESS: Loaded scenario '{args.scenario}'")
        logger.info(f"[STEP 1/5] Env keys: {list(env.keys())}")
        logger.info(f"[STEP 1/5] Code length: {len(env.get('full_code', ''))} chars")
    except Exception as e:
        logger.error(f"[STEP 1/5] FAILED: Error loading environment: {e}")
        logger.error(f"[STEP 1/5] Traceback:\n{traceback.format_exc()}")
        raise
    
    # Step 2: Prepare database
    logger.info(f"[STEP 2/5] Preparing database...")
    try:
        if args.db_path is None:
            args.db_path = env["db_path"]
            logger.info(f"[STEP 2/5] Using database from env config: {args.db_path}")
        else:
            logger.info(f"[STEP 2/5] Using provided database: {args.db_path}")
        
        if not os.path.exists(args.db_path):
            logger.error(f"[STEP 2/5] FAILED: Database file not found: {args.db_path}")
            raise FileNotFoundError(f"Database file {args.db_path} not found")
        
        db_size = os.path.getsize(args.db_path)
        logger.info(f"[STEP 2/5] SUCCESS: Database exists, size: {db_size} bytes")
    except Exception as e:
        logger.error(f"[STEP 2/5] FAILED: Database error: {e}")
        logger.error(f"[STEP 2/5] Traceback:\n{traceback.format_exc()}")
        raise

    # Step 3: Process and transform code
    logger.info(f"[STEP 3/5] Processing environment code...")
    try:
        code = env["full_code"]
        original_lines = len(code.split("\n"))
        
        # Add signal handling, resource monitoring, and shutdown logging at the beginning of the generated server
        signal_handling_code = '''import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Signal handling and resource monitoring for debugging shutdown reasons
import signal
import sys
import os
import atexit
import traceback
import threading
import time

# Use a mutable container to avoid global declaration issues
_awm_server_state = {
    "shutdown_reason": "unknown",
    "monitor_stop": False
}

def _set_shutdown_reason(reason):
    """Set the shutdown reason (called before uvicorn.run)."""
    _awm_server_state["shutdown_reason"] = reason

def _get_resource_info():
    """Get current resource usage information."""
    info = {}
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        info['max_rss_mb'] = rusage.ru_maxrss / 1024  # Convert to MB (on Linux it's KB)
        info['user_time'] = rusage.ru_utime
        info['sys_time'] = rusage.ru_stime
    except Exception as e:
        info['rusage_error'] = str(e)
    
    try:
        # Check memory from /proc/self/status
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    info['vm_rss_kb'] = line.split()[1]
                elif line.startswith('VmPeak:'):
                    info['vm_peak_kb'] = line.split()[1]
                elif line.startswith('Threads:'):
                    info['threads'] = line.split()[1]
    except Exception:
        pass
    
    try:
        # Check open file descriptors
        fd_count = len(os.listdir('/proc/self/fd'))
        info['open_fds'] = fd_count
    except Exception:
        pass
    
    try:
        # Check ulimits
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        info['fd_limit_soft'] = soft
        info['fd_limit_hard'] = hard
        soft_mem, hard_mem = resource.getrlimit(resource.RLIMIT_AS)
        info['mem_limit_soft'] = soft_mem if soft_mem != -1 else 'unlimited'
        info['mem_limit_hard'] = hard_mem if hard_mem != -1 else 'unlimited'
    except Exception:
        pass
    
    return info

def _resource_monitor_thread():
    """Background thread to periodically log resource usage."""
    last_log_time = time.time()
    while not _awm_server_state["monitor_stop"]:
        time.sleep(5)
        if time.time() - last_log_time >= 30:  # Log every 30 seconds
            info = _get_resource_info()
            print(f"[AWM_SERVER] Resource status: {info}", flush=True)
            last_log_time = time.time()

def _signal_handler(signum, frame):
    _awm_server_state["monitor_stop"] = True
    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    _awm_server_state["shutdown_reason"] = f"signal_{sig_name}"
    print(f"[AWM_SERVER] Received signal {sig_name} ({signum})", flush=True)
    print(f"[AWM_SERVER] Resource at signal: {_get_resource_info()}", flush=True)
    print(f"[AWM_SERVER] Stack trace at signal:", flush=True)
    traceback.print_stack(frame)
    sys.exit(0)

def _atexit_handler():
    _awm_server_state["monitor_stop"] = True
    print(f"[AWM_SERVER] Server shutdown - reason: {_awm_server_state['shutdown_reason']}", flush=True)
    print(f"[AWM_SERVER] PID: {os.getpid()}", flush=True)
    print(f"[AWM_SERVER] Final resource status: {_get_resource_info()}", flush=True)

# Register signal handlers
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)
atexit.register(_atexit_handler)

# Log initial resource info
print(f"[AWM_SERVER] Server process started, PID: {os.getpid()}", flush=True)
print(f"[AWM_SERVER] Initial resource status: {_get_resource_info()}", flush=True)

# Start resource monitor thread
_monitor_thread = threading.Thread(target=_resource_monitor_thread, daemon=True)
_monitor_thread.start()
'''
        new_code = [signal_handling_code]
        
        create_engine_found = False
        uvicorn_found = False
        
        for line in code.split("\n"):
            if 'create_engine(' in line:
                create_engine_found = True
                left = line.split('create_engine(')[0]
                sql_path = f"'sqlite:///{args.db_path}'"
                right = f"create_engine({sql_path}, connect_args={{'check_same_thread': False}})"
                line = f"{left}{right}"
                        
            if 'uvicorn.run(app' in line and not uvicorn_found:
                uvicorn_found = True
                
                # Detect the indentation of the original uvicorn.run line
                original_indent = len(line) - len(line.lstrip())
                indent_str = ' ' * original_indent
                
                # Generate the code to insert BEFORE uvicorn.run, using the same indentation
                pre_uvicorn_code = f'''
{indent_str}# AWM Server initialization
{indent_str}host = os.environ.get('HOST', '{args.host}')
{indent_str}port = os.environ.get('PORT', {args.port})
{indent_str}print(f'[AWM_SERVER] Server starting on host={{host}}, port={{port}}', flush=True)
{indent_str}
{indent_str}# Add a dedicated health endpoint that is independent of generated code
{indent_str}# This ensures health checks work reliably regardless of what the generated
{indent_str}# FastAPI app code does (middlewares, lifespan, error handlers, etc.)
{indent_str}@app.get("/awm_health", include_in_schema=False)
{indent_str}async def _awm_health_check():
{indent_str}    return {{"status": "ok"}}
{indent_str}
{indent_str}# Enable MCP server
{indent_str}from fastapi_mcp import FastApiMCP
{indent_str}mcp = FastApiMCP(app)
{indent_str}mcp.mount_http()
{indent_str}print("[AWM_SERVER] MCP server enabled at /mcp", flush=True)
{indent_str}print("[AWM_SERVER] Health endpoint: /awm_health", flush=True)
{indent_str}
{indent_str}# Mark shutdown reason as normal (will be overwritten by signal handler if killed)
{indent_str}_set_shutdown_reason("normal_exit")
'''
                new_code.append(pre_uvicorn_code)
                
                # Replace the uvicorn.run line with our version that uses host/port variables
                line = f'{indent_str}uvicorn.run(app, host=host, port=int(port))'
                
            new_code.append(line)

        new_code = "\n".join(new_code)
        processed_lines = len(new_code.split("\n"))
        
        logger.info(f"[STEP 3/5] Code processing stats:")
        logger.info(f"           - Original lines: {original_lines}")
        logger.info(f"           - Processed lines: {processed_lines}")
        logger.info(f"           - create_engine() found: {create_engine_found}")
        logger.info(f"           - uvicorn.run() found: {uvicorn_found}")
        
        if not uvicorn_found:
            logger.warning(f"[STEP 3/5] WARNING: uvicorn.run() not found in code - server may not start properly!")
        
        logger.info(f"[STEP 3/5] SUCCESS: Code processed")
    except Exception as e:
        logger.error(f"[STEP 3/5] FAILED: Code processing error: {e}")
        logger.error(f"[STEP 3/5] Traceback:\n{traceback.format_exc()}")
        raise

    # Step 4: Write temp server file
    logger.info(f"[STEP 4/5] Writing temp server file...")
    try:
        with open(args.temp_server_path, "w") as f:
            f.write(new_code)
        
        file_size = os.path.getsize(args.temp_server_path)
        logger.info(f"[STEP 4/5] SUCCESS: Temp server written to {args.temp_server_path} ({file_size} bytes)")
    except Exception as e:
        logger.error(f"[STEP 4/5] FAILED: Error writing temp server: {e}")
        logger.error(f"[STEP 4/5] Traceback:\n{traceback.format_exc()}")
        raise
    
    # Step 5: Set environment and exec
    logger.info(f"[STEP 5/5] Setting environment and executing server...")
    os.environ['PORT'] = str(args.port)
    os.environ['DATABASE_PATH'] = f"sqlite:///{args.db_path}"
    
    elapsed = time.time() - start_time
    logger.info(f"[STEP 5/5] Environment variables set:")
    logger.info(f"           - PORT={os.environ['PORT']}")
    logger.info(f"           - DATABASE_PATH={os.environ['DATABASE_PATH']}")
    logger.info(f"[STEP 5/5] Total setup time: {elapsed:.2f}s")
    logger.info(f"[STEP 5/5] Executing: {sys.executable} {args.temp_server_path}")
    logger.info(f"=" * 60)
    
    # Flush stdout/stderr to ensure logs are written before exec
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Use os.execv to replace the current process with the server process.
    # This avoids signal propagation issues that occur with os.system() when
    # running as a subprocess with redirected stdout/stderr.
    #
    # When using subprocess.Popen from awm_env.py, os.system() would spawn
    # a grandchild process that may not properly handle signals or I/O
    # redirection, leading to premature server shutdown.
    os.execv(sys.executable, [sys.executable, args.temp_server_path])


def run(config: Config):
    run_server(config)


if __name__ == "__main__":
    from simpleArgParser import parse_args
    config: Config = parse_args(Config)
    run(config)
