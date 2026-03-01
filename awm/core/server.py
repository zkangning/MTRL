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
        new_code = ['import warnings', 'warnings.filterwarnings("ignore", category=DeprecationWarning)']
        
        create_engine_found = False
        uvicorn_found = False
        
        for line in code.split("\n"):
            if 'create_engine(' in line:
                create_engine_found = True
                left = line.split('create_engine(')[0]
                sql_path = f"'sqlite:///{args.db_path}'"
                right = f"create_engine({sql_path}, connect_args={{'check_same_thread': False}})"
                line = f"{left}{right}"
                        
            if 'uvicorn.run(app' in line:
                uvicorn_found = True
                raw_code = f"""
                import os
                host = os.environ.get('HOST', '{args.host}')
                port = os.environ.get('PORT', {args.port})
                print(f'Server starting on port={{port}}')
                """
                lines = format_raw_code_to_lines(raw_code, indent=4)
                raw_code = f"""
                from fastapi_mcp import FastApiMCP
                mcp = FastApiMCP(app)
                mcp.mount_http()
                print("MCP server enabled, please visit http://{args.host}:{args.port}/mcp for the MCP service")
                """
                lines += format_raw_code_to_lines(raw_code, indent=4)

                line = f'    uvicorn.run(app, host=host, port=int(port))'
                new_code.extend(lines)
                
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
