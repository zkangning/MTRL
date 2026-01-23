import asyncio
import os
import sys
import json
import time
import shutil
from rllm.environments.tools.mcp_env import MCPEnvironment

# === 配置区域 ===
CACHE_DIR = "./search_cache_data_tmp"
# 为了测试效果，请确保这两个 URL 是可以访问的
URL_1 = "https://www.example.com"
URL_2 = "https://www.iana.org/help/example-domains"
# 两个不同的查询
QUERY_1 = "Python programming language"
QUERY_2 = "Rust programming language"

def clean_cache():
    if os.path.exists(CACHE_DIR):
        print(f"🧹 Cleaning up old cache directory: {CACHE_DIR}")
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)

def load_json_cache(filename):
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

async def main():
    if len(sys.argv) < 2:
        print("Usage: python test_granular_cache.py <bright_data_api_token>")
        sys.exit(1)

    api_token = sys.argv[1]
    clean_cache()

    # 配置 MCP
    mcp_command = "npx"
    mcp_args = ["-y", "@brightdata/mcp"]
    mcp_env = { "API_TOKEN": api_token, "GROUPS": "advanced_scraping", "PATH": os.environ.get("PATH", "") }

    print("\n🚀 Initializing MCP Environment...")
    env = MCPEnvironment(
        mcp_server_command=mcp_command,
        mcp_server_args=mcp_args,
        mcp_server_env=mcp_env,
        cache_dir=CACHE_DIR
    )

    try:
        # ==========================================
        # Phase 1: 基础单项缓存 (Seeding the Cache)
        # ==========================================
        print("\n" + "="*50)
        print("Phase 1: Seed Cache with Single Requests")
        print("="*50)
        
        action_seed = [
            {
                "id": "seed_search",
                "type": "function",
                "function": {
                    "name": "search_engine",
                    "arguments": json.dumps({"query": QUERY_1})
                }
            },
            {
                "id": "seed_scrape",
                "type": "function",
                "function": {
                    "name": "scrape_as_markdown",
                    "arguments": json.dumps({"url": URL_1})
                }
            }
        ]
        
        print(f"📡 Executing: Search('{QUERY_1}') & Scrape('{URL_1}')...")
        start = time.time()
        obs, _, _, _ = env.step(action_seed)
        print(f"⏱️  Time taken: {time.time() - start:.2f}s")
        
        # 验证文件写入
        search_data = load_json_cache("mcp_search_cache.json")
        md_data = load_json_cache("mcp_markdown_cache.json")
        
        if QUERY_1 in search_data:
            print(f"✅ Search Cache: Found key '{QUERY_1}'")
        else:
            print(f"❌ Search Cache: Key '{QUERY_1}' MISSING!")

        if URL_1 in md_data:
            print(f"✅ Markdown Cache: Found key '{URL_1}'")
        else:
            print(f"❌ Markdown Cache: Key '{URL_1}' MISSING!")


        # ==========================================
        # Phase 2: 细粒度 Batch 测试 (Granular Cache Logic)
        # ==========================================
        print("\n" + "="*50)
        print("Phase 2: Test Batch Tool with Mixed Cache (Hit + Miss)")
        print("="*50)
        print("💡 Scenario: Calling 'search_engine_batch' with:")
        print(f"   1. '{QUERY_1}' (Should HIT cache instantly)")
        print(f"   2. '{QUERY_2}' (Should MISS and fetch from network)")

        action_batch_search = [{
            "id": "batch_mixed",
            "type": "function",
            "function": {
                "name": "search_engine_batch",
                "arguments": json.dumps({
                    "queries": [
                        {"query": QUERY_1}, # Cached
                        {"query": QUERY_2}  # New
                    ]
                })
            }
        }]

        start = time.time()
        obs, _, _, _ = env.step(action_batch_search)
        duration = time.time() - start
        print(f"⏱️  Time taken: {duration:.2f}s")

        # 检查逻辑：
        # 1. search_engine_batch 的输出应该包含两个结果
        output_str = obs['tool_outputs']['batch_mixed']
        output_json = json.loads(output_str)
        if len(output_json) == 2:
            print(f"✅ Batch Output: Received 2 results as expected.")
        else:
            print(f"❌ Batch Output: Expected 2 results, got {len(output_json)}")

        # 2. 验证缓存文件是否更新了 QUERY_2
        search_data = load_json_cache("mcp_search_cache.json")
        if QUERY_2 in search_data:
            print(f"✅ Search Cache Update: Batch execution correctly wrote individual key '{QUERY_2}'.")
        else:
            print(f"❌ Search Cache Update: Key '{QUERY_2}' NOT found in cache file!")

        # ==========================================
        # Phase 3: 反向验证 (Interoperability)
        # ==========================================
        print("\n" + "="*50)
        print("Phase 3: Verify Cross-Tool Cache Hit")
        print("="*50)
        print(f"💡 Scenario: Calling SINGLE 'search_engine' for '{QUERY_2}'")
        print("   (This was just added via the BATCH tool in Phase 2)")
        print("   Expected: Instant return (Cache Hit)")

        action_verify = [{
            "id": "verify_single",
            "type": "function",
            "function": {
                "name": "search_engine",
                "arguments": json.dumps({"query": QUERY_2})
            }
        }]

        start = time.time()
        env.step(action_verify)
        duration = time.time() - start
        print(f"⏱️  Time taken: {duration:.4f}s")

        if duration < 0.5:
            print("✅ CACHE HIT! The batch tool successfully warmed the cache for single tool usage.")
        else:
            print("⚠️ CACHE MISS! Speed suggests network request occurred.")

        # ==========================================
        # Phase 4: Scrape Batch 测试
        # ==========================================
        print("\n" + "="*50)
        print("Phase 4: Scrape Batch Interoperability")
        print("="*50)
        print(f"💡 Calling 'scrape_batch' with [{URL_1}, {URL_2}]")
        
        action_scrape_batch = [{
            "id": "batch_scrape",
            "type": "function",
            "function": {
                "name": "scrape_batch",
                "arguments": json.dumps({
                    "urls": [URL_1, URL_2]
                })
            }
        }]

        start = time.time()
        env.step(action_scrape_batch)
        print(f"⏱️  Time taken: {time.time() - start:.2f}s")

        md_data = load_json_cache("mcp_markdown_cache.json")
        if URL_2 in md_data:
            print(f"✅ Markdown Cache: Batch scrape correctly added '{URL_2}' to markdown cache.")
        else:
            print(f"❌ Markdown Cache: '{URL_2}' missing.")

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        print("\n🛑 Closing Environment...")
        MCPEnvironment.cleanup_global_resources()
        print("Done.")

if __name__ == "__main__":
    asyncio.run(main())
