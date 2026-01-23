import asyncio
import os
import sys
import json
from rllm.environments.tools.mcp_env import MCPConnectionManager

# 使用一个稳定的、内容较少的 URL 进行测试，例如 example.com 或 wikipedia 首页
TEST_URL = "https://www.example.com"

async def main():
    if len(sys.argv) < 2:
        print("Usage: python get_mcp_format.py <bright_data_api_token>")
        sys.exit(1)

    api_token = sys.argv[1]
    
    # 配置 MCP
    mcp_command = "npx"
    mcp_args = ["-y", "@brightdata/mcp"]
    mcp_env = {
        "API_TOKEN": api_token,
        "GROUPS": "advanced_scraping",
        "PATH": os.environ.get("PATH", "")
    }

    print("正在启动 MCP Server...")
    manager = MCPConnectionManager(mcp_command, mcp_args, mcp_env)
    
    # try:
    manager.start()
    
    # 1. 测试 Scrape as Markdown
    print(f"\n[1/2] Testing 'scrape_as_markdown' for {TEST_URL}...")
    tool_calls_md = [{
        "id": "test_md",
        "type": "function",
        "function": {
            "name": "scrape_as_markdown",
            "arguments": json.dumps({"url": TEST_URL})
        }
    }]
    
    result_md = manager.execute_tool_calls(tool_calls_md)
    print("-" * 20 + " MARKDOWN RAW OUTPUT START " + "-" * 20)
    # 打印原始字符串，不做任何处理，以便观察是否包含 wrapper json
    print(result_md.get("test_md", "No output")) 
    print("-" * 20 + " MARKDOWN RAW OUTPUT END " + "-" * 22)

    # 2. 测试 Scrape as HTML
    print(f"\n[2/2] Testing 'scrape_as_html' for {TEST_URL}...")
    tool_calls_html = [{
        "id": "test_html",
        "type": "function",
        "function": {
            "name": "scrape_as_html",
            "arguments": json.dumps({"url": TEST_URL})
        }
    }]
    
    result_html = manager.execute_tool_calls(tool_calls_html)
    print("-" * 20 + " HTML RAW OUTPUT START " + "-" * 20)
    # 截取前1000个字符避免刷屏，但也足够看清头部格式
    html_content = result_html.get("test_html", "No output")
    print(html_content[:2000] + ("\n...[truncated]..." if len(html_content)>2000 else ""))
    print("-" * 20 + " HTML RAW OUTPUT END " + "-" * 22)

    # except Exception as e:
    #     print(f"\n❌ Error occurred: {e}")
    # finally:
    print("\nStopping MCP Server...")
    manager.stop()

if __name__ == "__main__":
    asyncio.run(main())
