# mcpuniverse/mcp/test_mcp_manager.py

import asyncio
import os

from mcpuniverse.common.context import Context
from mcpuniverse.mcp.manager import MCPManager


async def main():
    # 1. Context：用环境变量渲染模板
    ctx = Context(env=dict(os.environ))

    # 2. 使用默认配置（就是 mcp/configs/server_list.json）
    print("== 初始化 MCPManager ==")
    manager = MCPManager(config=None, context=ctx)

    # 3. 列出所有服务器
    print("\n== 已配置的服务器列表 ==")
    server_names = manager.list_server_names()
    print(server_names)

    # 4. 检查未指定参数（忽略 PORT）
    print("\n== 未指定参数检查 ==")
    unspecified = manager.list_unspecified_params(ignore_port=True)
    if unspecified:
        print("存在未指定参数的服务器：")
        for name, params in unspecified.items():
            print(f"  {name}: {params}")
    else:
        print("所有服务器模板参数（除 PORT 外）都已指定。")

    # 5. 计划测试的服务器（根据你当前环境可用的）
    test_servers = ["date", "echo", "weather"]
    test_servers = [s for s in test_servers if s in server_names]
    print("\n== 计划测试的服务器（stdio） ==")
    print(test_servers)

    for server in test_servers:
        print("\n====================")
        print(f"== 测试服务器: {server} (stdio) ==")

        # 5.1 list_tools
        try:
            tools_list = await manager.list_tools(server_names=server, transport="stdio")
            tools_for_server = tools_list[0]  # 单个 server 时是长度为 1 的列表

            print(f"[{server}] 工具列表（简要）：")
            for idx, tool in enumerate(tools_for_server):
                name = getattr(tool, "name", None) or getattr(tool, "tool_name", None)
                desc = getattr(tool, "description", None) or getattr(
                    tool, "description_md", None
                )
                if isinstance(tool, dict):
                    name = name or tool.get("name") or tool.get("tool_name")
                    desc = desc or tool.get("description")
                print(f"  {idx+1}. {name} - {desc}")
        except Exception as e:
            print(f"[{server}] list_tools 失败：{e}")
            continue

        # 5.2 使用你刚才打印出来的“真实工具名”
        if server == "date":
            # 这里我们直接调用 get_today_date、get_current_datetime_utc 和 get_date_in_timezone 各一次做演示
            test_cases = [
                ("get_today_date", {}),
                ("get_current_datetime_utc", {}),
                ("get_date_in_timezone", {"timezone_name": "Asia/Shanghai"}),
            ]
        elif server == "echo":
            # 已知工具名是 echo_tool，猜测参数为 text
            test_cases = [
                ("echo_tool", {"text": "hello from echo_tool"})
            ]
        elif server == "weather":
            # 已知工具有 get_alerts 和 get_forecast
            test_cases = [
                ("get_alerts", {"state": "CA"}),                # 加州天气预警
                ("get_forecast", {"latitude": 37.7749, "longitude": -122.4194}),  # 旧金山
            ]
        else:
            test_cases = []

        if not test_cases:
            print(f"[{server}] 暂不配置执行 demo 工具。")
            continue

        # 依次调用每个测试用例
        for tool_name, arguments in test_cases:
            print(f"\n[{server}] 尝试调用工具: {tool_name}，参数: {arguments}")
            try:
                result = await manager.execute(
                    server_name=server,
                    tool_name=tool_name,
                    arguments=arguments,
                    transport="stdio",
                )
                print(f"[{server}] 工具 `{tool_name}` 调用结果：")
                print(result)
            except Exception as e:
                print(f"[{server}] 工具 `{tool_name}` 调用失败：{e}")


if __name__ == "__main__":
    asyncio.run(main())
