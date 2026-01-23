# import logging
# import traceback

# from rllm.tools.tool_base import Tool, ToolOutput

# logger = logging.getLogger(__name__)


# class MCPTool(Tool):
#     def __init__(self, session, tool_name, tool_description, tool_schema):
#         self._tool_schema = tool_schema
#         self.session = session

#         super().__init__(name=tool_name, description=tool_description)

#     @property
#     def json(self):
#         return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self._tool_schema}}

#     async def async_forward(self, **kwargs) -> ToolOutput:
#         try:
#             logger.debug(f"Calling MCP tool: {self.name} with args: {kwargs}")

#             result = await self.session.call_tool(self.name, kwargs)
#             if hasattr(result, "content"):
#                 if hasattr(result.content, "text"):
#                     content_str = result.content.text
#                 elif isinstance(result.content, list) and hasattr(result.content[0], "text"):
#                     content_str = result.content[0].text
#                 else:
#                     content_str = str(result.content)
#             else:
#                 content_str = str(result)

#             logger.debug(f"MCP tool result: {content_str}")
#             return ToolOutput(name=self.name or "mcp_tool", output=content_str)
#         except Exception as e:
#             logger.debug(f"Error executing MCP tool {self.name}: {str(e)}")
#             traceback.print_exc()
#             return ToolOutput(
#                 name=self.name or "mcp_tool",
#                 error=f"Error calling MCP tool: {e}",
#             )
import logging
import traceback

from rllm.tools.tool_base import Tool, ToolOutput

logger = logging.getLogger(__name__)


class MCPTool(Tool):
    def __init__(self, session, tool_name, tool_description, tool_schema):
        self._tool_schema = tool_schema
        self.session = session

        super().__init__(name=tool_name, description=tool_description)

    # --- 新增代码开始: 处理 Ray/Pickle 序列化问题 ---
    def __getstate__(self):
        """
        在 Ray 将对象传输到 GPU Worker 时调用。
        我们需要移除不可序列化的 session (包含 asyncio.Task)。
        保留 schema 和 metadata 以便 Worker 可以生成 Prompt。
        """
        state = self.__dict__.copy()
        # session 包含活跃连接和 Task，无法跨进程传输，必须设为 None
        state['session'] = None
        return state

    def __setstate__(self, state):
        """
        在 Ray Worker 接收到对象时调用。
        """
        self.__dict__.update(state)
    # --- 新增代码结束 ---

    @property
    def json(self):
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self._tool_schema}}

    async def async_forward(self, **kwargs) -> ToolOutput:
        # --- 修改开始: 增加 session 检查 ---
        if self.session is None:
            # 如果在 Worker 节点上尝试执行工具，且没有 session，返回错误而不是崩溃
            error_msg = (
                f"MCPTool '{self.name}' cannot be executed strictly inside the Ray Worker "
                f"because the network session is not serializable. "
                f"Ensure tools are executed in the Environment or the Session is re-initialized."
            )
            logger.warning(error_msg)
            return ToolOutput(name=self.name or "mcp_tool", error=error_msg)
        # --- 修改结束 ---

        try:
            logger.debug(f"Calling MCP tool: {self.name} with args: {kwargs}")

            result = await self.session.call_tool(self.name, kwargs)
            if hasattr(result, "content"):
                if hasattr(result.content, "text"):
                    content_str = result.content.text
                elif isinstance(result.content, list) and hasattr(result.content[0], "text"):
                    content_str = result.content[0].text
                else:
                    content_str = str(result.content)
            else:
                content_str = str(result)

            logger.debug(f"MCP tool result: {content_str}")
            return ToolOutput(name=self.name or "mcp_tool", output=content_str)
        except Exception as e:
            logger.debug(f"Error executing MCP tool {self.name}: {str(e)}")
            traceback.print_exc()
            return ToolOutput(
                name=self.name or "mcp_tool",
                error=f"Error calling MCP tool: {e}",
            )

