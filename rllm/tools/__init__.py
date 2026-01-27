from rllm.tools.code_tools import (
    PythonInterpreter,
)
from rllm.tools.registry import ToolRegistry
from rllm.tools.web_tools import (
    FirecrawlTool,
    GoogleSearchTool,
    TavilyExtractTool,
    TavilySearchTool,
)
from rllm.tools.local_retrieval_tool import LocalRetrievalTool

# Define default tools dict
DEFAULT_TOOLS = {
    "python": PythonInterpreter,
    "google_search": GoogleSearchTool,
    "firecrawl": FirecrawlTool,
    "tavily-extract": TavilyExtractTool,
    "tavily-search": TavilySearchTool,
    "local_search": LocalRetrievalTool,
}

# Create the singleton registry instance and register all default tools
tool_registry = ToolRegistry()
tool_registry.register_all(DEFAULT_TOOLS)

__all__ = ["PythonInterpreter", "LocalRetrievalTool", "GoogleSearchTool", "FirecrawlTool", "TavilyExtractTool", "TavilySearchTool", "ToolRegistry", "tool_registry"]
