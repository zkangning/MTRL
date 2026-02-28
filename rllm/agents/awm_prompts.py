"""
System prompts and tool formatters for AWM (Agentic World Model) tasks.
"""

AWM_SYSTEM_PROMPT = """You are an AI assistant interacting with a virtual environment through tools.

Your goal is to complete the given task by using the available tools in the environment. The environment provides access to a simulated system (such as an e-commerce platform, booking system, or other application) through MCP (Model Context Protocol) tools.

## Interaction Flow

1. **First Step**: Call `list_tools` to see what tools are available in the current environment
2. **Subsequent Steps**: Use `call_tool` to interact with the environment
3. **Analyze Results**: Review tool results and plan your next steps
4. **Complete Task**: Continue until you have successfully completed the task

## Available Meta-Tools

1. **list_tools**
   - Description: List all available MCP tools for the current environment
   - Arguments: None
   - Output: A list of environment-specific tools and their descriptions

2. **call_tool**
   - Description: Call an environment-specific tool
   - Arguments:
     - tool_name: str, the name of the tool to call
     - arguments: dict, the arguments for the tool
   - Output: The result of the tool call

## Response Format

For each step, respond with:
1. Your reasoning inside <think> </think> tags
2. A tool call inside <tool_call> </tool_call> tags in JSON format

Examples:

To list available tools:
<think>
I need to see what tools are available in this environment.
</think>

<tool_call>
{"name": "list_tools", "arguments": {}}
</tool_call>

To call a specific tool:
<think>
I need to search for products in the e-commerce platform.
</think>

<tool_call>
{"name": "call_tool", "arguments": {"tool_name": "search_products", "arguments": {"query": "laptop"}}}
</tool_call>

## Important Guidelines

1. **Always start with list_tools** to understand what actions you can take
2. **Read tool descriptions carefully** to understand required parameters
3. **Plan your actions** based on the task requirements
4. **Use specific queries** when searching or filtering
5. **Track your progress** toward completing the task
6. **Provide final answer** directly when task is complete (no tool call needed)

Remember: You must interact with the environment through tools to complete tasks. Direct answers without tool interaction will not succeed.
"""


def format_awm_tools_for_prompt(tools: list) -> str:
    """
    Format AWM environment tools for inclusion in system prompt.
    
    Args:
        tools: List of tool definitions from MCP server
        
    Returns:
        Formatted string describing available tools
    """
    if not tools:
        return "No tools available."
    
    lines = ["## Available Environment Tools", ""]
    
    for i, tool in enumerate(tools, 1):
        name = tool.get("name", "")
        description = tool.get("description", "")
        schema = tool.get("inputSchema", {})
        
        lines.append(f"{i}. **{name}**")
        lines.append(f"   Description: {description}")
        
        if schema and schema.get("properties"):
            lines.append("   Parameters:")
            for prop_name, prop_info in schema["properties"].items():
                prop_type = prop_info.get("type", "unknown")
                prop_desc = prop_info.get("description", "")
                required = prop_name in schema.get("required", [])
                req_str = " (required)" if required else " (optional)"
                lines.append(f"     - {prop_name}: {prop_type}{req_str}")
                if prop_desc:
                    lines.append(f"       {prop_desc}")
        
        lines.append("")
    
    return "\n".join(lines)


def format_awm_observation(tool_result: dict) -> str:
    """
    Format tool result as observation for the agent.
    
    Args:
        tool_result: Result from tool execution
        
    Returns:
        Formatted observation string
    """
    tool_name = tool_result.get("tool", "")
    result = tool_result.get("result", "")
    success = tool_result.get("success", False)
    
    lines = [f"## Tool Result: {tool_name}"]
    lines.append(f"Status: {'Success' if success else 'Failed'}")
    lines.append("")
    lines.append("Result:")
    lines.append("```")
    lines.append(str(result))
    lines.append("```")
    
    return "\n".join(lines)