"""
System prompts and tool formatters for AWM (Agentic World Model) tasks.

This module provides the system prompt and tool formatters that are compatible
with the AWM (Agentic World Model) native implementation.
"""

# System prompt matching the AWM native implementation (awm/core/agent.py)
AWM_SYSTEM_PROMPT = """# MCP Tools

You are at a MCP environment. You need to call MCP tools to assist with the user query. At each step, you can only call one function. You have already logged in, and your user id is 1 if required for the MCP tool.

You are provided with TWO functions within <tools></tools> XML tags:
<tools>
1. list_tools
    - Description: List all available MCP tools for the current environment to help you finish the user task.
    - Arguments: None
    - Output: A list of MCP environment-specific tools and their descriptions

2. call_tool
    - Description: Call a MCP environment-specific tool
    - Arguments:
        - tool_name: str, required, the tool name in the list_tools output
        - arguments: str, required, the arguments for calling <tool_name>. You must pass a valid JSON string without any markdown fences or additional commentary. This JSON str will be parsed by the tool and executed. You can pass an empty JSON str if no arguments are required by <tool_name>.
    - Output: The result of the <tool_name> tool call
</tools>

You should always call list_tools function first to get the available tools, and should only call it once. You should always directly output the answer or summary at the final step instead of calling any function.

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Example Function Call #1:
<tool_call>
{"name": "list_tools", "arguments": null}
</tool_call>

Example Function Call #2:
<tool_call>
{"name": "call_tool", "arguments": {"tool_name": "get_weather", "arguments": "{\"city\": \"Beijing\"}"}}
</tool_call>
"""


def format_input_schema(schema: dict, indent_level: int = 6, parent_required: list | None = None) -> str:
    """Format input schema for tool description (matching AWM native implementation)."""
    if not schema:
        return ""
    
    result = ""
    indent_str = " " * indent_level
    properties = schema.get('properties', {})
    required_fields = parent_required if parent_required is not None else schema.get('required', [])
    
    for prop_name, prop_info in properties.items():
        is_required = prop_name in required_fields
        required_str = " (required)" if is_required else " (optional)"
        prop_type = prop_info.get('type', 'unknown')
        description = prop_info.get('description', '')
        default = prop_info.get('default')
        enum_values = prop_info.get('enum')
        nested_properties = prop_info.get('properties', {})
        nested_required = prop_info.get('required', [])
        
        result += f"{indent_str}- {prop_name}: {prop_type}{required_str}\n"
        if description:
            result += f"{indent_str}  Description: {description}\n"
        if default is not None:
            result += f"{indent_str}  Default: {default}\n"
        if enum_values:
            result += f"{indent_str}  Allowed values: {enum_values}\n"
        
        if prop_type == "object" and nested_properties:
            result += f"{indent_str}  Properties:\n"
            nested_schema = {"properties": nested_properties, "required": nested_required}
            result += format_input_schema(nested_schema, indent_level + 4, nested_required)
    
    return result


def format_awm_tools_for_prompt(tools: list) -> str:
    """
    Format AWM environment tools for inclusion in system prompt.
    Matches the AWM native implementation (awm/core/agent.py:format_tools_for_response).
    
    Args:
        tools: List of tool definitions from MCP server
        
    Returns:
        Formatted string describing available tools
    """
    # filter out list_tools meta-tool
    actual_tools = [t for t in tools if t.get('name') != 'list_tools']
    
    if not actual_tools:
        return "No tools available."
    
    docs_text = f"Available MCP Tools ({len(actual_tools)} tools):\n"
    docs_text += "=" * 80 + "\n\n"
    
    for i, tool in enumerate(actual_tools, 1):
        name = tool.get("name", "")
        description = tool.get("description", "")
        input_schema = tool.get("inputSchema", tool.get("input_schema", {}))
        
        # add mcp_tool_ prefix (matching native implementation)
        mcp_name = f"mcp_tool_{name}" if not name.startswith("mcp_tool_") else name
        
        # parse description for multi-line
        desc_lines = description.split('\n')
        first_line = desc_lines[0].strip() if desc_lines else "No description"
        more_desc = '\n'.join(line.strip() for line in desc_lines[1:]).strip() if len(desc_lines) > 1 else ""
        
        docs_text += f"{i}. {mcp_name}\n"
        docs_text += f"   Description: {first_line}\n"
        if more_desc:
            for line in more_desc.split('\n'):
                if line.strip():
                    docs_text += f"   {line}\n"
        
        if input_schema and input_schema.get('properties'):
            docs_text += f"   Parameters:\n"
            docs_text += format_input_schema(input_schema)
        else:
            docs_text += f"   Parameters: None\n"
        
        docs_text += "\n"
    
    return docs_text.strip()


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