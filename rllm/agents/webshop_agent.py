# Copyright 2025 RLLM Team
# Webshop Agent adapted for RLLM Multi-Task Training
# Tool-call version

import copy
import json
import logging
import re
from typing import Any, Dict, List, Optional

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory

logger = logging.getLogger(__name__)


# Tool definitions for Webshop task
WEBSHOP_TOOLS = [
    {
        "name": "search",
        "description": "Search for products with the given query. Use this only if a [Search] button appears in the observation. Note: If you wish to search and there's no [Search] button, click the [Back to Search] button instead.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for products"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "click",
        "description": "Click on a button or product. Use this only if a [button] is present in the observation. When you have identified the most suitable product, click the [Buy Now] button on its product page to finish the shopping task.",
        "parameters": {
            "type": "object",
            "properties": {
                "button": {
                    "type": "string",
                    "description": "Name of the button or product ASIN, don't add '[]' around the button name"
                }
            },
            "required": ["button"]
        }
    }
]


def format_tools_for_prompt(tools: List[Dict]) -> str:
    """Format tool definitions for inclusion in system prompt."""
    tool_strs = []
    for tool in tools:
        params = tool.get("parameters", {}).get("properties", {})
        param_strs = []
        for param_name, param_info in params.items():
            param_strs.append(f'"{param_name}": "{param_info.get("description", "")}"')
        params_json = "{" + ", ".join(param_strs) + "}"
        tool_strs.append(f"- {tool['name']}: {tool['description']}\n  Parameters: {params_json}")
    return "\n".join(tool_strs)


# System prompt for Webshop task (Tool-call version)
WEBSHOP_SYSTEM_PROMPT = f"""You are an expert shopping assistant in the WebShop environment.
Your goal is to find and purchase a product that matches ALL requirements in the instruction.

## Available Tools
{format_tools_for_prompt(WEBSHOP_TOOLS)}

## CRITICAL SHOPPING STRATEGIES (Read Carefully)
1. **BEWARE OF "TITLE TRAP" (Variant Blindness):**
   - Search result titles only show the *default* variant (e.g., "Sea Salt Flavor" or "Size Small").
   - **CRITICAL:** If a product title matches the *item type* (e.g., "Blue Diamond Almonds") but has the wrong attribute (e.g., wrong flavor/size), **YOU MUST CLICK IT**.
   - The specific option you need (e.g., "Pecan", "60x40x40cm") is likely hidden inside the product page as a selectable option. Do not skip products just because the title looks slightly off.

2. **SMART SEARCH QUERY:**
   - **NO PRICES:** Never include "price lower than X" or "under $50" in the search query. The search engine cannot process logic. Check prices AFTER clicking the product.
   - **NO COMPLEX SPECS:** Do not put complex dimensions (e.g., "60x40x40cm") in the search bar. Search for the object name (e.g., "folding ottoman") and check dimensions on the detail page.
   - **Keep it Simple:** Use 2-4 core keywords (e.g., "blue diamond almonds").

3. **AVOID LOOPS (State Awareness):**
   - **NEVER** execute the exact same action (same search query or same button click) twice in a row.
   - If a search fails, you **MUST** change the keywords significantly (e.g., remove adjectives, use synonyms).
   - If you are stuck, try clicking `Next >` to see more results instead of searching again.

4. **PURCHASE FLOW:**
   - Click product -> **Select Options** (click the specific flavor/size button) -> Check Price -> Buy Now.

## Response Format
You MUST respond with:
1. Your reasoning inside <think> </think> tags. **In your thought process, explicitly state if you are checking a product for hidden variants.**
2. A tool call inside <tool_call> </tool_call> tags in JSON format.

Example response:
<think>
The search result "Blue Diamond Almonds Sea Salt" is the right brand but wrong flavor. I need to click it because the "Pecan" flavor might be an option on the details page. I will not filter by price in the search bar.
</think>
<tool_call>
{{"name": "click", "arguments": {{"button": "B07HRFSNL4"}}}}
</tool_call>
"""


class WebshopAgent(BaseAgent):
    """
    Agent for Webshop shopping tasks using tool-call format.
    
    This agent handles the interaction with the Webshop environment,
    parsing observations and formatting actions as tool calls.
    """
    
    def __init__(
        self,
        system_prompt: str = WEBSHOP_SYSTEM_PROMPT,
        tools: List[Dict] = None,
        **kwargs
    ):
        """
        Initialize the Webshop Agent.
        
        Args:
            system_prompt: System prompt for the agent
            tools: List of tool definitions (defaults to WEBSHOP_TOOLS)
        """
        self.system_prompt = system_prompt
        self.tools = tools or WEBSHOP_TOOLS
        
        # Initialize state
        self._trajectory = Trajectory()
        self.messages: List[Dict[str, Any]] = []
        self.current_observation = None
        self._current_instruction = ""
        
        self.reset()
    
    def reset(self):
        """Reset the agent's state for a new episode."""
        self._trajectory = Trajectory()
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.current_observation = None
        self._current_instruction = ""
    
    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Update the agent's state based on environment feedback.
        
        Args:
            observation: The observation from the environment
            reward: The reward received
            done: Whether the episode is done
            info: Additional info from the environment
        """
        # Store current observation
        self.current_observation = observation
        
        # Extract instruction if available
        if info and "instruction" in info:
            self._current_instruction = info["instruction"]
        
        # Format observation as user message
        if isinstance(observation, dict):
            # Handle structured observation
            obs_content = observation.get("content", str(observation))
        else:
            obs_content = str(observation)
        
        # Add observation as user message
        self.messages.append({"role": "user", "content": obs_content})
        
        # Update last step if exists
        if self._trajectory.steps:
            self._trajectory.steps[-1].reward = reward
            self._trajectory.steps[-1].done = done
            self._trajectory.steps[-1].info = info or {}
    
    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Update the agent's state based on the model's response.
        
        Args:
            response: The response from the model
            
        Returns:
            Action object containing the parsed action
        """
        # Parse tool call from response
        parsed_action = self._parse_tool_call(response)
        
        # Add assistant message
        self.messages.append({"role": "assistant", "content": response})
        
        # Create new step
        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            action=parsed_action,
            model_response=response,
            observation=self.current_observation
        )
        self._trajectory.steps.append(new_step)
        
        return Action(action=parsed_action)
    
    def _parse_tool_call(self, response: str) -> str:
        """
        Parse the tool call from model response.
        
        Expected format: <think>...</think><tool_call>{"name": "...", "arguments": {...}}</tool_call>
        
        Args:
            response: The model's response
            
        Returns:
            The parsed action string in format "search[query]" or "click[button]"
        """
        # Try to extract tool call from <tool_call>...</tool_call> tags
        tool_call_match = re.search(
            r'<tool_call>\s*(.*?)\s*</tool_call>',
            response,
            re.DOTALL | re.IGNORECASE
        )
        
        if tool_call_match:
            tool_call_str = tool_call_match.group(1).strip()
            try:
                # Parse JSON
                tool_call = json.loads(tool_call_str)
                tool_name = tool_call.get("name", "").lower()
                arguments = tool_call.get("arguments", {})
                
                if tool_name == "search":
                    query = arguments.get("query", "")
                    return f"search[{query}]"
                elif tool_name == "click":
                    button = arguments.get("button", "")
                    return f"click[{button}]"
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool call JSON: {tool_call_str}")
        
        # Fallback: try to find JSON pattern directly
        json_match = re.search(r'\{[^{}]*"name"\s*:\s*"(search|click)"[^{}]*\}', response, re.IGNORECASE)
        if json_match:
            try:
                tool_call = json.loads(json_match.group(0))
                tool_name = tool_call.get("name", "").lower()
                arguments = tool_call.get("arguments", {})
                
                if tool_name == "search":
                    query = arguments.get("query", "")
                    return f"search[{query}]"
                elif tool_name == "click":
                    button = arguments.get("button", "")
                    return f"click[{button}]"
            except json.JSONDecodeError:
                pass
        
        # Fallback: try old format search[...] or click[...]
        search_match = re.search(r'search\[([^\]]+)\]', response, re.IGNORECASE)
        if search_match:
            return f"search[{search_match.group(1)}]"
        
        click_match = re.search(r'click\[([^\]]+)\]', response, re.IGNORECASE)
        if click_match:
            return f"click[{click_match.group(1)}]"
        
        # Last resort: return the response as-is (will likely fail)
        logger.warning(f"Could not parse tool call from response: {response[:200]}")
        return response[-100:] if len(response) > 100 else response
    
    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """Returns the current message history for the model."""
        return self.messages
    
    @property
    def trajectory(self) -> Trajectory:
        """Returns the trajectory recorded so far."""
        return self._trajectory


# Export tool definitions for use by environment
def get_webshop_tools() -> List[Dict]:
    """Get the tool definitions for Webshop environment."""
    return WEBSHOP_TOOLS
