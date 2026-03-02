"""
AWM (Agentic World Model) Agent for RLLM

This agent handles interaction with AWM-generated virtual environments
through MCP (Model Context Protocol) tools.

This implementation is aligned with the AWM native implementation (awm/core/agent.py).
"""

import copy
import json
import logging
import re
import time
from typing import Any, Dict, List

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.agents.awm_prompts import AWM_SYSTEM_PROMPT

# Import from AWM native code for compatibility
from awm.tools import tools_robust_json_loads

logger = logging.getLogger(__name__)


class AWMAgent(BaseAgent):
    """
    Agent for AWM (Agentic World Model) tasks.
    
    Handles multi-turn interactions with virtual environments through MCP tools.
    Supports the AWM-specific tool calling format with <think> and <tool_call> tags.
    """

    def __init__(
        self,
        system_prompt: str = AWM_SYSTEM_PROMPT,
        parser_name: str = "qwen",
        max_steps: int = 30,
    ):
        """
        Initialize the AWMAgent.

        Args:
            system_prompt: System prompt for the agent.
            parser_name: Name of the parser to use (kept for compatibility).
            max_steps: Maximum number of interaction steps.
        """
        self.system_prompt = system_prompt
        self.parser_name = parser_name
        self.max_steps = max_steps

        # Initialize state
        self._trajectory = Trajectory()
        self.messages: List[Dict[str, Any]] = []
        self.current_observation = None
        self.current_step = 0
        
        self.reset()

    def _format_observation_as_messages(self, obs: Any) -> List[Dict[str, str]]:
        """
        Format observation into messages for the chat history.
        
        Matches the AWM native implementation message format.
        """
        messages = []
        
        if isinstance(obs, dict):
            # Initial observation with system prompt and task
            if "system_prompt" in obs:
                # Update system prompt if provided
                self.messages[0]["content"] = obs["system_prompt"]
            
            if "task" in obs:
                # Initial task description (user query)
                messages.append({"role": "user", "content": obs['task']})
            
            # Tool results from environment
            if "tool_results" in obs:
                tool_results = obs["tool_results"]
                for result in tool_results:
                    tool_call_id = result.get("tool_call_id", f"call_{int(time.time() * 1000)}")
                    tool_output = result.get("result", "")
                    
                    # Use tool role format matching native implementation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": str(tool_output),
                    })
        elif isinstance(obs, str):
            messages.append({"role": "user", "content": obs})
        elif obs:
            messages.append({"role": "user", "content": str(obs)})

        return messages

    def update_from_env(self, observation: Any, reward: float, done: bool, info: Dict[str, Any], **kwargs):
        """
        Update the agent's state based on environment feedback.
        
        Args:
            observation: Observation from the environment
            reward: Reward for the previous action
            done: Whether the episode is done
            info: Additional information from the environment
        """
        self.current_observation = observation
        
        # Format observation into messages
        if isinstance(observation, dict):
            obs_messages = self._format_observation_as_messages(observation)
            self.messages.extend(obs_messages)
        elif observation:
            self.messages.append({"role": "user", "content": str(observation)})
        
        # Update previous step with reward/done if available
        if self._trajectory.steps:
            self._trajectory.steps[-1].reward = reward
            self._trajectory.steps[-1].done = done
            self._trajectory.steps[-1].info = info
        
        self.current_step = info.get("step", self.current_step) if info else self.current_step

    def _extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """
        Extract tool calls from the model response.
        
        Matches the AWM native implementation (awm/core/agent.py:parse_tool_calls).
        
        Expected format:
        <tool_call>
        {"name": "list_tools", "arguments": {}}
        </tool_call>
        
        Args:
            response: Model's response string
            
        Returns:
            List of tool call dictionaries with id, name, and arguments
        """
        # Be tolerant to non-string model outputs (e.g., None or structured
        # objects returned by some OpenAI-compatible backends).
        if response is None:
            response = ""
        elif not isinstance(response, str):
            logger.warning(
                "AWMAgent received non-string model response (%s); coercing to string.",
                type(response).__name__,
            )
            response = str(response)

        tool_calls = []
        pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for i, match in enumerate(matches):
            data = tools_robust_json_loads(match.strip())
            if not data:
                logger.warning(f"Failed to parse tool call JSON: {match[:100]}")
                continue
            
            # Handle list wrapping: [{"name": ..., "arguments": ...}]
            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], dict):
                    data = data[0]
                else:
                    continue
            
            if not isinstance(data, dict):
                continue
            
            name = data.get("name", "")
            arguments = data.get("arguments", {})
            
            # Handle mcp_tool_ prefix (matching native implementation)
            if name.startswith("mcp_tool_"):
                arguments = {
                    "tool_name": name,
                    "arguments": arguments if arguments else {},
                }
                name = "call_tool"
            
            tool_calls.append({
                "id": f"call_{int(time.time() * 1000)}_{i}",
                "name": name,
                "arguments": arguments,
            })
        
        return tool_calls

    def update_from_model(self, response: Any, **kwargs) -> Action:
        """
        Update the agent's state based on the model's response.
        
        Args:
            response: Model's response string
            **kwargs: Additional arguments
            
        Returns:
            Action object representing the action to take
        """
        # Normalize to string before storing in chat history and parsing.
        if response is None:
            response = ""
        elif not isinstance(response, str):
            response = str(response)

        # Add assistant message to chat history
        self.messages.append({"role": "assistant", "content": response})
        
        # Extract tool calls from response
        tool_calls = self._extract_tool_calls(response)
        
        # Create step in trajectory
        step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            action=tool_calls if tool_calls else response,
            model_response=response,
            observation=self.current_observation
        )
        self._trajectory.steps.append(step)
        
        # Return action
        if tool_calls:
            # Format as action string that environment can parse
            action_str = ""
            for tc in tool_calls:
                action_str += f"<tool_call>\n{json.dumps(tc)}\n</tool_call>\n"
            return Action(action=action_str.strip())
        else:
            # No tool calls - final answer
            return Action(action=response)

    def reset(self):
        """Reset the agent's state for a new episode."""
        self._trajectory = Trajectory()
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.current_observation = None
        self.current_step = 0

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """Return the current message history for the model."""
        return self.messages

    @property
    def trajectory(self) -> Trajectory:
        """Return the trajectory recorded so far."""
        return self._trajectory

    def get_current_state(self) -> Step:
        """Get the current step state."""
        if self._trajectory.steps:
            return self._trajectory.steps[-1]
        return Step()
