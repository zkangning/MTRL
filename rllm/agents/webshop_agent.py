# Copyright 2025 RLLM Team
# Webshop Agent adapted for RLLM Multi-Task Training

import copy
import logging
import re
from typing import Any, Dict, List

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory

logger = logging.getLogger(__name__)


# System prompt for Webshop task
WEBSHOP_SYSTEM_PROMPT = """You are a shopping assistant helping users find and purchase products on a web shopping platform.

You will be given a shopping task with specific requirements (e.g., product type, price range, features).
You need to navigate the website by searching for products and clicking on items to find the best match.

Available actions:
- search[keywords]: Search for products using keywords
- click[element]: Click on a button or link (e.g., click[Buy Now], click[< Prev], click[item_name])

You must respond in the following format:
<think>Your reasoning about what action to take next</think>
<action>Your action (e.g., search[red shoes] or click[Buy Now])</action>

Important:
- Read the instruction carefully to understand what product features are required
- Use search to find relevant products
- Click on products to view details and select options
- Click "Buy Now" when you find a product that matches all requirements
"""


class WebshopAgent(BaseAgent):
    """
    Agent for Webshop shopping tasks.
    
    This agent handles the interaction with the Webshop environment,
    parsing observations and formatting actions in the expected format.
    """
    
    def __init__(
        self,
        system_prompt: str = WEBSHOP_SYSTEM_PROMPT,
        **kwargs
    ):
        """
        Initialize the Webshop Agent.
        
        Args:
            system_prompt: System prompt for the agent
        """
        self.system_prompt = system_prompt
        
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
        # Parse action from response
        parsed_action = self._parse_action(response)
        
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
    
    def _parse_action(self, response: str) -> str:
        """
        Parse the action from model response.
        
        Expected format: <think>...</think><action>...</action>
        
        Args:
            response: The model's response
            
        Returns:
            The parsed action string
        """
        response_lower = response.lower()
        
        # Try to extract action from <action>...</action> tags
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = response_lower.find(start_tag)
        end_idx = response_lower.find(end_tag)
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            # Extract action content (use original case)
            action = response[start_idx + len(start_tag):end_idx].strip()
            return action
        
        # Fallback: try to find search[...] or click[...] patterns
        search_match = re.search(r'search\[([^\]]+)\]', response, re.IGNORECASE)
        if search_match:
            return f"search[{search_match.group(1)}]"
        
        click_match = re.search(r'click\[([^\]]+)\]', response, re.IGNORECASE)
        if click_match:
            return f"click[{click_match.group(1)}]"
        
        # Last resort: return the last part of response
        return response[-100:] if len(response) > 100 else response
    
    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """Returns the current message history for the model."""
        return self.messages
    
    @property
    def trajectory(self) -> Trajectory:
        """Returns the trajectory recorded so far."""
        return self._trajectory
