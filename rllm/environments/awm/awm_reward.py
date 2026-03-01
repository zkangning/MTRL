"""
AWM Reward Functions for RLLM

Provides reward computation for AWM tasks using:
1. Code-augmented LLM-as-a-Judge (AWMMCPRewardFn)
2. Purely code-based Judge (AWMMCPPureCodeRewardFn)

The verification code is generated during AWM dataset creation
(gen_verifier.jsonl and gen_verifier.pure_code.jsonl).
"""

import json
import logging
import os
import sqlite3
import tempfile
from typing import Any, Dict, Optional

from rllm.rewards.reward_types import RewardConfig, RewardInput, RewardOutput
from rllm.rewards.reward_fn import RewardFunction
from awm.core.verifier import execute_verification_code, VerificationMode

logger = logging.getLogger(__name__)


class AWMMCPPureCodeRewardFn:
    """
    Purely code-based reward function for AWM tasks.
    
    Uses verification code from gen_verifier.pure_code.jsonl which returns:
    - "complete": Task fully completed
    - "others": Task not completed or partially completed
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        
    def __call__(self, task_info: Dict[str, Any], action: str) -> RewardOutput:
        """
        Compute reward using pure code verification.
        
        Args:
            task_info: Dictionary containing:
                - verifier_code: Python code with verify function
                - db_path: Path to SQLite database
                - final_answer: Agent's final response
            action: Agent's action/response
            
        Returns:
            RewardOutput with reward and metadata
        """
        verifier_code = task_info.get("verifier_code", "")
        db_path = task_info.get("db_path", "")
        final_answer = task_info.get("final_answer", action)
        
        if not verifier_code or not db_path:
            logger.warning("Missing verifier_code or db_path for AWM reward computation")
            return RewardOutput(reward=0.0, metadata={"error": "missing_verifier_or_db"})
        
        try:
            # Execute verification code
            result = self._execute_verification(
                verifier_code, db_path, final_answer
            )
            
            # Parse result
            if result.get("execution_status") == "success":
                verification_result = result.get("result", {})
                status = verification_result.get("result", "others")
                
                if status == "complete":
                    reward = 1.0
                else:
                    reward = 0.0
                
                return RewardOutput(
                    reward=reward,
                    metadata={
                        "status": status,
                        "details": verification_result.get("details", {}),
                        "execution_status": "success",
                    }
                )
            else:
                # Execution error
                error_msg = result.get("error_message", "unknown error")
                logger.error(f"AWM verification execution error: {error_msg}")
                return RewardOutput(
                    reward=0.0,
                    metadata={
                        "execution_status": "error",
                        "error_message": error_msg,
                    }
                )
                
        except Exception as e:
            logger.error(f"Error in AWM reward computation: {e}")
            return RewardOutput(
                reward=0.0,
                metadata={"execution_status": "exception", "error": str(e)}
            )
    
    @staticmethod
    def _detect_verify_function_name(python_code: str, default: str) -> str:
        """
        Detect the verification function name from the generated code.
        
        Matches AWM native logic (awm/core/verifier.py): looks for 'def verify_...'
        and extracts the function name. Falls back to the default if not found.
        """
        for line in python_code.split('\n'):
            line = line.strip()
            if line.startswith('def verify_') and '(' in line:
                return line.split('(')[0].replace('def ', '').strip()
        return default

    def _execute_verification(
        self,
        python_code: str,
        db_path: str,
        final_answer: str,
        function_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute verification code using awm.core.verifier.
        
        Function name detection follows AWM native logic:
        - Default for VerificationMode.code: "verify_task_completion"
        - Auto-detect from code: looks for 'def verify_...()'
        """
        if function_name is None:
            function_name = self._detect_verify_function_name(
                python_code, default="verify_task_completion"
            )
        
        try:
            return execute_verification_code(
                python_code=python_code,
                function_name=function_name,
                initial_db_path=db_path,
                mode=VerificationMode.code
            )
        except Exception as e:
            return {
                "execution_status": "error",
                "error_message": f"Execution error: {str(e)}",
            }


class AWMMCPRewardFn:
    """
    Code-augmented LLM-as-a-Judge reward function for AWM tasks.
    
    Uses verification code from gen_verifier.jsonl which returns
    information for LLM to judge task completion.
    
    This version uses an external LLM to make the final judgment
    based on the verification results.
    """
    
    def __init__(self, config: Optional[RewardConfig] = None, llm_client=None):
        self.config = config or RewardConfig()
        self.llm_client = llm_client
        
    def __call__(self, task_info: Dict[str, Any], action: str) -> RewardOutput:
        """
        Compute reward using code-augmented LLM verification.
        
        Args:
            task_info: Dictionary containing:
                - verifier_code: Python code with verify function
                - db_path: Path to SQLite database
                - final_answer: Agent's final response
                - scenario: Scenario name
                - task: Task description
            action: Agent's action/response
            
        Returns:
            RewardOutput with reward and metadata
        """
        verifier_code = task_info.get("verifier_code", "")
        db_path = task_info.get("db_path", "")
        final_answer = task_info.get("final_answer", action)
        scenario = task_info.get("scenario", "")
        task = task_info.get("task", "")
        
        if not verifier_code or not db_path:
            logger.warning("Missing verifier_code or db_path for AWM reward computation")
            return RewardOutput(reward=0.0, metadata={"error": "missing_verifier_or_db"})
        
        try:
            # Execute verification code to get information dict
            result = self._execute_verification_sql(
                verifier_code, db_path
            )
            
            if result.get("execution_status") != "success":
                error_msg = result.get("error_message", "unknown error")
                logger.error(f"AWM SQL verification error: {error_msg}")
                return RewardOutput(
                    reward=0.0,
                    metadata={
                        "execution_status": "error",
                        "error_message": error_msg,
                    }
                )
            
            # Get verification info
            verification_info = result.get("result", {})
            
            # Use LLM to judge based on verification info
            if self.llm_client:
                judgment = self._llm_judge(
                    scenario, task, final_answer, verification_info
                )
                reward = 1.0 if judgment.get("completed", False) else 0.0
                
                return RewardOutput(
                    reward=reward,
                    metadata={
                        "execution_status": "success",
                        "verification_info": verification_info,
                        "llm_judgment": judgment,
                    }
                )
            else:
                # No LLM client - use heuristic based on verification info
                logger.warning("No LLM client provided for AWMMCPRewardFn, using heuristic")
                
                # Simple heuristic: check if info contains positive indicators
                info_str = json.dumps(verification_info).lower()
                positive_indicators = ["success", "complete", "found", "match"]
                negative_indicators = ["fail", "error", "not found", "missing"]
                
                positive_count = sum(1 for p in positive_indicators if p in info_str)
                negative_count = sum(1 for n in negative_indicators if n in info_str)
                
                reward = 1.0 if positive_count > negative_count else 0.0
                
                return RewardOutput(
                    reward=reward,
                    metadata={
                        "execution_status": "success",
                        "verification_info": verification_info,
                        "heuristic": True,
                    }
                )
                
        except Exception as e:
            logger.error(f"Error in AWM reward computation: {e}")
            return RewardOutput(
                reward=0.0,
                metadata={"execution_status": "exception", "error": str(e)}
            )
    
    def _execute_verification_sql(
        self,
        python_code: str,
        db_path: str,
        function_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute SQL-based verification code using awm.core.verifier.
        
        Function name detection follows AWM native logic:
        - Default for VerificationMode.sql: "verify_task"
        - Auto-detect from code: looks for 'def verify_...()'
        """
        if function_name is None:
            function_name = AWMMCPPureCodeRewardFn._detect_verify_function_name(
                python_code, default="verify_task"
            )
        
        try:
            return execute_verification_code(
                python_code=python_code,
                function_name=function_name,
                initial_db_path=db_path,
                mode=VerificationMode.sql
            )
        except Exception as e:
            return {
                "execution_status": "error",
                "error_message": f"Execution error: {str(e)}",
            }
    
    def _llm_judge(
        self,
        scenario: str,
        task: str,
        final_answer: str,
        verification_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use LLM to judge task completion based on verification info.
        
        Args:
            scenario: Scenario name
            task: Task description
            final_answer: Agent's final answer
            verification_info: Info dict from SQL verification
            
        Returns:
            Dictionary with judgment result
        """
        if not self.llm_client:
            return {"completed": False, "reason": "No LLM client available"}
        
        # Build judgment prompt
        prompt = f"""You are evaluating whether an AI agent has successfully completed a task.

Scenario: {scenario}
Task: {task}

Agent's Final Answer:
{final_answer}

Verification Information (from database queries):
{json.dumps(verification_info, indent=2)}

Based on the agent's final answer and the verification information, determine if the task has been successfully completed.

Respond with a JSON object in the following format:
{{
    "completed": true/false,
    "confidence": 0.0-1.0,
    "reason": "explanation of your judgment"
}}

Your response:"""
        
        try:
            # Call LLM for judgment
            response = self.llm_client.complete(prompt)
            
            # Parse response
            judgment = json.loads(response.strip())
            return judgment
            
        except Exception as e:
            logger.error(f"Error in LLM judgment: {e}")
            return {"completed": False, "reason": f"LLM judgment error: {e}"}


def awm_pure_code_reward_fn(task_info: Dict[str, Any], action: str) -> RewardOutput:
    """
    Convenience function for pure code-based AWM reward.
    """
    reward_fn = AWMMCPPureCodeRewardFn()
    return reward_fn(task_info, action)


def awm_mcp_reward_fn(task_info: Dict[str, Any], action: str, llm_client=None) -> RewardOutput:
    """
    Convenience function for MCP-based AWM reward with LLM judgment.
    """
    reward_fn = AWMMCPRewardFn(llm_client=llm_client)
    return reward_fn(task_info, action)
