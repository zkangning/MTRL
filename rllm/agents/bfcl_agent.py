import asyncio
import argparse
import logging
import re
import uuid
import json
import copy
from transformers import AutoTokenizer
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.utils import colorful_print
from rllm.agents.tool_agent import ToolAgent
from rllm.agents.agent import Action, Step, Trajectory, BaseAgent
# 导入上面修改后的 Env
from rllm.environments.tools.bfcl_env import BFCLEnvironment

class BFCLReadyAgent(ToolAgent):
    """
    针对 BFCL 环境定制的 Agent
    1. 能够接收 Environment 返回的 system prompt 并替换自己的。
    2. 能够处理多轮对话中的新 Query (Multi-turn)。
    """
    def _split_think_and_visible(self, text: str) -> tuple[str, str]:
        if not isinstance(text, str):
            return "", str(text)

        match = re.match(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
        
        if match:
            return match.group(1).strip("\n"), match.group(2).strip("\n")
        
        return "", text.strip()
    def update_from_env(self, observation, reward, done, info, **kwargs):
        # 调用父类更新 (追加 tool outputs, 更新 reward 等)
        super().update_from_env(observation, reward, done, info, **kwargs)
        
        # 1. 注入 BFCL 的 System Prompt
        # reset() 时 Env 会返回 info['system_prompt']
        if "system_prompt" in info and info["system_prompt"]:
            bfcl_sys_prompt = info["system_prompt"]
            # 查找并替换，或插入 System Message
            found_system = False
            for msg in self.messages:
                if msg["role"] == "system":
                    msg["content"] = bfcl_sys_prompt
                    found_system = True
                    break
            if not found_system:
                self.messages.insert(0, {"role": "system", "content": bfcl_sys_prompt})
            
            # 清空 Base Agent 可能自动生成的 tools_prompt，避免冲突
            self.tools_prompt = "" 

        # 2. 处理多轮对话的新问题 (New User Query)
        # 如果 observation 中包含 'question'，且不是 reset 时的初始问题（根据 step 判断或上下文判断）
        # 在 super().update_from_env() 中，如果 obs 是 tool_outputs，父类会处理为 Tool Message。
        # 如果 obs 是 question，父类可能不会自动处理（取决于 RLLM 版本），这里手动追加 User Message。
        # import pdb; pdb.set_trace()

        if isinstance(observation, dict) and "question" in observation:
            new_question = observation["question"]
            
            # 防止重复添加：如果最后一条消息已经是这个 question（通常发生在 reset 后的第一次 update），则跳过
            is_duplicate = False
            if self.messages and self.messages[-1]["role"] == "user" and self.messages[-1]["content"] == new_question:
                is_duplicate = True
            
            if not is_duplicate:
                # 这是一个新的用户任务
                colorful_print(f"Agent received new query: {new_question[:50]}...", "yellow")
                self.messages.append({"role": "user", "content": new_question})

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        覆盖父类实现：
        - 对 Qwen3 输出做裁剪：去掉历史中的 <think>...</think>
        - 只用可见内容解析工具调用 / finish
        - Step 中额外记录 thought / raw_model_response 便于分析
        """
        # 1. 先切掉 think
        think_content, visible_content = self._split_think_and_visible(response)

        tool_calls_dict = []
        assistant_content = visible_content

        # 2. 只用「可见部分」解析工具调用
        try:
            tool_calls = self.tool_parser.parse(assistant_content)
            tool_calls_dict = [
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": tool_call.to_dict(),
                }
                for tool_call in tool_calls
            ]
        except Exception as e:
            # logger.error(f"[BFCLReadyAgent] Failed to parse tool calls: {e}")
            tool_calls_dict = []

        # 3. assistant 消息：只把 visible_content 写入历史
        assistant_message = {"role": "assistant", "content": assistant_content}

        if tool_calls_dict:
            for call in tool_calls_dict:
                if isinstance(call.get("function", {}).get("arguments"), dict):
                    call["function"]["arguments"] = json.dumps(call["function"]["arguments"])
        else:
            # 没有工具调用，则构造 finish
            tool_calls_dict = [
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": "finish",
                        "arguments": {
                            "response": assistant_content,
                        },
                    },
                }
            ]

        # 4. 写入对话历史（不会包含 <think>）
        self.messages.append(assistant_message)

        # 5. 写入 trajectory 的 step，并额外挂载 thought 信息
        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            action=tool_calls_dict,
            model_response=assistant_content,  # 外显部分
            observation=self.current_observation,
        )
        # 附带思维链和原始输出，便于训练/调试
        setattr(new_step, "thought", think_content)
        setattr(new_step, "raw_model_response", response)

        self._trajectory.steps.append(new_step)

        return Action(action=tool_calls_dict)
