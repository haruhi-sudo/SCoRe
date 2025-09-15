# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import logging
import os
import re
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("agent_distill")
class AgentDistillLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level AgentDistillLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        print(f"Initialized tools: {cls.tools}")

        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length

        try:
            cls.system_prompt = tokenizer.apply_chat_template(
                [{
                }], add_generation_prompt=False, tokenize=True
            )
        except Exception as e:
            cls.system_prompt = tokenizer.apply_chat_template(
                [{
                    "role": "system",
                    "content": ""
                }], add_generation_prompt=False, tokenize=True
            )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        metrics = {}
        request_id = uuid4().hex
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, tools=None, add_generation_prompt=True, tokenize=True
            ),
        )
        response_mask = []
        tools_kwargs = kwargs.get("tools_kwargs", {})

        user_turns, assistant_turns = 0, 0
        end_tag = False

        # Create tool instances for this conversation
        tool_instances = {}
        try:
            # Initialize tools_kwargs and create tool instances for the conversation
            for tool_name, tool in self.tools.items():
                if tool_name not in tools_kwargs:
                    tools_kwargs[tool_name] = {}
                
                instance_id, _ = await tool.create(create_kwargs=tools_kwargs[tool_name].get("create_kwargs", {}))
                tool_instances[tool_name] = instance_id

            await self._execute_prompt_code(messages, tools_kwargs, tool_instances)

            while True:
                # The cycle is over
                if end_tag:
                    break
                with simple_timer("generate_sequences", metrics):
                    response_ids = await self.server_manager.generate(
                        request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
                    )
                prompt_ids += response_ids
                response_mask += [1] * len(response_ids)
                assistant_turns += 1

                # reach max response length
                if len(response_mask) >= self.response_length:
                    break

                # reach max assistant turns
                if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                    break

                # reach max user turns
                if self.max_user_turns and user_turns >= self.max_user_turns:
                    break

                final_answer_print, tool_call = await self.tool_parser.extract_tool_calls(response_ids)
                end_tag = final_answer_print

                if not tool_call and "<first_thought>" not in self.tokenizer.decode(response_ids): # The first_thought does not need the tool_call
                    break

                # call tools
                with simple_timer("tool_calls", metrics):
                    tool_response = await self._call_tool(tool_call, tools_kwargs, tool_instances)
                if isinstance(tool_response, Exception):
                    break

                tool_response_ids = await self.loop.run_in_executor(
                    None,
                    lambda messages=[tool_response]: self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True
                    ),
                )
                tool_response_ids = tool_response_ids[len(self.system_prompt) :]

                # NOTE: last turn should not be user turn, or the EOS token reward
                # can't be propagated to previous token in GAE.
                if len(response_mask) + len(tool_response_ids) >= self.response_length:
                    break
                
                # tool_response_ids = [198] + tool_response_ids 
                prompt_ids += tool_response_ids # 198 is "\n"
                response_mask += [0] * len(tool_response_ids)
                user_turns += 1

        finally:
            # Clean up tool instances
            for tool_name, instance_id in tool_instances.items():
                tool = self.tools[tool_name]
                await tool.release(instance_id)

        response_ids = prompt_ids[-len(response_mask) :]
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
        )
        return output

    async def _call_tool(self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], tool_instances: dict[str, str]) -> dict[str, str]:
        """Call tool and return tool response."""
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or 
            if not tool_call:
                return {
                    "role": "user",  # do not need the </tool_call> tag
                    "content": "Observation: None",
                }

            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            instance_id = tool_instances[tool_name]
            tool_execution_response, _, _ = await tool.execute(instance_id, tool_args)

            if tool_name == "web_search":
                search_text = json.loads(tool_execution_response.text)["result"]
                search_text = f"Observation: {search_text}"

                return {
                    "role": "user",  # do not need the </tool_call> tag
                    "content": search_text,
                }
        except Exception as e:
            logger.exception(f"Error when executing tool: {e}")
            return e

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        return {
            "role": "user",  # do not need the </tool_call> tag
            "content": tool_response_text,
        }

    async def _execute_prompt_code(self, messages: list, tools_kwargs: dict[str, Any], tool_instances: dict[str, str]):
        """Execute code from prompt messages using the agent distill tool."""
        # Look for code blocks in assistant messages
        for msg in messages:
            if msg.get("role") == "assistant" and "content" in msg:
                content = msg["content"]
                # Extract code blocks using regex
                code_blocks = re.findall(r"<code>(.*?)</code>", content, re.DOTALL)
                
                for code in code_blocks:
                    try:
                        if code == "" or "web_search(" in code:
                            continue
                        tool_call = FunctionCall(name="exec_python_code_block", arguments=json.dumps({"code": code}))
                        
                        # Use the existing _call_tool method to execute the code
                        result = await self._call_tool(tool_call, tools_kwargs, tool_instances)
                        
                    except Exception as e:
                        logger.warning(f"Error executing code from prompt: {e}")
                        continue
