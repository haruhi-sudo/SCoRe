# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
import io
import re
import sys
import json
import multiprocessing
import types
import ast
import pickle
import asyncio
import logging
import os
import subprocess
import signal
from contextlib import redirect_stdout
from typing import Dict, Any, Optional, Type, Tuple
from uuid import uuid4

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# This is the script that will be executed in the subprocess.
# It's designed to be self-contained to minimize dependencies.
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_CURRENT_DIR, 'executor_script.py'), 'r') as f:
    EXECUTOR_SCRIPT = f.read()


class AgentDistillTool(BaseTool):
    """
    A tool to execute Python code in an isolated subprocess, maintaining state
    across multiple calls for a single instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: Dict[str, Dict[str, Any]] = {}
        self.timeout = config.get("timeout", 20)
        self._semaphore = asyncio.Semaphore(config.get("max_tool_concurrent", 5))

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> Tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        
        # Create a long-running subprocess for this instance
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", "-c", EXECUTOR_SCRIPT,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        # Store the process in the instance dict
        self._instance_dict[instance_id] = {
            "proc": proc,
            "scope": {}
        }
        
        return instance_id, ToolResponse()

    async def python_interpreter_async(self, instance_id: str, code: str, scope: Dict[str, Any], timeout: int = 10) -> Dict:
        """
        Executes Python code in the subprocess associated with the instance.
        """
        if instance_id not in self._instance_dict:
            logger.error(f"python_interpreter_async: invalid instance ID: {instance_id}")
            return {"output": "Error: Invalid instance ID", "updated_scope": scope}
        
        proc = self._instance_dict[instance_id]["proc"]
        if proc.returncode is not None:
            logger.warning(f"python_interpreter_async: process has terminated: {instance_id}")
            try:
                new_proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-u", "-c", EXECUTOR_SCRIPT,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                self._instance_dict[instance_id]["proc"] = new_proc
            except Exception as e:
                logger.error(f"Failed to restart process: {e}")
                return {"output": "Error: Failed to restart process", "updated_scope": scope}
            
            # 重新执行代码
            return await self.python_interpreter_async(instance_id, code, scope, timeout)
        try:
            payload = {"code": code, "scope": scope, "timeout": timeout}
            payload_bytes = pickle.dumps(payload)
            
            # Send data with length prefix
            proc.stdin.write(len(payload_bytes).to_bytes(4, byteorder='little'))
            proc.stdin.write(payload_bytes)
            await proc.stdin.drain()

            # Read response length - use a more generous timeout for communication
            length_bytes = await asyncio.wait_for(
                proc.stdout.readexactly(4),
                timeout=timeout
            )
            
            if not length_bytes:
                return {"output": "Error: Process terminated unexpectedly", "updated_scope": scope}
                
            data_length = int.from_bytes(length_bytes, byteorder='little')
            
            # Read the actual response
            stdout_data = await asyncio.wait_for(
                proc.stdout.readexactly(data_length),
                timeout=timeout
            )

            if not stdout_data:
                return {"output": "No output.", "updated_scope": scope}

            try:
                result = pickle.loads(stdout_data)
            except (pickle.UnpicklingError, TypeError, ValueError) as e:
                return {"output": "Error: An unexpected error, try again.", "updated_scope": scope}
            
            return result

        except asyncio.IncompleteReadError:
            logger.warning("python_interpreter_async: incomplete read from subprocess")
            return {"output": "Error: Process communication error, try again", "updated_scope": scope}
        except asyncio.TimeoutError:
            # breakpoint()
            logger.warning(f"python_interpreter_async: timeout reached while executing generated Python code: {instance_id}.")
            try:
                proc.kill()
                await proc.wait()
                # Restart the subprocess
                new_proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-u", "-c", EXECUTOR_SCRIPT,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                self._instance_dict[instance_id]["proc"] = new_proc
                # self._instance_dict[instance_id]["scope"] = {}
            except ProcessLookupError:
                pass
            return {"output": f"Error: Timeout, check your code.", "updated_scope": scope}
        except Exception as e:
            # breakpoint()
            logger.warning(f"Error occurs while executing generated Python code: {e}")
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            return {"output": "Error: An unexpected error, try again.", "updated_scope": scope}

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[ToolResponse, dict, dict]:
        async with self._semaphore:
            code = parameters.get("code", "")
            scope = self._instance_dict[instance_id].get("scope", {})
            
            code_exec_res = await self.python_interpreter_async(instance_id, code, scope, self.timeout)
            
            code_exec_text = code_exec_res.get("output", "")
            tool_updated_scope = code_exec_res.get("updated_scope", {})
            
            # Update the scope for this instance
            self._instance_dict[instance_id]["scope"] = tool_updated_scope

            return ToolResponse(text=f"Observation: {code_exec_text}"), {}, {}
    async def release(self, instance_id: str, **kwargs) -> None:
        """Releases the resources associated with an instance."""
        if instance_id in self._instance_dict:
            proc = self._instance_dict[instance_id]["proc"]
            try:
                if proc.returncode is None:  # Process is still running
                    proc.kill()
                    await proc.wait()
            except ProcessLookupError:
                pass
            except Exception as e:
                logger.warning(f"Error terminating process: {e}")
            
            del self._instance_dict[instance_id]
