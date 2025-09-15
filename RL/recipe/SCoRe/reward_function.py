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

from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
from typing import Dict, Any, Optional, List

from openai import OpenAI
import os
import re
import json
import logging
from dotenv import load_dotenv

# Load environment
def load_env_config():
    """
    Load configuration from .env file in current directory.
    """
    # Load .env file
    load_dotenv()
    load_dotenv(".local_env", override=True)

    return {
        "api_base": os.getenv("REWARD_API_BASE"),
        "api_key": os.getenv("REWARD_API_KEY"),
        "model_name": ""
    }

# Load configuration
config = load_env_config()
    
# Configuration
API_BASE = os.getenv("REWARD_API_BASE", "http://localhost:30000")
API_KEY = os.getenv("REWARD_API_KEY", "EMPTY")
MAX_RETRIES = 3
BASE_DELAY = 2
MAX_WORKERS = 32
MODEL_NAME = os.getenv("REWARD_MODEL_NAME", "")

# Setup logging
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def call_llm_api(
    user_prompt: str,
    system_prompt: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: str = "",
    max_tokens: int = 50,
    temperature: float = 0.0,
    response_format: Optional[Dict] = None,
):
    # Use environment variables as defaults
    api_base = api_base or API_BASE
    api_key = api_key or API_KEY
    
    for attempt in range(MAX_RETRIES):
        try:
            client = OpenAI(api_key=api_key, base_url=api_base)
            try:
                models = client.models.list()
                dynamic_model_id = models.data[0].id if models.data else model_name
            except Exception:
                dynamic_model_id = model_name
            
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
            ]
            
            # Add response_format if provided
            if response_format:
                response = client.chat.completions.create(
                    model=dynamic_model_id,
                    messages=messages,
                    temperature=temperature,
                    extra_body={"max_completion_tokens": max_tokens},
                    response_format=response_format
                )
            else:
                response = client.chat.completions.create(
                    model=dynamic_model_id,
                    messages=messages,
                    temperature=temperature,
                    extra_body={"max_completion_tokens": max_tokens}
                )
            response_content = response.choices[0].message.content

            return response_content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"Exception in call_llm_api: {repr(e)}")
                delay = BASE_DELAY * (2**attempt)
                logger.info(f"Retrying in {delay} seconds...")
                sleep(delay)
            else:
                logger.error(f"Failed after {MAX_RETRIES} attempts. Error: {e}")

    return "None"


def judge_answer_correctness(question: str, predicted_answer: str, ground_truth: str) -> bool:
    """
    Use LLM to judge if the predicted answer matches the ground truth.
    """
    system_prompt = """You are an expert judge evaluating whether two answers are equivalent.
    You will be given a predicted answer and a ground truth answer.
    Your task is to determine if they are semantically equivalent, ignoring minor differences in formatting, spacing, or notation.
    You must respond in JSON format with a 'equivalent' field that is either true or false."""
    
    user_prompt = f"""Question: {question}

Predicted answer: {predicted_answer}
    
Ground truth answer: {ground_truth}
    
Are these answers semantically equivalent? Respond in JSON format with only the 'equivalent' field.

Sometimes, predicted_answer may contain some irrelevant content, please ignore it, as long as predicted_answer contains the final answer, it is considered correct.
Example: predicted_answer: {{The two sets are different because the sum of the remainders cannot equal the sum of the integers under the given constraints.}}.ground_truth: {{The two sets are different.}}. predicted_answer should be considered correct.

Example response: {{"equivalent": true}}
"""

    try:
        response = call_llm_api(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            api_base=config["api_base"],
            api_key=config["api_key"],
            model_name=config["model_name"],
            max_tokens=50,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        return "true" in response.strip().lower()
        
    except Exception as e:
        # If there's any error in calling the API or parsing JSON, fall back to strict comparison
        logger.error(f"Error in judge_answer_correctness: {e}")
        return predicted_answer.strip() == str(ground_truth).strip()


def judge_step_equality(prefix: str, step1: str, step1_observation: str, step2: str, step2_observation: str) -> bool:
    """
    Use LLM to judge step equality.
    """
    
    user_prompt = f"""There are two agent trajectories, which is a loop of (thought, code, observation). The two trajectories share the same prefix and are different only at the last step.

At the final step, there are two alternative candidate completions:

Step 1: contains a `<thought>` reasoning step and a `<code>` block representing the agent’s plan and action.
Step 2: contains another `<thought>` and `<code>` block with a different plan/action.

Each completion is followed by its **Observation**, which is the execution output of the `<code>` block:

`step1_observation`: output for the step 1
`step2_observation`: output for the step 2

Your task: Decide whether the two steps are functionally equivalent — meaning they lead to the same outcome or are interchangeable in solving the given problem, even if their wording or code differs.

Guidelines for equivalence:

1. If both approaches produce nearly identical or semantically equivalent results, mark them as **Equivalent**.
2. If there is a difference in logic, reasoning correctness, or resulting output (numerical or semantic), mark them as **Not Equivalent**.
3. Consider both the reasoning (`<thought>`) and the actual execution results in `step1` and `step2`.
4. Focus on correctness and functional interchangeability, not surface similarity.
5. Pay attention to the final step of observation. Because the agent's action is code, codes with the same function will generally output the same output.
6. Return your answer in JSON:
```json
{{
  "equivalent": true/false,
  "explanation": "Brief justification of your decision."
}}
```

Now, here are the two steps:

Shared prefix:
{prefix}

Step 1:
{step1}
Observation of Step 1: 
{step1_observation}

Step 2:
{step2}
Observation of Step 2: 
{step2_observation}
"""

    try:
        response = call_llm_api(
            user_prompt=user_prompt,
            system_prompt="",
            api_base=config["api_base"],
            api_key=config["api_key"],
            model_name=config["model_name"],
            max_tokens=500,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        return "true" in response.strip().lower()
        
    except Exception as e:
        # If there's any error in calling the API or parsing JSON, fall back to strict comparison
        logger.error(f"Error in judge_answer_correctness: {e}")
        return False


def compute_reward(solution_str: str, ground_truth: str, extra_info: Dict) -> float:
    """
    Compute reward for a single solution string and ground truth.
    """
    reward_score = 0.0
    try:
        if "final_answer_print" not in solution_str:
            return 0.0

        pattern = r'(?s).*Observation:(.*?)assistant.*'
        matches = re.findall(pattern, solution_str, re.DOTALL)
        if matches:
            last_user_obs = matches[-1].strip()
        else:
            return 0.0

        first_step_after_prefix = solution_str.split("assistant\n")[0]
        step1 = first_step_after_prefix.split("user\nObservation:")[0].strip()
        step1_obs = first_step_after_prefix.split("user\nObservation:")[-1].strip()
    except Exception as e:
        logger.error(f"Error in compute_reward when extracting the information: {e}")
        return 0.0

    try:
        # Use LLM-based judgment instead of strict string comparison
        question = extra_info["question"]
        if judge_answer_correctness(question, last_user_obs, ground_truth):
            reward_score = 1.0
        else:
            reward_score = 0.0  # format reward
        
        if reward_score < 1.0 and "final_answer_print" not in step1 and extra_info["chosen_step"] != "":
            # 判断是否和chosen_step一致
            if judge_step_equality(
                prefix=extra_info["prefix"],
                step1=step1, 
                step1_observation=step1_obs, 
                step2=extra_info["chosen_step"], 
                step2_observation=extra_info["chosen_step_obs"], 
            ):
                reward_score += 0.5
            elif judge_step_equality( # 判断是否和rejected_step一致
                prefix=extra_info["prefix"], 
                step1=step1, 
                step1_observation=step1_obs, 
                step2=extra_info["rejected_step"], 
                step2_observation=extra_info["rejected_step_obs"], 
            ):
                reward_score += 0
            else:
                reward_score += 0.1

    except Exception as e:
        logger.error(f"Error in compute_reward when call the reward model: {e}")
        reward_score = 0.0
    
    return reward_score

def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: Dict) -> float:
    """
    Compute reward score for agent distillation.
    
    Args:
        data_source: The data source identifier
        solution_str: The solution string to evaluate
        ground_truth: The ground truth answer
        extra_info: Additional information for scoring
        
    Returns:
        float: The computed reward score
    """
    try:
        return compute_reward(solution_str, ground_truth, extra_info)
    except Exception as e:
        logger.error(f"Error in compute_score: {e}")
        return 0.0

def compute_score_batch(
    data_sources: List[str], 
    solution_strs: List[str], 
    ground_truths: List[str], 
    extra_infos: List[Dict]
) -> List[float]:
    """
    Compute reward scores for a batch of solutions concurrently.
    
    Args:
        data_sources: List of data source identifiers
        solution_strs: List of solution strings to evaluate
        ground_truths: List of ground truth answers
        extra_infos: List of additional information for scoring
        
    Returns:
        List[float]: List of computed reward scores
    """
    results = [0.0] * len(data_sources)

    logger.info(f"Compute reward scores for a batch of solutions concurrently, batch_size: {len(data_sources)}")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_index = {}
        for i, (data_source, solution_str, ground_truth, extra_info) in enumerate(
            zip(data_sources, solution_strs, ground_truths, extra_infos)
        ):
            future = executor.submit(compute_score, data_source, solution_str, ground_truth, extra_info)
            future_to_index[future] = i

        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
            except Exception as e:
                logger.error(f"Error computing score for index {index}: {e}")
                results[index] = 0.0

    return results
