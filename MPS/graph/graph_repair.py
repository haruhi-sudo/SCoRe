import os
import re
import json
import threading
import logging
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from openai import OpenAI

from configuration import Configuration
from functions import (
    python_interpreter, answer_evaluate, call_llm_api, SearchTools
)
from prompts import REACT_SYSTEM_PROMPT, REPAIR_PROMPT, REPAIR_FIRST_THOUGHT_PROMPT, FIRST_THOUGHT_SYSTEM_PROMPT

# Create logger
logger = logging.getLogger(__name__)

MAX_REPAIR_ATTEMPTS = 3  # Maximum number of repair attempts

# Add a lock for thread-safe file writing
log_file_lock = threading.Lock()

# Client configuration for SFT model
API_KEY = "0"
BASE_URL = "http://0.0.0.0:8000/v1"
MODEL_NAME = "."
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

class AgentState(TypedDict):
    """代理的状态定义。"""
    python_scope: Dict[str, Any]
    is_finished: bool
    steps_log: List[Dict[str, Any]]  # All steps taken
    original_task_info: Dict[str, Any]  # Original task information
    iteration_count: int  # Number of iterations
    evaluation_result: Dict[str, Any]  # Evaluation result
    repair_attempt: int  # Number of repair attempts
    failed_experience: str  # Failed experience for repair
    max_iterations: int  # Max iterations from config
    messages: List[Dict[str, str]]  # Conversation messages
    previous_context: str  # Previous context
    repair_thought: str  # Repair thought
    repair_code: str  # Repair code
    correction_start_step: int  # Step where correction should start
    # Student inital failure
    repair_steps: List[Dict[str, Any]]


def create_step_config(base_config: RunnableConfig, step_name: str) -> RunnableConfig:
    """Create a new configuration for a specific step with its designated model"""
    cfg = Configuration.from_runnable_config(base_config)
    
    # If step_models is configured and this step has a specific configuration
    if cfg.step_models and step_name in cfg.step_models:
        step_model_config = cfg.step_models[step_name]
        
        # Create a new config with the specific model for this step
        step_config = base_config.copy() if base_config else {}
        if "configurable" not in step_config:
            step_config["configurable"] = {}
            
        # Apply the step-specific model configuration
        step_config["configurable"]["model_name"] = step_model_config["name"]
        if "temperature" in step_model_config:
            step_config["configurable"]["temperature"] = step_model_config["temperature"]
        if "max_tokens" in step_model_config:
            step_config["configurable"]["max_tokens"] = step_model_config["max_tokens"]
            
        return step_config
    
    # If no specific configuration for this step, return the base config
    return base_config

def reasoning_node(state: AgentState, config: RunnableConfig):
    """Reasoning node using SFT model similar to inference_api.py"""
    # Use existing messages or initialize with the original query
    messages = state.get("messages")
    
    # Call the SFT model
    try:
        llm_resp = client.chat.completions.create(messages=messages, model=MODEL_NAME)
        thought_code_content = llm_resp.choices[0].message.content

        # Extract code part from response
        code_match = re.search(r'<code>([\s\S]+)</code>', thought_code_content, re.DOTALL)
        if code_match:
            code_content = code_match.group(1).strip()
            thought_content = thought_code_content[:code_match.start()].strip()
            thought_content = thought_content.replace("<thought>", "").replace("</thought>", "")
        else:
            code_content = ''
            thought_content = thought_code_content.replace("<first_thought>", "").replace("</first_thought>", "")

        # Execute code if exists
        if code_content:
            # Use the persistent python_scope from state and update it with mock tools
            scope = state["python_scope"]
            query_match = re.search(r'web_search\("(.*?)"\)', code_content, re.DOTALL)
            if query_match:
                result = {}
                web_search_query = query_match.group(1)
                result["output"] = SearchTools.web_search(web_search_query)
                result["updated_scope"] = scope
            else:
                result = python_interpreter(code_content, scope)
            
            observation_content = result["output"]
            updated_scope = result["updated_scope"]
        else:
            observation_content = "None"
            updated_scope = state["python_scope"]

        # Update messages
        message_single_turn = [
            {"role": "assistant", "content": thought_code_content},
            {"role": "user", "content": observation_content}
        ]
        updated_messages = messages + message_single_turn
        
        # Check if finished
        is_finished = False
        if "final_answer_print" in thought_code_content.lower():
            is_finished = True

        # Log this step
        step_entry = {
            "thought": thought_content,
            "code": code_content,
            "observation": observation_content,
            "is_finished": is_finished
        }

        return {
            "messages": updated_messages,
            "python_scope": updated_scope,  # Updated scope with persistent variables
            "is_finished": is_finished,
            "steps_log": state["steps_log"] + [step_entry],  # Add step to log
            "iteration_count": state["iteration_count"] + 1,  # Increment iteration count
        }
        
    except Exception as e:
        task_id = state.get("original_task_info", {}).get("id", "unknown")
        logger.error(
            f"Error in reasoning_node for task {task_id}: {e}",
            extra={
                "task_id": task_id,
                "node": "reasoning_node",
                "error_type": type(e).__name__
            }
        )
        # Add error information to steps log
        error_entry = {
            "error": str(e),
            "node": "reasoning_node",
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "messages": messages,
            "python_scope": state["python_scope"],
            "is_finished": False,
            "steps_log": state["steps_log"] + [error_entry],
            "iteration_count": state["iteration_count"] + 1,
        }

def repair_generation_node(state: AgentState, config: RunnableConfig):
    """Repair node that generates the next thought and code based on failure experience"""
    step_config = create_step_config(config, "repair")
    cfg = Configuration.from_runnable_config(step_config)
    
    # Get the failed experience
    failed_experience = state.get("failed_experience", "None")

    # Get the original query
    original_query = state["original_task_info"].get("question", "")

    # Build previous context from steps log, only up to correction_start_step
    steps_log = state["steps_log"]
    correction_start_step = state.get("correction_start_step", -1)
    
    # If correction_start_step is valid, only include steps up to that point
    # Note: correction_start_step is 1-indexed, so we need to convert to 0-indexed
    if 1 <= correction_start_step <= len(steps_log):
        # Truncate steps_log to only include steps before the correction_start_step
        truncated_steps_log = steps_log[:correction_start_step-1]  # Convert to 0-indexed
        # Get the error step (1-indexed)
        error_step_index = correction_start_step - 1  # Convert to 0-indexed
        error_step = steps_log[error_step_index] if error_step_index < len(steps_log) else {}
    else:
        # Otherwise, use all steps
        truncated_steps_log = steps_log
        error_step = {}

    previous_context = ""
    for i, step in enumerate(truncated_steps_log):
        # Use 1-indexed for display
        previous_context += f"Step {i+1}:\n"
        if i == 0:
            first_thought = step["thought"].replace("<first_thought>", "").replace("</first_thought>", "").strip()
            previous_context += f"First Thought: {first_thought}\n"
            continue
        if "thought" in step:
            previous_context += f"Thought: {step['thought']}\n"
        if "code" in step:
            previous_context += f"Code:\n```python\n{step['code']}\n```\n"
        if "observation" in step:
            previous_context += f"Observation: {step['observation']}\n"
        previous_context += "\n"

    # If no relevant steps, set to "None"
    if not previous_context.strip():
        previous_context = "None"
        
    # Format the error step information
    error_step_info = "None"
    if error_step:
        if correction_start_step == 1: # First Thought mistake
            error_step_info = f"Step {correction_start_step} (Error Step):\n"
            error_step_info += f"First Thought: {error_step['thought']}\n"
        else:
            error_step_info = f"Step {correction_start_step} (Error Step):\n"
            if "thought" in error_step:
                error_step_info += f"Thought: {error_step['thought']}\n"
            if "code" in error_step:
                error_step_info += f"Code:\n```python\n{error_step['code']}\n```\n"
            if "observation" in error_step:
                error_step_info += f"Observation: {error_step['observation']}\n"

    if correction_start_step == 1: # First Thought mistake
        repair_prompt = REPAIR_FIRST_THOUGHT_PROMPT.format(
            original_query=original_query,
            failed_experience=failed_experience,
            previous_context=previous_context,
            error_step=error_step_info
        )    
        repair_system_prompt = FIRST_THOUGHT_SYSTEM_PROMPT
    else:
        repair_prompt = REPAIR_PROMPT.format(
            original_query=original_query,
            failed_experience=failed_experience,
            previous_context=previous_context,
            error_step=error_step_info
        )
        repair_system_prompt = REACT_SYSTEM_PROMPT

    try:
        # Call the LLM with the repair prompt using the reasoning model config
        repair_response = call_llm_api(
            user_prompt=repair_prompt,
            system_prompt=repair_system_prompt,
            api_base=cfg.api_base,
            api_key=cfg.api_key,
            model_name=cfg.model_name,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )

        # Extract Thought and Code. The thought is everything outside/before the code block.
        code_match = re.search(r'```python(.*?)```', repair_response, re.DOTALL)
        if code_match:
            repair_code = code_match.group(1).strip()
            # The thought is the part of the content before the code block starts.
            repair_thought = repair_response[:code_match.start()].strip()
        else:
            # If no code block is found, there's no code, and the whole content is the thought.
            repair_code = ''
            repair_thought = repair_response.strip()

        # Clean up keywords from the extracted thought
        repair_thought = repair_thought.replace('Code:', '').strip()
        
        return {
            "repair_thought": repair_thought,
            "repair_code": repair_code,
            "previous_context": previous_context  # Store previous context for training
        }

    except Exception as e:
        task_id = state.get("original_task_info", {}).get("id", "unknown")
        logger.error(
            f"Error in repair_generation_node for task {task_id}: {e}",
            extra={
                "task_id": task_id,
                "node": "repair_generation_node",
                "error_type": type(e).__name__
            }
        )
        
        # Add error information
        error_message = f"Error in repair generation: {str(e)}"
        
        return {
            "repair_thought": error_message,
            "repair_code": "",
            "previous_context": ""
        }

def repair_execution_node(state: AgentState, config: RunnableConfig):
    """Execute the repair thought and code"""
    repair_thought = state.get("repair_thought", "")
    repair_code = state.get("repair_code", "")
    previous_context = state.get("previous_context", "")  # Get previous context from state
    
    # Execute code if exists
    if repair_code:
        scope = state["python_scope"]
        
        # extract the web_search("...") query
        query_match = re.search(r'web_search\("(.*?)"\)', repair_code, re.DOTALL)
        if query_match:
            result = {}
            web_search_query = query_match.group(1)
            result["output"] = SearchTools.web_search(web_search_query)
            result["updated_scope"] = scope
        else:
            result = python_interpreter(repair_code, scope)

        observation_content = result["output"]
        updated_scope = result["updated_scope"]
    else:
        observation_content = "No Code Executed."
        updated_scope = state["python_scope"]
    
    # Update messages with repair step
    messages = state.get("messages", [])
    correction_start_step = state.get("correction_start_step", -1)

    # Remove steps after correction_start_step
    step_id = 0
    for idx, message in enumerate(messages):
        if message["role"] == "user":
            step_id += 1
        if step_id == correction_start_step:
            truncated_messages = messages[:idx+1]
            break

    repair_content = f"<thought>{repair_thought}</thought><code>{repair_code}</code>"
    message_single_turn = [
        {"role": "assistant", "content": repair_content},
        {"role": "user", "content": f"Observation: {observation_content}"}
    ]
    updated_messages = truncated_messages + message_single_turn
    
    # Check if finished
    is_finished = False
    if "final_answer_print" in repair_content.lower():
        is_finished = True

    # Log this step
    step_entry = {
        "thought": repair_thought,
        "code": repair_code,
        "observation": observation_content,
        "is_finished": is_finished
    }
    
    # Truncate steps_log to remove steps after correction_start_step
    steps_log = state["steps_log"]
    correction_start_step = state.get("correction_start_step", -1)
    correction_suggestion = state.get("failed_experience", "")
    
    # Note: correction_start_step is 1-indexed, so we need to convert to 0-indexed
    if 1 <= correction_start_step <= len(steps_log):
        # Truncate steps_log to only include steps before the correction_start_step
        truncated_steps_log = steps_log[:correction_start_step-1]  # Convert to 0-indexed
    else:
        # Keep all steps if correction_start_step is invalid
        truncated_steps_log = steps_log
    
    # Create learning episode if we have a repair attempt
    repair_steps = state.get("repair_steps", [])
    repair_attempt = state.get("repair_attempt", 0)
    
    # Only create learning episode if we have both student mistake and teacher correction
    if repair_attempt > 0:
        # Get the student's mistake that we're correcting
        # Note: correction_start_step is 1-indexed, so we need to convert to 0-indexed
        if 1 <= correction_start_step <= len(steps_log):
            # Get the failed step (0-indexed)
            failed_step_index = correction_start_step - 1
            if failed_step_index < len(steps_log):
                failed_step = steps_log[failed_step_index]
                
                # Create learning episode
                key_step = {
                    "teacher_guided_step": {
                        "thought": repair_thought,
                        "code": repair_code,
                        "observation": observation_content
                    },
                    "student_mistake": {
                        "thought": failed_step.get("thought", ""),
                        "code": failed_step.get("code", ""),
                        "observation": failed_step.get("observation", "")
                    },
                    "mistake_step_index": failed_step_index,
                    "correction_suggestion": correction_suggestion,
                    "context_before_intervention": previous_context,  # Use the previous_context from state
                    "is_valid": False  # Initially mark as not approved, will be validated later
                }
                repair_steps.append(key_step)
    
    return {
        "messages": updated_messages,
        "python_scope": updated_scope,
        "is_finished": is_finished,
        "steps_log": truncated_steps_log + [step_entry],  # Use truncated steps_log
        "iteration_count": state["iteration_count"] + 1,
        "repair_thought": "",  # Clear repair thought
        "repair_code": "",     # Clear repair code
        "previous_context": "", # Clear previous context
        "repair_steps": repair_steps  # Store learning trajectories
    }

def evaluation_node(state: AgentState, config: RunnableConfig):
    """Evaluate the correctness of the final answer."""
    # Create step-specific configuration for evaluation model
    step_config = create_step_config(config, "evaluation")
    cfg = Configuration.from_runnable_config(step_config)
    
    # Extract necessary information
    original_task_info = state["original_task_info"]
    steps_log = state["steps_log"]
    
    # Get the final answer from the last step if available
    generated_answer = ""
    if steps_log and len(steps_log) > 0:
        last_step = steps_log[-1]
        if not last_step.get("is_finished", False):
            # Mark examples as not approved if not finished
            repair_steps = state.get("repair_steps", [])
            for episode in repair_steps:
                episode["is_valid"] = False
                
            return {
                "evaluation_result": {"is_correct": False, "error_analysis": "Not finished"},
                "repair_steps": repair_steps
            }
        
        if "observation" in last_step:
            generated_answer = last_step["observation"]
    
    # Format the thought-code cycle history
    thought_code_cycle = ""
    for i, step in enumerate(steps_log):
        thought_code_cycle += f"Step {i+1}:\n"
        if i == 0:
            first_thought = step["thought"].replace("<first_thought>", "").replace("</first_thought>", "").strip()
            thought_code_cycle += f"First Thought: {first_thought}\n"
            continue
        if "thought" in step:
            thought_code_cycle += f"Thought: {step['thought']}\n"
        if "code" in step:
            thought_code_cycle += f"Code:\n```python\n{step['code']}\n```\n"
        if "observation" in step:
            thought_code_cycle += f"Observation: {step['observation']}\n"
        thought_code_cycle += "\n"
    
    try:
        evaluation_result_str = answer_evaluate(
            cfg, question=original_task_info.get("question", ""),
            true_answer=original_task_info.get("true_answer", ""),
            generated_answer=generated_answer,
            thought_code_cycle=thought_code_cycle
        )
        
        # Parse the evaluation result as JSON
        evaluation_result = json.loads(evaluation_result_str)

        # Update learning trajectories based on evaluation result
        repair_steps = state.get("repair_steps", [])
        if isinstance(evaluation_result, dict):
            is_correct = evaluation_result.get("is_correct", False)
            # Mark all repair steps as approved if the final answer is correct
            for episode in repair_steps:
                episode["is_valid"] = is_correct
        
        # Add evaluation result to the state
        return {
            "evaluation_result": evaluation_result,
            "repair_steps": repair_steps
        }
    except Exception as e:
        task_id = state.get("original_task_info", {}).get("id", "unknown")
        logger.error(
            f"Error in evaluation_node for task {task_id}: {e}",
            extra={
                "task_id": task_id,
                "node": "evaluation_node",
                "error_type": type(e).__name__
            }
        )
        
        # Return a default evaluation result indicating an error
        return {
            "evaluation_result": {
                "is_correct": False,
                "error_analysis": f"Evaluation failed: {str(e)}",
                "correction_start_step": None,
                "correction_suggestion": None
            },
            "repair_steps": state.get("repair_steps", [])
        }

def repair_analysis_node(state: AgentState, config: RunnableConfig):
    """Analyze the failure and prepare for repair"""
    # Create step-specific configuration for reasoning model
    step_config = create_step_config(config, "repair")
    cfg = Configuration.from_runnable_config(step_config)
    
    original_task_info = state["original_task_info"]
    evaluation_result = state["evaluation_result"]
    
    # Prepare failed experience from evaluation result
    failed_experience = ""
    correction_start_step = -1
    if isinstance(evaluation_result, dict):
        error_analysis = evaluation_result.get("error_analysis", "")
        correction_suggestion = evaluation_result.get("correction_suggestion", "")
        correction_start_step = evaluation_result.get("correction_start_step", -1)
        
        failed_experience = f"Error Analysis: {error_analysis}\n"
        failed_experience += f"Correction Suggestion: {correction_suggestion}\n"
        failed_experience += f"Correction Start Step: {correction_start_step}"
    
    # Update the state with failed experience and correction start step for the next reasoning cycle
    return {
        "failed_experience": failed_experience,
        "repair_attempt": state.get("repair_attempt", 0) + 1,
        "correction_start_step": correction_start_step,
        "iteration_count": 0  # Reset iteration count for the new repair attempt
    }

def should_continue(state: AgentState):
    max_iterations = state.get("max_iterations", 10)  # Default to 10 if not set
    # Check if we've reached the maximum iterations
    
    if state.get("iteration_count", 0) >= max_iterations:
        return "evaluate"
    # Check if the task is finished
    elif state.get("is_finished"):
        return "evaluate"
    else:
        return "continue"

def should_repair_or_end(state: AgentState):
    """Determine whether to repair or end based on evaluation result."""
    evaluation_result = state.get("evaluation_result", {})
    
    # Check if evaluation result is a dict with is_correct field
    if isinstance(evaluation_result, dict) and "is_correct" in evaluation_result:
        is_correct = evaluation_result["is_correct"]
        repair_attempts = state.get("repair_attempt", 0)
        
        # Check if correction_start_step is -1, which indicates a hard sample
        correction_start_step = evaluation_result.get("correction_start_step", -1)
        if correction_start_step == -1 and not is_correct:
            # Mark as hard sample and end
            return "end"
        
        # If answer is correct or we've reached max repair attempts, end
        if is_correct or repair_attempts >= MAX_REPAIR_ATTEMPTS:
            return "end"
        else:
            # Answer is incorrect and we have more repair attempts
            return "repair_analysis"
    else:
        # If we can't determine correctness, end
        return "end"

def should_end_after_repair(state: AgentState):
    """After repair, go back to reasoning."""
    return "reason_and_act"

# Build the graph with separate repair nodes
builder = StateGraph(AgentState, config_schema=RunnableConfig)
builder.add_node("reason_and_act", reasoning_node)
builder.add_node("evaluate_answer", evaluation_node)
builder.add_node("repair_analysis", repair_analysis_node)
builder.add_node("repair_generation", repair_generation_node)
builder.add_node("repair_execution", repair_execution_node)

builder.set_entry_point("reason_and_act")  # Start directly with reasoning
builder.add_conditional_edges(
    "reason_and_act",
    should_continue,
    {"continue": "reason_and_act", "evaluate": "evaluate_answer"}
)
builder.add_conditional_edges(
    "evaluate_answer",
    should_repair_or_end,
    {"repair_analysis": "repair_analysis", "end": END}
)
builder.add_edge("repair_analysis", "repair_generation")
builder.add_edge("repair_generation", "repair_execution")
# builder.add_edge("repair_execution", "reason_and_act")
builder.add_conditional_edges(
    "repair_execution",
    should_continue,
    {"continue": "reason_and_act", "evaluate": "evaluate_answer"}
)
graph = builder.compile()

# --- 运行入口 ---
def run_agent_repair(prompt: str, config: dict = None, original_task_info: dict = None): 
    run_config = {"configurable": config or {}}

    max_iterations = 10  # Default value
    if config and "processing" in config and "max_reasoning_iterations" in config["processing"]:
        max_iterations = config["processing"]["max_reasoning_iterations"]
    
    # Set recursion limit to prevent infinite loops
    # if "recursion_limit" not in run_config["configurable"]:
    run_config["recursion_limit"] = 200  # Increase from default 25
    
    initial_state = {
        "python_scope": {},  # Initialize empty python scope
        "is_finished": False,
        "steps_log": [],  # Initialize empty steps log
        "original_task_info": original_task_info or {"question": prompt},  # Store original task info
        "iteration_count": 0,  # Initialize iteration count
        "evaluation_result": {},  # Initialize evaluation result
        "repair_attempt": 0,  # Initialize repair attempt counter
        "failed_experience": "None",  # Initialize failed experience
        "messages": [{"role": "user", "content": prompt}],  # Initialize messages for SFT model
        "repair_thought": "",  # Initialize repair thought
        "repair_code": "",     # Initialize repair code
        "correction_start_step": -1,  # Initialize correction start step
        "repair_steps": []  # Initialize learning trajectories
    }
    
    final_state = graph.invoke(initial_state, config=run_config)
    
    # Save all steps to the single JSONL file
    save_steps_log(
        final_state['steps_log'], 
        prompt, 
        final_state['original_task_info'],
        final_state.get('evaluation_result', {}),
        config.get("logging", {}).get("log_file_path", "agent_logs.jsonl") if config else "agent_logs.jsonl",
        final_state.get('repair_steps', [])
    )
    
    return final_state

def save_steps_log(steps_log: List[Dict[str, Any]], original_prompt: str, 
                   original_task_info: dict = None, evaluation_result: dict = None,
                   log_file_path: str = "agent_logs.jsonl",
                   repair_steps: List[Dict[str, Any]] = None):
    """Save all steps to a single JSONL file (thread-safe)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_data = {
        "timestamp": timestamp,
        "steps": steps_log
    }
    
    # Include original task information if available
    if original_task_info:
        log_data["original_task_info"] = original_task_info
        
    # Include evaluation result if available
    if evaluation_result:
        log_data["evaluation_result"] = evaluation_result
        
    # Check if this is a hard sample (correction_start_step = -1)
    if (isinstance(evaluation_result, dict) and 
        evaluation_result.get("correction_start_step", -1) == -1 and
        evaluation_result.get("is_correct") == False):
        log_data["is_hard_sample"] = True
        
    # Include learning trajectories if available
    if repair_steps is not None:
        processed_repair_steps = []
        
        if len(repair_steps) > 0:
            final_trajectory_item = repair_steps[-1]
            
            if final_trajectory_item["is_valid"]:
                # Collect all student mistakes that led to success
                marked_student_mistakes = []
                
                # Process all trajectory items except the last one
                for trajectory_item in repair_steps[:-1]:
                    # Create a copy of the trajectory item to avoid modifying the original
                    processed_item = trajectory_item.copy()
                    marked_student_mistakes.append(processed_item["student_mistake"])
                    
                    # Check if this item's teacher guided step is in the final context
                    if processed_item["teacher_guided_step"]["thought"] not in final_trajectory_item["context_before_intervention"]:
                        processed_item["is_valid"] = False
                    else:
                        # Add student mistakes to the list
                        processed_item["student_mistake"] = marked_student_mistakes.copy()
                        marked_student_mistakes = []
                    
                    processed_repair_steps.append(processed_item)
                
                # Process the final trajectory item
                final_processed_item = final_trajectory_item.copy()
                marked_student_mistakes.append(final_processed_item["student_mistake"])
                final_processed_item["student_mistake"] = marked_student_mistakes.copy()
                processed_repair_steps.append(final_processed_item)
            else:
                # If final item is not approved, just copy all items as they are
                processed_repair_steps = [item.copy() for item in repair_steps]
        
        # Filter to only include approved learning episodes
        approved_repair_steps = [episode for episode in processed_repair_steps if episode.get("is_valid", False)]
        log_data["repair_steps"] = approved_repair_steps
    
    # Thread-safe write to the JSONL file
    with log_file_lock:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    # Example usage
    task = "Find the sum of all even numbers between 1 and 100."
    run_agent_repair(task)
