import json
import re
import yaml
import random

input_path = "output/math_mps_trajectories.jsonl"
already_sft_path = "output/math_mps_trajectories_llamafactory.json"
output_path = "output/math_mps_rl.json"

# If a item has been used in sft, it should not be used in rl
with open(already_sft_path) as f:
    already_sft_data = json.load(f)

already_sft_ids = [item["conversations"][0]["value"] for item in already_sft_data]
def extract_steps(text):
    """
    Extract steps from text, each containing thought, code, and observation.
    
    Args:
        text (str): Input text containing multiple steps
        
    Returns:
        list: A list of dictionaries, each containing step info with thought, code, observation
    """
    # Split text by "Step" but keep the step number
    step_blocks = re.split(r'(Step \d+:\n)', text)
    
    # Remove empty strings from the beginning if any
    if step_blocks and step_blocks[0] == '':
        step_blocks = step_blocks[1:]
        
    steps = []
    i = 0
    
    while i < len(step_blocks):
        # If current block is a step header
        if re.match(r'Step \d+:', step_blocks[i]):
            step_info = {
                'step': step_blocks[i].strip()
            }
            
            # Get the content of this step (next block)
            if i + 1 < len(step_blocks):
                content = step_blocks[i + 1]
                if i == 0:
                    step_info['thought'] = content.replace("First Thought:", "").strip()
                    step_info['code'] = ""
                    step_info['observation']  = "None"
                    steps.append(step_info)
                    i += 2
                    continue
                
                # Extract Thought
                thought_match = re.search(r"Thought:\s*(.*?)(?:\nCode:|\n```python)", content, re.DOTALL | re.IGNORECASE)
                step_info['thought'] = thought_match.group(1).strip() if thought_match else ""
                
                # Extract Code
                code_match = re.search(r"```python\s*(.*?)\s*```", content, re.DOTALL)
                step_info['code'] = code_match.group(1).strip() if code_match else ""
                
                # Extract Observation
                observation_match = re.search(r"Observation:\s*(.*?)(?:\n\n|$)", content, re.DOTALL | re.IGNORECASE)
                step_info['observation'] = observation_match.group(1).strip() if observation_match else ""
                
                steps.append(step_info)
            
            i += 2  # Move to next step
        else:
            i += 1  # Move to next block
    
    return steps


with open(input_path, "r") as f:
    data = [json.loads(line) for line in f]

all_saves = []
for idx, item in enumerate(data):
    repair_steps = item["repair_steps"]
    question = item['original_task_info']['question']
    gt_answer = item['original_task_info']['true_answer']

    if question in already_sft_ids: # If a item has been used in sft, it should not be used in rl
        continue

    if len(repair_steps) == 0:
        if "is_hard_sample" in item.keys():  # save part of hard examples for RL training
            keep_threshold = 0.5
        else: # easy sample
            keep_threshold = 0.1

        if random.random() > keep_threshold:
            continue
        
        all_saves.append({
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "chosen": "",
            "chosen_observation": "",
            "rejected": "",
            "rejected_observation": "",
            "correct_step": "",
            "gt_answer": gt_answer
        })

    else:
        for repair_steps_item in reversed(repair_steps):
            conversations = [{
                "role": "user",
                "content": question
            }]
            pre_steps = extract_steps(repair_steps_item["context_before_intervention"])

            for idx, pre_step in enumerate(pre_steps):
                if idx == 0:
                    conversations.append({
                        "role": "assistant",
                        "content": f"<first_thought>{pre_step['thought']}</first_thought>"
                    })
                else:
                    conversations.append({
                        "role": "assistant",
                        "content": f"<thought>{pre_step['thought']}</thought><code>{pre_step['code']}</code>"
                    })
                conversations.append({
                    "role": "user",
                    "content": f"Observation: {pre_step['observation']}".strip()
                })
            
            saves = {}

            if len(pre_steps) == 0:
                saves = {
                    "prompt": conversations,
                    "chosen": "",
                    "chosen_observation": "",
                    "rejected": "",
                    "rejected_observation": "",
                    "correct_step": "",
                    "gt_answer": gt_answer
                }
            else:
                chosen_step = repair_steps_item["teacher_guided_step"]
                
                saves["prompt"] = conversations
                saves["chosen"] = f"<thought>{chosen_step['thought']}</thought><code>{chosen_step['code']}</code>"
                saves["chosen_observation"] = f"Observation: {chosen_step['observation'].strip()}"
                
                for one_rejected_step in repair_steps_item["student_mistake"]:
                    saves["rejected"] = f"<thought>{one_rejected_step['thought']}</thought><code>{one_rejected_step['code']}</code>"

                    observation = one_rejected_step['observation'].strip()
                    saves["rejected_observation"] = f"Observation: {observation}"
                    
                    reason = repair_steps_item['correction_suggestion'].split("Correction Start Step:")[0]

                    chosen = saves['chosen'].split("<thought>")[-1]
                    saves["correct_step"] = f"<thought>{reason}Correct Step: {chosen}"
                    saves["gt_answer"] = gt_answer

                    break
            
            all_saves.append(saves)
            break

# save all_saves into a json file
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_saves, f, ensure_ascii=False, indent=4)
