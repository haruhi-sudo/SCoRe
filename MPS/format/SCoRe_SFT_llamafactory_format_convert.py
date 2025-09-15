import json
import re
import random

input_path = "output/math_mps_trajectories.jsonl"
output_path = "output/math_mps_trajectories_llamafactory.json"


with open(input_path, "r") as f:
    data = [json.loads(line) for line in f]

all_saves = []
for idx, item in enumerate(data):
    if idx > len(data)/2: # Only save half of the data for SFT
        break
    repair_steps = item["repair_steps"]
    if "\"is_correct\": true" not in json.dumps(item["evaluation_result"]):
        continue
    if len(repair_steps) == 0 and random.random() > 0.5:
        continue
    question = item['original_task_info']['question']
    conversations = [{
        "from": "human",
        "value": question
    }]
    for idx, step_item in enumerate(item["steps"]):
        if idx == 0:
            conversations.append({
                "from": "gpt",
                "value":  f"<first_thought>{step_item['thought'].replace('First-thought prefix:','').replace('First-thought:', '').strip()}</first_thought>"
            })
            conversations.append({
                "from": "human",
                "value": "Observation: None"
            })
            continue

        if "error" in step_item.keys():
            continue
        conversations.append({
            "from": "gpt",
            "value": f"<thought>{step_item['thought']}</thought><code>{step_item['code']}</code>"
        })
        conversations.append({
            "from": "human",
            "value": f"Observation: {step_item['observation']}".strip()
        })
    

    saves = {}
    saves["conversations"] = conversations[:-1] # The last observation is not needed for LLamaFactory, because observation is not used for learning.

    all_saves.append(saves)


with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_saves, f, ensure_ascii=False, indent=4)
