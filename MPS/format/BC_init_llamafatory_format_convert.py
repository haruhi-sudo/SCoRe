import json
import re

input_path = "output/math_bc_init_trajectories.jsonl"
output_path = "output/math_bc_init_trajectories_llamafactory.json"


with open(input_path, "r") as f:
    data = [json.loads(line) for line in f]

all_saves = []
for idx, item in enumerate(data):
    if "\": true" not in json.dumps(item["evaluation_result"]):
        continue
    if idx >= 15000:
        break
    question = item['original_task_info']['question']
    conversations = [{
        "from": "human",
        "value": question
    }]

    conversations.append({
        "from": "gpt",
        "value":  f"<first_thought>{item['first_thought'].replace('First-thought prefix:','').replace('First-thought:', '').strip()}</first_thought>"
    })
    conversations.append({
        "from": "human",
        "value": "Observation: None"
    })

    for step_item in item["steps"]:
        if "error" in step_item.keys():
            continue

        conversations.append({
            "from": "gpt",
            "value":  f"<thought>{step_item['thought'].replace('Thought: ', '')}</thought><code>{step_item['code']}</code>"
        })

        conversations.append({
            "from": "human",
            "value": f"Observation: {step_item['observation']}".strip()
        })
    

    saves = {}
    saves["conversations"] = conversations[:-1] # The last observation is not needed for LLamaFactory, because observation is not used for learning.

    all_saves.append(saves)

# save all_saves into a json file
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_saves, f, ensure_ascii=False, indent=4)
