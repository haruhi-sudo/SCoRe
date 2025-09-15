"""
Preprocess the agent distill dataset to parquet format
Supports different modes: standard, long_horizon, sparse_reward
"""

import argparse
import os
import re
import json

import datasets
from datasets import Dataset


def make_map_fn(split, mode="standard"):
    """
    Create a processing function based on the mode
    
    Args:
        split: Dataset split name
        mode: Processing mode (standard, long_horizon, sparse_reward)
    """
    def process_fn(example, idx):
        question = example["prompt"][0]["content"]
        solution = str(example["gt_answer"])
        prefix_text = ""
        
        # Handle prefix text differently based on mode
        if mode == "standard":
            for agent_step in example["prompt"]:
                prefix_text += f"{agent_step['content']}\n\n"
        else:
            prefix_text = ""

        # Configure prompt based on mode
        if mode == "long_horizon":
            prompt = example["prompt"][:1]  # Only first prompt element
        else:
            prompt = example["prompt"]  # All prompt elements

        # Configure extra info based on mode
        if mode == "standard":
            chosen_step = example["chosen"]
            chosen_step_obs = example["chosen_observation"]
            rejected_step = example["rejected"]
            rejected_step_obs = example["rejected_observation"]
        else:
            chosen_step = ""
            chosen_step_obs = ""
            rejected_step = ""
            rejected_step_obs = ""
        
        data = {
            "data_source": "agent_distill",
            "agent_name": "agent_distill",
            "prompt": prompt,
            "ability": None,
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": solution,
                "question": question,
                "prefix": prefix_text,
                "chosen_step": chosen_step,
                "chosen_step_obs": chosen_step_obs,
                "rejected_step": rejected_step,
                "rejected_step_obs": rejected_step_obs,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "exec_python_code_block": {
                        "create_kwargs": {
                            "ground_truth": solution,
                        },
                    },
                },
                "interaction_kwargs": {
                    "query": question,
                    "ground_truth": solution,
                },
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="my_data/SCoRe_math")
    parser.add_argument("--mode", choices=["standard", "long_horizon", "sparse_reward"], 
                        default="standard", help="Processing mode")
    parser.add_argument("--train_data_source", 
                        default="my_data/raw/math/SCoRe_train.json")
    parser.add_argument("--test_data_sources", nargs="+", 
                        default=["my_data/raw/math/SCoRe_val.json"])
    
    args = parser.parse_args()

    train_data_source = args.train_data_source
    test_data_sources = args.test_data_sources
    
    with open(train_data_source, "r") as f:
        train_data = json.load(f)
    train_dataset = Dataset.from_list(train_data)

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)

    # Process train dataset
    train_dataset = train_dataset.map(function=make_map_fn("train", args.mode), with_indices=True)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

    # Process test datasets
    for idx, test_data_source in enumerate(test_data_sources):
        with open(test_data_source, "r") as f:
            test_data = json.load(f)
        test_dataset = Dataset.from_list(test_data)
        
        test_dataset = test_dataset.map(function=make_map_fn(f"test_{idx}", args.mode), with_indices=True)
        
        filename = os.path.splitext(os.path.basename(test_data_source))[0]
        test_dataset.to_parquet(os.path.join(local_dir, f"{filename}.parquet"))
