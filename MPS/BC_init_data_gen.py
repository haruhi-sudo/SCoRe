import json
import logging
import yaml
import concurrent.futures
import os
import argparse
from typing import List, Dict, Any, Set
from functions import SearchTools
from graph import run_agent

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run math tasks from a JSONL file')
    parser.add_argument('--config', type=str, default='configs/BC_data_gen_config.yaml',
                    help='Path to the configuration file (default: configs/BC_data_gen_config.yaml)')
    args = parser.parse_args()
    
    # Load configuration from YAML file
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    main_log_file = config["logging"]["main_log_file"]
    log_dir = os.path.dirname(main_log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(main_log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Set the search cache file path from config
    if "search_cache_file" in config.get("logging", {}):
        SearchTools.set_cache_file_path(config["logging"]["search_cache_file"])

    def get_processed_ids(log_file_path: str) -> Set[str]:
        """Get a set of already processed task IDs from the log file"""
        processed_ids = set()
        
        # If log file doesn't exist, return empty set
        if not os.path.exists(log_file_path):
            # Create directory path if it doesn't exist
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            return processed_ids
        
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        if "original_task_info" in log_entry and "id" in log_entry["original_task_info"]:
                            processed_ids.add(log_entry["original_task_info"]["id"])
                    except json.JSONDecodeError:
                        # Skip invalid lines
                        continue
        except Exception as e:
            logger.warning(f"Warning: Error reading log file: {e}")
        
        return processed_ids

    def process_single_task(task_data: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Process a single task and return True if successful"""
        try:
            logger.info(f"处理任务: {task_data['id']}")
            # Run the agent with the math problem
            run_agent(
                prompt=task_data['question'],
                config=config,  # Pass the full config
                original_task_info={
                    "id": task_data['id'],
                    "question": task_data['question'],
                    "true_answer": task_data['true_answer']
                }
            )
            return True
        except Exception as e:
            logger.error(f"Error processing task {task_data['id']}: {e}")
            return False

    def run_tasks_from_file(config: Dict[str, Any]):
        """Run math tasks from a JSONL file with concurrent processing"""
        # Get already processed IDs
        processed_ids = get_processed_ids(config["logging"]["log_file_path"])
        tasks_to_process: List[Dict[str, Any]] = []
        
        # Load tasks to process
        with open(config["paths"]["data_file"], 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    task_data = json.loads(line.strip())
                    
                    # Skip if already processed
                    if task_data['id'] in processed_ids:
                        logger.info(f"⏭️  跳过已处理任务: {task_data['id']}")
                        continue
                    
                    tasks_to_process.append(task_data)
                    
                    # Break if we've reached the max tasks limit
                    if config["processing"]["max_tasks"] and len(tasks_to_process) >= config["processing"]["max_tasks"]:
                        break
                        
                except json.JSONDecodeError:
                    logger.warning(f"Warning: Skipping invalid JSON line")
                except Exception as e:
                    logger.error(f"Error loading task: {e}")
        
        if not tasks_to_process:
            logger.info("No new tasks to process")
            return
        
        logger.info(f"开始并发处理 {len(tasks_to_process)} 个任务，使用 {config['processing']['max_workers']} 个工作线程")
        
        # Process tasks concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=config["processing"]["max_workers"]) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(process_single_task, task_data, config): task_data 
                for task_data in tasks_to_process
            }
            
            # Collect results
            completed_tasks = 0
            failed_tasks = 0
            
            for future in concurrent.futures.as_completed(future_to_task):
                task_data = future_to_task[future]
                try:
                    success = future.result()
                    if success:
                        completed_tasks += 1
                    else:
                        failed_tasks += 1
                except Exception as e:
                    logger.error(f"Task {task_data['id']} generated an exception: {e}")
                    failed_tasks += 1
            
            logger.info(f"🏁 处理完成: {completed_tasks} 成功, {failed_tasks} 失败")

    # Run with configuration
    run_tasks_from_file(config)

if __name__ == "__main__":
    main()
