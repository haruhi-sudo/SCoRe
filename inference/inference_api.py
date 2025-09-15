import re
import json
import concurrent.futures
import threading
import logging
import os
from openai import OpenAI
import yaml
from tools.mock_tools import MockTools
from tools.search_tools import SearchTools
from tools.python_tools import CodeExecutor

# Load configuration from YAML file
with open('inference_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

API_KEY = config['API_KEY']
BASE_URL = config['BASE_URL']
MODEL_NAME = config['MODEL_NAME']
INPUT_FILE = config['INPUT_FILE']
OUTPUT_FILE = config['OUTPUT_FILE']
LOG_FILE = config['LOG_FILE']
MAX_WORKERS = config['MAX_WORKERS']
CACHE_FILE = config['CACHE_FILE']

# Configure SearchTools with cache file from config
SearchTools.set_cache_file(CACHE_FILE)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
file_lock = threading.Lock()

def thought_code_cycle(input_query, max_cycles=8):
    mock_tools = MockTools()
    executor = CodeExecutor(available_tools={
        "final_answer_print": mock_tools.final_answer_print
    })
    messages = [{"role": "user", "content": input_query}]
    cycle_status = False

    for idx in range(max_cycles):
        try:
            llm_resp = client.chat.completions.create(messages=messages, model=MODEL_NAME)
            thought_code_content = llm_resp.choices[0].message.content

            code_match = re.search(r'<code>([\s\S]*?)</code>', thought_code_content, re.DOTALL)
            if code_match:
                code_content = code_match.group(1).strip()
                thought_content = thought_code_content[:code_match.start()].strip()
            else:
                code_content = ''
                thought_content = thought_code_content.strip()

            if code_content:
                query_match = re.search(r'web_search\("(.*?)"\)', code_content, re.DOTALL)
                if query_match:
                    web_search_query = query_match.group(1)
                    observation_content = SearchTools.web_search(web_search_query)
                    observation_content = f"Observation: {observation_content}"
                else:
                    observation_content = executor.execute(code_content)
            else:
                observation_content = "Observation: None"

            message_single_turn = [
                {"role": "assistant", "content": thought_code_content},
                {"role": "user", "content": observation_content}
            ]
            messages.extend(message_single_turn)

            if ("error" in observation_content.lower() or "failed" in observation_content.lower()) and "web_search" not in thought_code_content.lower():
                cycle_status = False
            elif "final_answer_print" in thought_code_content.lower():
                cycle_status = True
                break
    
        except Exception as e:
            logging.error(f"处理查询时发生错误: {input_query[:50]}... 错误: {e}")
            cycle_status = False
            break

    return messages, cycle_status

def process_item_and_save(item):
    id = item.get("id", "")
    question = item.get("question", "")
    true_answer = item.get("true_answer", "")
    max_tries = 5

    while max_tries > 0:
        messages, cycle_status = thought_code_cycle(question)
        if cycle_status:
            break
        else:
            max_tries -= 1

    result = {
        "id": id,
        "question": question,
        "true_answer": true_answer,
        "messages": messages
    }

    if cycle_status and messages:
        try:
            gen_answer = messages[-1]["content"].split("Observation: ")[-1].strip()
            result["gen_answer"] = gen_answer
        except (IndexError, KeyError):
            result["gen_answer"] = "Error: Could not parse final answer."
    else:
        result["gen_answer"] = "Error"

    with file_lock:
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def load_processed_ids(filepath):
    processed_ids = set()
    if not os.path.exists(filepath):
        return processed_ids
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'question' in data:
                    processed_ids.add(data['question'])
            except (json.JSONDecodeError, KeyError):
                continue
    return processed_ids

def main():
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'), # 使用追加模式
            logging.StreamHandler()
        ]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    processed_ids = load_processed_ids(OUTPUT_FILE)
    logging.info(f"从 {OUTPUT_FILE} 加载了 {len(processed_ids)} 个已处理的查询。")

    try:
        with open(INPUT_FILE, encoding='utf-8') as f:
            input_queries = [json.loads(line) for line in f]
        logging.info(f"成功从 {INPUT_FILE} 加载 {len(input_queries)} 个查询。")
    except FileNotFoundError:
        logging.error(f"错误: 在 {INPUT_FILE} 未找到输入文件")
        return
    except json.JSONDecodeError:
        logging.error(f"错误: 无法从 {INPUT_FILE} 解码 JSON")
        return

    queries_to_process = [item for item in input_queries if item.get("question") not in processed_ids]
    logging.info(f"总查询数: {len(input_queries)}。已跳过: {len(processed_ids)}。本次运行待处理: {len(queries_to_process)}。")

    if not queries_to_process:
        logging.info("没有需要处理的新查询。")
        return

    processed_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix='Worker') as executor:
        future_to_item = {executor.submit(process_item_and_save, item): item for item in queries_to_process}

        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                future.result() # 检查是否有异常
                processed_count += 1
                logging.info(f"已处理 {processed_count}/{len(queries_to_process)} - 问题: {item.get('question', 'N/A')[:40]}...")
            except Exception as exc:
                logging.error(f'项目 "{item.get("question", "N/A")[:50]}..." 产生了一个异常: {exc}', exc_info=True)

    logging.info(f"处理完成。本次运行处理了 {processed_count} 个结果。")
    logging.info(f"详细日志已保存到 {LOG_FILE}")

if __name__ == "__main__":
    main()
