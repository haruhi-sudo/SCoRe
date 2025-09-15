import io
import multiprocessing
import types
import pickle
import ast
from contextlib import redirect_stdout

try:
    from ast import unparse
except ImportError:
    import astunparse
    unparse = astunparse.unparse

def _get_serializable_globals(globals_dict: dict, executed_code: str) -> dict:
    """
    过滤一个字典，只保留可以被序列化的对象。
    对于函数和导入语句，它会通过解析原始代码字符串来提取其源代码。

    Args:
        globals_dict: 包含代码执行后所有全局变量的字典。
        executed_code: 刚刚被执行的原始代码字符串。

    Returns:
        一个新的、只包含可序列化状态的字典。
    """
    serializable_globals = {}
    
    function_sources = {}
    import_statements = []
    try:
        tree = ast.parse(executed_code)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_sources[node.name] = unparse(node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                import_statements.append(unparse(node))
    except (SyntaxError, AttributeError):
        pass


    serializable_globals['__imports__'] = {
        '__type__': 'import_block',
        '__sources__': import_statements
    }

    for key, value in globals_dict.items():
        if key.startswith('__') or isinstance(value, (types.ModuleType, types.CodeType)):
            continue
        
        if isinstance(value, types.FunctionType):
            if key in function_sources:
                serializable_globals[key] = {
                    '__type__': 'function',
                    '__source__': function_sources[key]
                }
            continue

        try:
            pickle.dumps(value)
            serializable_globals[key] = value
        except (pickle.PicklingError, TypeError):
            pass
            
    return serializable_globals

def _execute_in_process(code: str, initial_globals: dict, result_queue: multiprocessing.Queue):
    """
    在独立的子进程中执行代码。
    """
    session_globals = {}
    
    # 这个循环的顺序很重要：先导入，再定义函数，最后加载变量
    
    # 1. 重建导入
    if '__imports__' in initial_globals and initial_globals['__imports__'].get('__type__') == 'import_block':
        for import_src in initial_globals['__imports__']['__sources__']:
            try:
                exec(import_src, session_globals)
            except Exception:
                pass # 如果导入失败，则跳过

    # 2. 重建函数和加载变量
    for key, value in initial_globals.items():
        if key == '__imports__': continue # 跳过我们已经处理过的导入块

        if isinstance(value, dict):
            if value.get('__type__') == 'function':
                try:
                    # 在 session_globals 中执行函数定义
                    exec(value['__source__'], session_globals)
                except Exception:
                    pass # 如果重建失败，则跳过
            else:
                 session_globals[key] = value # 普通字典
        else:
            # 直接复制其他可序列化的变量
            session_globals[key] = value

    # 重定向标准输出，以便捕获 print() 的内容
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            # 在这里执行用户代码，使用已经包含重建状态的 session_globals
            exec(code, session_globals)
        
        output = buffer.getvalue()
        
        # 将被执行的 `code` 传递给过滤函数以提取新的函数/导入
        filtered_globals = _get_serializable_globals(session_globals, code)
        
        # 从当前执行中提取新的导入
        new_imports = filtered_globals.get('__imports__', {}).get('__sources__', [])
        # 获取旧的导入
        old_imports = initial_globals.get('__imports__', {}).get('__sources__', [])
        # 合并并去重
        all_imports = list(dict.fromkeys(old_imports + new_imports))
        if all_imports:
            filtered_globals['__imports__']['__sources__'] = all_imports

        result_queue.put({
            "status": "success",
            "output": output,
            "globals": filtered_globals
        })

    except Exception as e:
        filtered_globals = _get_serializable_globals(session_globals, code)
        result_queue.put({
            "status": "exception",
            "error": f"{type(e).__name__}: {e}",
            "globals": filtered_globals
        })


class CodeExecutor:
    """
    一个更安全、更健壮的代码执行器，能在独立进程中运行代码，并处理超时和序列化问题。
    """
    def __init__(self, available_tools: dict = None):
        # session_globals 用于维护多次执行之间的状态
        self.session_globals = available_tools if available_tools is not None else {}

    def execute(self, code: str, timeout: int = 5) -> str:
        """
        在有超时限制的独立进程中执行代码。
        """
        result_queue = multiprocessing.Queue()
        
        process = multiprocessing.Process(
            target=_execute_in_process,
            args=(code, self.session_globals.copy(), result_queue)
        )
        
        process.start()
        process.join(timeout=timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            return f"Code execution failed: TimeoutError after {timeout} seconds. Check your code for infinite loops or long-running operations."

        try:
            result = result_queue.get_nowait()
            
            if 'globals' in result:
                self.session_globals.update(result['globals'])
            
            if result['status'] == 'success':
                if result['output']:
                    return f"Observation: {result['output']}"
                else:
                    return "Observation: No output"
            elif result['status'] == 'exception':
                return f"Observation: {result['error']}"
                
        except Exception as e:
            return f"Failed to retrieve execution result from process: {e}"
        
        return "Code execution finished with an unknown state."


if __name__ == "__main__":
    code_exec = CodeExecutor()

    # 第一次执行：导入模块，定义一个函数并计算
    code1 = """
import math
def my_func(x):
    # 这是一个可以被传递的函数
    return x * x + 1

result = my_func(10)
"""
    print("--- Executing code 1: Defining function and importing module ---")
    output1 = code_exec.execute(code1)
    print(output1)
    print(f"Current globals in main process: {code_exec.session_globals.keys()}")
    print("-" * 25, "\n")
    
    # 第二次执行：使用上一次执行中定义的函数和导入的模块
    code2 = "new_result = my_func(result) + math.sqrt(4)\nprint(f'The new result is {new_result}')"
    print("--- Executing code 2: Using transferred function and module ---")
    output2 = code_exec.execute(code2)
    print(output2)
    print(f"Current globals in main process: {code_exec.session_globals.keys()}")
    print("-" * 25, "\n")

    # 第三次执行：一个会产生异常的代码
    code3 = "z = 1 / 0"
    print("--- Executing code 3 ---")
    output3 = code_exec.execute(code3)
    print(output3)
    print("-" * 25, "\n")

    # 第四次执行：测试带别名的导入
    print("--- Executing code 4: Testing aliased import ---")
    try:
        code4 = "import numpy as np\narray = np.array([1, 2, 3])"
        output4 = code_exec.execute(code4)
        print(output4)
        print(f"Current globals in main process: {code_exec.session_globals.keys()}")
        print("-" * 25, "\n")
        
        # 第五次执行：使用上一次的别名导入
        code5 = "new_array = np.array([1, 2, 3]) * 2\nprint(f'The new array is {new_array}')\nprint(f'{my_func(result)}')"
        print("--- Executing code 5: Using the aliased import ---")
        output5 = code_exec.execute(code5)
        print(output5)
        print(f"Current globals in main process: {code_exec.session_globals.keys()}")
        print(f"Value of 'new_array': {code_exec.session_globals.get('new_array')}")
        print("-" * 25, "\n")
    except ImportError:
        print("Skipping numpy test because numpy is not installed.")

