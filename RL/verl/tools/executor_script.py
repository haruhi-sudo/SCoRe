# verl/tools/executor_script.py

import sys, json, pickle, io, types, ast, signal, inspect
from contextlib import redirect_stdout
import os, re
import time
from typing import Any, Optional, Dict, List
import requests
from pydantic import BaseModel
from dotenv import load_dotenv
import sqlite3

# Ensure the unparse function is available for compatibility.
try:
    from ast import unparse
except ImportError:
    try:
        import astunparse
        unparse = astunparse.unparse
    except ImportError:
        def unparse(node):
            import ast
            return ast.dump(node)

load_dotenv()

def timeout_handler(signum, frame):
    '''Raises a TimeoutError when the alarm signal is received.'''
    raise TimeoutError("Code execution timed out")

class MockTools:
    @staticmethod
    def final_answer_print(answer):
        """Only for final answer"""
        print(answer)


def _rebuild_scope(scope_dict):
    '''
    Reconstructs the execution scope from a dictionary. It specifically
    looks for function definitions saved as source code and re-executes them.
    '''
    scope = {}
    
    # 1. First, rebuild the imported modules.
    if '__imports__' in scope_dict and scope_dict['__imports__'].get('__type__') == 'import_block':
        for import_src in scope_dict['__imports__']['__sources__']:
            try:
                exec(import_src, scope)
            except Exception:
                pass

    # 2. Then, reconstruct functions from their source code.
    for key, value in scope_dict.items():
        if key == '__imports__': continue
        if isinstance(value, dict) and value.get('__type__') == 'function':
            try:
                exec(value['__source__'], scope)
            except Exception:
                pass

    # 3. Finally, load regular serializable variables.
    for key, value in scope_dict.items():
        # Skip non-simple variable types.
        if key == '__imports__' or (isinstance(value, dict) and value.get('__type__') == 'function'):
            continue
        try:
            pickle.dumps(value)
            scope[key] = value
        except (pickle.PicklingError, TypeError):
            pass
            
    return scope

def sanitize_value(obj):
    '''
    Recursively traverses an object to ensure all dictionary keys are
    serializable (str, int, float, bool, None).
    '''
    if isinstance(obj, dict):
        return {str(k): sanitize_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_value(elem) for elem in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_value(elem) for elem in obj)
    return obj

# --- Main Subprocess Execution Loop ---
# Keep the process alive and wait for commands
scope = {}
while True:
    timeout_occurred = False
    try:
        # Read length prefix (4 bytes)
        length_bytes = sys.stdin.buffer.read(4)
        if not length_bytes:
            break
            
        # Get the length of the incoming data
        data_length = int.from_bytes(length_bytes, byteorder='little')
        
        # Read the actual data
        input_data = sys.stdin.buffer.read(data_length)
        if not input_data:
            break
            
        data = pickle.loads(input_data)
        code = data["code"]
        scope_dict = data["scope"]
        timeout = data.get("timeout", 18)

        # Set the timeout alarm.
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        # --- Environment Reconstruction and Update ---
        # 1. Reconstruct the base environment from the incoming scope dictionary.
        new_scope = _rebuild_scope(scope_dict)
        # 2. Update the existing scope with the new one to maintain state
        scope.update(new_scope)
        
        # 3. Parse new import statements from the current code block.
        current_imports = []
        function_sources = {}
        try:
            tree = ast.parse(code)
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    current_imports.append(unparse(node))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    function_sources[node.name] = unparse(node)
        except (SyntaxError, TypeError):
            # Ignore if the code is incomplete or cannot be parsed.
            pass

        # 4. Merge the new import statements with the old ones.
        existing_imports = scope_dict.get('__imports__', {}).get('__sources__', [])
        # Use a dictionary to deduplicate while preserving order.
        all_imports = list(dict.fromkeys(existing_imports + current_imports))

        # Inject mock tools.
        scope['final_answer_print'] = MockTools.final_answer_print

        output_stream = io.StringIO()
        try:
            with redirect_stdout(output_stream):
                exec(code, scope)
            output = output_stream.getvalue() or "Execution successful, no output."
        except TimeoutError:
            timeout_occurred = True
            output = f"Error: Code execution timed out after {timeout} seconds"
        except Exception as e:
            output = f"Error: {e}"
        finally:
            signal.alarm(0) # Cancel the alarm.

        # --- Serialize the returning scope ---
        serializable_scope = {}
        
        # Only serialize scope if no timeout occurred to avoid further delays
        if not timeout_occurred:
            # Save all merged import statements.
            if all_imports:
                serializable_scope['__imports__'] = {
                    '__type__': 'import_block',
                    '__sources__': all_imports
                }

            for key, value in scope.items():
                # Exclude built-in variables and module objects.
                if key.startswith('__') or isinstance(value, types.ModuleType):
                    continue

                # For functions defined in the code, save their source code.
                if isinstance(value, types.FunctionType) and value.__module__ == '__main__':
                    # Use the source code we parsed earlier.
                    if key in function_sources:
                        serializable_scope[key] = {
                            '__type__': 'function',
                            '__source__': function_sources[key]
                        }
                # For other types, sanitize and then check if they are serializable.
                else:
                    try:
                        sanitized_value = sanitize_value(value)
                        pickle.dumps(sanitized_value)
                        serializable_scope[key] = sanitized_value
                    except (pickle.PicklingError, TypeError):
                        pass
        else:
            # On timeout, return minimal scope to avoid serialization overhead
            serializable_scope = {}

        result_data = {"output": output, "updated_scope": serializable_scope}
        
    except Exception as e:
        # Catch potential errors during the initialization phase.
        result_data = {"output": f"Error in executor script setup: {e}", "updated_scope": {}}
    
    # Send result back with length prefix
    try:
        result_bytes = pickle.dumps(result_data)
        sys.stdout.buffer.write(len(result_bytes).to_bytes(4, byteorder='little'))
        sys.stdout.buffer.write(result_bytes)
        sys.stdout.flush()
    except Exception as e:
        break

