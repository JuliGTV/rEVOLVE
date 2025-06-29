import pickle
from typing import Callable, Optional
import numpy as np
from GL_search import synth_GL
import sys
import tempfile
import os
import subprocess
import traceback
sys.path.append('../../')
from src.evaluation import Evaluation


class paramObj:
    """Parameter object for GL_search functions."""
    def __init__(self):
        # Default parameter values based on GL_search.py usage
        self.mode = 'GL'
        self.method = 'greedy'
        self.minDepth = False
        self.hv = 1  # vector heuristic
        self.hi = 1  # include inverse
        self.ht = 1  # include transpose
        self.hl = 1  # log of columns
        self.hr = 3  # scaling factor for heuristic
        self.wMax = 10  # maximum wait
        self.custom_heuristic = None

def evaluate(h: Optional[Callable[[np.ndarray], tuple[float, tuple[int]]]] = None, dataset_path: str = 'search_dataset.pkl'):

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    params = paramObj()
    params.mode = 'GL'
    params.method = 'greedy'
    params.minDepth = False
    params.hv = 1 ## vector
    params.hi = 1 ## include inverse
    params.ht = 1 ## include transpose
    params.hl = 1 ## log of cols 1 or sums 0
    params.hr = 3 # scaling factor for heuristic
    params.wMax = 10
    params.custom_heuristic = h

    total_gate_count = 0
    total_compute_time = 0
    gate_counts = []
    compute_times = []

    for matrix in dataset:
        n,gateCount,depth,procTime,check,circ = synth_GL(matrix,params)

        
        
        gate_counts.append(gateCount)
        compute_times.append(procTime)
        total_gate_count += gateCount
        total_compute_time += procTime
    
    avg_gate_count = total_gate_count / len(dataset)
    avg_compute_time = total_compute_time / len(dataset)

    print(total_compute_time)
    if total_compute_time > 100:
        return 0
    if total_compute_time > 10:
        return 80 - total_compute_time - avg_gate_count
    return 70 - avg_gate_count


def run_heuristic_safely(heuristic_code: str, timeout_seconds: int = 30) -> Callable:
    """
    Run a heuristic function definition safely in a subprocess with timeout.
    
    Args:
        heuristic_code: The heuristic function code as a string
        timeout_seconds: Maximum execution time in seconds
        
    Returns:
        A callable heuristic function or raises an exception
    """
    # Create a temporary file to execute
    with tempfile.NamedTemporaryFile(mode='w', suffix=".py", delete=False) as temp_file:
        # Write the heuristic code plus validation wrapper
        full_code = f"""
import numpy as np
import pickle
import sys
import traceback

# Required imports for heuristic functions
try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None

# Heuristic code
{heuristic_code}

# Function discovery and validation
try:
    # Look for a heuristic function
    heuristic_func = None
    
    # Common heuristic function names to search for
    possible_names = ['heuristic', 'compute_heuristic', 'gl_heuristic', 'matrix_heuristic']
    
    for name in possible_names:
        if name in globals() and callable(globals()[name]):
            heuristic_func = globals()[name]
            break
    
    # If not found by name, look for any callable that's not a built-in
    if heuristic_func is None:
        for name, obj in globals().items():
            if (callable(obj) and 
                not name.startswith('_') and 
                name not in ['pickle', 'sys', 'traceback', 'numpy', 'np', 'spearmanr']):
                heuristic_func = obj
                break
    
    if heuristic_func is None:
        raise RuntimeError("No suitable heuristic function found. Expected function names: " + 
                         ", ".join(possible_names))
    
    # Test the function with a simple matrix to validate signature
    test_matrix = np.array([[1, 0], [1, 1]], dtype=int)
    try:
        result = heuristic_func(test_matrix)
        # Result should be a number or tuple/list
        if not isinstance(result, (int, float, np.number, tuple, list)):
            raise RuntimeError(f"Heuristic function should return a number or tuple, got {{type(result)}}")
    except Exception as e:
        raise RuntimeError(f"Heuristic function failed test: {{str(e)}}")
    
    # Save the function for external use
    results = {{
        'success': True,
        'function_name': heuristic_func.__name__,
        'test_result': result
    }}
    
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
        
except Exception as e:
    # Save error
    results = {{
        'success': False,
        'error': str(e),
        'traceback': traceback.format_exc()
    }}
    
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
"""
        temp_file.write(full_code)
        temp_file_path = temp_file.name
    
    results_path = f"{temp_file_path}.results"
    
    try:
        # Run the script with timeout
        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            
            # Load results
            if os.path.exists(results_path):
                with open(results_path, 'rb') as f:
                    results = pickle.load(f)
                
                if results['success']:
                    # Return a wrapper function that safely executes the heuristic
                    return create_safe_heuristic_wrapper(heuristic_code, results['function_name'])
                else:
                    raise RuntimeError(f"Heuristic validation failed: {results['error']}")
            else:
                raise RuntimeError("Results file not found")
                
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise RuntimeError(f"Heuristic validation timed out after {timeout_seconds} seconds")
            
    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)


def create_safe_heuristic_wrapper(heuristic_code: str, function_name: str) -> Callable:
    """
    Create a safe wrapper function that executes the heuristic in a subprocess for each call.
    
    Args:
        heuristic_code: The validated heuristic code
        function_name: Name of the heuristic function
        
    Returns:
        A callable wrapper function
    """
    def safe_heuristic(matrix: np.ndarray):
        """Safe wrapper for heuristic function execution."""
        with tempfile.NamedTemporaryFile(mode='w', suffix=".py", delete=False) as temp_file:
            # Create execution code with proper indentation
            indented_heuristic_code = '\n'.join('    ' + line for line in heuristic_code.strip().split('\n'))
            exec_code = f"""
import numpy as np
import pickle
import sys
import traceback

# Load input matrix
with open('{temp_file.name}.input', 'rb') as f:
    matrix = pickle.load(f)

try:
    # Heuristic code
{indented_heuristic_code}
    
    # Execute the heuristic
    result = {function_name}(matrix)
    
    # Save result
    with open('{temp_file.name}.output', 'wb') as f:
        pickle.dump({{'success': True, 'result': result}}, f)
        
except Exception as e:
    # Save error
    with open('{temp_file.name}.output', 'wb') as f:
        pickle.dump({{'success': False, 'error': str(e)}}, f)
"""
            temp_file.write(exec_code)
            temp_file_path = temp_file.name
        
        input_path = f"{temp_file_path}.input"
        output_path = f"{temp_file_path}.output"
        
        try:
            # Save input matrix
            with open(input_path, 'wb') as f:
                pickle.dump(matrix, f)
            
            # Execute
            process = subprocess.Popen(
                [sys.executable, temp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            try:
                stdout, stderr = process.communicate(timeout=10)  # Shorter timeout for individual calls
                
                # Load result
                if os.path.exists(output_path):
                    with open(output_path, 'rb') as f:
                        result = pickle.load(f)
                    
                    if result['success']:
                        return result['result']
                    else:
                        raise RuntimeError(f"Heuristic execution failed: {result['error']}")
                else:
                    # Enhanced error message with subprocess output
                    error_msg = f"Output file not found. Return code: {process.returncode}"
                    if stderr:
                        error_msg += f", stderr: {stderr.decode()[:200]}"
                    if stdout:
                        error_msg += f", stdout: {stdout.decode()[:200]}"
                    raise RuntimeError(error_msg)
                    
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise RuntimeError("Heuristic execution timed out")
                
        finally:
            # Clean up
            for path in [temp_file_path, input_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)
    
    return safe_heuristic


def evaluate_heuristic_from_string(heuristic_code: str) -> Evaluation:
    """
    Evaluate a heuristic function defined as a string for the rEVOLVE framework.
    Uses subprocess execution for safety and robustness.
    
    Args:
        heuristic_code: Python code defining a heuristic function
        
    Returns:
        Evaluation object with fitness based on the evaluation function
    """
    # Create a temporary file to execute the entire evaluation in one subprocess
    with tempfile.NamedTemporaryFile(mode='w', suffix=".py", delete=False) as temp_file:
        # Write the complete evaluation script
        full_eval_code = f"""
import pickle
import numpy as np
import sys
import traceback
from typing import Callable, Optional

# Parameter object class
class paramObj:
    def __init__(self):
        self.mode = 'GL'
        self.method = 'greedy'
        self.minDepth = False
        self.hv = 1
        self.hi = 1
        self.ht = 1
        self.hl = 1
        self.hr = 3
        self.wMax = 10
        self.custom_heuristic = None

# Import GL_search functions
sys.path.append('.')
from GL_search import synth_GL

# Heuristic code
{heuristic_code}

def evaluate_single(h: Optional[Callable] = None, dataset_path: str = 'search_dataset.pkl'):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    params = paramObj()
    params.mode = 'GL'
    params.method = 'greedy'
    params.minDepth = False
    params.hv = 1
    params.hi = 1
    params.ht = 1
    params.hl = 1
    params.hr = 3
    params.wMax = 10
    params.custom_heuristic = h

    total_gate_count = 0
    total_compute_time = 0

    for matrix in dataset:
        n, gateCount, depth, procTime, check, circ = synth_GL(matrix, params)
        total_gate_count += gateCount
        total_compute_time += procTime
    
    avg_gate_count = total_gate_count / len(dataset)
    avg_compute_time = total_compute_time / len(dataset)

    if total_compute_time > 100:
        return 0
    if total_compute_time > 10:
        return 80 - total_compute_time - avg_gate_count
    return 70 - avg_gate_count

try:
    # Find the heuristic function
    heuristic_func = None
    possible_names = ['heuristic', 'compute_heuristic', 'gl_heuristic', 'matrix_heuristic']
    
    for name in possible_names:
        if name in globals() and callable(globals()[name]):
            heuristic_func = globals()[name]
            break
    
    if heuristic_func is None:
        for name, obj in globals().items():
            if (callable(obj) and 
                not name.startswith('_') and 
                name not in ['pickle', 'numpy', 'np', 'sys', 'traceback', 'paramObj', 'synth_GL', 'evaluate_single']):
                heuristic_func = obj
                break
    
    if heuristic_func is None:
        raise RuntimeError("No suitable heuristic function found")
    
    # Run the evaluation
    score = evaluate_single(heuristic_func)
    
    # Save result
    result = {{
        'success': True,
        'score': score,
        'function_name': heuristic_func.__name__
    }}
    
    with open('{temp_file.name}.result', 'wb') as f:
        pickle.dump(result, f)
        
except Exception as e:
    # Save error
    result = {{
        'success': False,
        'error': str(e),
        'traceback': traceback.format_exc()
    }}
    
    with open('{temp_file.name}.result', 'wb') as f:
        pickle.dump(result, f)
"""
        temp_file.write(full_eval_code)
        temp_file_path = temp_file.name
    
    result_path = f"{temp_file_path}.result"
    
    try:
        # Run the evaluation in subprocess with timeout
        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        try:
            stdout, stderr = process.communicate(timeout=60)  # 60 second timeout for full evaluation
            
            # Load results
            if os.path.exists(result_path):
                with open(result_path, 'rb') as f:
                    result = pickle.load(f)
                
                if result['success']:
                    score = result['score']
                    fitness = float(score)
                    
                    # Final safety check for fitness
                    if not np.isfinite(fitness):
                        fitness = 0.0
                    
                    return Evaluation(
                        fitness=fitness,
                        additional_data={
                            "score": f"{score:.6f}",
                            "validity": "valid",
                            "execution_method": "single_subprocess",
                            "function_name": result.get('function_name', 'unknown')
                        }
                    )
                else:
                    raise RuntimeError(f"Evaluation failed: {result['error']}")
            else:
                error_msg = f"Result file not found. Return code: {process.returncode}"
                if stderr:
                    error_msg += f", stderr: {stderr.decode()[:500]}"
                raise RuntimeError(error_msg)
                
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise RuntimeError("Evaluation timed out after 60 seconds")
            
    except Exception as e:
        return Evaluation(
            fitness=0.0,
            additional_data={
                "score": "0.0",
                "validity": "error", 
                "error": str(e),
                "execution_method": "single_subprocess"
            }
        )
    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(result_path):
            os.unlink(result_path)