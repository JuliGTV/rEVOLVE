#!/usr/bin/env python3
"""
Simplified heuristic evaluation function.
Self-contained script that returns average Spearman correlation across datasets.
"""

import pickle
import numpy as np
from scipy.stats import spearmanr
from typing import Callable, Any, List
from pathlib import Path


def evaluate(heuristic_func: Callable[[np.ndarray], Any]) -> float:
    """
    Evaluate a heuristic function on GL matrix datasets.
    
    Args:
        heuristic_func: Function that takes a numpy array (GL matrix) and returns
                       a sortable value (float, int, tuple, etc.)
    
    Returns:
        float: Average Spearman correlation coefficient across all datasets
    """
    # Dataset files (hardcoded for simplicity)
    dataset_files = [
        "gl_datasets/gl_4x4_1000samples_20250627_191708.pkl",
        "gl_datasets/gl_5x5_1000samples_20250627_191724.pkl", 
        "gl_datasets/gl_6x6_1000samples_20250627_191742.pkl"
    ]
    
    spearman_scores = []
    
    for dataset_file in dataset_files:
        # Load dataset
        with open(dataset_file, 'rb') as f:
            data = pickle.load(f)
        
        matrices = data['matrices']
        optimal_gate_counts = data['gate_counts']
        
        # Compute heuristic values
        heuristic_values = []
        for matrix in matrices:
            matrix_array = np.array(matrix)
            try:
                heuristic_value = heuristic_func(matrix_array)
                # Check for nan/inf values
                if isinstance(heuristic_value, (tuple, list)):
                    if any(not np.isfinite(x) for x in heuristic_value if isinstance(x, (int, float, np.number))):
                        heuristic_value = 0.0  # Fallback for invalid vector heuristics
                elif isinstance(heuristic_value, (int, float, np.number)):
                    if not np.isfinite(heuristic_value):
                        heuristic_value = 0.0  # Fallback for invalid scalar heuristics
                heuristic_values.append(heuristic_value)
            except Exception:
                # If heuristic computation fails, use 0.0 as fallback
                heuristic_values.append(0.0)
        
        # Convert vector heuristics to ranks for correlation
        if len(heuristic_values) > 0 and isinstance(heuristic_values[0], (tuple, list)):
            unique_values = sorted(set(heuristic_values))
            value_to_rank = {v: i for i, v in enumerate(unique_values)}
            numeric_heuristics = [value_to_rank[v] for v in heuristic_values]
        else:
            numeric_heuristics = heuristic_values
        
        # Compute Spearman correlation
        try:
            spearman_result = spearmanr(numeric_heuristics, optimal_gate_counts)
            if hasattr(spearman_result, 'correlation'):
                spearman_rho = float(spearman_result.correlation)
            else:
                spearman_rho, _ = spearman_result
                spearman_rho = float(spearman_rho)
            
            # Ensure correlation is finite
            if not np.isfinite(spearman_rho):
                spearman_rho = 0.0
                
        except:
            spearman_rho = 0.0
        
        spearman_scores.append(spearman_rho)
    
    # Return average Spearman correlation with safety check
    avg_correlation = float(np.mean(spearman_scores))
    if not np.isfinite(avg_correlation):
        avg_correlation = 0.0
    return avg_correlation


# Example usage and test functions
def heuristic_sum_entries(matrix: np.ndarray) -> float:
    """Simple sum of all matrix entries."""
    return float(np.sum(matrix))


def heuristic_log_column_sums(matrix: np.ndarray) -> float:
    """Sum of logarithms of column sums."""
    col_sums = np.sum(matrix, axis=0)
    col_sums = np.maximum(col_sums, 1e-10)  # Avoid log(0)
    return float(np.sum(np.log(col_sums)))


def heuristic_vector_column_sums(matrix: np.ndarray) -> tuple:
    """Vector of sorted column sums."""
    col_sums = np.sum(matrix, axis=0)
    return tuple(sorted(col_sums))


# rEVOLVE framework integration
import tempfile
import os
import subprocess
import sys
import traceback
import sys
sys.path.append('../../')
from src.evaluation import Evaluation


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
            # Create execution code
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
    {heuristic_code}
    
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
                    raise RuntimeError("Output file not found")
                    
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
    Uses restricted namespace execution for safety.
    
    Args:
        heuristic_code: Python code defining a heuristic function
        
    Returns:
        Evaluation object with fitness based on Spearman correlation
    """
    try:
        # Create a dictionary to serve as the local namespace
        local_namespace = {'np': np, 'numpy': np}
        
        # Execute the code string in the local namespace with minimal builtins
        safe_globals = {
            '__builtins__': {
                '__import__': __import__,
                'float': float,
                'int': int,
                'tuple': tuple,
                'list': list,
                'sorted': sorted,
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'max': max,
                'min': min,
                'sum': sum,
            }
        }
        exec(heuristic_code, safe_globals, local_namespace)
        
        # Look for heuristic function
        heuristic_func = None
        possible_names = ['heuristic', 'compute_heuristic', 'gl_heuristic', 'matrix_heuristic']
        
        for name in possible_names:
            if name in local_namespace and callable(local_namespace[name]):
                heuristic_func = local_namespace[name]
                break
        
        # If not found by name, look for any callable
        if heuristic_func is None:
            for name, obj in local_namespace.items():
                if (callable(obj) and 
                    not name.startswith('_') and 
                    name not in ['np', 'numpy']):
                    heuristic_func = obj
                    break
        
        if heuristic_func is None:
            raise RuntimeError("No suitable heuristic function found")
        
        # Test the function
        test_matrix = np.array([[1, 0], [1, 1]], dtype=int)
        test_result = heuristic_func(test_matrix)
        if not isinstance(test_result, (int, float, np.number, tuple, list)):
            raise RuntimeError(f"Heuristic function should return a number or tuple, got {type(test_result)}")
        
        # Evaluate the heuristic using the existing evaluation logic
        spearman_correlation = evaluate(heuristic_func)
        
        # Convert to fitness (higher is better, so use correlation directly)
        fitness = float(spearman_correlation)
        
        # Final safety check for fitness
        if not np.isfinite(fitness):
            fitness = 0.0
        
        return Evaluation(
            fitness=fitness,
            additional_data={
                "spearman_correlation": f"{spearman_correlation:.6f}",
                "validity": "valid",
                "function_name": heuristic_func.__name__
            }
        )
        
    except Exception as e:
        return Evaluation(
            fitness=0.0,
            additional_data={
                "spearman_correlation": "0.0",
                "validity": "error",
                "error": str(e)
            }
        )


if __name__ == "__main__":
    # Test the evaluation function
    print("Testing evaluate() function:")
    print(f"Sum of entries: {evaluate(heuristic_sum_entries):.4f}")
    print(f"Log column sums: {evaluate(heuristic_log_column_sums):.4f}")
    print(f"Vector column sums: {evaluate(heuristic_vector_column_sums):.4f}")
    
    # Test the string evaluation function
    print("\nTesting evaluate_heuristic_from_string():")
    test_heuristic = """
def heuristic(matrix):
    import numpy as np
    return float(np.sum(matrix))
"""
    result = evaluate_heuristic_from_string(test_heuristic)
    print(f"String heuristic fitness: {result.fitness:.4f}")