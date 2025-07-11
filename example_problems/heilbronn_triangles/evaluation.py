import itertools
import numpy as np
import tempfile
import os
import subprocess
import sys
import pickle
import traceback
from typing import Dict, Any

from src.evaluation import Evaluation
import logfire

def check_inside_triangle(points: np.ndarray):
  """Checks that all points are inside the triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2)."""
  for (x, y) in points:
    # Check if point is inside the equilateral triangle
    inside = (y >= 0) and (np.sqrt(3) * x <= np.sqrt(3) - y) and (y <= np.sqrt(3) * x)
    if not inside:
      return False, f'Point ({x}, {y}) is outside the equilateral triangle.'
  
  # All points are inside
  return True, 'All points are inside the equilateral triangle.'

def triangle_area(a: np.array, b: np.array, c: np.array) -> float:
  return np.abs(a[0]*(b[1] - c[1]) + b[0]*(c[1] - a[1]) + c[0]*(a[1] - b[1])) / 2

def evaluate_points_detailed(found_points: np.ndarray, n=11):
    """
    Evaluate points and return detailed results including failure reasons.
    
    Returns:
        Tuple of (fitness, validity, error_message)
    """
    if len(found_points) != n:
        return 0.0, "invalid", f"Expected {n} points but got {len(found_points)}"
    
    a = np.array([0, 0])
    b = np.array([1, 0])
    c = np.array([0.5, np.sqrt(3)/2])

    is_inside, msg = check_inside_triangle(found_points)
    if not is_inside:
        return 0.0, "invalid", msg
    
    min_triangle_area = min([triangle_area(p1,p2,p3) for p1, p2, p3 in itertools.combinations(found_points, 3)])
    # Normalize the minimum triangle area (since the equilateral triangle is not unit).
    min_area_normalized = min_triangle_area / triangle_area(a, b, c)

    return min_area_normalized, "valid", "All constraints satisfied"

def evaluate_points(found_points: np.ndarray, n=11):
    """Legacy function for backwards compatibility"""
    fitness, _, _ = evaluate_points_detailed(found_points, n)
    return fitness


def run_solution_safely_with_timing(solution_code: str, n: int = 11, timeout_seconds: int = 30) -> tuple[np.ndarray, float, bool]:
    """
    Run a solution code safely in a subprocess with timeout and return timing information.
    
    Args:
        solution_code: The solution code as a string containing find_points() function
        n: Number of points to generate
        timeout_seconds: Maximum execution time in seconds
        
    Returns:
        Tuple of (points, execution_time, timeout_occurred)
    """
    import time
    start_time = time.time()
    timeout_occurred = False
    
    with logfire.span("subprocess_execution", timeout_seconds=timeout_seconds):
        # Create a temporary file to execute
        with tempfile.NamedTemporaryFile(mode='w', suffix=".py", delete=False) as temp_file:
            # Write the solution code plus evaluation wrapper
            full_code = f"""
# Solution code
{solution_code}

# Additional imports needed for evaluation
import numpy as np
import pickle
import sys
import traceback
import time

# Evaluation wrapper
start_time = time.time()
try:
    if 'find_points' in globals():
        points = find_points()
    else:
        raise RuntimeError("No find_points() function found")
    
    execution_time = time.time() - start_time
    
    # Convert to numpy array if needed
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    
    # Save results
    results = {{
        'points': points,
        'execution_time': execution_time,
        'success': True
    }}
    
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
        
except Exception as e:
    execution_time = time.time() - start_time
    # Save error
    results = {{
        'success': False,
        'error': str(e),
        'execution_time': execution_time,
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
            with logfire.span("subprocess_run"):
                process = subprocess.Popen(
                    [sys.executable, temp_file_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                try:
                    stdout, stderr = process.communicate(timeout=timeout_seconds)
                    
                    execution_time = time.time() - start_time
                    
                    logfire.info("Subprocess completed", 
                                return_code=process.returncode,
                                stdout_length=len(stdout),
                                stderr_length=len(stderr),
                                execution_time=execution_time)
                    
                    # Load results
                    if os.path.exists(results_path):
                        with open(results_path, 'rb') as f:
                            results = pickle.load(f)
                        
                        if results['success']:
                            return results['points'], results['execution_time'], timeout_occurred
                        else:
                            logfire.error("Solution execution failed", 
                                         error=results['error'],
                                         execution_time=results.get('execution_time', execution_time))
                            raise RuntimeError(f"Solution execution failed: {results['error']}")
                    else:
                        logfire.error("Results file not found", 
                                     results_path=results_path,
                                     execution_time=execution_time)
                        raise RuntimeError("Results file not found")
                        
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    timeout_occurred = True
                    execution_time = timeout_seconds  # Use timeout as execution time
                    logfire.error("Solution timed out", 
                                 timeout_seconds=timeout_seconds,
                                 execution_time=execution_time)
                    # Return special values indicating timeout instead of raising exception
                    return np.array([]), execution_time, timeout_occurred
                    
        finally:
            # Clean up
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            if os.path.exists(results_path):
                os.unlink(results_path)


def run_solution_safely(solution_code: str, n: int = 11, timeout_seconds: int = 30) -> np.ndarray:
    """
    Run a solution code safely in a subprocess with timeout.
    
    Args:
        solution_code: The solution code as a string containing find_points() function
        n: Number of points to generate
        timeout_seconds: Maximum execution time in seconds
        
    Returns:
        np.ndarray of points or raises an exception
    """
    with logfire.span("subprocess_execution", timeout_seconds=timeout_seconds):
        # Create a temporary file to execute
        with tempfile.NamedTemporaryFile(mode='w', suffix=".py", delete=False) as temp_file:
            # Write the solution code plus evaluation wrapper
            full_code = f"""
# Solution code
{solution_code}

# Additional imports needed for evaluation
import numpy as np
import pickle
import sys
import traceback

# Evaluation wrapper
try:
    if 'find_points' in globals():
        points = find_points()
    else:
        raise RuntimeError("No find_points() function found")
    
    # Convert to numpy array if needed
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    
    # Save results
    results = {{
        'points': points,
        'success': True
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
            with logfire.span("subprocess_run"):
                process = subprocess.Popen(
                    [sys.executable, temp_file_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                try:
                    stdout, stderr = process.communicate(timeout=timeout_seconds)
                    
                    logfire.info("Subprocess completed", 
                                return_code=process.returncode,
                                stdout_length=len(stdout),
                                stderr_length=len(stderr))
                    
                    # Load results
                    if os.path.exists(results_path):
                        with open(results_path, 'rb') as f:
                            results = pickle.load(f)
                        
                        if results['success']:
                            return results['points']
                        else:
                            logfire.error("Solution execution failed", 
                                         error=results['error'])
                            raise RuntimeError(f"Solution execution failed: {results['error']}")
                    else:
                        logfire.error("Results file not found", 
                                     results_path=results_path)
                        raise RuntimeError("Results file not found")
                        
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    logfire.error("Solution timed out", 
                                 timeout_seconds=timeout_seconds)
                    raise RuntimeError(f"Solution timed out after {timeout_seconds} seconds")
                    
        finally:
            # Clean up
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            if os.path.exists(results_path):
                os.unlink(results_path)


def evaluate_heilbronn_triangles(solution: str, n: int = 11) -> Evaluation:
    """
    Evaluate a Heilbronn triangles solution.
    
    Args:
        solution: Python code as a string that implements find_points(n) function
        n: Number of points to generate
        
    Returns:
        Evaluation object with fitness and metadata
    """
    import time
    start_time = time.time()
    
    with logfire.span("heilbronn_evaluation", solution_length=len(solution), num_points=n):
        try:
            # Run the solution safely
            with logfire.span("solution_execution"):
                points, execution_time, timeout_occurred = run_solution_safely_with_timing(solution, n, timeout_seconds=30)
            
            # Handle timeout case
            if timeout_occurred:
                total_time = time.time() - start_time
                logfire.error("Heilbronn evaluation timeout", 
                             execution_time=execution_time,
                             timeout_seconds=30,
                             total_time=total_time)
                
                return Evaluation(
                    fitness=0.0,
                    additional_data={
                        "num_points": "0",
                        "min_area_normalized": "0.0",
                        "validity": "timeout",
                        "error": f"Solution timed out after {30} seconds",
                        "execution_time_seconds": f"{execution_time:.3f}",
                        "timeout_occurred": "true",
                        "points": "[]"
                    }
                )
            
            # Convert to numpy array if needed
            if not isinstance(points, np.ndarray):
                points = np.array(points)
            
            # Evaluate the points using detailed function
            with logfire.span("fitness_calculation"):
                fitness, validity_status, failure_reason = evaluate_points_detailed(points, n)
            
            total_time = time.time() - start_time
            
            logfire.info("Heilbronn evaluation success", 
                        fitness=fitness, 
                        num_points=len(points),
                        validity_status=validity_status,
                        failure_reason=failure_reason,
                        execution_time=execution_time,
                        timeout_occurred=timeout_occurred,
                        total_time=total_time)
            
            additional_data = {
                "num_points": f"{len(points)}",
                "min_area_normalized": f"{fitness:.6f}",
                "validity": validity_status,
                "execution_time_seconds": f"{execution_time:.3f}",
                "timeout_occurred": "true" if timeout_occurred else "false",
                "points": str(points.tolist())
            }
            
            # Add failure reason if invalid
            if validity_status == "invalid":
                additional_data["failure_reason"] = failure_reason
            
            return Evaluation(
                fitness=float(fitness),
                additional_data=additional_data
            )
            
        except Exception as e:
            total_time = time.time() - start_time
            
            # Check if this was a timeout exception
            timeout_occurred = "timeout" in str(e).lower()
            
            logfire.error("Heilbronn evaluation error", 
                         error=str(e), 
                         error_type=type(e).__name__,
                         solution_length=len(solution),
                         total_time=total_time,
                         timeout_occurred=timeout_occurred)
            
            return Evaluation(
                fitness=0.0,
                additional_data={
                    "num_points": "0",
                    "min_area_normalized": "0.0",
                    "validity": "error",
                    "error": str(e),
                    "execution_time_seconds": f"{total_time:.3f}",
                    "timeout_occurred": "true" if timeout_occurred else "false",
                    "points": "[]"
                }
            )