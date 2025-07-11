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

def check_inside_triangle(points: np.ndarray):
  """Checks that all points are inside the triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2)."""
  for (x, y) in points:
    return (y >= 0) and (np.sqrt(3) * x <= np.sqrt(3) - y) and (y <= np.sqrt(3) * x), f'Point ({x}, {y}) is outside the equilateral triangle.'

def triangle_area(a: np.array, b: np.array, c: np.array) -> float:
  return np.abs(a[0]*(b[1] - c[1]) + b[0]*(c[1] - a[1]) + c[0]*(a[1] - b[1])) / 2

def evaluate_points(found_points: np.ndarray, n=11):
    if len(found_points) != n:
        return 0
    
    a = np.array([0, 0])
    b = np.array([1, 0])
    c = np.array([0.5, np.sqrt(3)/2])


    is_inside, msg = check_inside_triangle(found_points)
    if not is_inside:
        return 0
    

    min_triangle_area = min([triangle_area(p1,p2,p3) for p1, p2, p3 in itertools.combinations(found_points, 3)])
    # Normalize the minimum triangle area (since the equilateral triangle is not unit).
    min_area_normalized = min_triangle_area / triangle_area(a, b, c)

    return min_area_normalized


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
    # Create a temporary file to execute
    with tempfile.NamedTemporaryFile(mode='w', suffix=".py", delete=False) as temp_file:
        # Write the solution code plus evaluation wrapper
        full_code = f"""
import numpy as np
import pickle
import sys
import traceback

# Solution code
{solution_code}

# Evaluation wrapper
try:
    if 'find_points' in globals():
        points = find_points({n})
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
                    return results['points']
                else:
                    raise RuntimeError(f"Solution execution failed: {results['error']}")
            else:
                raise RuntimeError("Results file not found")
                
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
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
    try:
        # Run the solution safely
        points = run_solution_safely(solution, n, timeout_seconds=30)
        
        # Convert to numpy array if needed
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        
        # Evaluate the points using existing function
        fitness = evaluate_points(points, n)
        
        return Evaluation(
            fitness=float(fitness),
            additional_data={
                "num_points": f"{len(points)}",
                "min_area_normalized": f"{fitness:.6f}",
                "validity": "valid" if fitness > 0 else "invalid",
                "points": points.tolist()
            }
        )
        
    except Exception as e:
        return Evaluation(
            fitness=0.0,
            additional_data={
                "num_points": "0",
                "min_area_normalized": "0.0",
                "validity": "error",
                "error": str(e),
                "points": []
            }
        )