"""
Circle packing evaluation for the simple evolutionary system.
Evaluates solutions for packing 26 circles in a unit square to maximize sum of radii.
"""

import numpy as np
import tempfile
import os
import subprocess
import sys
import pickle
import traceback
from typing import Dict, Any

from src.evaluation import Evaluation


def validate_packing(centers: np.ndarray, radii: np.ndarray) -> bool:
    """
    Validate that circles don't overlap and are inside the unit square.
    
    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    
    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]
    
    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
            return False
    
    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-6:  # Allow for tiny numerical errors
                return False
    
    return True


def run_solution_safely(solution_code: str, timeout_seconds: int = 30) -> tuple:
    """
    Run a solution code safely in a subprocess with timeout.
    
    Args:
        solution_code: The solution code as a string
        timeout_seconds: Maximum execution time in seconds
        
    Returns:
        (centers, radii, sum_radii) or raises an exception
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
    if 'run_packing' in globals():
        centers, radii, sum_radii = run_packing()
    else:
        # Try to find the main function
        main_func = None
        for name in globals():
            if callable(globals()[name]) and name.startswith('construct') or name.startswith('pack'):
                main_func = globals()[name]
                break
        
        if main_func is None:
            raise RuntimeError("No run_packing() function or suitable main function found")
        
        result = main_func()
        if len(result) == 3:
            centers, radii, sum_radii = result
        else:
            raise RuntimeError("Function should return (centers, radii, sum_radii)")
    
    # Save results
    results = {{
        'centers': centers,
        'radii': radii,
        'sum_radii': sum_radii,
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
                    return results['centers'], results['radii'], results['sum_radii']
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


def evaluate_circle_packing(solution: str) -> Evaluation:
    """
    Evaluate a circle packing solution.
    
    Args:
        solution: Python code as a string that implements circle packing
        
    Returns:
        Evaluation object with fitness and metadata
    """
    TARGET_VALUE = 2.635  # AlphaEvolve benchmark for n=26
    
    try:
        # Run the solution safely
        centers, radii, sum_radii = run_solution_safely(solution, timeout_seconds=30)
        
        # Convert to numpy arrays if needed
        if not isinstance(centers, np.ndarray):
            centers = np.array(centers)
        if not isinstance(radii, np.ndarray):
            radii = np.array(radii)
        
        # Validate solution
        if centers.shape != (26, 2) or radii.shape != (26,):
            raise ValueError(f"Invalid shapes: centers={centers.shape}, radii={radii.shape}")
        
        valid = validate_packing(centers, radii)
        
        if not valid:
            return Evaluation(
                fitness=0.0,
                additional_data={
                    "sum_radii": "0.0",
                    "target_ratio": "0.0",
                    "validity": "invalid",
                    "error": "Invalid packing (overlaps or out of bounds)"
                }
            )
        
        # Calculate fitness
        actual_sum = np.sum(radii)
        target_ratio = actual_sum / TARGET_VALUE
        
        # Fitness is the sum of radii (higher is better)
        fitness = float(actual_sum)
        
        return Evaluation(
            fitness=fitness,
            additional_data={
                "sum_radii": f"{actual_sum:.6f}",
                "target_ratio": f"{target_ratio:.6f}",
                "validity": "valid",
                "target_value": f"{TARGET_VALUE}"
            }
        )
        
    except Exception as e:
        return Evaluation(
            fitness=0.0,
            additional_data={
                "sum_radii": "0.0",
                "target_ratio": "0.0",
                "validity": "error",
                "error": str(e)
            }
        )