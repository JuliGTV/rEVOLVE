#!/usr/bin/env python3
"""
Simple script to visualize the best circle packing solution from any output folder.

Usage: python visualize_best_circle_packing.py <output_folder_path>
"""

import sys
import os
import json
import tempfile
import subprocess
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

def find_best_solution(population_json_path):
    """Find the organism with the highest fitness from population.json"""
    with open(population_json_path, 'r') as f:
        population = json.load(f)
    
    best_organism = max(population, key=lambda org: org['evaluation']['fitness'])
    return best_organism

def run_solution_safely(solution_code: str, timeout_seconds: int = 30) -> tuple:
    """
    Run a solution code safely in a subprocess with timeout, similar to evaluation.py
    
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
            if callable(globals()[name]) and (name.startswith('construct') or name.startswith('pack')):
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


def visualize_solution(solution_code: str, fitness: float, title: str = "Circle Packing Solution"):
    """
    Visualize a circle packing solution.
    
    Args:
        solution_code: The Python code that implements the solution
        fitness: The fitness score of the solution  
        title: Title for the plot
    """
    try:
        # Run the solution safely in a subprocess
        centers, radii, sum_radii = run_solution_safely(solution_code)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw unit square
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Draw square border
        square = Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(square)
        
        # Draw circles
        colors = plt.cm.viridis(np.linspace(0, 1, len(centers)))
        for i, (center, radius, color) in enumerate(zip(centers, radii, colors)):
            circle = Circle(center, radius, alpha=0.6, facecolor=color, edgecolor='black', linewidth=0.5)
            ax.add_patch(circle)
            
            # Add circle number
            if radius > 0.02:  # Only show numbers for larger circles
                ax.text(center[0], center[1], str(i), ha='center', va='center', 
                       fontsize=8, fontweight='bold')
        
        ax.set_title(f"{title}\nFitness: {fitness:.6f} | Sum of radii: {sum_radii:.6f} (Target: 2.635)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        plt.tight_layout()
        plt.show()
        
        print(f"Fitness: {fitness:.6f}")
        print(f"Sum of radii: {sum_radii:.6f}")
        print(f"Target ratio: {sum_radii/2.635:.6f} ({sum_radii/2.635*100:.2f}%)")
        
    except Exception as e:
        print(f"Error visualizing solution: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python visualize_best_circle_packing.py <output_folder_path>")
        print("\nExample:")
        print("  python visualize_best_circle_packing.py outputs/2025-06-18_13-34-35_circle_packing")
        return
    
    output_folder = sys.argv[1]
    population_json_path = os.path.join(output_folder, "population.json")
    
    if not os.path.exists(population_json_path):
        print(f"Error: population.json not found in {output_folder}")
        return
    
    try:
        best_organism = find_best_solution(population_json_path)
        print(f"Found best solution with fitness: {best_organism['evaluation']['fitness']:.6f}")
        
        visualize_solution(
            best_organism['solution'], 
            best_organism['evaluation']['fitness'],
            f"Best Solution from {os.path.basename(output_folder)}"
        )
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()