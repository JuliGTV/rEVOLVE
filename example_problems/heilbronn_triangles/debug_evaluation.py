#!/usr/bin/env python3
"""
Debug script to investigate why evaluations are returning 0 fitness
"""

import sys
import os
import tempfile

# Add the project root to the path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from example_problems.heilbronn_triangles.evaluation import evaluate_heilbronn_triangles
from example_problems.heilbronn_triangles.spec import INITIAL_CODE

def test_initial_code():
    """Test the initial code to ensure it works"""
    print("=== Testing Initial Code ===")
    result = evaluate_heilbronn_triangles(INITIAL_CODE, 11)
    print(f"Fitness: {result.fitness}")
    print(f"Additional data: {result.additional_data}")
    print()

def test_simple_code():
    """Test a very simple code that should work"""
    print("=== Testing Simple Code ===")
    simple_code = '''
def find_points():
    # Simple grid pattern inside the triangle
    import numpy as np
    
    points = []
    # Add some points inside the triangle
    points.append((0.1, 0.1))
    points.append((0.2, 0.1))
    points.append((0.3, 0.1))
    points.append((0.4, 0.1))
    points.append((0.5, 0.1))
    points.append((0.6, 0.1))
    points.append((0.7, 0.1))
    points.append((0.8, 0.1))
    points.append((0.1, 0.2))
    points.append((0.2, 0.2))
    points.append((0.3, 0.2))
    
    return points
    '''
    
    result = evaluate_heilbronn_triangles(simple_code, 11)
    print(f"Fitness: {result.fitness}")
    print(f"Additional data: {result.additional_data}")
    print()

def test_broken_code():
    """Test code that should fail"""
    print("=== Testing Broken Code ===")
    broken_code = '''
def find_points():
    # This will cause an error
    return undefined_variable
    '''
    
    result = evaluate_heilbronn_triangles(broken_code, 11)
    print(f"Fitness: {result.fitness}")
    print(f"Additional data: {result.additional_data}")
    print()

def test_no_function():
    """Test code without the expected function"""
    print("=== Testing Code Without find_points Function ===")
    no_func_code = '''
# This code doesn't have find_points function
x = 5
y = 10
result = x + y
    '''
    
    result = evaluate_heilbronn_triangles(no_func_code, 11)
    print(f"Fitness: {result.fitness}")
    print(f"Additional data: {result.additional_data}")
    print()

def test_timeout_code():
    """Test code that should timeout"""
    print("=== Testing Code That Times Out ===")
    timeout_code = '''
def find_points():
    # This will timeout
    import time
    time.sleep(35)  # Longer than timeout
    return [(0.1, 0.1)]
    '''
    
    result = evaluate_heilbronn_triangles(timeout_code, 11)
    print(f"Fitness: {result.fitness}")
    print(f"Additional data: {result.additional_data}")
    print()

def test_modified_initial_code():
    """Test slightly modified initial code (simulating evolution)"""
    print("=== Testing Modified Initial Code ===")
    
    # Simulate what might happen during evolution - slight modification
    modified_code = INITIAL_CODE.replace("iters=100", "iters=50")
    
    result = evaluate_heilbronn_triangles(modified_code, 11)
    print(f"Fitness: {result.fitness}")
    print(f"Additional data: {result.additional_data}")
    print()

if __name__ == "__main__":
    print("=== Debugging Heilbronn Triangles Evaluation ===\n")
    
    test_initial_code()
    test_simple_code()
    test_broken_code()
    test_no_function()
    test_modified_initial_code()
    
    # Don't test timeout code by default as it takes too long
    # test_timeout_code()
    
    print("=== Debug Complete ===")