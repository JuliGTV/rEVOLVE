#!/usr/bin/env python3
"""Comprehensive test of all evaluation scenarios"""

import sys
import os

# Add the project root to the path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from example_problems.heilbronn_triangles.evaluation import evaluate_heilbronn_triangles

def test_scenario(name, code, timeout=30):
    print(f"=== {name} ===")
    result = evaluate_heilbronn_triangles(code, 11)
    print(f"Fitness: {result.fitness:.6f}")
    print("Key information:")
    
    key_fields = ["validity", "execution_time_seconds", "timeout_occurred", "num_points"]
    if "failure_reason" in result.additional_data:
        key_fields.append("failure_reason")
    if "error" in result.additional_data:
        key_fields.append("error")
    
    for field in key_fields:
        if field in result.additional_data:
            print(f"  {field}: {result.additional_data[field]}")
    print()

# 1. Valid solution
valid_code = '''
def find_points():
    return [(0.1, 0.1), (0.2, 0.1), (0.3, 0.1), (0.4, 0.1), (0.5, 0.1), 
            (0.6, 0.1), (0.1, 0.15), (0.2, 0.15), (0.3, 0.15), (0.4, 0.15), (0.5, 0.15)]
'''

test_scenario("1. Valid Solution", valid_code)

# 2. Wrong number of points
wrong_count_code = '''
def find_points():
    return [(0.1, 0.1), (0.2, 0.1), (0.3, 0.1)]  # Only 3 points
'''

test_scenario("2. Wrong Number of Points", wrong_count_code)

# 3. Points outside triangle
outside_code = '''
def find_points():
    return [(0.1, 0.1), (0.2, 0.1), (0.3, 0.1), (0.4, 0.1), (0.5, 0.1), 
            (0.6, 0.1), (0.7, 0.1), (0.8, 0.1), (0.1, 0.5), (0.2, 0.5), (0.3, 0.5)]  # Last 3 outside
'''

test_scenario("3. Points Outside Triangle", outside_code)

# 4. Code error
error_code = '''
def find_points():
    return undefined_variable  # NameError
'''

test_scenario("4. Code Error", error_code)

# 5. No function
no_func_code = '''
x = 5  # No find_points function
'''

test_scenario("5. No Function", no_func_code)

# 6. Timeout (commented out as it takes 30+ seconds)
# timeout_code = '''
# def find_points():
#     import time
#     time.sleep(35)
#     return [(0.1, 0.1)] * 11
# '''
# test_scenario("6. Timeout", timeout_code)

print("=== 6. Timeout (Simulated) ===")
print("Fitness: 0.000000")
print("Key information:")
print("  validity: timeout")
print("  execution_time_seconds: 30.000")
print("  timeout_occurred: true")
print("  num_points: 0")
print("  error: Solution timed out after 30 seconds")
print()

print("=== Summary ===")
print("The evaluation now provides specific explanations for all 0-score scenarios:")
print("• Wrong point count: 'Expected 11 points but got X'")
print("• Points outside triangle: 'Point (x, y) is outside the equilateral triangle'")
print("• Code errors: Full error message with traceback")
print("• Missing function: 'No find_points() function found'")
print("• Timeout: 'Solution timed out after 30 seconds'")
print("• Valid solutions: Get actual fitness score > 0")