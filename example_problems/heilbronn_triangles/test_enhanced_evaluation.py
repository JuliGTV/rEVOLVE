#!/usr/bin/env python3
"""Test the enhanced evaluation with detailed failure explanations"""

import sys
import os

# Add the project root to the path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from example_problems.heilbronn_triangles.evaluation import evaluate_heilbronn_triangles

def test_case(name, code):
    print(f"=== {name} ===")
    result = evaluate_heilbronn_triangles(code, 11)
    print(f"Fitness: {result.fitness:.6f}")
    print("Additional data:")
    for key, value in result.additional_data.items():
        if key != 'points':  # Skip printing the full points array
            print(f"  {key}: {value}")
    print()

# Test case: No function found
no_func_code = '''
# This code has no find_points function
x = 5
y = 10
'''

test_case("No Function Found", no_func_code)

# Test case: Valid solution
valid_code = '''
def find_points():
    # Valid points inside triangle
    return [(0.1, 0.1), (0.2, 0.1), (0.3, 0.1), (0.4, 0.1), (0.5, 0.1), 
            (0.6, 0.1), (0.1, 0.15), (0.2, 0.15), (0.3, 0.15), (0.4, 0.15), (0.5, 0.15)]
'''

test_case("Valid Solution", valid_code)

# Test case: Empty return
empty_code = '''
def find_points():
    return []
'''

test_case("Empty Return", empty_code)

# Test case: Too many points
too_many_code = '''
def find_points():
    # Return 15 points instead of 11
    return [(0.1, 0.1)] * 15
'''

test_case("Too Many Points", too_many_code)