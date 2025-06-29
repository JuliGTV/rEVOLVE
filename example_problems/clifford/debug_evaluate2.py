#!/usr/bin/env python3
"""
Debug the evaluate2.py subprocess execution to see what's failing.
"""

import numpy as np
from evaluate2 import evaluate_heuristic_from_string

# Test with the same heuristic from spec2.py
test_heuristic = '''
def heuristic(matrix):
    import numpy as np
    return tuple(np.concatenate((np.sum(matrix,axis=0),np.sum(matrix,axis=0))))
'''

print("Testing heuristic evaluation...")
print("Heuristic code:")
print(test_heuristic)
print()

try:
    result = evaluate_heuristic_from_string(test_heuristic)
    print(f"Result: {result}")
    print(f"Fitness: {result.fitness}")
    print(f"Additional data: {result.additional_data}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()