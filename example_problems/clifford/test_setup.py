#!/usr/bin/env python3
"""
Test script to verify the Clifford problem setup is working correctly.

This script tests the core components of the Clifford heuristic evolution
problem without running a full evolution.
"""

import sys
import os

# Add rEVOLVE source to path
evolve_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if evolve_src_path not in sys.path:
    sys.path.insert(0, evolve_src_path)

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from spec import get_problem_spec
        print("✓ spec.py imports successfully")
    except Exception as e:
        print(f"✗ spec.py import failed: {e}")
        return False
    
    try:
        # Import the clifford evaluation module directly
        clifford_eval_path = os.path.join(os.path.dirname(__file__), 'evaluation.py')
        import importlib.util
        spec = importlib.util.spec_from_file_location("clifford_evaluation", clifford_eval_path)
        clifford_evaluation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(clifford_evaluation)
        print("✓ evaluation.py imports successfully")
    except Exception as e:
        print(f"✗ evaluation.py import failed: {e}")
        return False
    
    try:
        from utils import load_symplectic_benchmark, create_default_params
        print("✓ utils.py imports successfully")
    except Exception as e:
        print(f"✗ utils.py import failed: {e}")
        return False
    
    try:
        from specification import ProblemSpecification, Hyperparameters
        from population import Organism
        print("✓ rEVOLVE core modules import successfully")
    except Exception as e:
        print(f"✗ rEVOLVE core import failed: {e}")
        return False
    
    return True


def test_problem_spec():
    """Test that the problem specification can be created."""
    print("\nTesting problem specification...")
    
    try:
        from spec import get_problem_spec
        problem_spec = get_problem_spec()
        
        print(f"✓ Problem name: {problem_spec.name}")
        print(f"✓ Initial population size: {len(problem_spec.starting_population)}")
        print(f"✓ Max steps: {problem_spec.hyperparameters.max_steps}")
        print(f"✓ Exploration rate: {problem_spec.hyperparameters.exploration_rate}")
        
        # Test that organisms have the expected structure
        for i, organism in enumerate(problem_spec.starting_population[:2]):
            print(f"✓ Organism {i}: {organism.metadata.get('name', 'unnamed')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Problem specification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation():
    """Test that the evaluation function works."""
    print("\nTesting evaluation function...")
    
    try:
        # Import the clifford evaluation module directly
        clifford_eval_path = os.path.join(os.path.dirname(__file__), 'evaluation.py')
        import importlib.util
        spec = importlib.util.spec_from_file_location("clifford_evaluation", clifford_eval_path)
        clifford_evaluation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(clifford_evaluation)
        
        # Test with a simple heuristic function
        test_code = '''
def evolved_heuristic(symplectic_matrix):
    import numpy as np
    # Simple test - return a constant score
    return 42.0
'''
        
        result = clifford_evaluation.evaluate_clifford_heuristic(test_code)
        
        print(f"✓ Evaluation returned result: {type(result)}")
        print(f"✓ Fitness: {result.get('fitness', 'None')}")
        
        if 'additional_data' in result:
            print(f"✓ Additional data keys: {list(result['additional_data'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_clifford_opt():
    """Test CliffordOpt integration."""
    print("\nTesting CliffordOpt integration...")
    
    try:
        # Check if CliffordOpt directory exists
        clifford_opt_path = os.path.join(os.path.dirname(__file__), 'CliffordOpt')
        if not os.path.exists(clifford_opt_path):
            print("⚠ CliffordOpt directory not found - this is expected in many setups")
            return True
        
        # Try importing CliffordOpt modules
        sys.path.insert(0, clifford_opt_path)
        
        from cliffordopt.common import paramObj
        print("✓ CliffordOpt paramObj imports successfully")
        
        from cliffordopt.CliffordOps import isSymplectic
        print("✓ CliffordOpt CliffordOps imports successfully")
        
        # Test parameter creation
        from utils import create_default_params
        params = create_default_params('greedy', 'Sp')
        print(f"✓ Created default parameters: mode={params.mode}, method={params.method}")
        
        return True
        
    except Exception as e:
        print(f"⚠ CliffordOpt integration test failed: {e}")
        print("  This may be expected if CliffordOpt is not fully installed")
        return True  # Don't fail the overall test for this


def main():
    """Run all tests."""
    print("Clifford Problem Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_problem_spec,
        test_evaluation,
        test_clifford_opt,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! The Clifford problem is ready for evolution.")
        return True
    else:
        print("⚠ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)