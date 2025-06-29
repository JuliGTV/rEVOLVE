#!/usr/bin/env python3
"""
Test spec2.py with evaluate2.py for a very short evolutionary run.
"""

import sys
import os
import asyncio

# Add the project root to the path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from src.evolve import AsyncEvolver
from src.specification import ProblemSpecification
from src.population import Organism
from spec2 import get_clifford_heuristic_spec, get_clifford_heuristic_evolver_config

# Rebuild the pydantic models to resolve forward references
ProblemSpecification.model_rebuild()
Organism.model_rebuild()


async def test_spec2_evolution():
    """Test spec2.py with a very short evolutionary run."""
    
    print("Testing spec2.py with evaluate2.py")
    print("=" * 40)
    
    # Get problem specification and evolver configuration
    spec = get_clifford_heuristic_spec()
    evolver_config = get_clifford_heuristic_evolver_config()
    
    # Override for very short test run
    spec.hyperparameters.max_steps = 3
    evolver_config['max_concurrent'] = 2  # Reduce concurrency for testing
    
    print("Test Configuration:")
    print(f"  Max steps: {spec.hyperparameters.max_steps}")
    print(f"  Max concurrent: {evolver_config['max_concurrent']}")
    print(f"  Using evaluate2.py with subprocess execution")
    print()
    
    # Show initial population fitness
    print("Initial population:")
    for i, organism in enumerate(spec.starting_population):
        fitness = organism.evaluation.fitness
        additional = organism.evaluation.additional_data
        validity = additional.get('validity', 'unknown')
        score = additional.get('score', 'N/A')
        print(f"  Heuristic {i+1}: fitness = {fitness:.4f} (score: {score}, {validity})")
    print()
    
    # Create the evolver
    evolver = AsyncEvolver(spec, **evolver_config)
    
    # Run evolution
    print("Starting short evolution test...")
    try:
        population = await evolver.evolve()
        
        # Show results
        best_organism = population.get_best()
        
        print("\nTest completed!")
        print(f"Best fitness: {best_organism.evaluation.fitness:.6f}")
        print(f"Population size: {len(population.get_population())}")
        
        # Print additional data
        if best_organism.evaluation.additional_data:
            print("\nBest heuristic metrics:")
            for key, value in best_organism.evaluation.additional_data.items():
                print(f"  {key}: {value}")
        
        print("\nTest passed! spec2.py and evaluate2.py are working correctly.")
        return True
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point for the test."""
    success = asyncio.run(test_spec2_evolution())
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()