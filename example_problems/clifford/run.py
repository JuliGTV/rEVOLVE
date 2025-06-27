#!/usr/bin/env python3
"""
Run the Clifford heuristic evolution experiment using the rEVOLVE framework.

This script evolves heuristic functions for CNOT synthesis from GL matrices,
optimizing for Spearman correlation with actual minimum gate counts.
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
from spec import get_clifford_heuristic_spec, get_clifford_heuristic_evolver_config

# Rebuild the pydantic models to resolve forward references
ProblemSpecification.model_rebuild()
Organism.model_rebuild()


async def run_evolution(max_steps=None):
    """Run the Clifford heuristic evolution experiment asynchronously."""
    
    # Get problem specification and evolver configuration
    spec = get_clifford_heuristic_spec()
    evolver_config = get_clifford_heuristic_evolver_config()
    
    # Override max_steps for testing if provided
    if max_steps is not None:
        spec.hyperparameters.max_steps = max_steps
        print(f"Overriding max_steps to {max_steps} for testing")
    
    print("Clifford Heuristic Evolution Experiment")
    print("=" * 45)
    print("Goal: Evolve heuristics for CNOT synthesis from GL matrices")
    print(f"Target: Spearman correlation > {spec.hyperparameters.target_fitness}")
    print()
    
    print("Configuration:")
    print(f"  Max steps: {spec.hyperparameters.max_steps}")
    print(f"  Exploration rate: {spec.hyperparameters.exploration_rate}")
    print(f"  Elitism rate: {spec.hyperparameters.elitism_rate}")
    print(f"  Max concurrent: {evolver_config['max_concurrent']}")
    print(f"  Model mix: {evolver_config['model_mix']}")
    print(f"  Target fitness: {spec.hyperparameters.target_fitness}")
    print()
    
    # Show initial population fitness
    print("Initial population:")
    for i, organism in enumerate(spec.starting_population):
        fitness = organism.evaluation.fitness
        additional = organism.evaluation.additional_data
        validity = additional.get('validity', 'unknown')
        print(f"  Heuristic {i+1}: fitness = {fitness:.4f} ({validity})")
    print()
    
    # Create the evolver with consolidated configuration
    evolver = AsyncEvolver(spec, **evolver_config)
    
    # Run evolution
    print("Starting evolution...")
    try:
        population = await evolver.evolve()
        
        # Generate report
        report_dir = evolver.report()
        
        best_organism = population.get_best()
        
        print("\nEvolution completed!")
        print(f"Best fitness: {best_organism.evaluation.fitness:.6f}")
        print(f"Target achievement: {best_organism.evaluation.fitness/spec.hyperparameters.target_fitness:.3f}")
        print(f"Report generated in: {report_dir}")
        
        # Print additional data
        if best_organism.evaluation.additional_data:
            print("\nBest heuristic metrics:")
            for key, value in best_organism.evaluation.additional_data.items():
                print(f"  {key}: {value}")
        
        # Show the best heuristic code
        print("\nBest heuristic code:")
        print("-" * 40)
        print(best_organism.solution)
        print("-" * 40)
        
        return population
        
    except KeyboardInterrupt:
        print("\nEvolution interrupted by user")
        if evolver.population.get_population():
            best = evolver.population.get_best()
            print(f"Best fitness so far: {best.evaluation.fitness:.6f}")
    
    except Exception as e:
        print(f"\nError during evolution: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point that runs the async evolution."""
    
    # Check for test mode
    test_mode = "--test" in sys.argv
    if test_mode:
        print("Running in test mode with reduced steps...")
        asyncio.run(run_evolution(max_steps=5))  # Very short test run
    else:
        asyncio.run(run_evolution())


if __name__ == "__main__":
    main()