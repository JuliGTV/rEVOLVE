#!/usr/bin/env python3
"""
Run the circle packing experiment using the simple evolutionary system.

This script demonstrates how to use the rEVOLVE system to optimize circle packing,
replicating the experiment from the circle_packing_openevolve folder but using
the simpler evolutionary framework.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Add the project root to the path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from src.evolve import AsyncEvolver
from example_problems.circle_packing.spec import get_circle_packing_spec, get_circle_packing_evolver_config
import asyncio


async def run_evolution():
    """Run the circle packing evolution experiment asynchronously."""
    
    # Get problem specification and evolver configuration
    spec = get_circle_packing_spec()
    evolver_config = get_circle_packing_evolver_config()
    
    print("Circle Packing Evolution Experiment")
    print("=" * 40)
    print("Goal: Pack 26 circles in a unit square to maximize sum of radii")
    print(f"Target: {spec.hyperparameters.target_fitness} (AlphaEvolve benchmark)")
    print()
    
    print("Configuration:")
    print(f"  Max steps: {spec.hyperparameters.max_steps}")
    print(f"  Max concurrent: {evolver_config['max_concurrent']}")
    print(f"  Model mix: {evolver_config['model_mix']}")
    print(f"  Target fitness: {spec.hyperparameters.target_fitness}")
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
        print(f"Target ratio: {best_organism.evaluation.fitness/spec.hyperparameters.target_fitness:.6f}")
        print(f"Report generated in: {report_dir}")
        
        # Print additional data
        if best_organism.evaluation.additional_data:
            print("\nAdditional metrics:")
            for key, value in best_organism.evaluation.additional_data.items():
                print(f"  {key}: {value}")
        

        

        
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
    asyncio.run(run_evolution())


if __name__ == "__main__":
    main()