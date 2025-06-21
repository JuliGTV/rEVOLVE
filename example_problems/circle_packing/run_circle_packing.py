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

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from evolve3 import Evolver3
from .spec import get_circle_packing_spec





def main():
    """Run the circle packing evolution experiment."""
    
    print("Circle Packing Evolution Experiment")
    print("=" * 50)
    print("Goal: Pack 26 circles in a unit square to maximize sum of radii")
    print("Target: 2.635 (AlphaEvolve benchmark)")
    print()
    
    # Get the problem specification
    spec = get_circle_packing_spec()
    
    # Create the evolver
    evolver = Evolver3(spec)
    
    # Run evolution
    print("Starting evolution...")
    try:
        population = evolver.evolve()

        best_organism = population.get_best()
        
        print("\nEvolution completed!")
        print(f"Best fitness: {best_organism.evaluation.fitness:.6f}")
        print(f"Target ratio: {best_organism.evaluation.fitness/2.635:.6f}")
        
        # Print additional data
        if best_organism.evaluation.additional_data:
            print("\nAdditional metrics:")
            for key, value in best_organism.evaluation.additional_data.items():
                print(f"  {key}: {value}")
        
        evolver.report()
        
        # Save the best solution
        output_file = "best_circle_packing_solution.py"
        with open(output_file, 'w') as f:
            f.write("# Best circle packing solution found by evolution\n")
            f.write(f"# Fitness: {best_organism.evaluation.fitness:.6f}\n")
            f.write(f"# Target ratio: {best_organism.evaluation.fitness/2.635:.6f}\n\n")
            f.write(best_organism.solution)
        
        print(f"\nBest solution saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\nEvolution interrupted by user")
        if evolver.population.organisms:
            best = max(evolver.population.organisms, key=lambda o: o.evaluation.fitness if o.evaluation else 0)
            print(f"Best fitness so far: {best.evaluation.fitness:.6f}")
    
    except Exception as e:
        print(f"\nError during evolution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()