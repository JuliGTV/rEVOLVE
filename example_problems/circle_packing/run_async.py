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

from src_async.evolve import AsyncEvolver
from src_async.specification import ProblemSpecification, Hyperparameters
from src_async.population import Organism
from src_async.evaluation import Evaluation
from example_problems.circle_packing.spec import get_circle_packing_spec
from example_problems.circle_packing.evaluation import evaluate_circle_packing
import asyncio

# Rebuild the pydantic model to resolve forward references
ProblemSpecification.model_rebuild()


def visualize_solution(solution_code: str, title: str = "Circle Packing Solution"):
    """
    Visualize a circle packing solution.
    
    Args:
        solution_code: The Python code that implements the solution
        title: Title for the plot
    """
    try:
        # Execute the solution code
        local_vars = {}
        exec(solution_code, {"np": np, "numpy": np}, local_vars)
        
        if 'run_packing' not in local_vars:
            print("No run_packing function found in solution")
            return
        
        # Get the packing
        centers, radii, sum_radii = local_vars['run_packing']()
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw unit square
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Draw square border
        from matplotlib.patches import Rectangle
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
        
        ax.set_title(f"{title}\nSum of radii: {sum_radii:.6f} (Target: 2.635)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        plt.tight_layout()
        plt.show()
        
        print(f"Sum of radii: {sum_radii:.6f}")
        print(f"Target ratio: {sum_radii/2.635:.6f} ({sum_radii/2.635*100:.2f}%)")
        
    except Exception as e:
        print(f"Error visualizing solution: {e}")


async def run_evolution():
    """Run the circle packing evolution experiment asynchronously."""
    
    print("Circle Packing Evolution Experiment (Async)")
    print("=" * 50)
    print("Goal: Pack 26 circles in a unit square to maximize sum of radii")
    print("Target: 2.635 (AlphaEvolve benchmark)")
    print()
    
    # Get the sync problem specification and convert to async
    sync_spec = get_circle_packing_spec()
    
    # Convert to async specification
    spec = ProblemSpecification(
        name=sync_spec.name,
        systemprompt=sync_spec.systemprompt,
        evaluator=evaluate_circle_packing,  # Same evaluator works
        starting_population=[
            Organism(
                solution=org.solution,
                evaluation=Evaluation(
                    fitness=org.evaluation.fitness,
                    additional_data=org.evaluation.additional_data
                )
            ) for org in sync_spec.starting_population
        ],
        hyperparameters=Hyperparameters(
            exploration_rate=sync_spec.hyperparameters.exploration_rate,
            elitism_rate=sync_spec.hyperparameters.elitism_rate,
            max_steps=sync_spec.hyperparameters.max_steps,
            target_fitness=sync_spec.hyperparameters.target_fitness,
            reason=sync_spec.hyperparameters.reason
        )
    )
    
    # Create the evolver
    evolver = AsyncEvolver(spec,
                           checkpoint_dir="checkpoints_async",
                           max_concurrent=20,
                           model_mix={"deepseek:deepseek-reasoner": 0.01, "deepseek:deepseek-chat": 0.99},
                           big_changes_rate=0.2,
                           best_model="deepseek:deepseek-reasoner",
                           max_children_per_organism=20)
    
    # Run evolution
    print("Starting evolution...")
    try:
        population = await evolver.evolve()
        
        # Generate report
        report_dir = evolver.report()
        
        best_organism = population.get_best()
        
        print("\nEvolution completed!")
        print(f"Best fitness: {best_organism.evaluation.fitness:.6f}")
        print(f"Target ratio: {best_organism.evaluation.fitness/2.635:.6f}")
        print(f"Report generated in: {report_dir}")
        
        # Print additional data
        if best_organism.evaluation.additional_data:
            print("\nAdditional metrics:")
            for key, value in best_organism.evaluation.additional_data.items():
                print(f"  {key}: {value}")
        
        # Visualize the best solution
        print("\nVisualizing best solution...")
        visualize_solution(best_organism.solution, "Best Circle Packing Solution")
        
        # Save the best solution
        output_file = "best_circle_packing_solution_async.py"
        with open(output_file, 'w') as f:
            f.write("# Best circle packing solution found by async evolution\n")
            f.write(f"# Fitness: {best_organism.evaluation.fitness:.6f}\n")
            f.write(f"# Target ratio: {best_organism.evaluation.fitness/2.635:.6f}\n\n")
            f.write(best_organism.solution)
        
        print(f"\nBest solution saved to: {output_file}")
        
        return population
        
    except KeyboardInterrupt:
        print("\nEvolution interrupted by user")
        if evolver.population.get_population():
            best = evolver.population.get_best()
            print(f"Best fitness so far: {best.evaluation.fitness:.6f}")
            visualize_solution(best.solution, "Best Solution So Far")
    
    except Exception as e:
        print(f"\nError during evolution: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point that runs the async evolution."""
    asyncio.run(run_evolution())


if __name__ == "__main__":
    main()