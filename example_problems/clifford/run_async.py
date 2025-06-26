#!/usr/bin/env python3
"""
Execute Clifford heuristic evolution using the rEVOLVE framework.

This script runs the evolutionary optimization of heuristic functions
for Clifford quantum circuit synthesis using AsyncEvolver.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add rEVOLVE source to path
evolve_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if evolve_src_path not in sys.path:
    sys.path.insert(0, evolve_src_path)

from evolve import AsyncEvolver
from spec import get_problem_spec


async def main():
    """
    Main execution function for Clifford heuristic evolution.
    """
    print("=" * 80)
    print("CLIFFORD HEURISTIC EVOLUTION")
    print("=" * 80)
    print()
    
    # Get problem specification
    print("Loading problem specification...")
    problem_spec = get_problem_spec()
    
    print(f"Problem: {problem_spec.name}")
    print(f"Initial population size: {len(problem_spec.starting_population)}")
    print(f"Target fitness: {problem_spec.hyperparameters.target_fitness}")
    print(f"Max steps: {problem_spec.hyperparameters.max_steps}")
    print()
    
    # Create evolution directory for outputs
    evolution_dir = Path(__file__).parent / "evolution_results"
    evolution_dir.mkdir(exist_ok=True)
    
    # Initialize AsyncEvolver
    evolver = AsyncEvolver(
        problem_specification=problem_spec,
        output_dir=str(evolution_dir),
        reason=True,  # Enable reasoning mode for better LLM explanations
        max_concurrent=3,  # Adjust based on API rate limits
        model_mix={
            "openai:gpt-4": 0.7,      # Primary model for reliability
            "openai:gpt-4-turbo": 0.3  # Secondary for diversity
        }
    )
    
    print(f"Evolution outputs will be saved to: {evolution_dir}")
    print()
    
    # Run evolution
    start_time = time.time()
    
    try:
        print("Starting evolution...")
        print("This may take 30-60 minutes depending on LLM response times.")
        print("Press Ctrl+C to stop evolution gracefully.")
        print()
        
        final_population = await evolver.evolve()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print()
        print("=" * 80)
        print("EVOLUTION COMPLETED")
        print("=" * 80)
        print(f"Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print(f"Final population size: {len(final_population.organisms)}")
        
        # Find and display best organism
        best_organism = final_population.get_best_organism()
        if best_organism:
            print(f"Best fitness achieved: {best_organism.fitness}")
            print(f"Best organism ID: {best_organism.id}")
            print()
            print("Best heuristic function:")
            print("-" * 40)
            print(best_organism.solution)
            print("-" * 40)
        
        # Save final results
        final_results_path = evolution_dir / "final_results.txt"
        with open(final_results_path, 'w') as f:
            f.write(f"Clifford Heuristic Evolution Results\n")
            f.write(f"=====================================\n\n")
            f.write(f"Evolution time: {elapsed_time:.1f} seconds\n")
            f.write(f"Final population size: {len(final_population.organisms)}\n")
            f.write(f"Best fitness: {best_organism.fitness if best_organism else 'None'}\n\n")
            
            if best_organism:
                f.write(f"Best Heuristic Function:\n")
                f.write(f"------------------------\n")
                f.write(best_organism.solution)
                f.write(f"\n\nMetadata: {best_organism.metadata}\n")
        
        print(f"Results summary saved to: {final_results_path}")
        
    except KeyboardInterrupt:
        print("\n\nEvolution stopped by user.")
        print("Partial results may be available in the checkpoint files.")
        
    except Exception as e:
        print(f"\nError during evolution: {e}")
        print("Check the error logs for details.")
        raise


def run_evolution():
    """
    Wrapper function to run the async evolution.
    """
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nEvolution interrupted.")
    except Exception as e:
        print(f"Evolution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check if we're in the correct directory
    if not os.path.exists("spec.py"):
        print("Error: Please run this script from the clifford problem directory.")
        print("Expected files: spec.py, evaluation.py, utils.py")
        sys.exit(1)
    
    # Check if CliffordOpt is available
    clifford_opt_path = Path(__file__).parent / "CliffordOpt"
    if not clifford_opt_path.exists():
        print("Warning: CliffordOpt directory not found.")
        print("Please ensure CliffordOpt is installed in the CliffordOpt/ subdirectory.")
        print("Evolution may fail without it.")
    
    run_evolution()