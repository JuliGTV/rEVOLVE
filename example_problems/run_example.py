from src.specification import ProblemSpecification
from src.evolve import Evolver
from typing import Optional



def run_example(spec: ProblemSpecification, max_steps: Optional[int] = None, target_fitness: Optional[int] = None):
    if max_steps:
        spec.hyperparameters.max_steps = max_steps
    if target_fitness:
        spec.hyperparameters.target_fitness = target_fitness
    
    try:
        e = Evolver(spec)
        e.evolve()
        e.report()
    except Exception as e:
  
        raise e

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        problem_name = sys.argv[1]
        
        if problem_name == "circle_packing":
            from example_problems.circle_packing.spec import get_circle_packing_spec
            spec = get_circle_packing_spec()
        elif problem_name == "guess_the_votes":
            from example_problems.guess_the_votes.spec import spec
        else:
            print(f"Unknown problem: {problem_name}")
            print("Available problems: circle_packing, guess_the_votes")
            sys.exit(1)
        
        # Optional command line arguments
        max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else None
        target_fitness = float(sys.argv[3]) if len(sys.argv) > 3 else None
        
        run_example(spec, max_steps, target_fitness)
    else:
        print("Usage: python run_example.py <problem_name> [max_steps] [target_fitness]")
        print("Available problems: circle_packing, guess_the_votes")