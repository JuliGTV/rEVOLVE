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
    except Exception as e:
  
        raise e

if __name__ == "__main__":
    from example_problems.guess_the_votes.spec import spec
    run_example(spec)