from src.specification import ProblemSpecification
from src.evolve import evolve
from typing import Optional



def run_example(spec: ProblemSpecification, max_steps: Optional[int] = None, target_fitness: Optional[int] = None):
    if max_steps:
        spec.hyperparameters.max_steps = max_steps
    if target_fitness:
        spec.hyperparameters.target_fitness = target_fitness
    
    try:
        e = evolve(spec)
        print(e)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    from example_problems.guess_the_votes.spec import spec
    run_example(spec)