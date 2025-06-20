from src_async.specification import ProblemSpecification, Hyperparameters
from src_async.evolve import AsyncEvolver, Evolver
from src_async.population import Organism
from src_async.evaluation import Evaluation
from typing import Optional
import asyncio


async def run_example_async(spec: ProblemSpecification, max_steps: Optional[int] = None, target_fitness: Optional[int] = None, max_concurrent: int = 5, model_mix: dict = None):
    """Run async example with concurrent LLM calls"""
    if max_steps:
        spec.hyperparameters.max_steps = max_steps
    if target_fitness:
        spec.hyperparameters.target_fitness = target_fitness
    
    try:
        e = AsyncEvolver(spec, max_concurrent=max_concurrent, model_mix=model_mix)
        await e.evolve()
        e.report()
    except Exception as e:
        raise e


def run_example_sync(spec: ProblemSpecification, max_steps: Optional[int] = None, target_fitness: Optional[int] = None, max_concurrent: int = 5, model_mix: dict = None):
    """Run sync wrapper version"""
    if max_steps:
        spec.hyperparameters.max_steps = max_steps
    if target_fitness:
        spec.hyperparameters.target_fitness = target_fitness
    
    try:
        e = Evolver(spec, max_concurrent=max_concurrent, model_mix=model_mix)
        e.evolve()
        e.report()
    except Exception as e:
        raise e


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        problem_name = sys.argv[1]
        
        if problem_name == "circle_packing":
            # Import the sync version and convert to async
            from example_problems.circle_packing.spec import get_circle_packing_spec
            from example_problems.circle_packing.evaluation import evaluate_circle_packing
            sync_spec = get_circle_packing_spec()
            
            # Convert to async version
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
        elif problem_name == "guess_the_votes":
            # Import the sync version and convert to async
            from example_problems.guess_the_votes.spec import spec as sync_spec
            from example_problems.guess_the_votes.evaluation import evaluate_guess_the_votes
            spec = ProblemSpecification(
                name=sync_spec.name,
                systemprompt=sync_spec.systemprompt,
                evaluator=evaluate_guess_the_votes,
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
        else:
            print(f"Unknown problem: {problem_name}")
            print("Available problems: circle_packing, guess_the_votes")
            sys.exit(1)
        
        # Optional command line arguments
        max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else None
        target_fitness = float(sys.argv[3]) if len(sys.argv) > 3 else None
        max_concurrent = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        use_sync = "--sync" in sys.argv
        use_mixed_models = "--mixed" in sys.argv
        
        # Example model mix: 70% fast mini, 30% reasoning o4
        model_mix = {"gpt-4.1-mini": 0.7, "gpt-4o": 0.3} if use_mixed_models else None
        
        if use_sync:
            print(f"Running in sync mode with max_concurrent={max_concurrent}")
            if model_mix:
                print(f"Using mixed models: {model_mix}")
            run_example_sync(spec, max_steps, target_fitness, max_concurrent, model_mix)
        else:
            print(f"Running in async mode with max_concurrent={max_concurrent}")
            if model_mix:
                print(f"Using mixed models: {model_mix}")
            asyncio.run(run_example_async(spec, max_steps, target_fitness, max_concurrent, model_mix))
    else:
        print("Usage: python run_example_async.py <problem_name> [max_steps] [target_fitness] [max_concurrent] [--sync] [--mixed]")
        print("Available problems: circle_packing, guess_the_votes")
        print("Use --sync flag to use sync wrapper instead of native async")
        print("Use --mixed flag to use mixed models (70% fast, 30% reasoning)")