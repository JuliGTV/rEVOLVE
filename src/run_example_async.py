from src.specification import ProblemSpecification, Hyperparameters
from src.evolve import AsyncEvolver, Evolver
from src.population import Organism
from src.evaluation import Evaluation
from typing import Optional
import asyncio


async def run_example_async(spec: ProblemSpecification, max_steps: Optional[int] = None, target_fitness: Optional[int] = None, 
                          max_concurrent: int = 5, model_mix: dict = None, big_changes_rate: float = 0.25, 
                          best_model: str = "gpt-4o", max_children_per_organism: int = 10):
    """Run async example with concurrent LLM calls"""
    if max_steps:
        spec.hyperparameters.max_steps = max_steps
    if target_fitness:
        spec.hyperparameters.target_fitness = target_fitness
    
    try:
        e = AsyncEvolver(spec, max_concurrent=max_concurrent, model_mix=model_mix, 
                        big_changes_rate=big_changes_rate, best_model=best_model,
                        max_children_per_organism=max_children_per_organism)
        await e.evolve()
        e.report()
    except Exception as e:
        raise e


def run_example_sync(spec: ProblemSpecification, max_steps: Optional[int] = None, target_fitness: Optional[int] = None, 
                    max_concurrent: int = 5, model_mix: dict = None, big_changes_rate: float = 0.25,
                    best_model: str = "gpt-4o", max_children_per_organism: int = 10):
    """Run sync wrapper version"""
    if max_steps:
        spec.hyperparameters.max_steps = max_steps
    if target_fitness:
        spec.hyperparameters.target_fitness = target_fitness
    
    try:
        e = Evolver(spec, max_concurrent=max_concurrent, model_mix=model_mix,
                   big_changes_rate=big_changes_rate, best_model=best_model,
                   max_children_per_organism=max_children_per_organism)
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
        
        # Parse flags
        use_sync = "--sync" in sys.argv
        use_mixed_models = "--mixed" in sys.argv
        use_exploit = "--exploit" in sys.argv
        
        # Configuration
        # Example model mix: 70% fast mini, 30% reasoning o4
        model_mix = {"gpt-4.1-mini": 0.7, "gpt-4o": 0.3} if use_mixed_models else None
        
        # Evolution parameters from evolve3
        big_changes_rate = 0.4  # Higher rate for more exploration
        best_model = "gpt-4o"  # Use reasoning model for best organism exploitation
        max_children_per_organism = 10
        
        print(f"Running in {'sync' if use_sync else 'async'} mode with:")
        print(f"  max_concurrent: {max_concurrent}")
        print(f"  big_changes_rate: {big_changes_rate}")
        print(f"  best_model: {best_model}")
        print(f"  max_children_per_organism: {max_children_per_organism}")
        if model_mix:
            print(f"  model_mix: {model_mix}")
        if use_exploit:
            print(f"  best_organism_exploitation: enabled")
        
        if use_sync:
            run_example_sync(spec, max_steps, target_fitness, max_concurrent, model_mix, 
                           big_changes_rate, best_model, max_children_per_organism)
        else:
            asyncio.run(run_example_async(spec, max_steps, target_fitness, max_concurrent, model_mix,
                                        big_changes_rate, best_model, max_children_per_organism))
    else:
        print("Usage: python run_example_async.py <problem_name> [max_steps] [target_fitness] [max_concurrent] [flags]")
        print("Available problems: circle_packing, guess_the_votes")
        print("Flags:")
        print("  --sync    Use sync wrapper instead of native async")
        print("  --mixed   Use mixed models (70% fast, 30% reasoning)")
        print("  --exploit Enable best organism exploitation (always enabled now)")