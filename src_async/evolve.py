from src_async.population import Organism, Population
from src_async.specification import ProblemSpecification
from src_async.mutate import generate_async
from src_async.prompt import Promptgenerator
import logfire
import os
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import asyncio
from typing import List, Tuple


class AsyncEvolver:
    def __init__(self, specification: ProblemSpecification, checkpoint_dir: str = None, max_concurrent: int = 5, model_mix: dict = None):
        self.specification = specification
        self.checkpoint_dir = checkpoint_dir or "checkpoints"
        self.checkpoint_file = os.path.join(self.checkpoint_dir, f"{specification.name.replace(' ', '_')}_checkpoint.pkl")
        self.max_concurrent = max_concurrent
        
        # Model mixing: allows using different models with different probabilities
        # Example: {"gpt-4.1-mini": 0.8, "gpt-4o": 0.2} for 80% fast, 20% reasoning
        self.model_mix = model_mix or {"gpt-4.1-mini": 1.0}
        
        # Try to load from checkpoint first
        if os.path.exists(self.checkpoint_file):
            self.population, self.current_step = self._load_checkpoint()
            logfire.info("Loaded from checkpoint", 
                        specification_name=specification.name,
                        current_step=self.current_step,
                        population_size=len(self.population.get_population()))
        else:
            self.population = Population(
                pop = specification.starting_population,
                exploration_rate = specification.hyperparameters.exploration_rate,
                elitism_rate = specification.hyperparameters.elitism_rate
            )
            self.current_step = 0
            
        self.target = specification.hyperparameters.target_fitness if specification.hyperparameters.target_fitness else float('inf')
        self.max_steps = specification.hyperparameters.max_steps
        self.reason = specification.hyperparameters.reason
        self.prompt_gen = Promptgenerator(specification.systemprompt, self.reason)
        
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        logfire.info("AsyncEvolver initialized", 
                    specification_name=specification.name,
                    target_fitness=self.target,
                    max_steps=self.max_steps,
                    starting_step=self.current_step,
                    max_concurrent=self.max_concurrent,
                    model_mix=self.model_mix)

    async def evolve(self) -> Population:
        """Main async evolution loop with streaming concurrent LLM calls"""
        step = self.current_step
        
        try:
            # Use a queue-based approach for continuous processing
            active_tasks = set()
            
            while self.population.get_best().evaluation.fitness < self.target and step < self.max_steps:
                # Fill up to max_concurrent slots with new tasks
                while len(active_tasks) < self.max_concurrent and step < self.max_steps:
                    mutatee = self.population.get_next()
                    prompt = self.prompt_gen.generate_prompt(mutatee)
                    
                    # Create async task for LLM call with metadata
                    task = asyncio.create_task(
                        self._generate_and_evaluate_async(prompt, mutatee.id),
                        name=f"step_{step + 1}_parent_{mutatee.id}"
                    )
                    task.step_number = step + 1  # Store step number for later
                    task.parent_id = mutatee.id   # Store parent ID for later
                    active_tasks.add(task)
                    step += 1
                
                # Wait for at least one task to complete
                if active_tasks:
                    done, pending = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                    
                    # Process completed tasks immediately
                    for task in done:
                        task_step = task.step_number
                        parent_id = task.parent_id
                        
                        try:
                            result = await task
                            mutated_solution, evaluation = result
                            
                            self.population.add(Organism(
                                solution=mutated_solution, 
                                evaluation=evaluation, 
                                parent_id=parent_id
                            ))
                            
                            current_best = self.population.get_best().evaluation.fitness
                            logfire.info(f"Step completed {task_step}\n"
                                         f"with fitness {evaluation.fitness}\n"
                                         f"and current best fitness {current_best}\n")
                            
                            # Check if we've reached the target
                            if evaluation.fitness >= self.target:
                                # Cancel remaining tasks if target reached
                                for pending_task in pending:
                                    pending_task.cancel()
                                break
                                
                        except Exception as e:
                            logfire.error(f"Error in evolution step {task_step}: {str(e)}")
                    
                    # Update active tasks to only include pending ones
                    active_tasks = pending
                    
                    # Save checkpoint every 10 completed steps
                    if len(self.population.get_population()) % 10 == 0:
                        self._save_checkpoint(len(self.population.get_population()))
            
            # Wait for any remaining active tasks to complete
            if active_tasks:
                logfire.info(f"Waiting for {len(active_tasks)} remaining tasks to complete...")
                done, _ = await asyncio.wait(active_tasks)
                
                for task in done:
                    task_step = task.step_number
                    parent_id = task.parent_id
                    
                    try:
                        result = await task
                        mutated_solution, evaluation = result
                        
                        self.population.add(Organism(
                            solution=mutated_solution, 
                            evaluation=evaluation, 
                            parent_id=parent_id
                        ))
                        
                        current_best = self.population.get_best().evaluation.fitness
                        logfire.info(f"Final step completed {task_step}\n"
                                     f"with fitness {evaluation.fitness}\n"
                                     f"and current best fitness {current_best}\n")
                        
                    except Exception as e:
                        logfire.error(f"Error in final evolution step {task_step}: {str(e)}")
                    
        except Exception as e:
            logfire.error(f"Critical error in evolution loop: {str(e)}")
            logfire.info(f"Population state saved to checkpoint: {self.checkpoint_file}")
            raise
        finally:
            # Always save final checkpoint
            self._save_checkpoint(step)

        logfire.info(f"Evolution completed\n"
                     f"with {step} steps\n"
                     f"and best fitness {self.population.get_best().evaluation.fitness}\n"
                     f"and average fitness {self.population.calculate_average_fitness()}\n"
                     )
        return self.population

    def _select_model(self) -> str:
        """Select a model based on the configured mix"""
        import random
        r = random.random()
        cumulative = 0.0
        
        for model, probability in self.model_mix.items():
            cumulative += probability
            if r <= cumulative:
                return model
        
        # Fallback to first model
        return list(self.model_mix.keys())[0]

    async def _generate_and_evaluate_async(self, prompt: str, parent_id: int) -> Tuple[str, object]:
        """Generate a mutation and evaluate it asynchronously"""
        selected_model = self._select_model()
        is_reasoning = self.reason or "o4" in selected_model
        
        logfire.debug(f"Selected model {selected_model} for parent {parent_id} (reasoning: {is_reasoning})")
        
        mutated = await generate_async(prompt, model=selected_model, reasoning=is_reasoning)
        evaluation = self.specification.evaluator(mutated)
        return mutated, evaluation

    def report(self):
        """Generate report - identical to sync version"""
        # Create directory with datetime and problem name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        problem_name = self.specification.name.replace(" ", "_").replace("(", "").replace(")", "")
        dir_name = f"{timestamp}_{problem_name}"
        
        # Create the directory inside outputs
        outputs_dir = "outputs"
        os.makedirs(outputs_dir, exist_ok=True)
        full_dir_path = os.path.join(outputs_dir, dir_name)
        os.makedirs(full_dir_path, exist_ok=True)
        
        # 1. Create visualization
        viz_path = os.path.join(full_dir_path, "population_visualization")
        self.population.visualize_population(filename=viz_path, view=False)
        
        # 2. Create fitness progression plot
        organisms = self.population.get_population()
        organisms_sorted = sorted(organisms, key=lambda x: x.id)
        
        # Track best fitness at each step
        best_fitness_progression = []
        current_best = float('-inf')
        
        for org in organisms_sorted:
            if org.evaluation.fitness > current_best:
                current_best = org.evaluation.fitness
            best_fitness_progression.append(current_best)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(best_fitness_progression) + 1), best_fitness_progression, 'b-', linewidth=2)
        plt.xlabel('Generation (Organism ID)')
        plt.ylabel('Best Fitness Score')
        plt.title('Fitness Progression Over Time')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fitness_plot_path = os.path.join(full_dir_path, "fitness_progression.png")
        plt.savefig(fitness_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Serialize population (try JSON first, fallback to pickle)
        try:
            # Try to serialize as JSON
            population_data = []
            for org in self.population.get_population():
                org_data = {
                    "id": org.id,
                    "parent_id": org.parent_id,
                    "solution": org.solution,
                    "evaluation": {
                        "fitness": org.evaluation.fitness,
                        "additional_data": org.evaluation.additional_data
                    },
                    "children": org.children
                }
                population_data.append(org_data)
            
            with open(os.path.join(full_dir_path, "population.json"), "w") as f:
                json.dump(population_data, f, indent=2)
                
        except Exception as e:
            # Fallback to pickle
            logfire.warning(f"JSON serialization failed: {e}, using pickle instead")
            with open(os.path.join(full_dir_path, "population.pkl"), "wb") as f:
                pickle.dump(self.population.get_population(), f)
        
        # 4. Create markdown report
        best_organism = self.population.get_best()
        num_organisms = len(self.population.get_population())
        avg_fitness = self.population.calculate_average_fitness()
        
        # Get hyperparameters
        hyperparams = self.specification.hyperparameters
        
        markdown_content = f"""# Evolution Report

## Problem Information
- **Problem Name**: {self.specification.name}
- **Timestamp**: {timestamp}

## Hyperparameters
- **Exploration Rate**: {hyperparams.exploration_rate}
- **Elitism Rate**: {hyperparams.elitism_rate}
- **Max Steps**: {hyperparams.max_steps}
- **Target Fitness**: {hyperparams.target_fitness if hyperparams.target_fitness is not None else 'None'}
- **Reason**: {hyperparams.reason}
- **Max Concurrent**: {self.max_concurrent}

## Population Statistics
- **Number of Organisms**: {num_organisms}
- **Best Fitness Score**: {best_organism.evaluation.fitness}
- **Average Fitness Score**: {avg_fitness:.4f}

## Fitness Progression
![Fitness Progression](fitness_progression.png)

## Population Visualization
![Population Visualization](population_visualization.gv.png)

## Best Solution
```
{best_organism.solution}
```

## Additional Data from Best Solution
```json
{json.dumps(best_organism.evaluation.additional_data, indent=2)}
```

## Files in this Report
- `population_visualization.gv` / `population_visualization.gv.png` - Visual representation of the population
- `fitness_progression.png` - Plot showing fitness improvement over generations
- `population.json` or `population.pkl` - Serialized population data
- `report.md` - This report file
"""
        
        with open(os.path.join(full_dir_path, "report.md"), "w") as f:
            f.write(markdown_content)
            
        logfire.info(f"Report generated in directory: {full_dir_path}")
        
        # Clean up checkpoint file after successful completion
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            logfire.info("Checkpoint file cleaned up after successful completion")
            
        return full_dir_path
    
    def _save_checkpoint(self, step: int):
        """Save current population state and step to checkpoint file"""
        checkpoint_data = {
            'population': self.population,
            'step': step,
            'specification_name': self.specification.name,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            logfire.debug(f"Checkpoint saved at step {step}")
        except Exception as e:
            logfire.error(f"Failed to save checkpoint: {str(e)}")
    
    def _load_checkpoint(self):
        """Load population state and step from checkpoint file"""
        try:
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            population = checkpoint_data['population']
            step = checkpoint_data['step']
            
            logfire.info(f"Checkpoint loaded from {checkpoint_data['timestamp']}, "
                        f"resuming from step {step}")
            
            return population, step
        except Exception as e:
            logfire.error(f"Failed to load checkpoint: {str(e)}")
            raise


# For backward compatibility, provide a sync wrapper
class Evolver(AsyncEvolver):
    def __init__(self, specification: ProblemSpecification, checkpoint_dir: str = None, max_concurrent: int = 5, model_mix: dict = None):
        super().__init__(specification, checkpoint_dir, max_concurrent, model_mix)
    
    def evolve(self) -> Population:
        """Sync wrapper that runs the async evolution"""
        return asyncio.run(super().evolve())