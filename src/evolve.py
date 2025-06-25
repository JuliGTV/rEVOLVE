from src.population import Organism, Population
from src.specification import ProblemSpecification
from src.mutate import generate_async
from src.prompt import Promptgenerator
import logfire
import os
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import asyncio
from typing import List, Tuple


class AsyncEvolver:
    def __init__(self, specification: ProblemSpecification, checkpoint_dir: str = None, max_concurrent: int = 20, 
                 model_mix: dict = None, big_changes_rate: float = 0.2, best_model: str = "gpt-4o", 
                 max_children_per_organism: int = 20, population_path: str = None):
        self.specification = specification
        self.checkpoint_dir = checkpoint_dir or "checkpoints"
        self.checkpoint_file = os.path.join(self.checkpoint_dir, f"{specification.name.replace(' ', '_')}_async_checkpoint.pkl")
        self.max_concurrent = max_concurrent
        
        # Model mixing: allows using different models with different probabilities
        # Example: {"gpt-4.1-mini": 0.8, "gpt-4o": 0.2} for 80% fast, 20% reasoning
        self.model_mix = model_mix or {"deepseek:deepseek-reasoner": 0.01, "deepseek:deepseek-chat": 0.99}
        
        # New features from evolve3
        self.big_changes_rate = big_changes_rate  # Probability of asking for large vs small changes
        self.best_model = best_model  # Model to use for exploiting best organisms
        self.max_children_per_organism = max_children_per_organism  # Cap on children per organism
        
        # Try to load from checkpoint first, then from population_path, then create new
        if os.path.exists(self.checkpoint_file):
            self.population, self.current_step = self._load_checkpoint()
            logfire.info("Loaded from checkpoint", 
                        specification_name=specification.name,
                        current_step=self.current_step,
                        population_size=len(self.population.get_population()))
        elif population_path and os.path.exists(population_path):
            self.population = Population.from_pickle(
                population_path,
                exploration_rate=specification.hyperparameters.exploration_rate,
                elitism_rate=specification.hyperparameters.elitism_rate
            )
            self.current_step = 0
            logfire.info("Loaded from population file", 
                        specification_name=specification.name,
                        population_path=population_path,
                        population_size=len(self.population.get_population()),
                        best_fitness=self.population.get_best().evaluation.fitness)
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
                    model_mix=self.model_mix,
                    big_changes_rate=self.big_changes_rate,
                    best_model=self.best_model,
                    max_children_per_organism=self.max_children_per_organism,
                    population_path=population_path)

    async def evolve(self) -> Population:
        """Main async evolution loop with streaming concurrent LLM calls"""
        step = self.current_step
        
        try:
            # Use a queue-based approach for continuous processing
            active_tasks = set()
            target_reached = False
            
            # Main evolution loop - continue until all tasks are processed
            while active_tasks or (not target_reached and step < self.max_steps and self.population.get_best().evaluation.fitness < self.target):
                # Fill up to max_concurrent slots with new tasks (only if we haven't reached limits)
                should_create_new_tasks = (
                    not target_reached and 
                    step < self.max_steps and 
                    self.population.get_best().evaluation.fitness < self.target
                )
                
                if should_create_new_tasks:
                    while len(active_tasks) < self.max_concurrent and step < self.max_steps:
                        # Select organism with child cap enforcement
                        mutatee = self._select_organism_with_child_cap()
                        
                        # Determine change type based on probability
                        change_type = self._determine_change_type()
                        
                        # Generate prompt with specific change type
                        prompt = self.prompt_gen.generate_prompt(mutatee, change_type=change_type)
                        
                        # Create async task for LLM call with metadata
                        task = asyncio.create_task(
                            self._generate_and_evaluate_async_with_metadata(prompt, mutatee.id, change_type, step + 1),
                            name=f"step_{step + 1}_parent_{mutatee.id}"
                        )
                        task.step_number = step + 1  # Store step number for later
                        task.parent_id = mutatee.id   # Store parent ID for later
                        task.change_type = change_type  # Store change type for later
                        active_tasks.add(task)
                        step += 1
                
                # Wait for at least one task to complete (if we have any active tasks)
                if active_tasks:
                    done, pending = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                else:
                    # No more tasks to process, exit the loop
                    break
                
                # Process completed tasks immediately
                for task in done:
                    task_step = task.step_number
                    parent_id = task.parent_id
                    change_type = task.change_type
                    
                    try:
                        result = await task
                        
                        # Handle different task types
                        if change_type == "EXPLOITATION":
                            # Exploitation task returns an Organism directly
                            exploitation_organism = result
                            self.population.add(exploitation_organism)
                            
                            logfire.info(f"Exploitation step completed {task_step}\n"
                                         f"with fitness {exploitation_organism.evaluation.fitness}\n"
                                         f"from organism {exploitation_organism.parent_id}\n"
                                         f"current best fitness: {self.population.get_best().evaluation.fitness}")
                            
                            # Check if we've reached the target
                            if exploitation_organism.evaluation.fitness >= self.target:
                                target_reached = True
                                logfire.info(f"Target fitness {self.target} reached! Cancelling remaining tasks...")
                                # Cancel remaining tasks if target reached
                                for pending_task in pending:
                                    pending_task.cancel()
                        else:
                            # Regular task returns (solution, evaluation, creation_info)
                            mutated_solution, evaluation, creation_info = result
                            
                            # Check if this is a new best organism
                            current_best_fitness = self.population.get_best().evaluation.fitness
                            is_new_best = evaluation.fitness > current_best_fitness
                            
                            # Add the new organism with creation info
                            new_organism = Organism(
                                solution=mutated_solution, 
                                evaluation=evaluation, 
                                parent_id=parent_id,
                                creation_info=creation_info
                            )
                            self.population.add(new_organism)
                            
                            logfire.info(f"Step completed {task_step}\n"
                                         f"with fitness {evaluation.fitness}\n"
                                         f"change_type: {change_type}\n"
                                         f"model: {creation_info['model']}\n"
                                         f"current best fitness: {self.population.get_best().evaluation.fitness}\n"
                                         f"is_new_best: {is_new_best}")
                            
                            # If we have a new best organism, exploit it asynchronously (only if we're still running)
                            if is_new_best and not target_reached and step < self.max_steps:
                                exploitation_task = asyncio.create_task(
                                    self._exploit_best_organism_async(new_organism, task_step),
                                    name=f"exploit_best_{task_step}_org_{new_organism.id}"
                                )
                                exploitation_task.step_number = task_step
                                exploitation_task.parent_id = new_organism.id
                                exploitation_task.change_type = "EXPLOITATION"
                                
                                # Add exploitation task to active tasks if we have room
                                if len(pending) < self.max_concurrent:
                                    pending.add(exploitation_task)
                                    logfire.info(f"Started exploitation task for new best organism {new_organism.id}")
                            
                            # Check if we've reached the target
                            if evaluation.fitness >= self.target:
                                target_reached = True
                                logfire.info(f"Target fitness {self.target} reached! Cancelling remaining tasks...")
                                # Cancel remaining tasks if target reached
                                for pending_task in pending:
                                    pending_task.cancel()
                            
                    except Exception as e:
                        logfire.error(f"Error in evolution step {task_step}: {str(e)}")
                
                # Update active tasks to only include pending ones
                active_tasks = pending
                
                # Save checkpoint every 10 completed steps
                if len(self.population.get_population()) % 10 == 0:
                    self._save_checkpoint(len(self.population.get_population()))
            
            # Log completion
            logfire.info(f"Evolution loop completed. Final step count: {step}, "
                        f"Active tasks remaining: {len(active_tasks)}, "
                        f"Target reached: {target_reached}")
                    
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
    
    def _determine_change_type(self) -> str:
        """Determine whether to make big or small changes based on probability"""
        import random
        if random.random() < self.big_changes_rate:
            return "LARGE QUALITATIVE CHANGE"
        else:
            return "SMALL ITERATIVE IMPROVEMENT"
    
    def _select_organism_with_child_cap(self) -> Organism:
        """Select organism ensuring it doesn't exceed max children limit"""
        max_attempts = 50  # Prevent infinite loop
        attempts = 0
        
        mutatee = self.population.get_next()
        while mutatee.children >= self.max_children_per_organism and attempts < max_attempts:
            attempts += 1
            mutatee = self.population.get_next()
        
        if attempts > 0:
            logfire.debug(f"Skipped {attempts} organisms with {self.max_children_per_organism}+ children, "
                         f"selected organism {mutatee.id} with {mutatee.children} children")
        
        return mutatee

    async def _generate_and_evaluate_async(self, prompt: str, parent_id: int) -> Tuple[str, object]:
        """Generate a mutation and evaluate it asynchronously"""
        selected_model = self._select_model()
        is_reasoning = self.reason or "o4" in selected_model
        
        logfire.debug(f"Selected model {selected_model} for parent {parent_id} (reasoning: {is_reasoning})")
        
        mutated = await generate_async(prompt, model=selected_model, reasoning=is_reasoning)
        evaluation = self.specification.evaluator(mutated)
        return mutated, evaluation
    
    async def _generate_and_evaluate_async_with_metadata(self, prompt: str, parent_id: int, change_type: str, step_number: int) -> Tuple[str, object, dict]:
        """Generate a mutation and evaluate it asynchronously with creation metadata"""
        from src.evaluation import Evaluation
        
        selected_model = self._select_model()
        is_reasoning = self.reason or "o4" in selected_model
        
        logfire.debug(f"Step {step_number}: Selected model {selected_model} for parent {parent_id} "
                     f"(change_type: {change_type}, reasoning: {is_reasoning})")
        
        mutated = await generate_async(prompt, model=selected_model, reasoning=is_reasoning)
        sync_evaluation = self.specification.evaluator(mutated)
        
        # Convert sync evaluation to async evaluation
        evaluation = Evaluation(
            fitness=sync_evaluation.fitness,
            additional_data=sync_evaluation.additional_data
        )
        
        # Create creation info
        creation_info = {
            "model": selected_model,
            "change_type": change_type,
            "step": step_number,
            "is_reasoning": is_reasoning,
            "big_changes_rate": self.big_changes_rate
        }
        
        return mutated, evaluation, creation_info
    
    async def _exploit_best_organism_async(self, best_organism: Organism, step_number: int) -> Organism:
        """Generate a small improvement on the best organism using the best model"""
        from src.evaluation import Evaluation
        
        prompt = self.prompt_gen.generate_prompt(best_organism, change_type="SMALL ITERATIVE IMPROVEMENT")
        is_reasoning = self.reason or "o4" in self.best_model
        
        logfire.info(f"Step {step_number}: Exploiting best organism {best_organism.id} "
                    f"(fitness: {best_organism.evaluation.fitness}) with {self.best_model}")
        
        mutated = await generate_async(prompt, model=self.best_model, reasoning=is_reasoning)
        sync_evaluation = self.specification.evaluator(mutated)
        
        # Convert sync evaluation to async evaluation
        evaluation = Evaluation(
            fitness=sync_evaluation.fitness,
            additional_data=sync_evaluation.additional_data
        )
        
        # Create creation info for exploitation
        creation_info = {
            "model": self.best_model,
            "change_type": "SMALL ITERATIVE IMPROVEMENT",
            "step": step_number,
            "is_reasoning": is_reasoning,
            "exploitation": True,
            "exploited_organism_id": best_organism.id,
            "exploited_organism_fitness": best_organism.evaluation.fitness
        }
        
        exploitation_organism = Organism(
            solution=mutated,
            evaluation=evaluation,
            parent_id=best_organism.id,
            creation_info=creation_info
        )
        
        return exploitation_organism

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
        
        # 3. Serialize population (always create both JSON and pickle)
        # Always create pickle file
        with open(os.path.join(full_dir_path, "population.pkl"), "wb") as f:
            pickle.dump(self.population.get_population(), f)
        
        # Try to serialize as JSON with all fields
        try:
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
                    "children": org.children,
                    "creation_info": org.creation_info if hasattr(org, 'creation_info') else None
                }
                population_data.append(org_data)
            
            with open(os.path.join(full_dir_path, "population.json"), "w") as f:
                json.dump(population_data, f, indent=2)
                
        except Exception as e:
            logfire.warning(f"JSON serialization failed: {e}, but pickle was created successfully")
        
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
    def __init__(self, specification: ProblemSpecification, checkpoint_dir: str = None, max_concurrent: int = 5, 
                 model_mix: dict = None, big_changes_rate: float = 0.25, best_model: str = "gpt-4o", 
                 max_children_per_organism: int = 10, population_path: str = None):
        super().__init__(specification, checkpoint_dir, max_concurrent, model_mix, big_changes_rate, 
                        best_model, max_children_per_organism, population_path)
    
    def evolve(self) -> Population:
        """Sync wrapper that runs the async evolution"""
        return asyncio.run(super().evolve())