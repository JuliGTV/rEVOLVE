from src.population import Organism, Population
from src.specification import ProblemSpecification
from src.mutate import generate
from src.prompt import Promptgenerator
import logfire
import os
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import random


class Evolver2:
    def __init__(self, specification: ProblemSpecification, checkpoint_dir: str = None):
        self.specification = specification
        self.checkpoint_dir = checkpoint_dir or "checkpoints"
        self.checkpoint_file = os.path.join(self.checkpoint_dir, f"{specification.name.replace(' ', '_')}_evolver2_checkpoint.pkl")
        
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
        
        logfire.info("Evolver2 initialized", 
                    specification_name=specification.name,
                    target_fitness=self.target,
                    max_steps=self.max_steps,
                    starting_step=self.current_step)

    def _choose_model(self, iteration_type: str = "normal") -> str:
        """Choose model based on iteration type and probabilities"""
        if iteration_type in ["large_change", "small_change"]:
            return "openai:o4-mini-2025-04-16"
        
        # For normal iterations: o4-mini 1/100, gpt-4.1 ~14%, gpt-4.1-mini rest
        rand = random.random()
        if rand < 0.01:  # 1/100
            return "openai:o4-mini-2025-04-16"
        elif rand < 0.2:  
            return "openai:gpt-4.1"
        else:
            return "openai:gpt-4.1-mini"

    def _determine_big_changes_param(self, organism: Organism) -> float:
        """Determine big_changes parameter based on organism fitness vs average"""
        avg_fitness = self.population.calculate_average_fitness()
        if organism.evaluation.fitness < avg_fitness:
            return 0.6  # Below average - more big changes
        else:
            return 0.2  # Above average - fewer big changes

    def _apply_iterative_small_changes(self, best_organism: Organism) -> Organism:
        """Apply small changes iteratively until no improvement"""
        current_organism = best_organism
        improvement_count = 0
        
        while True:
            # Generate prompt with big_changes=0 for small changes
            prompt = self.prompt_gen.generate_prompt(current_organism, big_changes=0)
            mutated = generate(prompt, model="openai:o4-mini-2025-04-16", reasoning=True)
            evaluation = self.specification.evaluator(mutated)
            
            # Check if it's an improvement
            if evaluation.fitness > current_organism.evaluation.fitness:
                # Create new organism with improvement
                new_organism = Organism(
                    solution=mutated, 
                    evaluation=evaluation, 
                    parent_id=current_organism.id
                )
                self.population.add(new_organism)
                current_organism = new_organism
                improvement_count += 1
                
                logfire.info(f"Small change improvement #{improvement_count}, "
                           f"fitness: {evaluation.fitness}")
            else:
                # No improvement, stop iterative process
                logfire.info(f"Small change iteration stopped after {improvement_count} improvements")
                break
                
        return current_organism

    def evolve(self) -> Population:
        step = self.current_step
        
        try:
            while self.population.get_best().evaluation.fitness < self.target and step < self.max_steps:
                step += 1
                
                try:

                    
                    if step in [2,5] or step % 100 == 0:  # 1/100 - Large change to best solution
                        iteration_type = "large_change"
                        mutatee = self.population.get_best()
                        model = self._choose_model(iteration_type)
                        prompt = self.prompt_gen.generate_prompt(mutatee, big_changes=1)  # Force large changes
                        
                        logfire.info(f"Step {step}: Large change to best solution (fitness: {mutatee.evaluation.fitness})")
                        
                    elif step % 100 == 50:  # Next 1/100 - Small iterative changes to best solution
                        iteration_type = "small_change"
                        
                        logfire.info(f"Step {step}: Small iterative changes to best solution")
                        
                        # Apply iterative small changes
                        final_organism = self._apply_iterative_small_changes(self.population.get_best())
                        
                        # Continue with normal evolution step
                        mutatee = self.population.get_next()
                        model = self._choose_model("normal")
                        big_changes_param = self._determine_big_changes_param(mutatee)
                        prompt = self.prompt_gen.generate_prompt(mutatee, big_changes=big_changes_param)
                        
                    else:  # Rest of the time - Normal evolution
                        iteration_type = "normal"
                        mutatee = self.population.get_next()
                        model = self._choose_model(iteration_type)
                        big_changes_param = self._determine_big_changes_param(mutatee)
                        prompt = self.prompt_gen.generate_prompt(mutatee, big_changes=big_changes_param)
                        
                        logfire.debug(f"Step {step}: Normal evolution, model: {model}, "
                                    f"big_changes: {big_changes_param}")
                    
                    # Generate mutation (skip if we already did iterative changes)
                    if iteration_type != "small_change":
                        # Set reasoning=True for o4-mini models (they are reasoning models), False for others
                        is_reasoning_model = True if "o4-mini" in model else False
                        mutated = generate(prompt, model=model, reasoning=is_reasoning_model)
                        evaluation = self.specification.evaluator(mutated)
                        self.population.add(Organism(solution=mutated, evaluation=evaluation, parent_id=mutatee.id))
                        
                        current_best_fitness = self.population.get_best().evaluation.fitness
                        logfire.info(f"Step {step} completed with fitness {evaluation.fitness}, "
                                   f"current best: {current_best_fitness}, "
                                   f"model: {model}, type: {iteration_type}")
                    
                    # Save checkpoint every 10 steps
                    if step % 10 == 0:
                        self._save_checkpoint(step)
                        
                except Exception as e:
                    logfire.error(f"Error in evolution step {step}: {str(e)}")
                    self._save_checkpoint(step)
                    raise
                    
        except Exception as e:
            logfire.error(f"Critical error in evolution loop: {str(e)}")
            logfire.info(f"Population state saved to checkpoint: {self.checkpoint_file}")
            raise
        finally:
            # Always save final checkpoint
            self._save_checkpoint(step)

        logfire.info(f"Evolution completed with {step} steps, "
                     f"best fitness: {self.population.get_best().evaluation.fitness}, "
                     f"average fitness: {self.population.calculate_average_fitness()}")
        return self.population

    def report(self):
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
                    }
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
        
        markdown_content = f"""# Evolution Report (Evolver2 - Stochastic)

## Problem Information
- **Problem Name**: {self.specification.name}
- **Timestamp**: {timestamp}
- **Evolver**: Evolver2 (with stochastic behaviors)

## Hyperparameters
- **Exploration Rate**: {hyperparams.exploration_rate}
- **Elitism Rate**: {hyperparams.elitism_rate}
- **Max Steps**: {hyperparams.max_steps}
- **Target Fitness**: {hyperparams.target_fitness if hyperparams.target_fitness is not None else 'None'}
- **Reason**: {hyperparams.reason}

## Stochastic Behaviors
- **1/100 iterations**: Large change to best solution using openai:o4-mini-2025-04-16
- **1/100 iterations**: Small iterative changes to best solution using openai:o4-mini-2025-04-16 (until no improvement)
- **Rest**: Normal evolution with model selection (openai:o4-mini-2025-04-16: 1/100, openai:gpt-4.1: ~14%, openai:gpt-4.1-mini: rest)
- **Big changes parameter**: 0.6 for below-average fitness, 0.2 for above-average fitness

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