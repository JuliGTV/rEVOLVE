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


class Evolver:
    def __init__(self, specification: ProblemSpecification):
        self.specification = specification
        self.population = Population(
            pop = specification.starting_population,
            exploration_rate = specification.hyperparameters.exploration_rate,
            elitism_rate = specification.hyperparameters.elitism_rate
        )
        self.target = specification.hyperparameters.target_fitness if specification.hyperparameters.target_fitness else float('inf')
        self.max_steps = specification.hyperparameters.max_steps
        self.reason = specification.hyperparameters.reason
        self.prompt_gen = Promptgenerator(specification.systemprompt, self.reason)
        logfire.info("Evolver initialized", 
                    specification_name=specification.name,
                    target_fitness=self.target,
                    max_steps=self.max_steps)

    def evolve(self) -> Population:
        step = 0
        best_fitness = float('-inf')
        while self.population.get_best().evaluation.fitness < self.target and step < self.max_steps:
            step += 1
            mutatee = self.population.get_next()
            prompt = self.prompt_gen.generate_prompt(mutatee)
            mutated = generate(prompt, self.reason)
            evaluation = self.specification.evaluator(mutated)
            self.population.add(Organism(solution=mutated, evaluation=evaluation, parent_id=mutatee.id))
            
            current_best = self.population.get_best().evaluation.fitness
            logfire.info(f"Step completed {step}\n"
                         f"with fitness {evaluation.fitness}\n"
                         f"and current best fitness {current_best}\n"
                         ) 


        logfire.info(f"Evolution completed\n"
                     f"with {step} steps\n"
                     f"and best fitness {self.population.get_best().evaluation.fitness}\n"
                     f"and average fitness {self.population.calculate_average_fitness()}\n"
                     )
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
        return full_dir_path
