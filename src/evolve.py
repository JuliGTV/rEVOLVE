from src.population import Organism, Population
from src.specification import ProblemSpecification
from src.mutate import generate
from src.prompt import Promptgenerator
import logfire


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
            print(mutated)
            evaluation = self.specification.evaluator(mutated)
            print(evaluation)
            self.population.add(Organism(solution=mutated, evaluation=evaluation, parent_id=mutatee.id))
            
            current_best = self.population.get_best().evaluation.fitness
            logfire.info(f"Step completed {step}\n"
                         f"with fitness {evaluation.fitness}\n"
                         f"and current best fitness {current_best}\n"
                         ) 

        final_stats = {
            "steps": step,
            "best_fitness": self.population.get_best().evaluation.fitness,
            "average_fitness": self.population.calculate_average_fitness()
        }
        logfire.info("Evolution completed", **final_stats)
        return self.population

