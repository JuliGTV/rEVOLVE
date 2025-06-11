from src.population import Organism, Population
from src.specification import ProblemSpecification
from src.mutate import generate
from src.prompt import Promptgenerator



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

    def evolve(self) -> Population:
        step = 0
        while self.population.get_best().evaluation.fitness < self.target and step < self.max_steps:
            step += 1
            mutatee = self.population.get_next()
            prompt = self.prompt_gen.generate_prompt(mutatee)
            mutated = generate(prompt, self.reason)
            evaluation = self.specification.evaluator(mutated)
            self.population.add(Organism(solution=mutated, evaluation=evaluation, parent_id=mutatee.id))
            print(f"Step {step} complete")
            print(f"Best fitness: {self.population.get_best().evaluation.fitness}")

        print(f"Evolved {step} steps"
              f"with best fitness {self.population.get_best().evaluation.fitness}"
              f"and average fitness {self.population.calculate_average_fitness()}"
              )
        return self.population

