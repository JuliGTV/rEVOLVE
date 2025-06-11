from population import Organism, Population
from specification import ProblemSpecification
from mutate import generate
from prompt import Promptgenerator



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
        while self.population.get_best().fitness < self.target and step < self.max_steps:
            step += 1
            mutatee = self.population.get_next()
            prompt = self.prompt_gen.generate_prompt(mutatee)
            mutated = generate(prompt, self.reason)
            fitness = self.specification.evaluator(mutated)
            self.population.add(Organism(mutated, fitness, parent_id=mutatee.id))

        print(f"Evolved {step} steps"
              f"with best fitness {self.population.get_best().fitness}"
              f"and average fitness {self.population.get_average().fitness}"
              )
        return self.population

