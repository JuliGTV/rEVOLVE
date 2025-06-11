from pydantic import BaseModel
import random
from typing import Optional
from src.evaluation import Evaluation
class Organism(BaseModel):
    solution: str
    evaluation: Evaluation
    id: Optional[int] = None
    parent_id: Optional[int] = None


class Population:
    def __init__(self,exploration_rate: float = 0.4, elitism_rate: float = 0.1, pop=[]):
        self.population = pop
        self.id_counter = 1
        self.exploration_rate = exploration_rate
        self.elitism_rate = elitism_rate

    def add(self, organism: Organism):
        organism.id = self.id_counter
        self.population.append(organism)
        self.id_counter += 1

    def get_best(self) -> Organism:
        return max(self.population, key=lambda x: x.evaluation.fitness)
    
    def get_random(self) -> Organism:
        return random.choice(self.population)
    
    def get_weighted_random(self) -> Organism:

        weights = [organism.evaluation.fitness + 2 for organism in self.population]
        return random.choices(self.population, weights=weights, k=1)[0]
    
    def get_next(self) -> Organism:
        "implements a very simple genetic algorithm"
        r =random.random()
        if r < self.exploration_rate:
            return self.get_random()
        if r > 1-self.elitism_rate:
            return self.get_best()
        else:
            return self.get_weighted_random()
       
    def get_id(self, id: int) -> Organism:
        return next((organism for organism in self.population if organism.id == id), None)
    
    def get_population(self) -> list[Organism]:
        return self.population
    
    def calculate_average_fitness(self) -> float:
        return sum(organism.evaluation.fitness for organism in self.population) / len(self.population)
    

