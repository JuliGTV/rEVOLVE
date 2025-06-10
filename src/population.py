from pydantic import BaseModel
import random

class Organism(BaseModel):
    solution: str
    fitness: float


class Population:
    def __init__(self,exploration_rate: float = 0.4, elitism_rate: float = 0.1, pop=[]):
        self.pop = pop

    def add(self, organism: Organism):
        self.pop.append(organism)

    def get_best(self) -> Organism:
        return max(self.pop, key=lambda x: x.fitness)
    
    def get_random(self) -> Organism:
        return random.choice(self.pop)
    
    def get_weighted_random(self) -> Organism:
        weights = [organism.fitness for organism in self.pop]
        return random.choices(self.pop, weights=weights, k=1)[0]
    
    def get_next(self) -> Organism:
        "implements a very simple genetic algorithm"
        r =random.random()
        if r < self.exploration_rate:
            return self.get_random()
        if r > 1-self.elitism_rate:
            return self.get_best()
        else:
            return self.get_weighted_random()
       
    

