from pydantic import BaseModel
import random
from typing import Optional, List
from src.evaluation import Evaluation
from graphviz import Digraph
import colorsys


class Organism(BaseModel):
    solution: str
    evaluation: Evaluation
    id: Optional[int] = None
    parent_id: Optional[int] = None


class Population:
    def __init__(self, exploration_rate: float = 0.2, elitism_rate: float = 0.1, pop: List[Organism] = None):
        self.population = pop or []
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
        weights = [organism.evaluation.fitness + 1 for organism in self.population]
        return random.choices(self.population, weights=weights, k=1)[0]
    
    def get_next(self) -> Organism:
        "implements a very simple genetic algorithm"
        r = random.random()
        if r < self.exploration_rate:
            return self.get_random()
        if r > 1 - self.elitism_rate:
            return self.get_best()
        return self.get_weighted_random()
       
    def get_id(self, id: int) -> Optional[Organism]:
        return next((org for org in self.population if org.id == id), None)
    
    def get_population(self) -> List[Organism]:
        return self.population
    
    def calculate_average_fitness(self) -> float:
        return sum(org.evaluation.fitness for org in self.population) / len(self.population)
    
    def visualize_population(self, filename: str = 'population.gv', view: bool = False) -> Digraph:
        """
        Create a Graphviz directed graph of the population.
        Nodes are colored by fitness on a red-to-green gradient.
        An edge from parent to child is drawn when parent_id is set.
        """
        dot = Digraph(comment='Population')
        # compute fitness range
        fitness_values = [org.evaluation.fitness for org in self.population]
        min_fit, max_fit = min(fitness_values), max(fitness_values)

        for org in self.population:
            # normalize fitness to [0,1]
            if max_fit > min_fit:
                norm = (org.evaluation.fitness - min_fit) / (max_fit - min_fit)
            else:
                norm = 0.5
            # hue from red (0deg) to green (120deg)
            hue = (120 * norm) / 360.0
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            hexcolor = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

            dot.node(str(org.id), label=str(org.id), style='filled', fillcolor=hexcolor)

            # add edge from parent
            if org.parent_id is not None:
                dot.edge(str(org.parent_id), str(org.id))

        dot.render(filename, format='png', view=view)
        return dot

    

