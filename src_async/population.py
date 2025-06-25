from pydantic import BaseModel
import random
import pickle
from typing import Optional, List
from src_async.evaluation import Evaluation
from graphviz import Digraph
import colorsys


class Organism(BaseModel):
    solution: str
    evaluation: Evaluation
    id: Optional[int] = None
    parent_id: Optional[int] = None
    creation_info: Optional[dict] = None
    children: int = 0


class Population:
    def __init__(self, exploration_rate: float = 0.2, elitism_rate: float = 0.1, pop: List[Organism] = None):
        self.population = []
        self.id_counter = 1
        self.exploration_rate = exploration_rate
        self.elitism_rate = elitism_rate
        
        # Add initial population through add() method to ensure IDs are assigned
        if pop:
            for organism in pop:
                self.add(organism)
    
    @classmethod
    def from_pickle(cls, pickle_path: str, exploration_rate: float = 0.2, elitism_rate: float = 0.1):
        """Load population from a pickle file"""
        with open(pickle_path, 'rb') as f:
            organisms = pickle.load(f)
        
        # Create new population instance
        population = cls(exploration_rate=exploration_rate, elitism_rate=elitism_rate)
        
        # Add organisms and set the ID counter to the highest ID + 1
        max_id = 0
        for organism in organisms:
            # Don't use add() method to preserve original IDs
            population.population.append(organism)
            if organism.id and organism.id > max_id:
                max_id = organism.id
        
        population.id_counter = max_id + 1
        return population

    def add(self, organism: Organism):
        organism.id = self.id_counter
        
        # Add child number to creation_info if this organism has a parent
        if organism.parent_id is not None:
            parent = self.get_id(organism.parent_id)
            if parent is not None:
                # Child number is current children count + 1 (since we haven't incremented yet)
                child_number = parent.children + 1
                
                # Add child_number to creation_info
                if organism.creation_info is None:
                    organism.creation_info = {}
                organism.creation_info["child_number"] = child_number
                
                # Increment parent's children count
                parent.children += 1
        
        self.population.append(organism)
        self.id_counter += 1

    def get_best(self) -> Organism:
        return max(self.population, key=lambda x: x.evaluation.fitness)
    
    def get_random(self) -> Organism:
        return random.choice(self.population)
    
    def get_weighted_random(self) -> Organism:
        weights = [organism.evaluation.fitness ** 4 + 1 for organism in self.population]
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
        Nodes are colored by organism ID on a gradient (blue to red).
        Node labels show the fitness score.
        An edge from parent to child is drawn when parent_id is set.
        Organisms that were historically the best are highlighted with thick borders.
        The overall best organism is highlighted with a gold border.
        """
        dot = Digraph(comment='Population')
        
        # Find organisms that were historically the best at some point
        organisms_sorted = sorted(self.population, key=lambda x: x.id)
        historically_best_ids = set()
        current_best_fitness = float('-inf')
        
        for org in organisms_sorted:
            if org.evaluation.fitness > current_best_fitness:
                current_best_fitness = org.evaluation.fitness
                historically_best_ids.add(org.id)
        
        # Find the overall best organism
        overall_best = self.get_best()
        
        # compute ID range for coloring
        id_values = [org.id for org in self.population]
        min_id, max_id = min(id_values), max(id_values)

        for org in self.population:
            # normalize ID to [0,1] for color mapping
            if max_id > min_id:
                norm = (org.id - min_id) / (max_id - min_id)
            else:
                norm = 0.5
            # hue from blue (240deg) to red (0deg)
            hue = (240 * (1 - norm)) / 360.0
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            hexcolor = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

            # Display fitness score as the node label
            fitness_label = f"{org.evaluation.fitness:.2f}"
            
            # Determine node styling based on historical significance
            if org.id == overall_best.id:
                # Overall best: gold border, thick
                dot.node(str(org.id), label=fitness_label, style='filled', 
                        fillcolor=hexcolor, color='gold', penwidth='4')
            elif org.id in historically_best_ids:
                # Historically best: thick black border
                dot.node(str(org.id), label=fitness_label, style='filled', 
                        fillcolor=hexcolor, color='black', penwidth='3')
            else:
                # Regular organism
                dot.node(str(org.id), label=fitness_label, style='filled', fillcolor=hexcolor)

            # add edge from parent
            if org.parent_id is not None:
                dot.edge(str(org.parent_id), str(org.id))

        dot.render(filename, format='png', view=view)
        return dot