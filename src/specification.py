from typing import Callable, Optional
from pydantic import BaseModel
from src.population import Organism
from src.evaluation import Evaluation



class Hyperparameters(BaseModel):
    exploration_rate: float = 0.2
    elitism_rate: float = 0.1
    max_steps: int = 100
    target_fitness: Optional[float] = None
    reason: bool = False


class ProblemSpecification(BaseModel):
    name: str
    systemprompt: str
    evaluator: Callable[[str], Evaluation]
    starting_population: list[Organism] 
    hyperparameters: Hyperparameters = Hyperparameters()


