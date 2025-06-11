
from typing import Callable, Optional
from pydantic import BaseModel
from src.population import Organism



class Hyperparameters(BaseModel):
    exploration_rate: float = 0.4
    elitism_rate: float = 0.1
    max_steps: int = 100
    target_fitness: Optional[float] = None
    reason: bool = False

class Evaluation(BaseModel):
    fitness: float
    additional_data: dict[str, str] = {}

class ProblemSpecification(BaseModel):
    name: str
    systemprompt: str
    evaluator: Callable[[str], Evaluation]
    starting_population: list[Organism] 
    hyperparameters: Hyperparameters = Hyperparameters()


