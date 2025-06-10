
from typing import Callable, Optional
from pydantic import BaseModel



class Hyperparameters(BaseModel):
    exploration_rate: float = 0.4
    elitism_rate: float = 0.1
    max_steps: int = 100
    target_fitness: Optional[float] = None
    reason: bool = False


class ProblemSpecification(BaseModel):
    name: str
    systemprompt: str
    evaluator: Callable[[str], float]
    starting_population: list[str] 
    hyperparameters: Hyperparameters


