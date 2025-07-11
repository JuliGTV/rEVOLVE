from typing import Callable, Optional, TYPE_CHECKING
from pydantic import BaseModel
from src.evaluation import Evaluation

if TYPE_CHECKING:
    from src.population import Organism


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
    starting_population: list["Organism"] 
    hyperparameters: Hyperparameters = Hyperparameters()