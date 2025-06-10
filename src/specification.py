
from typing import Callable, Optional
from pydantic import BaseModel

class ProblemSpecification(BaseModel):
    name: str
    systemprompt: str
    evlator: Callable[[str], float]
    sample_solution: str
    target_fitness: Optional[float] = None
    max_steps: Optional[int] = None
