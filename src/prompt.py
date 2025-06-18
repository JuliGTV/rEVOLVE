from src.population import Organism
import random
import logfire
from typing import Optional

class Promptgenerator:
    def __init__(self, systemprompt: str, reason: bool = False, big_changes: float = 0.25):
        self.systemprompt = systemprompt
        self.reason = reason
        self.big_changes = big_changes

    def generate_prompt(self, organism: Organism, big_changes: Optional[float] = None) -> str:
        if big_changes is None:
            big_changes = self.big_changes

        otpt = f"""

{self.systemprompt}

{self._previous_solution(organism)}

{self._format()}
"""
        return otpt


    def _previous_solution(self, organism: Organism) -> str:

        changes = "SMALL ITERATIVE IMPROVEMENT"
        if random.random() < self.big_changes:
            changes = "LARGE QUALITATIVE CHANGE"
        
        logfire.debug(f"Changes: {changes}")

        otpt = f"""
Look at this solution to the problem:
```
{organism.solution}
```
It achieved a fitness of {organism.evaluation.fitness}.
Here is some additional data from the evaluation:
{organism.evaluation.additional_data}

You should propose a new solution that is a {changes} on this one.
Only improve it with respect to the fitness as defined above. No other criteria will be considered.
"""
        return otpt
    
    def _format(self) -> str:
        if not self.reason:
            return "Your answer should just be the new solution in code, with no additional text or formatting."
        else:
            return r"You should output any reasoning you need, and then the new solution in code, with no additional text or formatting. Use a markdown code block to enclose the solution."