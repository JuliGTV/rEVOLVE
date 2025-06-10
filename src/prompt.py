from population import Organism


class Promptgenerator:
    def __init__(self, systemprompt: str, reason: bool = False):
        self.systemprompt = systemprompt

    def generate_prompt(self, organism: Organism) -> str:
        otpt = f"""
{self.systemprompt}

{self._previous_solution(organism)}

{self._format()}
"""
        return otpt


    def _previous_solution(self, organism: Organism) -> str:
        otpt = f"""
Look at this solution to the problem:
{organism.solution}

It achieved a fitness of {organism.fitness}.

You should propose a new solution that is a SMALL ITERATIVE IMPROVEMENT on this one.
"""
        return otpt
    
    def _format(self) -> str:
        if not self.reason:
            return "Your answer should just be the new solution in code, with no additional text or formatting."
        else:
            return r"You should output any reasoning you need, and then the new solution in code, with no additional text or formatting. Use /n---/n to separate the reasoning from the solution."
        

