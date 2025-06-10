from population import Organism


class Prompgenerator:
    def __init__(self, systemprompt: str):
        self.systemprompt = systemprompt

    def generate_prompt(self, organism: Organism) -> str:
        otpt = f"""
{self.systemprompt}

{self._previous_solution(organism)}
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