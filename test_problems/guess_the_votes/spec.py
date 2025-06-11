from src.specification import ProblemSpecification
from test_problems.guess_the_votes.test import test_guess_the_votes

baseline_solution = """def guess_the_votes(s,v):
 from itertools import product
 r={k:set()for k in v}
 a=list(s)
 o=list(v)
 A=[x for x in product(range(len(o)),repeat=len(a))if all(sum(s[a[i]]for i in range(len(a))if x[i]==j)==v[o[j]]for j in range(len(o)))]
 for i,n in enumerate(a):
  p={o[x[i]]for x in A}
  if len(p)==1:r[p.pop()].add(n)
 return r"""


def evaluate(solution:str) -> int:

    try:
        # Create a dictionary to serve as the local namespace
        local_namespace = {}
        
        # Execute the code string in the local namespace
        exec(solution, {}, local_namespace)
        
        # Check if 'f' is defined and is callable
        if 'guess_the_votes' in local_namespace and callable(local_namespace['guess_the_votes']):
            if test_guess_the_votes(local_namespace['guess_the_votes']):
                if len(solution) >= len(baseline_solution):
                    return 0
                return len(baseline_solution) - len(solution)
            else:
                return -1
        else:
            return -1
    except Exception as e:
        return -1





spec = ProblemSpecification(
    name="guess_the_votes (code golf)",
    systemprompt="""
    You are a master python programmer, and code golf champion.
    You will be given a problem, and a solution (or attempt at a solution) to the problem.
    Your task will be to improve the given solution.

    Your solution will be run on a series of test cases. 
    If it does not pass them all it will be scored as -1.
    If it passes them and is the same length or longer than the baseline solution, it will be scored as 0.
    Otherwise it will be scored as the difference between the length of the solution and the baseline solution which is 330 characters.



""",
    starting_population=[
        "def guess_the_votes(shares, votes):\n    return {vote: set() for vote in votes}\n",
        "def guess_the_votes(shares, votes):\n    return {vote: set() for vote in votes}\n",
        "def guess_the_votes(shares, votes):\n    return {vote: set() for vote in votes}\n",
        "def guess_the_votes(shares, votes):\n    return {vote: set() for vote in votes}\n",
    ],
)



if __name__ == "__main__":
    print(len(baseline_solution))
    print(evaluate(baseline_solution))