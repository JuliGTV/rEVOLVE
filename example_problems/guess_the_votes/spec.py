from src.specification import ProblemSpecification, Hyperparameters
from src.population import Organism
from example_problems.guess_the_votes.evaluation import evaluate        

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




sysprompt = """
    You are a master python programmer, and code golf champion.
    You will be given a problem, and a solution (or attempt at a solution) to the problem.
    Your task will be to improve the given solution.

    Your solution will be run on a series of test cases. 
    If it does not pass them all it will be scored as -1.
    If it passes them and is the same length or longer than the baseline solution, it will be scored as 0.
    Otherwise it will be scored as the difference between the length of the solution and the baseline solution which is 330 characters.


"""

hyperparameters = Hyperparameters(
    max_steps=10,
    target_fitness=0,
    exploration_rate=0.1,
    elitism_rate=0.2,
    reason=True
)


spec = ProblemSpecification(
    name="guess_the_votes (code golf)",
    systemprompt=str(sysprompt),
    evaluator=evaluate,
    starting_population=[
        Organism(solution=str(baseline_solution), evaluation=evaluate(str(baseline_solution)), id=0)
    ],
    hyperparameters=hyperparameters
)



if __name__ == "__main__":
    print("Baseline solution length:", len(baseline_solution))
    print("Baseline solution score:", evaluate(baseline_solution))
    
    sample_files = ['o4high.py', 'o4high_golf.py', 'gpt4o.py', 'gpt4o_golf.py']
    for file in sample_files:
        with open(f'test_problems/guess_the_votes/sample_solutions/{file}', 'r') as f:
            solution = f.read()
        print(f"\n{file}:")
        print("Length:", str(len(solution)))
        print("Score:", str(evaluate(solution)))
    