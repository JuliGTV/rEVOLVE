from src.evaluation import Evaluation

from example_problems.guess_the_votes.test import test_guess_the_votes


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


def evaluate(solution:str) -> Evaluation:

    try:
        # Create a dictionary to serve as the local namespace
        local_namespace = {}
        
        # Execute the code string in the local namespace
        exec(solution, {}, local_namespace)
        
        # Check if 'f' is defined and is callable
        if 'guess_the_votes' in local_namespace and callable(local_namespace['guess_the_votes']):
            function_detected = True
            result = test_guess_the_votes(eval('guess_the_votes', local_namespace, local_namespace))
            if type(result) == bool:
                if len(solution) >= len(baseline_solution):
                    fitness = 0
                else:
                    fitness = len(baseline_solution) - len(solution)
            else:
                fitness = -1
        else:
            function_detected = False
            fitness = -1
    except Exception as e:
        function_detected = False
        print(e)
        fitness = -1

    additional_data = {
        "length": str(len(solution)),
        "function_detected": str(function_detected)
    }
    if 'result' in locals():
        additional_data["result"] = str(result)
    return Evaluation(fitness=fitness, additional_data=additional_data)
    