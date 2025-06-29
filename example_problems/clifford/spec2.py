



INITIAL_SOLUTION = '''
def heuristic(matrix):
    import numpy as np
    return tuple(np.concatenate((np.sum(matrix,axis=0),np.sum(matrix,axis=0)))) 
'''

PROMPT = '''
You are a quantum computing expert.

You are working on synthesizing CNOT circuits to implement given parity matrices (GL matrices) using the fewest possible CNOTs.

You are doing this using greedy search algorithms which require a heuristic function that approximates the cost of synthesizing the circuit.

You will be given a function heuristic(matrix: np.ndarray) -> tuple[float] that takes a symplectic matrix built from a GL matrix and returns a tuple of floats that will be used to decide where the next CNOT gate should be placed.
Your task is to improve this heuristic function.

the symplective matrix is built from the GL matrix using the following function:

```
def symCNOT(U):
    """CNOT circuit from binary invertible matrix U"""
    m,n = U.shape
    U2 = np.zeros((2*n,2*n),dtype=int)
    U2[:n,:n] = U
    Uinv = binMatInv(U.T) # this is a function that inverts a binary matrix
    U2[n:,n:] = Uinv
    return U2

```

Here is some relevant information:

- your function must return a tuple of floats
- your function will be evaluated by running greedy search on a dataset of GL matrices and measuring the average number of gates synthesized, and the time taken.
- The goal is to use the fewest gates possible, while not taking too long.
- when you return a tuple, these will be compared lexicographically to make decisions during the search
- operators need only be sythesized up to permutation, so any permutation of the identity matrix has cost 0
- The score will be the calculated as 70 - (average number of gates synthesized)   where 70 has been chosen as a weak baseline.
- Every second over 10 that it takes to synthesize the dataset will be penalized by 1 point.
- if it takes more than 20 seconds or the code does not run properly, the score will be 0.


Domain knowledge:
- originally people used float heuristics:
- the sum of the elements of the matrix is a good basic heuristic
- another is a metric based on the sum of the logarithms of the column sums of
the parity matrix. This method gives priority to eliminating entries in columns which are ‘almost
done’ and have weight close to 1
- it is also useful to consider the inverse and transpose of the parity matrix when calculating their
heuristics because these can be synthesized using the same number of CNOT gates and may have
lower heuristics.
- it has been found that heuristics which return tuples (e.g. by looking at columns seperately) are less likely to be caught in local minima during the search


'''

# rEVOLVE framework imports
import sys
sys.path.append('../../')
from src.specification import ProblemSpecification, Hyperparameters
from src.population import Organism
from src.evaluation import Evaluation
from evaluate2 import evaluate_heuristic_from_string





# Configuration for Clifford heuristic evolution
CLIFFORD_CONFIG = {
    "exploration_rate": 0.1,
    "elitism_rate": 0.3,
    "max_steps": 1000,
    "target_fitness": 0.95,
    "reason": True,
    "max_concurrent": 15,
    "model_mix": {"deepseek:deepseek-reasoner": 0.1, "deepseek:deepseek-chat": 0.9},
    "big_changes_rate": 0.4,
    "best_model": "deepseek:deepseek-reasoner",
    "max_children_per_organism": 30,
    "checkpoint_dir": "evolution_results/checkpoints",
    "population_path": None,
}


def get_clifford_heuristic_spec() -> ProblemSpecification:
    """Get the Clifford heuristic evolution problem specification."""
    
    # Create initial population
    initial_heuristics = [INITIAL_SOLUTION]
    starting_population = []
    
    for i, heuristic_code in enumerate(initial_heuristics):
        try:
            evaluation = evaluate_heuristic_from_string(heuristic_code)
            organism = Organism(solution=heuristic_code, evaluation=evaluation)
            starting_population.append(organism)
            print(f"Initial heuristic {i+1}: fitness = {evaluation.fitness:.4f}")
        except Exception as e:
            print(f"Failed to evaluate initial heuristic {i+1}: {e}")
            fallback_eval = Evaluation(fitness=0.0, additional_data={"error": str(e)})
            organism = Organism(solution=heuristic_code, evaluation=fallback_eval)
            starting_population.append(organism)
    
    # Configure hyperparameters
    hyperparameters = Hyperparameters(
        exploration_rate=CLIFFORD_CONFIG["exploration_rate"],
        elitism_rate=CLIFFORD_CONFIG["elitism_rate"],
        max_steps=CLIFFORD_CONFIG["max_steps"],
        target_fitness=CLIFFORD_CONFIG["target_fitness"],
        reason=CLIFFORD_CONFIG["reason"]
    )
    
    return ProblemSpecification(
        name="clifford_heuristic",
        systemprompt=PROMPT,
        evaluator=evaluate_heuristic_from_string,
        starting_population=starting_population,
        hyperparameters=hyperparameters
    )


def get_clifford_heuristic_evolver_config() -> dict:
    """Get the AsyncEvolver configuration."""
    return {
        "checkpoint_dir": CLIFFORD_CONFIG["checkpoint_dir"],
        "max_concurrent": CLIFFORD_CONFIG["max_concurrent"],
        "model_mix": CLIFFORD_CONFIG["model_mix"],
        "big_changes_rate": CLIFFORD_CONFIG["big_changes_rate"],
        "best_model": CLIFFORD_CONFIG["best_model"],
        "max_children_per_organism": CLIFFORD_CONFIG["max_children_per_organism"],
        "population_path": CLIFFORD_CONFIG["population_path"]
    }


if __name__ == "__main__":
    # Test the problem specification
    print("Testing Clifford heuristic problem specification...")
    spec = get_clifford_heuristic_spec()
    config = get_clifford_heuristic_evolver_config()
    
    print(f"Problem name: {spec.name}")
    print(f"Starting population size: {len(spec.starting_population)}")
    print(f"Target fitness: {spec.hyperparameters.target_fitness}")

