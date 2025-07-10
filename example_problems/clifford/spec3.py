INITIAL_SOLUTION = '''
#!/usr/bin/env python3
"""
search.py - Self-contained CNOT circuit synthesis using GL greedy heuristic.
"""

import numpy as np
import time


def search(U):
    """
    Synthesize a CNOT circuit for an n×n invertible binary matrix U.

    Returns:
        n (int): number of qubits,
        gate_count (int): number of CNOT gates,
        depth (int): circuit depth,
        proc_time (float): synthesis time in seconds,
        check (bool): verification of correctness,
        circ (str): circuit operations string.
    """
    # Fixed parameters
    params = {
        'mode': 'GL',
        'method': 'greedy',
        'minDepth': False,
        'hv': True,
        'hi': True,
        'ht': True,
        'hl': True,
        'hr': 3,
        'wMax': 10
    }
    sym = sym_cnot(U)
    return synth_main(sym, params)


def synth_main(U, params):
    n = U.shape[1] // 2
    start = time.time()
    op_list, UC = csynth_greedy(U, params)
    # prepend single-qubit Clifford and permutation layer
    sqc_ops = mat2SQC(UC)
    full_ops = sqc_ops + op_list
    depth = len(op_list_layers(full_ops))
    gate_count = count_entangling(full_ops)
    proc_time = time.time() - start
    circ = op_list_to_string(full_ops)
    check = sym_test(U, full_ops)
    return n, gate_count, depth, proc_time, check, circ


def csynth_greedy(A, params):
    n2 = A.shape[1]
    n = n2 // 2
    U = A.copy()
    op_list = []
    h_min = None
    curr_wait = 0
    while True:
        h_val, h_tuple = gl_heuristic(U, params)
        if h_val <= 1e-5:
            break
        best = None
        best_dh = None
        best_B = None
        for op in gl_options(U):
            B = apply_op(U, op)
            hB_val, hB_tuple = gl_heuristic(B, params)
            key = (hB_tuple, hB_val) if params['hv'] else (hB_val, hB_tuple)
            dh = (0, key)
            if best_dh is None or dh < best_dh:
                best_dh = dh
                best = (hB_tuple, hB_val, op)
                best_B = B
        _, hB_val, op_chosen = best
        U = best_B
        op_list.append(op_chosen)
        if h_min is None or best < h_min:
            curr_wait = 0
            h_min = best
        else:
            curr_wait += 1
        if params['wMax'] > 0 and curr_wait > params['wMax']:
            # abort and return identity mapping
            return [], A
    op_list = list(reversed(op_list))
    return op_list, U


def gl_heuristic(U, params):
    if params['ht']:
        sA = np.concatenate([U.sum(axis=0), U.sum(axis=1)])
    else:
        sA = U.sum(axis=0)
    w = tuple(sorted(int(x) for x in sA))
    Ls = len(sA)
    if params['hl']:
        h0 = np.log(sA).sum() / Ls
    else:
        h0 = (U.sum() / len(U) - 1)
    return params['hr'] * h0, w


def gl_options(U):
    n = U.shape[1] // 2
    top = U[:n, :n]
    prod = top.T @ top
    opts = []
    rows, cols = np.nonzero(prod)
    for i, j in zip(rows, cols):
        if i != j:
            opts.append(('CX', (i, j)))
    return opts


def apply_op(U, op):
    typ, q = op
    U_new = U.copy()
    n = U.shape[1] // 2
    if typ == 'CX':
        i, j = q
        U_new[:, j] ^= U_new[:, i]
        U_new[:, n + i] ^= U_new[:, n + j]
    elif typ == 'QPerm':
        ix = list(q)
        perm = ix + [p + len(ix) for p in ix]
        U_new = U_new[:, perm]
    return U_new


def op_list_layers(op_list):
    layers = []
    for typ, q in op_list:
        qubits = set(q) if isinstance(q, tuple) else {q}
        for layer in layers:
            if qubits.isdisjoint(layer):
                layer.update(qubits)
                break
        else:
            layers.append(set(qubits))
    return layers


def count_entangling(op_list):
    return sum(1 for typ, q in op_list if typ == 'CX')


def op_list_to_string(op_list, sep=' '):
    parts = []
    for typ, q in op_list:
        if isinstance(q, tuple):
            parts.append(f'{typ}:{q[0]},{q[1]}')
        else:
            parts.append(f'{typ}:{q}')
    return sep.join(parts)


def sym_cnot(U):
    n = U.shape[0]
    S = np.zeros((2*n, 2*n), dtype=int)
    S[:n, :n] = U % 2
    inv = bin_mat_inv(U.T % 2)
    S[n:, n:] = inv % 2
    return S


def bin_mat_inv(A):
    A = A.copy() % 2
    n = A.shape[0]
    I = np.eye(n, dtype=int)
    aug = np.concatenate([A, I], axis=1)
    for i in range(n):
        if aug[i, i] == 0:
            for j in range(i+1, n):
                if aug[j, i] == 1:
                    aug[[i, j]] = aug[[j, i]]
                    break
            else:
                raise ValueError("Matrix is singular and cannot be inverted.")
        for j in range(n):
            if j != i and aug[j, i] == 1:
                aug[j] ^= aug[i]
    return aug[:, n:]

# Single-qubit Clifford layer extraction

SQC_tostr = {
    '1001': 'I', '0110': 'H',
    '1101': 'S', '1011': 'HSH',
    '1110': 'HS', '0111': 'SH'
}

def sym_r2(U):
    n = U.shape[0] // 2
    R2 = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            F = np.array([[U[i, j], U[i, j+n]],
                          [U[i+n, j], U[i+n, j+n]]], dtype=int)
            det = (F[0,0]*F[1,1] + F[0,1]*F[1,0]) % 2
            if det == 1:
                R2[i,j] = 1
    return R2


def perm_mat_to_ix(P):
    if (P.sum(axis=0) != 1).any() or (P.sum(axis=1) != 1).any():
        return None
    return [int(np.where(P[i])[0][0]) for i in range(P.shape[0])]


def ix_rev(ix):
    r = [0]*len(ix)
    for i, j in enumerate(ix):
        r[j] = i
    return r


def Fmat(U, i, j):
    n = U.shape[0] // 2
    return np.array([[U[i, j], U[i, j+n]],
                     [U[i+n, j], U[i+n, j+n]]], dtype=int)


def SQC2str(A):
    key = ''.join(str(int(x)) for x in A.ravel())
    return SQC_tostr.get(key, 'I')


def CList2opList(CList):
    ops = []
    for i, M in enumerate(CList):
        c = SQC2str(M)
        if c != 'I':
            ops.append((c, (i,)))
    return ops


def mat2SQC(UC):
    n = UC.shape[0] // 2
    UR2 = sym_r2(UC)
    ix = perm_mat_to_ix(UR2)
    if ix is None:
        ix = list(range(n))
    ixR = ix_rev(ix)
    CList = [Fmat(UC, i, ix[i]) for i in ixR]
    ops = CList2opList(CList)
    if ixR != list(range(n)):
        ops.insert(0, ('QPerm', tuple(ixR)))
    return ops


def op_list2sym(op_list, n):
    U = np.eye(2*n, dtype=int)
    for op in op_list:
        U = apply_op(U, op)
    return U


def sym_test(U, ops):
    n = U.shape[1] // 2
    return np.array_equal(U, op_list2sym(ops, n))

'''


PROMPT = '''
You are a quantum computing expert.

You are working on synthesizing CNOT circuits to implement given parity matrices (GL matrices) using the fewest possible CNOTs.

You will be given a complete search function search(matrix: np.ndarray) that takes a GL matrix and returns synthesis results in the format: (n, gate_count, depth, proc_time, check, circ)

Your task is to improve this search function to synthesize circuits more efficiently.



Here is some relevant information:

- Your function must be named 'search' and take a single GL matrix parameter
- The function will be evaluated by running it on a dataset of GL matrices and measuring the average number of gates synthesized and time taken
- The goal is to use the fewest gates possible, while not taking too long
- The score will be calculated as 70 - (average number of gates synthesized) where 70 has been chosen as a weak baseline
- Every second over 10 that it takes to synthesize the dataset will be penalized by 1 point
- If it takes more than 20 seconds or the code does not run properly, or it fails to synthesize any of the circuits in the dataset, the score will be 0
- Your function must return: (n, gate_count, depth, proc_time, check, circ) where check=True indicates successful synthesis


Domain knowledge:
Optimization strategies:
- Improve the heuristic function used in greedy search
- Modify search parameters (wMax, hr, etc.)
- Try different search strategies beyond greedy
- Optimize the order of operations
- Use different matrix representations or transformations
- Implement early termination conditions
- Balance between solution quality and computation time

Heuristic insights:
- the sum of the elements of the matrix is a good basic heuristic
- another is a metric based on the sum of the logarithms of the column sums of
the parity matrix. This method gives priority to eliminating entries in columns which are ‘almost
done’ and have weight close to 1
- it is also useful to consider the inverse and transpose of the parity matrix when calculating their
heuristics because these can be synthesized using the same number of CNOT gates and may have
lower heuristics.
- it has been found that heuristics which return tuples (e.g. by looking at columns seperately) are less likely to be caught in local minima during the search

Your search function should be self-contained and include all necessary helper functions.
'''

# rEVOLVE framework imports
import sys
sys.path.append('../../')
from src.specification import ProblemSpecification, Hyperparameters
from src.population import Organism
from src.evaluation import Evaluation
from evaluate3 import evaluate_heuristic_from_string





# Configuration for Clifford heuristic evolution
CLIFFORD_CONFIG = {
    "exploration_rate": 0.1,
    "elitism_rate": 0.8,
    "max_steps": 100,
    "target_fitness": 70,
    "reason": True,
    "max_concurrent": 5,
    "model_mix": {"deepseek:deepseek-reasoner": 1},
    "big_changes_rate": 0.4,
    "best_model": "deepseek:deepseek-reasoner",
    "max_children_per_organism": 15,
    "checkpoint_dir": "evolution_results/checkpoints",
    "population_path": None,
}


def get_clifford_heuristic_spec() -> ProblemSpecification:
    """Get the Clifford search function evolution problem specification."""
    
    # Create initial population
    initial_search_functions = [INITIAL_SOLUTION]
    starting_population = []
    
    for i, search_code in enumerate(initial_search_functions):
        try:
            evaluation = evaluate_heuristic_from_string(search_code)
            organism = Organism(solution=search_code, evaluation=evaluation)
            starting_population.append(organism)
            print(f"Initial search function {i+1}: fitness = {evaluation.fitness:.4f}")
        except Exception as e:
            print(f"Failed to evaluate initial search function {i+1}: {e}")
            fallback_eval = Evaluation(fitness=0.0, additional_data={"error": str(e)})
            organism = Organism(solution=search_code, evaluation=fallback_eval)
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
        name="clifford_search",
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
    print("Testing Clifford search function evolution problem specification...")
    spec = get_clifford_heuristic_spec()
    config = get_clifford_heuristic_evolver_config()
    
    print(f"Problem name: {spec.name}")
    print(f"Starting population size: {len(spec.starting_population)}")
    print(f"Target fitness: {spec.hyperparameters.target_fitness}")

