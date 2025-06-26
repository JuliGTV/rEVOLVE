"""
Problem specification for evolving Clifford circuit synthesis heuristics.

This module defines the rEVOLVE problem specification for evolving improved
heuristic functions for quantum Clifford circuit synthesis using CliffordOpt.
"""

import numpy as np
from typing import Dict, Any, List
import sys
import os

# Add rEVOLVE source to path
evolve_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if evolve_src_path not in sys.path:
    sys.path.insert(0, evolve_src_path)

from specification import ProblemSpecification, Hyperparameters
from population import Organism
# Import from local evaluation module (avoid conflict with rEVOLVE's evaluation.py)
clifford_eval_path = os.path.join(os.path.dirname(__file__), 'evaluation.py')
import importlib.util
spec = importlib.util.spec_from_file_location("clifford_evaluation", clifford_eval_path)
clifford_evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clifford_evaluation)
evaluate_clifford_heuristic = clifford_evaluation.evaluate_clifford_heuristic


# System prompt for LLM to understand the Clifford synthesis problem
SYSTEM_PROMPT = """You are an expert in quantum computing and mathematical optimization, tasked with evolving improved heuristic functions for Clifford quantum circuit synthesis.

## Background: Quantum Circuit Synthesis

Clifford circuits are a fundamental class of quantum circuits that:
- Can be efficiently simulated on classical computers
- Are essential for quantum error correction codes
- Form building blocks for many quantum algorithms
- Have well-understood mathematical structure

## The Synthesis Problem

Given a target Clifford operation represented as a 2n×2n symplectic matrix over GF(2), find the shortest sequence of elementary gates (CNOT, Hadamard, Phase) that implements this operation.

**Mathematical Context:**
- Symplectic matrices U satisfy: U * J * U^T = J (where J is the symplectic form)
- They represent how Clifford operations transform Pauli operators
- Synthesis algorithms use heuristics to guide search through the exponentially large space

## Current Approach Limitations

Existing synthesis algorithms (greedy, A*) use simple hand-crafted heuristics that analyze:
- Row/column sums of matrix blocks
- Rank properties of submatrices  
- Simple transformations (transpose, inverse)

These heuristics are likely suboptimal and miss complex mathematical patterns.

## Your Task: Evolve Better Heuristics

You must create Python functions that take a symplectic matrix and return a numerical score indicating synthesis difficulty or priority. Higher scores typically indicate "easier" matrices to reduce.

**Function Interface:**
```python
def evolved_heuristic(symplectic_matrix: np.ndarray) -> float:
    \"\"\"
    Heuristic function for Clifford circuit synthesis.
    
    Args:
        symplectic_matrix: 2n×2n binary symplectic matrix (numpy array)
                          representing n-qubit Clifford operation
    
    Returns:
        float: Heuristic score (higher = easier to synthesize)
    \"\"\"
    # Your evolved analysis here
    return score
```

**Key Considerations:**
1. **Mathematical Insight**: Analyze symplectic structure, block patterns, rank properties
2. **Computational Efficiency**: Functions will be called thousands of times
3. **Generalization**: Must work across different circuit sizes (2-16 qubits)
4. **Numerical Stability**: Return finite, meaningful numerical values

**Available Libraries:**
- numpy (as np): For matrix operations and linear algebra
- Standard math operations: min, max, sum, len, abs, etc.

**Matrix Structure:** 
For n qubits, the 2n×2n symplectic matrix has block structure:
```
[ X_block  | XZ_block ]
[ ZX_block | Z_block  ]
```
where each block is n×n.

**Success Metrics:**
- Primary: Reduce average gate count compared to baseline methods
- Secondary: Maintain reasonable computation time
- Robustness: Consistent performance across circuit sizes

**Examples of Mathematical Features to Consider:**
- Block-wise analysis of X, Z, XZ, ZX components
- Matrix rank and nullspace properties
- Eigenvalue/eigenvector structure (if computationally feasible)
- Symmetry and pattern detection
- Information-theoretic measures (entropy, etc.)
- Graph-theoretic properties when viewed as adjacency matrices

Generate functions that discover novel mathematical insights into symplectic matrix structure that can guide more efficient circuit synthesis."""


def create_initial_heuristic_population() -> List[Organism]:
    """
    Create initial population of heuristic functions for evolution.
    
    Returns:
        List of Organism objects with different heuristic approaches
    """
    
    # Simple baseline heuristic
    baseline_code = """def evolved_heuristic(symplectic_matrix):
    import numpy as np
    
    # Simple baseline: prefer matrices with more zeros
    total_ones = np.sum(symplectic_matrix)
    matrix_size = symplectic_matrix.size
    sparsity_score = (matrix_size - total_ones) / matrix_size
    
    return sparsity_score * 100
"""

    # Block-based analysis heuristic
    block_code = """def evolved_heuristic(symplectic_matrix):
    import numpy as np
    
    n = symplectic_matrix.shape[0] // 2
    
    # Analyze block structure
    x_block = symplectic_matrix[:n, :n]
    z_block = symplectic_matrix[n:, n:]
    xz_block = symplectic_matrix[:n, n:]
    zx_block = symplectic_matrix[n:, :n]
    
    # Score based on block sparsity
    x_sparsity = 1.0 - (np.sum(x_block) / x_block.size)
    z_sparsity = 1.0 - (np.sum(z_block) / z_block.size)
    off_diag_density = (np.sum(xz_block) + np.sum(zx_block)) / (2 * n * n)
    
    # Combine scores - prefer sparse diagonal blocks, some off-diagonal structure
    score = (x_sparsity + z_sparsity) * 50 + off_diag_density * 25
    
    return score
"""

    # Row/column analysis heuristic
    row_col_code = """def evolved_heuristic(symplectic_matrix):
    import numpy as np
    
    # Analyze row and column patterns
    row_sums = np.sum(symplectic_matrix, axis=1)
    col_sums = np.sum(symplectic_matrix, axis=0)
    
    # Prefer uniform distribution of ones
    row_variance = np.var(row_sums)
    col_variance = np.var(col_sums)
    
    # Score inversely related to variance (lower variance = more uniform = better)
    uniformity_score = 100 / (1 + row_variance + col_variance)
    
    return uniformity_score
"""

    # Rank-based heuristic
    rank_code = """def evolved_heuristic(symplectic_matrix):
    import numpy as np
    
    n = symplectic_matrix.shape[0] // 2
    
    # Analyze rank properties
    try:
        full_rank = np.linalg.matrix_rank(symplectic_matrix)
        
        # Look at submatrix ranks
        x_block = symplectic_matrix[:n, :n]
        z_block = symplectic_matrix[n:, n:]
        
        x_rank = np.linalg.matrix_rank(x_block)
        z_rank = np.linalg.matrix_rank(z_block)
        
        # Heuristic: prefer certain rank relationships
        rank_score = (x_rank + z_rank) / (2 * n) * 100
        
        return rank_score
        
    except:
        # Fallback to simple density measure
        return (1.0 - np.sum(symplectic_matrix) / symplectic_matrix.size) * 100
"""

    # Entropy-based heuristic
    entropy_code = """def evolved_heuristic(symplectic_matrix):
    import numpy as np
    
    def binary_entropy(prob):
        if prob == 0 or prob == 1:
            return 0
        return -prob * np.log2(prob) - (1-prob) * np.log2(1-prob)
    
    # Calculate entropy of different parts
    total_ones = np.sum(symplectic_matrix)
    total_size = symplectic_matrix.size
    overall_entropy = binary_entropy(total_ones / total_size)
    
    # Row-wise entropy
    row_entropies = []
    for row in symplectic_matrix:
        if len(row) > 0:
            prob = np.sum(row) / len(row)
            row_entropies.append(binary_entropy(prob))
    
    avg_row_entropy = np.mean(row_entropies) if row_entropies else 0
    
    # Combine measures - prefer moderate entropy (not too structured, not too random)
    target_entropy = 0.5  # Moderate entropy
    entropy_score = 100 * (1 - abs(overall_entropy - target_entropy))
    
    return entropy_score + avg_row_entropy * 20
"""

    organisms = []
    
    # Create organisms with different heuristic functions
    heuristics = [
        ("baseline_sparsity", baseline_code),
        ("block_analysis", block_code), 
        ("row_col_uniformity", row_col_code),
        ("rank_analysis", rank_code),
        ("entropy_based", entropy_code)
    ]
    
    for i, (name, code) in enumerate(heuristics):
        organism = Organism(
            id=f"init_{i}",
            solution=code,
            fitness=None,  # Will be evaluated
            parent_id=None,
            metadata={
                "name": name,
                "type": "initial_population",
                "description": f"Initial heuristic function: {name}"
            }
        )
        organisms.append(organism)
    
    return organisms


def get_problem_spec() -> ProblemSpecification:
    """
    Create the rEVOLVE problem specification for Clifford heuristic evolution.
    
    Returns:
        ProblemSpecification object configured for heuristic evolution
    """
    
    # Create initial population
    starting_population = create_initial_heuristic_population()
    
    # Configure hyperparameters for evolution
    hyperparameters = Hyperparameters(
        exploration_rate=0.25,    # Higher exploration for discovering novel approaches
        elitism_rate=0.15,        # Keep some best performers
        max_steps=100,            # Allow sufficient evolution time
        target_fitness=None,      # No specific target - continuous improvement
        min_fitness_improvement=1.0,  # Require meaningful improvements
        patience=20,              # Stop if no improvement for 20 generations
    )
    
    return ProblemSpecification(
        name="clifford_heuristic_evolution",
        systemprompt=SYSTEM_PROMPT,
        evaluator=evaluate_clifford_heuristic,
        starting_population=starting_population,
        hyperparameters=hyperparameters,
        metadata={
            "problem_type": "heuristic_optimization",
            "domain": "quantum_computing",
            "target": "clifford_circuit_synthesis",
            "evaluation_metrics": ["synthesis_success_rate", "gate_count_reduction", "computation_time"],
            "baseline_methods": ["greedy", "astar"],
            "test_datasets": ["Sp_small.txt", "Sp_large.txt"]
        }
    )