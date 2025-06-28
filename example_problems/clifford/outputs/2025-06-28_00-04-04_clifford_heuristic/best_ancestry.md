# Best Organism Ancestry Analysis

This document traces the complete ancestry of the fittest organism (ID: 334) with fitness 0.88426700.

Each section shows an ancestor in the lineage, from the fittest organism back to the original ancestor.
Organisms marked with * were the best fitness when they were created.

---

## Ancestor #1: Organism 334*

| Property | Value |
|----------|-------|
| **ID** | 334* |
| **Fitness** | 0.88426700 |
| **Best at Time** | 0.88426700 |
| **Parent ID** | 318 |
| **Was Best When Created** | Yes |

### Solution Code

```python

def heuristic(matrix):
    """Improved column-focused heuristic with enhanced sparsity and column distribution terms."""
    import numpy as np
    
    def get_heuristic(m):
        n = m.shape[0]
        col_nonzeros = np.count_nonzero(m, axis=0)
        
        # Column completion with log2 weighting
        col_completion = np.sum(np.log2(col_nonzeros + 1))
        
        # Enhanced column distribution penalty using squared differences
        col_dist = np.sum((np.sum(m, axis=0) - 1)**2)
        
        # More meaningful sparsity term using log2
        sparsity = np.log2(np.count_nonzero(m) + 1) - np.log2(n)
        
        return (col_completion, col_dist, sparsity)
    
    variants = [
        matrix,
        np.linalg.inv(matrix),
        matrix.T,
        np.linalg.inv(matrix.T)
    ]
    
    return min(get_heuristic(v) for v in variants)

```

---

## Ancestor #2: Organism 318*

| Property | Value |
|----------|-------|
| **ID** | 318* |
| **Fitness** | 0.88426524 |
| **Best at Time** | 0.88426524 |
| **Parent ID** | 285 |
| **Was Best When Created** | Yes |

### Solution Code

```python

def heuristic(matrix):
    """Improved column-focused heuristic with log2 scaling and sparsity term."""
    import numpy as np
    
    def get_heuristic(m):
        n = m.shape[0]
        col_nonzeros = np.count_nonzero(m, axis=0)
        
        # Column completion with log2 weighting (more aggressive scaling)
        col_completion = np.sum(np.log2(col_nonzeros + 1))
        
        # Column distribution penalty
        col_dist = np.sum(np.abs(np.sum(m, axis=0) - 1))
        
        # Small sparsity term to break ties
        sparsity = np.count_nonzero(m) / (n*n)
        
        return (col_completion, col_dist, sparsity)
    
    variants = [
        matrix,
        np.linalg.inv(matrix),
        matrix.T,
        np.linalg.inv(matrix.T)
    ]
    
    return min(get_heuristic(v) for v in variants)

```

---

## Ancestor #3: Organism 285*

| Property | Value |
|----------|-------|
| **ID** | 285* |
| **Fitness** | 0.88352227 |
| **Best at Time** | 0.88352227 |
| **Parent ID** | 62 |
| **Was Best When Created** | Yes |

### Solution Code

```python

def heuristic(matrix):
    """Simplified column-focused heuristic with aggressive logarithmic scaling."""
    import numpy as np
    
    def get_heuristic(m):
        n = m.shape[0]
        col_nonzeros = np.count_nonzero(m, axis=0)
        
        # Column completion with logarithmic weighting (prioritize nearly-complete columns)
        col_completion = np.sum(np.log(col_nonzeros + 1))
        
        # Column distribution penalty (how far from single non-zero per column)
        col_dist = np.sum(np.abs(np.sum(m, axis=0) - 1))
        
        return (col_completion, col_dist)
    
    variants = [
        matrix,
        np.linalg.inv(matrix),
        matrix.T,
        np.linalg.inv(matrix.T)
    ]
    
    return min(get_heuristic(v) for v in variants)

```

---

## Ancestor #4: Organism 62

| Property | Value |
|----------|-------|
| **ID** | 62 |
| **Fitness** | -0.68353896 |
| **Best at Time** | 0.88173780 |
| **Parent ID** | 57 |
| **Was Best When Created** | No |

### Solution Code

```python

def heuristic(matrix):
    """Advanced heuristic considering column/row basis proximity and interactions."""
    import numpy as np
    
    def get_heuristic(m):
        n = m.shape[0]
        col_sums = np.sum(m, axis=0)
        row_sums = np.sum(m, axis=1)
        
        # Column completion: how close each column is to a basis vector
        col_completion = np.sum([min(np.sum(col), np.sum(1-col)) for col in m.T])
        
        # Row completion similarly
        row_completion = np.sum([min(np.sum(row), np.sum(1-row)) for row in m])
        
        # Interaction term: measures how columns overlap
        interaction = np.sum(np.abs(m @ m.T - np.eye(n)))
        
        # Column distribution (prioritize nearly-complete columns)
        col_dist = np.sum(np.abs(col_sums - 1))
        
        # Nonzeros with logarithmic weighting (smaller changes matter more near completion)
        log_nonzeros = np.sum(np.log(np.count_nonzero(m, axis=0) + 1))
        
        return (col_completion + row_completion, interaction, log_nonzeros, col_dist)
    
    variants = [
        matrix,
        np.linalg.inv(matrix),
        matrix.T,
        np.linalg.inv(matrix.T),
        matrix @ matrix.T,
        np.linalg.inv(matrix @ matrix.T)
    ]
    
    return min(get_heuristic(v) for v in variants)

```

---

## Ancestor #5: Organism 57*

| Property | Value |
|----------|-------|
| **ID** | 57* |
| **Fitness** | 0.88173780 |
| **Best at Time** | 0.88173780 |
| **Parent ID** | 42 |
| **Was Best When Created** | Yes |

### Solution Code

```python

def heuristic(matrix):
    """Improved heuristic considering column sums and variants."""
    import numpy as np
    
    def get_heuristic(m):
        col_sums = np.sum(m, axis=0)
        nonzeros = np.count_nonzero(m)
        # Sort column sums to make order irrelevant
        sorted_cols = tuple(sorted(col_sums))
        # Add small term for column weight distribution
        col_imbalance = np.sum(np.abs(col_sums - 1))
        return (nonzeros, col_imbalance, *sorted_cols)
    
    h_original = get_heuristic(matrix)
    h_inverse = get_heuristic(np.linalg.inv(matrix))
    h_transpose = get_heuristic(matrix.T)
    
    return min(h_original, h_inverse, h_transpose)

```

---

## Ancestor #6: Organism 42*

| Property | Value |
|----------|-------|
| **ID** | 42* |
| **Fitness** | 0.87499566 |
| **Best at Time** | 0.87499566 |
| **Parent ID** | 7 |
| **Was Best When Created** | Yes |

### Solution Code

```python

def heuristic(matrix):
    """Sum of non-zero entries across all matrix variants."""
    import numpy as np
    
    def count_nonzeros(m):
        return np.count_nonzero(m)
    
    h_original = count_nonzeros(matrix)
    h_inverse = count_nonzeros(np.linalg.inv(matrix))
    h_transpose = count_nonzeros(matrix.T)
    
    return min(h_original, h_inverse, h_transpose)

```

---

## Ancestor #7: Organism 7

| Property | Value |
|----------|-------|
| **ID** | 7 |
| **Fitness** | -0.16898660 |
| **Best at Time** | 0.86588408 |
| **Parent ID** | 2 |
| **Was Best When Created** | No |

### Solution Code

```python

def heuristic(matrix):
    """Sum of logarithms of column sums, considering inverse/transpose variants."""
    import numpy as np
    def calculate_heuristic(m):
        col_sums = np.sum(m, axis=0)
        col_sums = np.maximum(col_sums, 1e-10)
        return float(np.sum(np.log(col_sums)))
    
    h_original = calculate_heuristic(matrix)
    h_inverse = calculate_heuristic(np.linalg.inv(matrix))
    h_transpose = calculate_heuristic(matrix.T)
    
    return min(h_original, h_inverse, h_transpose)

```

---

## Ancestor #8: Organism 2*

| Property | Value |
|----------|-------|
| **ID** | 2* |
| **Fitness** | 0.78484999 |
| **Best at Time** | 0.78484999 |
| **Parent ID** | None |
| **Was Best When Created** | Yes |

### Solution Code

```python
def heuristic(matrix):
    """Sum of logarithms of column sums (H_prod metric)."""
    import numpy as np
    col_sums = np.sum(matrix, axis=0)
    col_sums = np.maximum(col_sums, 1e-10)  # Avoid log(0)
    return float(np.sum(np.log(col_sums)))

```

---
