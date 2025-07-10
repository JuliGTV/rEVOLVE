# Best Organism Ancestry Analysis

This document traces the complete ancestry of the fittest organism (ID: 23) with fitness 31.93333333.

Each section shows an ancestor in the lineage, from the fittest organism back to the original ancestor.
Organisms marked with * were the best fitness when they were created.

---

## Ancestor #1: Organism 23*

| Property | Value |
|----------|-------|
| **ID** | 23* |
| **Fitness** | 31.93333333 |
| **Best at Time** | 31.93333333 |
| **Parent ID** | 2 |
| **Was Best When Created** | Yes |

### Solution Code

```python

def heuristic(matrix):
    import numpy as np
    n = matrix.shape[0] // 2
    original = matrix[:n,:n]
    inverse = matrix[n:,n:]
    
    def weighted_sums(m):
        col_sums = np.sum(m, axis=0)
        row_sums = np.sum(m, axis=1)
        log_col = np.log(col_sums + 1)
        log_row = np.log(row_sums + 1)
        return np.concatenate((col_sums, row_sums, log_col, log_row))
    
    original_sums = weighted_sums(original)
    inverse_sums = weighted_sums(inverse)
    
    return tuple(np.minimum(original_sums, inverse_sums))

```

---

## Ancestor #2: Organism 2*

| Property | Value |
|----------|-------|
| **ID** | 2* |
| **Fitness** | 4.83333333 |
| **Best at Time** | 4.83333333 |
| **Parent ID** | 1 |
| **Was Best When Created** | Yes |

### Solution Code

```python

def heuristic(matrix):
    import numpy as np
    n = matrix.shape[0] // 2
    original = matrix[:n,:n]
    inverse = matrix[n:,n:]
    return tuple(np.concatenate((np.sum(original,axis=0), np.sum(original,axis=1), np.sum(inverse,axis=0), np.sum(inverse,axis=1))))

```

---

## Ancestor #3: Organism 1

| Property | Value |
|----------|-------|
| **ID** | 1 |
| **Fitness** | -2.10000000 |
| **Best at Time** | -2.10000000 |
| **Parent ID** | None |
| **Was Best When Created** | No |

### Solution Code

```python

def heuristic(matrix):
    import numpy as np
    return tuple(np.concatenate((np.sum(matrix,axis=0),np.sum(matrix,axis=0)))) 

```

---
