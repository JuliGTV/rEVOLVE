# Best Organism Ancestry Analysis

This document traces the complete ancestry of the fittest organism (ID: 101) with fitness 9.20000000.

Each section shows an ancestor in the lineage, from the fittest organism back to the original ancestor.
Organisms marked with * were the best fitness when they were created.

---

## Ancestor #1: Organism 101*

| Property | Value |
|----------|-------|
| **ID** | 101* |
| **Fitness** | 9.20000000 |
| **Best at Time** | 9.20000000 |
| **Parent ID** | 15 |
| **Was Best When Created** | Yes |

### Solution Code

```python

def heuristic(matrix):
    import numpy as np
    n = matrix.shape[0] // 2
    U = matrix[:n, :n]
    Uinv = matrix[n:, n:].T
    a = np.sum(U) - n
    b = np.sum(Uinv) - n
    return (min(a, b), max(a, b))

```

---

## Ancestor #2: Organism 15

| Property | Value |
|----------|-------|
| **ID** | 15 |
| **Fitness** | 0.00000000 |
| **Best at Time** | 4.00998386 |
| **Parent ID** | 9 |
| **Was Best When Created** | No |

### Solution Code

```python

def heuristic(matrix):
    import numpy as np
    n = matrix.shape[0] // 2
    parity_matrix = matrix[:n, :n]
    
    def binMatInv(mat):
        return np.linalg.inv(mat).astype(int) % 2
    
    inv_parity = binMatInv(parity_matrix)
    
    def get_metrics(mat):
        col_sums = np.sum(mat, axis=0)
        row_sums = np.sum(mat, axis=1)
        log_col = np.log(col_sums + 1e-10)
        log_row = np.log(row_sums + 1e-10)
        col_vars = np.var(mat, axis=0)
        row_vars = np.var(mat, axis=1)
        return np.concatenate((
            col_sums,
            row_sums,
            log_col,
            log_row,
            col_vars,
            row_vars,
            [np.sum(np.abs(mat))]
        ))
    
    metrics = np.concatenate((
        get_metrics(parity_matrix),
        get_metrics(inv_parity)
    ))
    return tuple(sorted(metrics)[::2])  # Take every other metric to keep tuple size reasonable

```

---

## Ancestor #3: Organism 9

| Property | Value |
|----------|-------|
| **ID** | 9 |
| **Fitness** | 0.00000000 |
| **Best at Time** | 2.19241126 |
| **Parent ID** | 3 |
| **Was Best When Created** | No |

### Solution Code

```python

def heuristic(matrix):
    import numpy as np
    n = matrix.shape[0] // 2
    parity_matrix = matrix[:n, :n]
    inv_parity = binMatInv(parity_matrix)
    
    def get_metrics(mat):
        col_sums = np.sum(mat, axis=0)
        row_sums = np.sum(mat, axis=1)
        log_col = np.log(col_sums + 1e-10)
        log_row = np.log(row_sums + 1e-10)
        return np.concatenate((col_sums, row_sums, log_col, log_row))
    
    metrics = np.concatenate((
        get_metrics(parity_matrix),
        get_metrics(parity_matrix.T),
        get_metrics(inv_parity),
        get_metrics(inv_parity.T)
    ))
    return tuple(sorted(metrics))

```

---

## Ancestor #4: Organism 3*

| Property | Value |
|----------|-------|
| **ID** | 3* |
| **Fitness** | 1.92228573 |
| **Best at Time** | 1.92228573 |
| **Parent ID** | 1 |
| **Was Best When Created** | Yes |

### Solution Code

```python

def heuristic(matrix):
    import numpy as np
    n = matrix.shape[0] // 2
    parity_matrix = matrix[:n, :n]
    col_sums = np.sum(parity_matrix, axis=0)
    row_sums = np.sum(parity_matrix, axis=1)
    trans_col_sums = np.sum(parity_matrix.T, axis=0)
    trans_row_sums = np.sum(parity_matrix.T, axis=1)
    return tuple(sorted(np.concatenate((col_sums, row_sums, trans_col_sums, trans_row_sums))))

```

---

## Ancestor #5: Organism 1

| Property | Value |
|----------|-------|
| **ID** | 1 |
| **Fitness** | -4.91145062 |
| **Best at Time** | -4.91145062 |
| **Parent ID** | None |
| **Was Best When Created** | No |

### Solution Code

```python

def heuristic(matrix):
    import numpy as np
    return tuple(sorted(np.concatenate((np.sum(matrix,axis=0),np.sum(matrix,axis=0))))) 

```

---
