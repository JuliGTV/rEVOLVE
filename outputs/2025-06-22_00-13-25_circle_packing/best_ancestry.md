# Best Organism Ancestry Analysis

This document traces the complete ancestry of the fittest organism (ID: 9067) with fitness 2.63598309.

Each section shows an ancestor in the lineage, from the fittest organism back to the original ancestor.
Organisms marked with * were the best fitness when they were created.

---

## Ancestor #1: Organism 9067*

| Property | Value |
|----------|-------|
| **ID** | 9067* |
| **Fitness** | 2.63598309 |
| **Best at Time** | 2.63598309 |
| **Parent ID** | 9021 |
| **Was Best When Created** | Yes |

### Solution Code

```python

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Place 4 larger corner circles with slightly increased radius
    corners = [(0,0), (0,1), (1,0), (1,1)]
    for i in range(4):
        centers[i] = corners[i]
        radii[i] = 0.233  # Slightly increased
    
    # Place 12 edge circles with optimized positions and radii
    edge_positions = [0.29, 0.455, 0.71]  # Adjusted spacing
    edge_radii = [0.143, 0.129, 0.123]  # More optimized variation
    for i in range(3):
        centers[4+i] = [edge_positions[i], 0]
        centers[7+i] = [edge_positions[i], 1]
        centers[10+i] = [0, edge_positions[i]]
        centers[13+i] = [1, edge_positions[i]]
        radii[4+i] = edge_radii[i]
        radii[7+i] = edge_radii[i]
        radii[10+i] = edge_radii[i]
        radii[13+i] = edge_radii[i]
    
    # Place remaining 10 circles in a more optimized pattern
    inner_positions = [
        (0.24, 0.24), (0.455, 0.24), (0.665, 0.24), (0.865, 0.24),
        (0.345, 0.415), (0.555, 0.415), (0.765, 0.415),
        (0.24, 0.565), (0.455, 0.565), (0.665, 0.565)
    ]
    inner_radii = [0.113, 0.109, 0.106, 0.102,
                  0.108, 0.106, 0.103,
                  0.107, 0.105, 0.102]
    for idx in range(16, 26):
        centers[idx] = inner_positions[idx-16]
        radii[idx] = inner_radii[idx-16]
    
    # Optimization with constraints
    def constraint_boundary(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        return np.concatenate([centers[:,0] - radii, 
                             centers[:,1] - radii,
                             1 - centers[:,0] - radii,
                             1 - centers[:,1] - radii])
    
    def constraint_overlap(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        constraints = []
        for i in range(26):
            for j in range(i+1, 26):
                dist = np.linalg.norm(centers[i] - centers[j])
                constraints.append(dist - radii[i] - radii[j])
        return np.array(constraints)
    
    def objective(x):
        return -np.sum(x[52:])
    
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0,1)]*52 + [(0, 0.25)]*26
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]
    
    # Fine-tuned optimization parameters for better convergence
    res = minimize(objective, x0, bounds=bounds, constraints=constraints,
                  method='SLSQP', options={'maxiter': 150000, 'ftol': 1e-13, 'eps': 1e-11})
    
    optimized = res.x
    centers = optimized[:52].reshape(26, 2)
    radii = optimized[52:]
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

```

---

## Ancestor #2: Organism 9021*

| Property | Value |
|----------|-------|
| **ID** | 9021* |
| **Fitness** | 2.63598308 |
| **Best at Time** | 2.63598308 |
| **Parent ID** | 8882 |
| **Was Best When Created** | Yes |

### Solution Code

```python

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Place 4 larger corner circles with increased radius
    corners = [(0,0), (0,1), (1,0), (1,1)]
    for i in range(4):
        centers[i] = corners[i]
        radii[i] = 0.232  # Slightly increased
    
    # Place 12 edge circles with optimized positions and radii
    edge_positions = [0.295, 0.46, 0.705]  # Adjusted spacing
    edge_radii = [0.142, 0.128, 0.122]  # More optimized variation
    for i in range(3):
        centers[4+i] = [edge_positions[i], 0]
        centers[7+i] = [edge_positions[i], 1]
        centers[10+i] = [0, edge_positions[i]]
        centers[13+i] = [1, edge_positions[i]]
        radii[4+i] = edge_radii[i]
        radii[7+i] = edge_radii[i]
        radii[10+i] = edge_radii[i]
        radii[13+i] = edge_radii[i]
    
    # Place remaining 10 circles in a more optimized pattern
    inner_positions = [
        (0.245, 0.245), (0.46, 0.245), (0.66, 0.245), (0.86, 0.245),
        (0.35, 0.41), (0.56, 0.41), (0.76, 0.41),
        (0.245, 0.56), (0.46, 0.56), (0.66, 0.56)
    ]
    inner_radii = [0.112, 0.108, 0.105, 0.101,
                  0.107, 0.105, 0.102,
                  0.106, 0.104, 0.101]
    for idx in range(16, 26):
        centers[idx] = inner_positions[idx-16]
        radii[idx] = inner_radii[idx-16]
    
    # Optimization with constraints
    def constraint_boundary(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        return np.concatenate([centers[:,0] - radii, 
                             centers[:,1] - radii,
                             1 - centers[:,0] - radii,
                             1 - centers[:,1] - radii])
    
    def constraint_overlap(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        constraints = []
        for i in range(26):
            for j in range(i+1, 26):
                dist = np.linalg.norm(centers[i] - centers[j])
                constraints.append(dist - radii[i] - radii[j])
        return np.array(constraints)
    
    def objective(x):
        return -np.sum(x[52:])
    
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0,1)]*52 + [(0, 0.25)]*26
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]
    
    # Fine-tuned optimization parameters
    res = minimize(objective, x0, bounds=bounds, constraints=constraints,
                  method='SLSQP', options={'maxiter': 100000, 'ftol': 1e-12, 'eps': 1e-10})
    
    optimized = res.x
    centers = optimized[:52].reshape(26, 2)
    radii = optimized[52:]
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

```

---

## Ancestor #3: Organism 8882*

| Property | Value |
|----------|-------|
| **ID** | 8882* |
| **Fitness** | 2.63598308 |
| **Best at Time** | 2.63598308 |
| **Parent ID** | 7482 |
| **Was Best When Created** | Yes |

### Solution Code

```python

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Place 4 larger corner circles with increased radius
    corners = [(0,0), (0,1), (1,0), (1,1)]
    for i in range(4):
        centers[i] = corners[i]
        radii[i] = 0.23  # Increased from 0.225
    
    # Place 12 edge circles with optimized positions and radii
    edge_positions = [0.30, 0.45, 0.70]  # Adjusted spacing
    edge_radii = [0.14, 0.125, 0.12]  # More optimized variation
    for i in range(3):
        centers[4+i] = [edge_positions[i], 0]
        centers[7+i] = [edge_positions[i], 1]
        centers[10+i] = [0, edge_positions[i]]
        centers[13+i] = [1, edge_positions[i]]
        radii[4+i] = edge_radii[i]
        radii[7+i] = edge_radii[i]
        radii[10+i] = edge_radii[i]
        radii[13+i] = edge_radii[i]
    
    # Place remaining 10 circles in a more hexagonal pattern
    inner_positions = [
        (0.25, 0.25), (0.45, 0.25), (0.65, 0.25), (0.85, 0.25),
        (0.35, 0.40), (0.55, 0.40), (0.75, 0.40),
        (0.25, 0.55), (0.45, 0.55), (0.65, 0.55)
    ]
    inner_radii = [0.108, 0.105, 0.102, 0.098,
                  0.104, 0.102, 0.099,
                  0.103, 0.101, 0.098]
    for idx in range(16, 26):
        centers[idx] = inner_positions[idx-16]
        radii[idx] = inner_radii[idx-16]
    
    # Optimization with constraints
    def constraint_boundary(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        return np.concatenate([centers[:,0] - radii, 
                             centers[:,1] - radii,
                             1 - centers[:,0] - radii,
                             1 - centers[:,1] - radii])
    
    def constraint_overlap(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        constraints = []
        for i in range(26):
            for j in range(i+1, 26):
                dist = np.linalg.norm(centers[i] - centers[j])
                constraints.append(dist - radii[i] - radii[j])
        return np.array(constraints)
    
    def objective(x):
        return -np.sum(x[52:])
    
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0,1)]*52 + [(0, 0.25)]*26
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]
    
    # More optimization iterations with adjusted tolerance
    res = minimize(objective, x0, bounds=bounds, constraints=constraints,
                  method='SLSQP', options={'maxiter': 50000, 'ftol': 1e-10, 'eps': 1e-8})
    
    optimized = res.x
    centers = optimized[:52].reshape(26, 2)
    radii = optimized[52:]
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

```

---

## Ancestor #4: Organism 7482

| Property | Value |
|----------|-------|
| **ID** | 7482 |
| **Fitness** | 2.61783349 |
| **Best at Time** | 2.63303522 |
| **Parent ID** | 2558 |
| **Was Best When Created** | No |

### Solution Code

```python

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Place 4 larger corner circles with increased radius
    corners = [(0,0), (0,1), (1,0), (1,1)]
    for i in range(4):
        centers[i] = corners[i]
        radii[i] = 0.225  # Slightly increased from 0.22
    
    # Place 12 edge circles with more varied radii and positions
    edge_positions = [0.31, 0.44, 0.69]  # Adjusted spacing
    edge_radii = [0.135, 0.125, 0.115]  # More variation
    for i in range(3):
        centers[4+i] = [edge_positions[i], 0]
        centers[7+i] = [edge_positions[i], 1]
        centers[10+i] = [0, edge_positions[i]]
        centers[13+i] = [1, edge_positions[i]]
        radii[4+i] = edge_radii[i]
        radii[7+i] = edge_radii[i]
        radii[10+i] = edge_radii[i]
        radii[13+i] = edge_radii[i]
    
    # Place remaining 10 circles with varied radii in optimized positions
    inner_positions = [
        (0.22, 0.22), (0.42, 0.22), (0.62, 0.22), (0.82, 0.22),
        (0.32, 0.38), (0.52, 0.38), (0.72, 0.38),
        (0.22, 0.52), (0.42, 0.52), (0.62, 0.52)
    ]
    inner_radii = [0.105, 0.102, 0.098, 0.095,
                  0.101, 0.099, 0.097,
                  0.100, 0.098, 0.096]
    for idx in range(16, 26):
        centers[idx] = inner_positions[idx-16]
        radii[idx] = inner_radii[idx-16]
    
    # Optimization with constraints
    def constraint_boundary(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        return np.concatenate([centers[:,0] - radii, 
                             centers[:,1] - radii,
                             1 - centers[:,0] - radii,
                             1 - centers[:,1] - radii])
    
    def constraint_overlap(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        constraints = []
        for i in range(26):
            for j in range(i+1, 26):
                dist = np.linalg.norm(centers[i] - centers[j])
                constraints.append(dist - radii[i] - radii[j])
        return np.array(constraints)
    
    def objective(x):
        return -np.sum(x[52:])
    
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0,1)]*52 + [(0, 0.25)]*26
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]
    
    # More optimization iterations with adjusted tolerance
    res = minimize(objective, x0, bounds=bounds, constraints=constraints,
                  method='SLSQP', options={'maxiter': 35000, 'ftol': 1e-9})
    
    optimized = res.x
    centers = optimized[:52].reshape(26, 2)
    radii = optimized[52:]
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

```

---

## Ancestor #5: Organism 2558

| Property | Value |
|----------|-------|
| **ID** | 2558 |
| **Fitness** | 2.61783349 |
| **Best at Time** | 2.63072975 |
| **Parent ID** | 2217 |
| **Was Best When Created** | No |

### Solution Code

```python

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Place 4 larger corner circles
    corners = [(0,0), (0,1), (1,0), (1,1)]
    for i in range(4):
        centers[i] = corners[i]
        radii[i] = 0.22  # Increased from 0.215
    
    # Place 12 edge circles with varied radii
    edge_positions = [0.32, 0.45, 0.68]  # Adjusted positions
    edge_radii = [0.13, 0.12, 0.11]  # Varied sizes
    for i in range(3):
        centers[4+i] = [edge_positions[i], 0]
        centers[7+i] = [edge_positions[i], 1]
        centers[10+i] = [0, edge_positions[i]]
        centers[13+i] = [1, edge_positions[i]]
        radii[4+i] = edge_radii[i]
        radii[7+i] = edge_radii[i]
        radii[10+i] = edge_radii[i]
        radii[13+i] = edge_radii[i]
    
    # Place remaining 10 circles in optimized positions
    inner_positions = [
        (0.2, 0.2), (0.4, 0.2), (0.6, 0.2), (0.8, 0.2),
        (0.3, 0.35), (0.5, 0.35), (0.7, 0.35),
        (0.2, 0.5), (0.4, 0.5), (0.6, 0.5)
    ]
    for idx in range(16, 26):
        centers[idx] = inner_positions[idx-16]
        radii[idx] = 0.095  # Slightly increased
    
    # Optimization with constraints
    def constraint_boundary(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        return np.concatenate([centers[:,0] - radii, 
                             centers[:,1] - radii,
                             1 - centers[:,0] - radii,
                             1 - centers[:,1] - radii])
    
    def constraint_overlap(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        constraints = []
        for i in range(26):
            for j in range(i+1, 26):
                dist = np.linalg.norm(centers[i] - centers[j])
                constraints.append(dist - radii[i] - radii[j])
        return np.array(constraints)
    
    def objective(x):
        return -np.sum(x[52:])
    
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0,1)]*52 + [(0, 0.25)]*26
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]
    
    # More optimization iterations
    res = minimize(objective, x0, bounds=bounds, constraints=constraints,
                  method='SLSQP', options={'maxiter': 30000, 'ftol': 1e-8})
    
    optimized = res.x
    centers = optimized[:52].reshape(26, 2)
    radii = optimized[52:]
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

```

---

## Ancestor #6: Organism 2217

| Property | Value |
|----------|-------|
| **ID** | 2217 |
| **Fitness** | 2.61492641 |
| **Best at Time** | 2.63058838 |
| **Parent ID** | 1944 |
| **Was Best When Created** | No |

### Solution Code

```python

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Place 4 large corner circles with increased radius
    corners = [(0,0), (0,1), (1,0), (1,1)]
    for i in range(4):
        centers[i] = corners[i]
        radii[i] = 0.215
    
    # Place 12 edge circles with adjusted positions and radii
    edge_positions = [0.34, 0.5, 0.66]  # Farther from corners
    for i in range(3):
        centers[4+i] = [edge_positions[i], 0]
        centers[7+i] = [edge_positions[i], 1]
        centers[10+i] = [0, edge_positions[i]]
        centers[13+i] = [1, edge_positions[i]]
        radii[4+i] = 0.12  # Reduced to avoid overlaps
        radii[7+i] = 0.12
        radii[10+i] = 0.12
        radii[13+i] = 0.12
    
    # Place remaining 10 circles with increased radius
    hex_positions = []
    for i in range(4):
        for j in range(4):
            x = 0.17 + i*0.21
            y = 0.17 + j*0.21
            if j % 2 == 1:
                x += 0.105
            if len(hex_positions) < 10:  # Ensure exactly 10 circles
                hex_positions.append((x,y))
    
    for idx in range(16, 26):
        centers[idx] = hex_positions[idx-16]
        radii[idx] = 0.09  # Increased radius
    
    # Optimization with constraints
    def constraint_boundary(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        return np.concatenate([centers[:,0] - radii, 
                             centers[:,1] - radii,
                             1 - centers[:,0] - radii,
                             1 - centers[:,1] - radii])
    
    def constraint_overlap(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        constraints = []
        for i in range(26):
            for j in range(i+1, 26):
                dist = np.linalg.norm(centers[i] - centers[j])
                constraints.append(dist - radii[i] - radii[j])
        return np.array(constraints)
    
    def objective(x):
        return -np.sum(x[52:])
    
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0,1)]*52 + [(0, 0.25)]*26
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]
    
    # Increased iterations for better convergence
    res = minimize(objective, x0, bounds=bounds, constraints=constraints,
                  method='SLSQP', options={'maxiter': 20000})
    
    optimized = res.x
    centers = optimized[:52].reshape(26, 2)
    radii = optimized[52:]
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

```

---

## Ancestor #7: Organism 1944*

| Property | Value |
|----------|-------|
| **ID** | 1944* |
| **Fitness** | 2.63058830 |
| **Best at Time** | 2.63058830 |
| **Parent ID** | 1576 |
| **Was Best When Created** | Yes |

### Solution Code

```python

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Place 4 large corner circles with optimized radius
    corners = [(0,0), (0,1), (1,0), (1,1)]
    for i in range(4):
        centers[i] = corners[i]
        radii[i] = 0.212
    
    # Place 8 medium edge circles with optimized positions and radii
    edge_positions = [0.23, 0.5, 0.77]
    for i in range(3):
        centers[4+i] = [edge_positions[i], 0]
        centers[7+i] = [edge_positions[i], 1]
        centers[10+i] = [0, edge_positions[i]]
        centers[13+i] = [1, edge_positions[i]]
        radii[4+i] = 0.126
        radii[7+i] = 0.126
        radii[10+i] = 0.128
        radii[13+i] = 0.128
    
    # Place remaining 14 circles in optimized hexagonal pattern
    hex_positions = []
    for i in range(4):
        for j in range(4):
            x = 0.17 + i*0.21
            y = 0.17 + j*0.21
            if j % 2 == 1:
                x += 0.105
            if i < 3 or j < 3:  # Only place 14 circles
                hex_positions.append((x,y))
    
    for idx in range(16, 26):
        centers[idx] = hex_positions[idx-16]
        radii[idx] = 0.088 if idx < 22 else 0.078
    
    # Optimization with constraints
    def constraint_boundary(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        return np.concatenate([centers[:,0] - radii, 
                             centers[:,1] - radii,
                             1 - centers[:,0] - radii,
                             1 - centers[:,1] - radii])
    
    def constraint_overlap(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        constraints = []
        for i in range(26):
            for j in range(i+1, 26):
                dist = np.linalg.norm(centers[i] - centers[j])
                constraints.append(dist - radii[i] - radii[j])
        return np.array(constraints)
    
    def objective(x):
        return -np.sum(x[52:])
    
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0,1)]*52 + [(0, 0.25)]*26
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]
    
    res = minimize(objective, x0, bounds=bounds, constraints=constraints,
                  method='SLSQP', options={'maxiter': 10000})
    
    optimized = res.x
    centers = optimized[:52].reshape(26, 2)
    radii = optimized[52:]
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

```

---

## Ancestor #8: Organism 1576

| Property | Value |
|----------|-------|
| **ID** | 1576 |
| **Fitness** | 2.63058829 |
| **Best at Time** | 2.63058830 |
| **Parent ID** | 1517 |
| **Was Best When Created** | No |

### Solution Code

```python

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Place 4 large corner circles with optimized radius
    corners = [(0,0), (0,1), (1,0), (1,1)]
    for i in range(4):
        centers[i] = corners[i]
        radii[i] = 0.21
    
    # Place 8 medium edge circles with optimized positions and radii
    edge_positions = [0.22, 0.5, 0.78]
    for i in range(3):
        centers[4+i] = [edge_positions[i], 0]
        centers[7+i] = [edge_positions[i], 1]
        centers[10+i] = [0, edge_positions[i]]
        centers[13+i] = [1, edge_positions[i]]
        radii[4+i] = 0.128
        radii[7+i] = 0.128
        radii[10+i] = 0.13
        radii[13+i] = 0.13
    
    # Place remaining 14 circles in optimized hexagonal pattern
    hex_positions = []
    for i in range(4):
        for j in range(4):
            x = 0.16 + i*0.22
            y = 0.16 + j*0.22
            if j % 2 == 1:
                x += 0.11
            hex_positions.append((x,y))
    
    for idx in range(16, 26):
        centers[idx] = hex_positions[idx-16]
        radii[idx] = 0.085 if idx < 22 else 0.075
    
    # Optimization with constraints
    def constraint_boundary(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        return np.concatenate([centers[:,0] - radii, 
                             centers[:,1] - radii,
                             1 - centers[:,0] - radii,
                             1 - centers[:,1] - radii])
    
    def constraint_overlap(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        constraints = []
        for i in range(26):
            for j in range(i+1, 26):
                dist = np.linalg.norm(centers[i] - centers[j])
                constraints.append(dist - radii[i] - radii[j])
        return np.array(constraints)
    
    def objective(x):
        return -np.sum(x[52:])
    
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0,1)]*52 + [(0, 0.25)]*26
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]
    
    res = minimize(objective, x0, bounds=bounds, constraints=constraints,
                  method='SLSQP', options={'maxiter': 10000})
    
    optimized = res.x
    centers = optimized[:52].reshape(26, 2)
    radii = optimized[52:]
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

```

---

## Ancestor #9: Organism 1517*

| Property | Value |
|----------|-------|
| **ID** | 1517* |
| **Fitness** | 2.63058829 |
| **Best at Time** | 2.63058829 |
| **Parent ID** | 804 |
| **Was Best When Created** | Yes |

### Solution Code

```python

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Place 4 large corner circles with optimized radius
    corners = [(0,0), (0,1), (1,0), (1,1)]
    for i in range(4):
        centers[i] = corners[i]
        radii[i] = 0.208
    
    # Place 8 medium edge circles with optimized positions and radii
    edge_positions = [0.22, 0.5, 0.78]
    for i in range(3):
        centers[4+i] = [edge_positions[i], 0]
        centers[7+i] = [edge_positions[i], 1]
        centers[10+i] = [0, edge_positions[i]]
        centers[13+i] = [1, edge_positions[i]]
        radii[4+i] = 0.126
        radii[7+i] = 0.126
        radii[10+i] = 0.126
        radii[13+i] = 0.126
    
    # Place remaining 14 circles in optimized hexagonal pattern
    hex_positions = []
    for i in range(4):
        for j in range(4):
            x = 0.16 + i*0.22
            y = 0.16 + j*0.22
            if j % 2 == 1:
                x += 0.11
            hex_positions.append((x,y))
    
    for idx in range(16, 26):
        centers[idx] = hex_positions[idx-16]
        radii[idx] = 0.082 if idx < 22 else 0.072
    
    # Optimization with constraints
    def constraint_boundary(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        return np.concatenate([centers[:,0] - radii, 
                             centers[:,1] - radii,
                             1 - centers[:,0] - radii,
                             1 - centers[:,1] - radii])
    
    def constraint_overlap(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        constraints = []
        for i in range(26):
            for j in range(i+1, 26):
                dist = np.linalg.norm(centers[i] - centers[j])
                constraints.append(dist - radii[i] - radii[j])
        return np.array(constraints)
    
    def objective(x):
        return -np.sum(x[52:])
    
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0,1)]*52 + [(0, 0.25)]*26
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]
    
    res = minimize(objective, x0, bounds=bounds, constraints=constraints,
                  method='SLSQP', options={'maxiter': 10000})
    
    optimized = res.x
    centers = optimized[:52].reshape(26, 2)
    radii = optimized[52:]
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

```

---

## Ancestor #10: Organism 804

| Property | Value |
|----------|-------|
| **ID** | 804 |
| **Fitness** | 2.61831171 |
| **Best at Time** | 2.62675676 |
| **Parent ID** | 187 |
| **Was Best When Created** | No |

### Solution Code

```python

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Place 4 large corner circles with slightly adjusted radius
    corners = [(0,0), (0,1), (1,0), (1,1)]
    for i in range(4):
        centers[i] = corners[i]
        radii[i] = 0.207
    
    # Place 8 medium edge circles with adjusted positions and radii
    edge_positions = [0.23, 0.5, 0.77]
    for i in range(3):
        centers[4+i] = [edge_positions[i], 0]
        centers[7+i] = [edge_positions[i], 1]
        centers[10+i] = [0, edge_positions[i]]
        centers[13+i] = [1, edge_positions[i]]
        radii[4+i] = 0.125
        radii[7+i] = 0.125
        radii[10+i] = 0.125
        radii[13+i] = 0.125
    
    # Place remaining 14 circles in hexagonal pattern
    hex_positions = []
    for i in range(4):
        for j in range(4):
            x = 0.15 + i*0.23
            y = 0.15 + j*0.23
            if j % 2 == 1:
                x += 0.115
            hex_positions.append((x,y))
    
    for idx in range(16, 26):
        centers[idx] = hex_positions[idx-16]
        radii[idx] = 0.08 if idx < 22 else 0.07
    
    # Optimization with constraints
    def constraint_boundary(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        return np.concatenate([centers[:,0] - radii, 
                             centers[:,1] - radii,
                             1 - centers[:,0] - radii,
                             1 - centers[:,1] - radii])
    
    def constraint_overlap(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        constraints = []
        for i in range(26):
            for j in range(i+1, 26):
                dist = np.linalg.norm(centers[i] - centers[j])
                constraints.append(dist - radii[i] - radii[j])
        return np.array(constraints)
    
    def objective(x):
        return -np.sum(x[52:])
    
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0,1)]*52 + [(0, 0.25)]*26
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]
    
    res = minimize(objective, x0, bounds=bounds, constraints=constraints,
                  method='SLSQP', options={'maxiter': 5000})
    
    optimized = res.x
    centers = optimized[:52].reshape(26, 2)
    radii = optimized[52:]
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

```

---

## Ancestor #11: Organism 187

| Property | Value |
|----------|-------|
| **ID** | 187 |
| **Fitness** | 2.60621872 |
| **Best at Time** | 2.62675676 |
| **Parent ID** | 151 |
| **Was Best When Created** | No |

### Solution Code

```python

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Place 4 large corner circles
    corners = [(0,0), (0,1), (1,0), (1,1)]
    for i in range(4):
        centers[i] = corners[i]
        radii[i] = 0.2
    
    # Place 8 medium edge circles
    edge_positions = [0.25, 0.5, 0.75]
    for i in range(3):
        centers[4+i] = [edge_positions[i], 0]
        centers[7+i] = [edge_positions[i], 1]
        centers[10+i] = [0, edge_positions[i]]
        centers[13+i] = [1, edge_positions[i]]
        radii[4+i] = 0.12
        radii[7+i] = 0.12
        radii[10+i] = 0.12
        radii[13+i] = 0.12
    
    # Place remaining 14 circles in center grid
    grid_positions = [0.2, 0.4, 0.6, 0.8]
    for i in range(4):
        for j in range(4):
            idx = 16 + i*4 + j
            if idx < 26:
                centers[idx] = [grid_positions[i], grid_positions[j]]
                radii[idx] = 0.07
    
    # Optimization with constraints
    def constraint_boundary(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        return np.concatenate([centers[:,0] - radii, 
                             centers[:,1] - radii,
                             1 - centers[:,0] - radii,
                             1 - centers[:,1] - radii])
    
    def constraint_overlap(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        constraints = []
        for i in range(26):
            for j in range(i+1, 26):
                dist = np.linalg.norm(centers[i] - centers[j])
                constraints.append(dist - radii[i] - radii[j])
        return np.array(constraints)
    
    def objective(x):
        return -np.sum(x[52:])
    
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0,1)]*52 + [(0, 0.25)]*26
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]
    
    res = minimize(objective, x0, bounds=bounds, constraints=constraints,
                  method='SLSQP', options={'maxiter': 5000})
    
    optimized = res.x
    centers = optimized[:52].reshape(26, 2)
    radii = optimized[52:]
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

```

---

## Ancestor #12: Organism 151

| Property | Value |
|----------|-------|
| **ID** | 151 |
| **Fitness** | 2.55832697 |
| **Best at Time** | 2.62451029 |
| **Parent ID** | 95 |
| **Was Best When Created** | No |

### Solution Code

```python

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Place larger central circle
    centers[0] = [0.5, 0.5]
    radii[0] = 0.22
    
    # Place 6 circles in hexagonal pattern with adjusted radius
    hex_radius = 0.32
    for i in range(6):
        angle = 2 * np.pi * i / 6
        centers[i+1] = [0.5 + hex_radius * np.cos(angle), 
                        0.5 + hex_radius * np.sin(angle)]
        radii[i+1] = 0.11
    
    # Place larger circles in the corners
    corners = [(0,0), (0,1), (1,0), (1,1)]
    for i in range(4):
        centers[i+7] = corners[i]
        radii[i+7] = 0.17
    
    # Place remaining circles along edges with adjusted positions and sizes
    edge_positions = [0.15, 0.35, 0.65, 0.85]
    for i in range(4):
        centers[11+i] = [edge_positions[i], 0]
        centers[15+i] = [edge_positions[i], 1]
        centers[19+i] = [0, edge_positions[i]]
        centers[22+i] = [1, edge_positions[i]]
        radii[11+i] = 0.08
        radii[15+i] = 0.08
        radii[19+i] = 0.08
        radii[22+i] = 0.08
    
    # Optimization with constraints
    def constraint_boundary(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        return np.concatenate([centers[:,0] - radii, 
                              centers[:,1] - radii,
                              1 - centers[:,0] - radii,
                              1 - centers[:,1] - radii])
    
    def constraint_overlap(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        constraints = []
        for i in range(26):
            for j in range(i+1, 26):
                dist = np.linalg.norm(centers[i] - centers[j])
                constraints.append(dist - radii[i] - radii[j])
        return np.array(constraints)
    
    def objective(x):
        return -np.sum(x[52:])
    
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0,1)]*52 + [(0, 0.3)]*26
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]
    
    res = minimize(objective, x0, bounds=bounds, constraints=constraints,
                  method='trust-constr', options={'maxiter': 2000})
    
    optimized = res.x
    centers = optimized[:52].reshape(26, 2)
    radii = optimized[52:]
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

```

---

## Ancestor #13: Organism 95*

| Property | Value |
|----------|-------|
| **ID** | 95* |
| **Fitness** | 2.60549165 |
| **Best at Time** | 2.60549165 |
| **Parent ID** | 37 |
| **Was Best When Created** | Yes |

### Solution Code

```python

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Place large central circle
    centers[0] = [0.5, 0.5]
    radii[0] = 0.2
    
    # Place 6 circles in hexagonal pattern around center
    hex_radius = 0.3
    for i in range(6):
        angle = 2 * np.pi * i / 6
        centers[i+1] = [0.5 + hex_radius * np.cos(angle), 
                        0.5 + hex_radius * np.sin(angle)]
        radii[i+1] = 0.1
    
    # Place 4 circles in the corners
    corners = [(0,0), (0,1), (1,0), (1,1)]
    for i in range(4):
        centers[i+7] = corners[i]
        radii[i+7] = 0.15
    
    # Place remaining circles along edges
    edge_positions = [0.2, 0.4, 0.6, 0.8]
    for i in range(4):
        centers[11+i] = [edge_positions[i], 0]
        centers[15+i] = [edge_positions[i], 1]
        centers[19+i] = [0, edge_positions[i]]
        centers[22+i] = [1, edge_positions[i]]  # Adjusted to avoid index overflow
        radii[11+i] = 0.07
        radii[15+i] = 0.07
        radii[19+i] = 0.07
        radii[22+i] = 0.07
    
    # Optimization with constraints
    def constraint_boundary(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        return np.concatenate([centers[:,0] - radii, 
                              centers[:,1] - radii,
                              1 - centers[:,0] - radii,
                              1 - centers[:,1] - radii])
    
    def constraint_overlap(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        constraints = []
        for i in range(26):
            for j in range(i+1, 26):
                dist = np.linalg.norm(centers[i] - centers[j])
                constraints.append(dist - radii[i] - radii[j])
        return np.array(constraints)
    
    def objective(x):
        return -np.sum(x[52:])
    
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0,1)]*52 + [(0, 0.25)]*26
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]
    
    res = minimize(objective, x0, bounds=bounds, constraints=constraints,
                  method='SLSQP', options={'maxiter': 1000})
    
    optimized = res.x
    centers = optimized[:52].reshape(26, 2)
    radii = optimized[52:]
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

```

---

## Ancestor #14: Organism 37

| Property | Value |
|----------|-------|
| **ID** | 37 |
| **Fitness** | 0.00000000 |
| **Best at Time** | 2.03203368 |
| **Parent ID** | 1 |
| **Was Best When Created** | No |

### Solution Code

```python

import numpy as np
from scipy.optimize import minimize

def run_packing():
    # Initial placement with variable radii and edge optimization
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Place large central circle
    centers[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    # Place 8 circles in hexagonal pattern around center
    hex_radius = 0.5 - 0.15
    for i in range(6):
        angle = 2 * np.pi * i / 6
        centers[i+1] = [0.5 + hex_radius * np.cos(angle), 
                        0.5 + hex_radius * np.sin(angle)]
        radii[i+1] = 0.12
    
    # Place 4 circles in the corners
    corners = [(0,0), (0,1), (1,0), (1,1)]
    for i in range(4):
        centers[i+7] = corners[i]
        radii[i+7] = 0.18
    
    # Place 4 circles along each edge (total 16)
    edge_positions = [0.2, 0.4, 0.6, 0.8]
    for i in range(4):
        centers[11+i] = [edge_positions[i], 0]
        centers[15+i] = [edge_positions[i], 1]
        centers[19+i] = [0, edge_positions[i]]
        centers[23+i] = [1, edge_positions[i]]
        radii[11+i] = 0.08
        radii[15+i] = 0.08
        radii[19+i] = 0.08
        radii[23+i] = 0.08
    
    # Optimization function
    def objective(x):
        centers = x[:52].reshape(26, 2)
        radii = x[52:]
        
        # Boundary constraints
        if np.any(centers - radii < 0) or np.any(centers + radii > 1):
            return -np.sum(radii) + 1000  # Penalty
        
        # Overlap constraints
        for i in range(26):
            for j in range(i+1, 26):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < radii[i] + radii[j]:
                    return -np.sum(radii) + 1000  # Penalty
        
        return -np.sum(radii)
    
    # Initial guess
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Bounds
    bounds = [(0,1)]*52 + [(0, 0.25)]*26
    
    # Optimize
    res = minimize(objective, x0, bounds=bounds, method='L-BFGS-B', 
                  options={'maxiter': 1000})
    
    optimized = res.x
    centers = optimized[:52].reshape(26, 2)
    radii = optimized[52:]
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

```

---

## Ancestor #15: Organism 1*

| Property | Value |
|----------|-------|
| **ID** | 1* |
| **Fitness** | 1.78046489 |
| **Best at Time** | 1.78046489 |
| **Parent ID** | None |
| **Was Best When Created** | Yes |

### Solution Code

```python
import numpy as np

def run_packing():
    """
    Construct arrangement of 26 circles in unit square to maximize sum of radii.
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # Central circle
    centers[0] = [0.5, 0.5]
    
    # Ring of 8 circles
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.25 * np.cos(angle), 0.5 + 0.25 * np.sin(angle)]
    
    # Ring of 16 circles
    for i in range(16):
        angle = 2 * np.pi * i / 16
        centers[i + 9] = [0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle)]
    
    # One additional circle
    centers[25] = [0.5, 0.85]
    
    # Ensure all circles are well within the unit square
    centers = np.clip(centers, 0.1, 0.9)
    
    # Calculate radii
    n = centers.shape[0]
    radii = np.ones(n) * 0.05  # Start with small radii
    
    # Limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y, 0.08)
    
    # Limit by distance to other circles
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if radii[i] + radii[j] > dist * 0.95:  # Leave small gap
                scale = (dist * 0.95) / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale
    
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

```

---
