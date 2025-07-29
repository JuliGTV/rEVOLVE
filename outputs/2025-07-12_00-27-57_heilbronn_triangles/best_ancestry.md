# Best Organism Ancestry Analysis

This document traces the complete ancestry of the fittest organism (ID: 1002) with fitness 0.03252312.

Each section shows an ancestor in the lineage, from the fittest organism back to the original ancestor.
Organisms marked with * were the best fitness when they were created.

---

## Ancestor #1: Organism 1002*

| Property | Value |
|----------|-------|
| **ID** | 1002* |
| **Fitness** | 0.03252312 |
| **Best at Time** | 0.03252312 |
| **Parent ID** | 835 |
| **Was Best When Created** | Yes |

### Solution Code

```python

from __future__ import annotations

import math
import random
import itertools
from typing import List, Tuple
import numpy as np

SQRT3 = math.sqrt(3.0)

def _area(p, q, r) -> float:
    return abs((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])) * 0.5

comb = list(itertools.combinations(range(11), 3))

def _min_triangle_area(pts: np.ndarray) -> float:
    best = float("inf")
    for tri in comb:
        i, j, k = tri
        a = _area(pts[i], pts[j], pts[k])
        if a < best:
            best = a
    return best

_A = np.array([0.0, 0.0])
_B = np.array([1.0, 0.0])
_C = np.array([0.5, SQRT3 / 2.0])

def _random_point() -> np.ndarray:
    while True:
        u, v = random.random(), random.random()
        if u + v <= 1.0:
            break
    return (1 - u - v) * _A + u * _B + v * _C

def _project_to_simplex(p: np.ndarray) -> np.ndarray:
    M = np.column_stack((_B - _A, _C - _A))
    uv = np.linalg.lstsq(M, p - _A, rcond=None)[0]
    u, v = uv
    w = 1.0 - u - v
    
    eps = 1e-8
    u = max(eps, min(1.0-eps, u))
    v = max(eps, min(1.0-eps, v))
    if u + v > 1.0 - eps:
        s = u + v
        u = (1.0 - eps) * u / s
        v = (1.0 - eps) * v / s
    
    return (1 - u - v) * _A + u * _B + v * _C

def _initial_configuration(n_pts: int = 11) -> np.ndarray:
    pts = []
    pts.append(_A.copy())
    pts.append(_B.copy())
    pts.append(_C.copy())
    
    golden_angle = math.pi * (3 - math.sqrt(5))
    for i in range(1, n_pts - 2):
        theta = golden_angle * i
        r = math.sqrt(i / (n_pts - 3))
        u = 0.5 + r * math.cos(theta) * 0.5
        v = 0.5 + r * math.sin(theta) * 0.5
        if u > 0 and v > 0 and u + v < 1:
            pts.append((1 - u - v) * _A + u * _B + v * _C)
    
    while len(pts) < n_pts:
        u = random.uniform(0.1, 0.9)
        v = random.uniform(0.1, 0.9)
        if u + v < 0.95:
            pt = (1 - u - v) * _A + u * _B + v * _C
            if not any(np.allclose(pt, x, atol=1e-6) for x in pts):
                pts.append(pt)
    return np.array(pts[:n_pts])

def _anneal(n_pts: int = 11,
            iters: int = 40000,
            start_temp: float = 0.2,
            end_temp: float = 1e-10,
            seed: int | None = 0) -> np.ndarray:
    rng = random.Random(seed)
    pts = _initial_configuration(n_pts)
    best_pts = pts.copy()
    best_val = _min_triangle_area(best_pts)

    for t in range(iters):
        T = start_temp * (end_temp / start_temp) ** (t / iters)
        step_size = 0.15 * (1 - t/iters) + 0.001
        idx = rng.randrange(n_pts)
        trial = pts.copy()
        trial[idx] += rng.gauss(0, step_size), rng.gauss(0, step_size)
        trial[idx] = _project_to_simplex(trial[idx])

        trial_val = _min_triangle_area(trial)
        delta = trial_val - best_val

        if delta > 0 or rng.random() < math.exp(delta / max(T, 1e-12)):
            pts = trial
            if trial_val > best_val:
                best_val = trial_val
                best_pts = trial.copy()

    return best_pts

def _local_perturb(pts: np.ndarray, steps: int = 3000) -> np.ndarray:
    best_pts = pts.copy()
    best_val = _min_triangle_area(best_pts)
    n_pts = len(pts)
    
    for step_idx in range(steps):
        trial = best_pts.copy()
        idx = random.randrange(n_pts)
        step = 0.06 * (1 - step_idx/steps)
        trial[idx] += random.gauss(0, step), random.gauss(0, step)
        trial[idx] = _project_to_simplex(trial[idx])
        trial_val = _min_triangle_area(trial)
        if trial_val > best_val:
            best_val = trial_val
            best_pts = trial.copy()
    
    return best_pts

def find_points(seed: int | None = 0) -> List[Tuple[float, float]]:
    pts = _anneal(n_pts=11, iters=40000, seed=seed)
    pts = _local_perturb(pts, steps=3000)
    return [(float(f"{x:.8f}"), float(f"{y:.8f}")) for x, y in pts]

```

---

## Ancestor #2: Organism 835

| Property | Value |
|----------|-------|
| **ID** | 835 |
| **Fitness** | 0.01684375 |
| **Best at Time** | 0.02456154 |
| **Parent ID** | 486 |
| **Was Best When Created** | No |

### Solution Code

```python

from __future__ import annotations

import math
import random
import itertools
from typing import List, Tuple
import numpy as np

SQRT3 = math.sqrt(3.0)

def _area(p, q, r) -> float:
    return abs((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])) * 0.5

comb = list(itertools.combinations(range(11), 3))

def _min_triangle_area(pts: np.ndarray) -> float:
    best = float("inf")
    for tri in comb:
        i, j, k = tri
        a = _area(pts[i], pts[j], pts[k])
        if a < best:
            best = a
    return best

_A = np.array([0.0, 0.0])
_B = np.array([1.0, 0.0])
_C = np.array([0.5, SQRT3 / 2.0])

def _random_point() -> np.ndarray:
    while True:
        u, v = random.random(), random.random()
        if u + v <= 1.0:
            break
    return (1 - u - v) * _A + u * _B + v * _C

def _project_to_simplex(p: np.ndarray) -> np.ndarray:
    M = np.column_stack((_B - _A, _C - _A))
    uv = np.linalg.lstsq(M, p - _A, rcond=None)[0]
    u, v = uv
    w = 1.0 - u - v
    
    eps = 1e-8
    u = max(eps, min(1.0-eps, u))
    v = max(eps, min(1.0-eps, v))
    if u + v > 1.0 - eps:
        s = u + v
        u = (1.0 - eps) * u / s
        v = (1.0 - eps) * v / s
    
    return (1 - u - v) * _A + u * _B + v * _C

def _initial_configuration(n_pts: int = 11) -> np.ndarray:
    pts = []
    pts.append(_A.copy())
    pts.append(_B.copy())
    pts.append(_C.copy())
    
    # Fibonacci spiral pattern
    golden_angle = math.pi * (3 - math.sqrt(5))
    for i in range(1, n_pts - 2):
        theta = golden_angle * i
        r = math.sqrt(i / (n_pts - 3))
        u = 0.5 + r * math.cos(theta) * 0.5
        v = 0.5 + r * math.sin(theta) * 0.5
        if u > 0 and v > 0 and u + v < 1:
            pts.append((1 - u - v) * _A + u * _B + v * _C)
    
    while len(pts) < n_pts:
        u = random.uniform(0.1, 0.9)
        v = random.uniform(0.1, 0.9)
        if u + v < 0.95:
            pt = (1 - u - v) * _A + u * _B + v * _C
            if not any(np.allclose(pt, x, atol=1e-6) for x in pts):
                pts.append(pt)
    return np.array(pts[:n_pts])

def _anneal(n_pts: int = 11,
            iters: int = 20000,
            start_temp: float = 0.1,
            end_temp: float = 1e-8,
            seed: int | None = 0) -> np.ndarray:
    rng = random.Random(seed)
    pts = _initial_configuration(n_pts)
    best_pts = pts.copy()
    best_val = _min_triangle_area(best_pts)

    for t in range(iters):
        T = start_temp * math.exp(math.log(end_temp / start_temp) * t / iters)
        step_size = 0.1 * (1 - (t/iters)**2) + 0.001
        idx = rng.randrange(n_pts)
        trial = pts.copy()
        trial[idx] += rng.gauss(0, step_size), rng.gauss(0, step_size)
        trial[idx] = _project_to_simplex(trial[idx])

        trial_val = _min_triangle_area(trial)
        delta = trial_val - best_val

        if delta > 0 or rng.random() < math.exp(delta / max(T, 1e-12)):
            pts = trial
            if trial_val > best_val:
                best_val = trial_val
                best_pts = trial.copy()

    return best_pts

def _local_perturb(pts: np.ndarray, steps: int = 5000) -> np.ndarray:
    best_pts = pts.copy()
    best_val = _min_triangle_area(best_pts)
    n_pts = len(pts)
    
    for step_idx in range(steps):
        trial = best_pts.copy()
        idx = random.randrange(n_pts)
        step = 0.05 * (1 - (step_idx/steps)**4)
        trial[idx] += random.gauss(0, step), random.gauss(0, step)
        trial[idx] = _project_to_simplex(trial[idx])
        trial_val = _min_triangle_area(trial)
        if trial_val > best_val:
            best_val = trial_val
            best_pts = trial.copy()
    
    return best_pts

def find_points(seed: int | None = 0) -> List[Tuple[float, float]]:
    pts = _anneal(n_pts=11, iters=30000, seed=seed)
    pts = _local_perturb(pts, steps=5000)
    return [(float(f"{x:.8f}"), float(f"{y:.8f}")) for x, y in pts]

```

---

## Ancestor #3: Organism 486

| Property | Value |
|----------|-------|
| **ID** | 486 |
| **Fitness** | 0.01372984 |
| **Best at Time** | 0.02456154 |
| **Parent ID** | 267 |
| **Was Best When Created** | No |

### Solution Code

```python

from __future__ import annotations

import math
import random
import itertools
from typing import List, Tuple
import numpy as np

SQRT3 = math.sqrt(3.0)

def _area(p, q, r) -> float:
    return abs((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])) * 0.5

comb = list(itertools.combinations(range(11), 3))

def _min_triangle_area(pts: np.ndarray) -> float:
    best = float("inf")
    for tri in comb:
        i, j, k = tri
        a = _area(pts[i], pts[j], pts[k])
        if a < best:
            best = a
    return best

_A = np.array([0.0, 0.0])
_B = np.array([1.0, 0.0])
_C = np.array([0.5, SQRT3 / 2.0])

def _random_point() -> np.ndarray:
    while True:
        u, v = random.random(), random.random()
        if u + v <= 1.0:
            break
    return (1 - u - v) * _A + u * _B + v * _C

def _project_to_simplex(p: np.ndarray) -> np.ndarray:
    M = np.column_stack((_B - _A, _C - _A))
    uv = np.linalg.lstsq(M, p - _A, rcond=None)[0]
    u, v = uv
    w = 1.0 - u - v
    
    # Clamp to triangle with small epsilon margin
    eps = 1e-8
    u = max(eps, min(1.0-eps, u))
    v = max(eps, min(1.0-eps, v))
    if u + v > 1.0 - eps:
        s = u + v
        u = (1.0 - eps) * u / s
        v = (1.0 - eps) * v / s
    
    return (1 - u - v) * _A + u * _B + v * _C

def _initial_configuration(n_pts: int = 11) -> np.ndarray:
    pts = []
    # Include vertices
    pts.append(_A.copy())
    pts.append(_B.copy())
    pts.append(_C.copy())
    # Include points along edges
    for i in range(1, 4):
        pts.append((i/4.0) * _A + (1-i/4.0) * _B)
        pts.append((i/4.0) * _B + (1-i/4.0) * _C)
        pts.append((i/4.0) * _C + (1-i/4.0) * _A)
    # Add center point
    pts.append((_A + _B + _C)/3.0)
    # Add remaining points in a more uniform pattern
    while len(pts) < n_pts:
        u = random.uniform(0.2, 0.8)
        v = random.uniform(0.1, 0.7)
        if u + v < 0.95:
            pt = (1 - u - v) * _A + u * _B + v * _C
            if not any(np.allclose(pt, x, atol=1e-6) for x in pts):
                pts.append(pt)
    return np.array(pts[:n_pts])

def _anneal(n_pts: int = 11,
            iters: int = 20000,
            start_temp: float = 0.1,
            end_temp: float = 1e-8,
            seed: int | None = 0) -> np.ndarray:
    rng = random.Random(seed)
    pts = _initial_configuration(n_pts)
    best_pts = pts.copy()
    best_val = _min_triangle_area(best_pts)

    for t in range(iters):
        T = start_temp * (end_temp / start_temp) ** (t / (iters - 1))
        step_size = 0.05 * (1 - (t/iters)**3) + 0.001
        idx = rng.randrange(n_pts)
        trial = pts.copy()
        trial[idx] += rng.uniform(-step_size, step_size), rng.uniform(-step_size, step_size)
        trial[idx] = _project_to_simplex(trial[idx])

        trial_val = _min_triangle_area(trial)
        delta = trial_val - best_val

        if delta > 0 or rng.random() < math.exp(delta / max(T, 1e-12)):
            pts = trial
            if trial_val > best_val:
                best_val = trial_val
                best_pts = trial.copy()

    return best_pts

def _local_perturb(pts: np.ndarray, steps: int = 3000) -> np.ndarray:
    best_pts = pts.copy()
    best_val = _min_triangle_area(best_pts)
    n_pts = len(pts)
    
    for step_idx in range(steps):
        trial = best_pts.copy()
        idx = random.randrange(n_pts)
        step = 0.02 * (1 - (step_idx/steps)**3)
        trial[idx] += random.uniform(-step, step), random.uniform(-step, step)
        trial[idx] = _project_to_simplex(trial[idx])
        trial_val = _min_triangle_area(trial)
        if trial_val > best_val:
            best_val = trial_val
            best_pts = trial.copy()
    
    return best_pts

def find_points(seed: int | None = 0) -> List[Tuple[float, float]]:
    pts = _anneal(n_pts=11, iters=20000, seed=seed)
    pts = _local_perturb(pts, steps=3000)
    return [(float(f"{x:.8f}"), float(f"{y:.8f}")) for x, y in pts]

```

---

## Ancestor #4: Organism 267

| Property | Value |
|----------|-------|
| **ID** | 267 |
| **Fitness** | 0.00000000 |
| **Best at Time** | 0.01880514 |
| **Parent ID** | 215 |
| **Was Best When Created** | No |

### Solution Code

```python

from __future__ import annotations

import math
import random
import itertools
from typing import List, Tuple
import numpy as np

SQRT3 = math.sqrt(3.0)

def _area(p, q, r) -> float:
    return abs((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])) * 0.5

comb = list(itertools.combinations(range(11), 3))

def _min_triangle_area(pts: np.ndarray) -> float:
    best = float("inf")
    for tri in comb:
        i, j, k = tri
        a = _area(pts[i], pts[j], pts[k])
        if a < best:
            best = a
    return best

_A = np.array([0.0, 0.0])
_B = np.array([1.0, 0.0])
_C = np.array([0.5, SQRT3 / 2.0])

def _random_point() -> np.ndarray:
    while True:
        u, v = random.random(), random.random()
        if u + v <= 1.0:
            break
    return (1 - u - v) * _A + u * _B + v * _C

def _project_to_simplex(p: np.ndarray) -> np.ndarray:
    M = np.column_stack((_B - _A, _C - _A))
    uv = np.linalg.lstsq(M, p - _A, rcond=None)[0]
    u, v = uv
    w = 1.0 - u - v
    if u >= 0 and v >= 0 and w >= 0:
        return p
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))
    if u + v > 1.0:
        s = u + v
        u /= s
        v /= s
    return (1 - u - v) * _A + u * _B + v * _C

def _initial_configuration(n_pts: int = 11) -> np.ndarray:
    pts = []
    # Include vertices
    pts.append(_A.copy())
    pts.append(_B.copy())
    pts.append(_C.copy())
    # Include points along edges
    for i in range(1, 4):
        pts.append((i/4.0) * _A + (1-i/4.0) * _B)
        pts.append((i/4.0) * _B + (1-i/4.0) * _C)
        pts.append((i/4.0) * _C + (1-i/4.0) * _A)
    # Add center point
    pts.append((_A + _B + _C)/3.0)
    # Add remaining points in a more uniform pattern
    while len(pts) < n_pts:
        u = random.uniform(0.2, 0.8)
        v = random.uniform(0.1, 0.7)
        if u + v < 0.95:
            pt = (1 - u - v) * _A + u * _B + v * _C
            if not any(np.allclose(pt, x) for x in pts):
                pts.append(pt)
    return np.array(pts[:n_pts])

def _anneal(n_pts: int = 11,
            iters: int = 20000,
            start_temp: float = 0.1,
            end_temp: float = 1e-8,
            seed: int | None = 0) -> np.ndarray:
    rng = random.Random(seed)
    pts = _initial_configuration(n_pts)
    best_pts = pts.copy()
    best_val = _min_triangle_area(best_pts)

    for t in range(iters):
        T = start_temp * (end_temp / start_temp) ** (t / (iters - 1))
        step_size = 0.05 * (1 - (t/iters)**3) + 0.001
        idx = rng.randrange(n_pts)
        trial = pts.copy()
        trial[idx] += rng.uniform(-step_size, step_size), rng.uniform(-step_size, step_size)
        trial[idx] = _project_to_simplex(trial[idx])

        trial_val = _min_triangle_area(trial)
        delta = trial_val - best_val

        if delta > 0 or rng.random() < math.exp(delta / max(T, 1e-12)):
            pts = trial
            if trial_val > best_val:
                best_val = trial_val
                best_pts = trial.copy()

    return best_pts

def _local_perturb(pts: np.ndarray, steps: int = 3000) -> np.ndarray:
    best_pts = pts.copy()
    best_val = _min_triangle_area(best_pts)
    n_pts = len(pts)
    
    for step_idx in range(steps):
        trial = best_pts.copy()
        idx = random.randrange(n_pts)
        step = 0.02 * (1 - (step_idx/steps)**3)
        trial[idx] += random.uniform(-step, step), random.uniform(-step, step)
        trial[idx] = _project_to_simplex(trial[idx])
        trial_val = _min_triangle_area(trial)
        if trial_val > best_val:
            best_val = trial_val
            best_pts = trial.copy()
    
    return best_pts

def find_points(seed: int | None = 0) -> List[Tuple[float, float]]:
    pts = _anneal(n_pts=11, iters=20000, seed=seed)
    pts = _local_perturb(pts, steps=3000)
    return [(float(f"{x:.8f}"), float(f"{y:.8f}")) for x, y in pts]

```

---

## Ancestor #5: Organism 215*

| Property | Value |
|----------|-------|
| **ID** | 215* |
| **Fitness** | 0.01819040 |
| **Best at Time** | 0.01819040 |
| **Parent ID** | 130 |
| **Was Best When Created** | Yes |

### Solution Code

```python

from __future__ import annotations

import math
import random
import itertools
from typing import List, Tuple
import numpy as np

SQRT3 = math.sqrt(3.0)

def _area(p, q, r) -> float:
    return abs((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])) * 0.5

comb = list(itertools.combinations(range(11), 3))

def _min_triangle_area(pts: np.ndarray) -> float:
    best = float("inf")
    for tri in comb:
        i, j, k = tri
        a = _area(pts[i], pts[j], pts[k])
        if a < best:
            best = a
    return best

_A = np.array([0.0, 0.0])
_B = np.array([1.0, 0.0])
_C = np.array([0.5, SQRT3 / 2.0])

def _random_point() -> np.ndarray:
    while True:
        u, v = random.random(), random.random()
        if u + v <= 1.0:
            break
    return (1 - u - v) * _A + u * _B + v * _C

def _project_to_simplex(p: np.ndarray) -> np.ndarray:
    M = np.column_stack((_B - _A, _C - _A))
    uv = np.linalg.lstsq(M, p - _A, rcond=None)[0]
    u, v = uv
    w = 1.0 - u - v
    if u >= 0 and v >= 0 and w >= 0:
        return p
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))
    if u + v > 1.0:
        s = u + v
        u /= s
        v /= s
    return (1 - u - v) * _A + u * _B + v * _C

def _initial_configuration(n_pts: int = 11) -> np.ndarray:
    pts = []
    # Include vertices
    pts.append(_A.copy())
    pts.append(_B.copy())
    pts.append(_C.copy())
    # Include midpoints
    pts.append(0.5 * (_A + _B))
    pts.append(0.5 * (_B + _C))
    pts.append(0.5 * (_C + _A))
    # Hexagonal grid for remaining points
    layers = 4
    for layer in range(layers):
        for i in range(layer + 1):
            u = (i + 0.5 * (layer % 2)) / (layers - 0.5)
            v = (layer - i + 0.5 * (layer % 2)) / (2 * layers)
            pt = (1 - u - v) * _A + u * _B + v * _C
            # Avoid duplicates
            if not any(np.allclose(pt, x) for x in pts):
                pts.append(pt)
            if len(pts) >= n_pts:
                return np.array(pts[:n_pts])
    # Add random points if needed
    while len(pts) < n_pts:
        pt = _random_point()
        if not any(np.allclose(pt, x) for x in pts):
            pts.append(pt)
    return np.array(pts[:n_pts])

def _anneal(n_pts: int = 11,
            iters: int = 15000,
            start_temp: float = 0.05,
            end_temp: float = 1e-7,
            seed: int | None = 0) -> np.ndarray:
    rng = random.Random(seed)
    pts = _initial_configuration(n_pts)
    best_pts = pts.copy()
    best_val = _min_triangle_area(best_pts)

    for t in range(iters):
        T = start_temp * (end_temp / start_temp) ** (t / (iters - 1))
        step_size = 0.04 * (1 - (t/iters)**2) + 0.001
        idx = rng.randrange(n_pts)
        trial = pts.copy()
        trial[idx] += rng.uniform(-step_size, step_size), rng.uniform(-step_size, step_size)
        trial[idx] = _project_to_simplex(trial[idx])

        trial_val = _min_triangle_area(trial)
        delta = trial_val - best_val

        if delta > 0 or rng.random() < math.exp(delta / max(T, 1e-12)):
            pts = trial
            if trial_val > best_val:
                best_val = trial_val
                best_pts = trial.copy()

    return best_pts

def _local_perturb(pts: np.ndarray, steps: int = 2000) -> np.ndarray:
    best_pts = pts.copy()
    best_val = _min_triangle_area(best_pts)
    n_pts = len(pts)
    
    for step_idx in range(steps):
        trial = best_pts.copy()
        idx = random.randrange(n_pts)
        step = 0.015 * (1 - (step_idx/steps)**2)
        trial[idx] += random.uniform(-step, step), random.uniform(-step, step)
        trial[idx] = _project_to_simplex(trial[idx])
        trial_val = _min_triangle_area(trial)
        if trial_val > best_val:
            best_val = trial_val
            best_pts = trial.copy()
    
    return best_pts

def find_points(seed: int | None = 0) -> List[Tuple[float, float]]:
    pts = _anneal(n_pts=11, iters=15000, seed=seed)
    pts = _local_perturb(pts, steps=2000)
    return [(float(f"{x:.8f}"), float(f"{y:.8f}")) for x, y in pts]

```

---

## Ancestor #6: Organism 130*

| Property | Value |
|----------|-------|
| **ID** | 130* |
| **Fitness** | 0.01556142 |
| **Best at Time** | 0.01556142 |
| **Parent ID** | 61 |
| **Was Best When Created** | Yes |

### Solution Code

```python

from __future__ import annotations

import math
import random
import itertools
from typing import List, Tuple
import numpy as np

SQRT3 = math.sqrt(3.0)

def _area(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    return abs(np.cross(q - p, r - p)) * 0.5

def _min_triangle_area(pts: np.ndarray) -> float:
    best = float("inf")
    for i, j, k in itertools.combinations(range(len(pts)), 3):
        a = _area(pts[i], pts[j], pts[k])
        if a < best:
            best = a
    return best

_A = np.array([0.0, 0.0])
_B = np.array([1.0, 0.0])
_C = np.array([0.5, SQRT3 / 2.0])

def _random_point() -> np.ndarray:
    while True:
        u, v = random.random(), random.random()
        if u + v <= 1.0:
            break
    return (1 - u - v) * _A + u * _B + v * _C

def _project_to_simplex(p: np.ndarray) -> np.ndarray:
    M = np.column_stack((_B - _A, _C - _A))
    uv = np.linalg.lstsq(M, p - _A, rcond=None)[0]
    u, v = uv
    w = 1.0 - u - v
    if u >= 0 and v >= 0 and w >= 0:
        return p
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))
    if u + v > 1.0:
        s = u + v
        u /= s
        v /= s
    return (1 - u - v) * _A + u * _B + v * _C

def _initial_configuration(n_pts: int = 11) -> np.ndarray:
    pts = []
    # Hexagonal grid with 4 layers
    layers = 4
    for layer in range(layers):
        for i in range(layer + 1):
            u = (i + 0.5 * (layer % 2)) / (layers - 0.5)
            v = (layer - i + 0.5 * (layer % 2)) / (2 * layers)
            pts.append((1 - u - v) * _A + u * _B + v * _C)
    # Add some random points if needed
    while len(pts) < n_pts:
        pts.append(_random_point())
    return np.array(pts[:n_pts])

def _anneal(n_pts: int = 11,
            iters: int = 8000,
            start_temp: float = 0.05,
            end_temp: float = 1e-7,
            seed: int | None = 0) -> np.ndarray:
    rng = random.Random(seed)
    pts = _initial_configuration(n_pts)
    best_pts = pts.copy()
    best_val = _min_triangle_area(best_pts)

    for t in range(iters):
        T = start_temp * (end_temp / start_temp) ** (t / (iters - 1))
        step_size = 0.03 * (1 - (t/iters)**2) + 0.001
        idx = rng.randrange(n_pts)
        trial = pts.copy()
        trial[idx] += rng.uniform(-step_size, step_size), rng.uniform(-step_size, step_size)
        trial[idx] = _project_to_simplex(trial[idx])

        trial_val = _min_triangle_area(trial)
        delta = trial_val - best_val

        if delta > 0 or rng.random() < math.exp(delta / max(T, 1e-12)):
            pts = trial
            if trial_val > best_val:
                best_val = trial_val
                best_pts = trial.copy()

    return best_pts

def _local_perturb(pts: np.ndarray, steps: int = 1000) -> np.ndarray:
    best_pts = pts.copy()
    best_val = _min_triangle_area(best_pts)
    n_pts = len(pts)
    
    for _ in range(steps):
        trial = best_pts.copy()
        idx = random.randrange(n_pts)
        step = 0.015 * (1 - (_/steps)**2)
        trial[idx] += random.uniform(-step, step), random.uniform(-step, step)
        trial[idx] = _project_to_simplex(trial[idx])
        trial_val = _min_triangle_area(trial)
        if trial_val > best_val:
            best_val = trial_val
            best_pts = trial.copy()
    
    return best_pts

def find_points(seed: int | None = 0) -> List[Tuple[float, float]]:
    pts = _anneal(n_pts=11, iters=8000, seed=seed)
    pts = _local_perturb(pts, steps=1000)
    return [(float(f"{x:.8f}"), float(f"{y:.8f}")) for x, y in pts]

```

---

## Ancestor #7: Organism 61*

| Property | Value |
|----------|-------|
| **ID** | 61* |
| **Fitness** | 0.01514223 |
| **Best at Time** | 0.01514223 |
| **Parent ID** | 18 |
| **Was Best When Created** | Yes |

### Solution Code

```python

from __future__ import annotations

import math
import random
import itertools
from typing import List, Tuple

import numpy as np

SQRT3 = math.sqrt(3.0)

def _area(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    return abs(np.cross(q - p, r - p)) * 0.5

def _min_triangle_area(pts: np.ndarray) -> float:
    best = float("inf")
    for i, j, k in itertools.combinations(range(len(pts)), 3):
        a = _area(pts[i], pts[j], pts[k])
        if a < best:
            best = a
    return best

_A = np.array([0.0, 0.0])
_B = np.array([1.0, 0.0])
_C = np.array([0.5, SQRT3 / 2.0])

def _random_point() -> np.ndarray:
    while True:
        u, v = random.random(), random.random()
        if u + v <= 1.0:
            break
    return (1 - u - v) * _A + u * _B + v * _C

def _project_to_simplex(p: np.ndarray) -> np.ndarray:
    M = np.column_stack((_B - _A, _C - _A))
    uv = np.linalg.lstsq(M, p - _A, rcond=None)[0]
    u, v = uv
    w = 1.0 - u - v
    if u >= 0 and v >= 0 and w >= 0:
        return p
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))
    if u + v > 1.0:
        s = u + v
        u /= s
        v /= s
    return (1 - u - v) * _A + u * _B + v * _C

def _initial_configuration(n_pts: int = 11) -> np.ndarray:
    pts = []
    layers = 3
    for layer in range(layers):
        for i in range(layer + 1):
            u = (i + 0.5 * (layer % 2)) / (layers - 0.5)
            v = (layer - i + 0.5 * (layer % 2)) / (2 * layers)
            pts.append((1 - u - v) * _A + u * _B + v * _C)
    while len(pts) < n_pts:
        pts.append(_random_point())
    return np.array(pts[:n_pts])

def _anneal(n_pts: int = 11,
            iters: int = 6000,
            start_temp: float = 0.02,
            end_temp: float = 1e-6,
            seed: int | None = 0) -> np.ndarray:
    rng = random.Random(seed)
    pts = _initial_configuration(n_pts)
    best_pts = pts.copy()
    best_val = _min_triangle_area(best_pts)

    for t in range(iters):
        T = start_temp * (end_temp / start_temp) ** (t / (iters - 1))
        step_size = 0.02 * (1 - t/iters) + 0.001
        idx = rng.randrange(n_pts)
        trial = pts.copy()
        trial[idx] += rng.uniform(-step_size, step_size), rng.uniform(-step_size, step_size)
        trial[idx] = _project_to_simplex(trial[idx])

        trial_val = _min_triangle_area(trial)
        delta = trial_val - best_val

        if delta > 0 or rng.random() < math.exp(delta / max(T, 1e-12)):
            pts = trial
            if trial_val > best_val:
                best_val = trial_val
                best_pts = trial.copy()

    return best_pts

def _local_perturb(pts: np.ndarray, steps: int = 500) -> np.ndarray:
    best_pts = pts.copy()
    best_val = _min_triangle_area(best_pts)
    n_pts = len(pts)
    
    for _ in range(steps):
        trial = best_pts.copy()
        idx = random.randrange(n_pts)
        step = 0.01 * (1 - _/steps)
        trial[idx] += random.uniform(-step, step), random.uniform(-step, step)
        trial[idx] = _project_to_simplex(trial[idx])
        trial_val = _min_triangle_area(trial)
        if trial_val > best_val:
            best_val = trial_val
            best_pts = trial.copy()
    
    return best_pts

def find_points(seed: int | None = 0) -> List[Tuple[float, float]]:
    pts = _anneal(n_pts=11, iters=6000, seed=seed)
    pts = _local_perturb(pts, steps=500)
    return [(float(f"{x:.8f}"), float(f"{y:.8f}")) for x, y in pts]

```

---

## Ancestor #8: Organism 18*

| Property | Value |
|----------|-------|
| **ID** | 18* |
| **Fitness** | 0.00635695 |
| **Best at Time** | 0.00635695 |
| **Parent ID** | 1 |
| **Was Best When Created** | Yes |

### Solution Code

```python

from __future__ import annotations

import math
import random
import itertools
from typing import List, Tuple

import numpy as np

SQRT3 = math.sqrt(3.0)

def _area(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    return abs(np.cross(q - p, r - p)) * 0.5

def _min_triangle_area(pts: np.ndarray) -> float:
    best = float("inf")
    for i, j, k in itertools.combinations(range(len(pts)), 3):
        a = _area(pts[i], pts[j], pts[k])
        if a < best:
            best = a
    return best

_A = np.array([0.0, 0.0])
_B = np.array([1.0, 0.0])
_C = np.array([0.5, SQRT3 / 2.0])

def _random_point() -> np.ndarray:
    while True:
        u, v = random.random(), random.random()
        if u + v <= 1.0:
            break
    return (1 - u - v) * _A + u * _B + v * _C

def _project_to_simplex(p: np.ndarray) -> np.ndarray:
    M = np.column_stack((_B - _A, _C - _A))
    uv = np.linalg.lstsq(M, p - _A, rcond=None)[0]
    u, v = uv
    w = 1.0 - u - v
    if u >= 0 and v >= 0 and w >= 0:
        return p
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))
    if u + v > 1.0:
        s = u + v
        u /= s
        v /= s
    return (1 - u - v) * _A + u * _B + v * _C

def _initial_configuration(n_pts: int = 11) -> np.ndarray:
    pts = []
    for i in range(n_pts):
        u = (i + 0.5) / n_pts
        v = 0.5 * (1 - u)
        pts.append((1 - u - v) * _A + u * _B + v * _C)
    return np.array(pts)

def _anneal(n_pts: int = 11,
            iters: int = 5000,
            start_temp: float = 0.02,
            end_temp: float = 1e-5,
            step_size: float = 0.02,
            seed: int | None = 0) -> np.ndarray:
    rng = random.Random(seed)
    pts = _initial_configuration(n_pts)
    best_pts = pts.copy()
    best_val = _min_triangle_area(best_pts)

    for t in range(iters):
        T = start_temp * (end_temp / start_temp) ** (t / (iters - 1))
        idx = rng.randrange(n_pts)
        trial = pts.copy()
        trial[idx] += rng.uniform(-step_size, step_size), rng.uniform(-step_size, step_size)
        trial[idx] = _project_to_simplex(trial[idx])

        trial_val = _min_triangle_area(trial)
        delta = trial_val - best_val

        if delta > 0 or rng.random() < math.exp(delta / max(T, 1e-12)):
            pts = trial
            if trial_val > best_val:
                best_val = trial_val
                best_pts = trial.copy()

    return best_pts

def _local_perturb(pts: np.ndarray, steps: int = 1000) -> np.ndarray:
    best_pts = pts.copy()
    best_val = _min_triangle_area(best_pts)
    n_pts = len(pts)
    
    for _ in range(steps):
        trial = best_pts.copy()
        idx = random.randrange(n_pts)
        trial[idx] += random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01)
        trial[idx] = _project_to_simplex(trial[idx])
        trial_val = _min_triangle_area(trial)
        if trial_val > best_val:
            best_val = trial_val
            best_pts = trial.copy()
    
    return best_pts

def find_points(seed: int | None = 0) -> List[Tuple[float, float]]:
    pts = _anneal(n_pts=11, iters=5000, seed=seed)
    pts = _local_perturb(pts, steps=1000)
    return [(float(f"{x:.8f}"), float(f"{y:.8f}")) for x, y in pts]

```

---

## Ancestor #9: Organism 1*

| Property | Value |
|----------|-------|
| **ID** | 1* |
| **Fitness** | 0.00573459 |
| **Best at Time** | 0.00573459 |
| **Parent ID** | None |
| **Was Best When Created** | Yes |

### Solution Code

```python

"""
Heilbronn problem in an equilateral triangle – n = 11

The search routine tries to *maximize* the area of the *smallest*
triangle determined by any 3 of the chosen points.

Author: ChatGPT (o3 family)
Date  : 2025-07-11
"""

from __future__ import annotations

import math
import random
import itertools
from typing import List, Tuple

import numpy as np
try:
    from scipy.optimize import minimize
except ImportError:               # allow the code to run even without SciPy
    minimize = None               # local SA is still useful

# --------------------------------------------------------------------------- #
#  Geometry helpers
# --------------------------------------------------------------------------- #

SQRT3 = math.sqrt(3.0)

def _area(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    """Signed 2-D area of the triangle p-q-r divided by 2."""
    return abs(np.cross(q - p, r - p)) * 0.5

def _min_triangle_area(pts: np.ndarray) -> float:
    """Return the smallest triangle area determined by any 3 points."""
    best = float("inf")
    for i, j, k in itertools.combinations(range(len(pts)), 3):
        a = _area(pts[i], pts[j], pts[k])
        if a < best:
            best = a
            # Early exit: if we dip below current best we can break sooner
            # when called from the optimiser, but keep it simple here.
    return best

# --------------------------------------------------------------------------- #
#  Sampling inside an equilateral triangle
# --------------------------------------------------------------------------- #

_A = np.array([0.0, 0.0])
_B = np.array([1.0, 0.0])
_C = np.array([0.5, SQRT3 / 2.0])

def _random_point() -> np.ndarray:
    """Uniform random point inside the reference equilateral triangle."""
    # Exponential-map trick: generate two randoms, keep the point if u+v<=1
    while True:
        u, v = random.random(), random.random()
        if u + v <= 1.0:
            break
    return (1 - u - v) * _A + u * _B + v * _C

def _project_to_simplex(p: np.ndarray) -> np.ndarray:
    """
    Project a tentative point back into the triangle by clamping its
    barycentric coordinates.
    """
    # Express p in barycentric coords wrt (_A, _B, _C)
    M = np.column_stack((_B - _A, _C - _A))
    uv = np.linalg.lstsq(M, p - _A, rcond=None)[0]
    u, v = uv
    w = 1.0 - u - v
    # If already inside, return unchanged
    if u >= 0 and v >= 0 and w >= 0:
        return p
    # Otherwise, snap to the closest point on the triangle (quadratic-program style)
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))
    if u + v > 1.0:
        s = u + v
        u /= s
        v /= s
    return (1 - u - v) * _A + u * _B + v * _C

# --------------------------------------------------------------------------- #
#  Simulated-annealing style global search
# --------------------------------------------------------------------------- #

def _anneal(n_pts: int = 11,
            iters: int = 50_000,
            start_temp: float = 0.05,
            end_temp: float = 1e-4,
            step_size: float = 0.04,
            seed: int | None = 0) -> np.ndarray:
    """
    Crude SA: perturb one random point at each step; accept if area improves,
    or with Metropolis probability exp((ΔA)/T) otherwise.
    """
    rng = random.Random(seed)
    pts = np.array([_random_point() for _ in range(n_pts)])
    best_pts = pts.copy()
    best_val = _min_triangle_area(best_pts)

    for t in range(iters):
        T = start_temp * (end_temp / start_temp) ** (t / (iters - 1))
        idx = rng.randrange(n_pts)
        trial = pts.copy()
        trial[idx] += rng.uniform(-step_size, step_size), rng.uniform(-step_size, step_size)
        trial[idx] = _project_to_simplex(trial[idx])

        trial_val = _min_triangle_area(trial)
        delta = trial_val - best_val

        if delta > 0 or rng.random() < math.exp(delta / max(T, 1e-12)):
            pts = trial
            if trial_val > best_val:
                best_val = trial_val
                best_pts = trial.copy()

    return best_pts

# --------------------------------------------------------------------------- #
#  Optional local polish (requires SciPy)
# --------------------------------------------------------------------------- #

def _local_polish(pts: np.ndarray, method: str = "Nelder-Mead") -> np.ndarray:
    """Simple local optimisation: maximise *negative* min-area."""
    n_pts = len(pts)
    x0 = pts.ravel()

    def f(x: np.ndarray) -> float:
        P = x.reshape((-1, 2))
        return -_min_triangle_area(P)          # minimise negative area

    res = minimize(
        f,
        x0,
        method=method,
        options=dict(maxiter=5_000, fatol=1e-10, xatol=1e-10)
    ) if minimize else None

    if res is not None and res.success:
        out = res.x.reshape((n_pts, 2))
        # In rare cases Nelder-Mead wanders outside; project back.
        for i in range(n_pts):
            out[i] = _project_to_simplex(out[i])
        return out
    return pts

# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #

def find_points(seed: int | None = 0) -> List[Tuple[float, float]]:
    """
    Heuristic solver for the Heilbronn problem with n = 11 inside the
    reference equilateral triangle.

    Parameters
    ----------
    seed : int or None
        RNG seed for reproducibility.  Pass None for stochastic runs.

    Returns
    -------
    coords : list[tuple[float, float]]
        A list of 11 (x, y) pairs.  All coordinates lie in the triangle
        with vertices (0, 0), (1, 0), (0.5, √3/2).
    """
    # 1) global crude search
    pts = _anneal(n_pts=11, iters=100, seed=seed)

    # 2) local deterministic polish (if SciPy available)
    # pts = _local_polish(pts)  # Skip for now to avoid timeout

    # 3) final rounding for readability
    return [(float(f"{x:.8f}"), float(f"{y:.8f}")) for x, y in pts]




```

---
