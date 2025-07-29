# Best Organism Ancestry Analysis

This document traces the complete ancestry of the fittest organism (ID: 392) with fitness 0.02752820.

Each section shows an ancestor in the lineage, from the fittest organism back to the original ancestor.
Organisms marked with * were the best fitness when they were created.

---

## Ancestor #1: Organism 392*

| Property | Value |
|----------|-------|
| **ID** | 392* |
| **Fitness** | 0.02752820 |
| **Best at Time** | 0.02752820 |
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
    pts.extend([_A, _B, _C])
    pts.append((_A + _B) / 2)
    pts.append((_A + _C) / 2)
    pts.append((_B + _C) / 2)
    pts.append((_A + _B + _C) / 3)
    while len(pts) < n_pts:
        pts.append(_random_point())
    return np.array(pts)

n_pts_fixed = 11
triplets = list(itertools.combinations(range(n_pts_fixed), 3))
point_to_triplets = [[] for _ in range(n_pts_fixed)]
for idx, (i, j, k) in enumerate(triplets):
    point_to_triplets[i].append(idx)
    point_to_triplets[j].append(idx)
    point_to_triplets[k].append(idx)

def _anneal(n_pts: int = 11,
            iters: int = 30000,
            start_temp: float = 0.2,
            end_temp: float = 1e-8,
            seed: int | None = 0) -> np.ndarray:
    rng = random.Random(seed)
    pts = _initial_configuration(n_pts)
    
    areas_arr = np.zeros(len(triplets))
    for idx, (i, j, k) in enumerate(triplets):
        areas_arr[idx] = _area(pts[i], pts[j], pts[k])
    current_val = np.min(areas_arr)
    
    best_pts = pts.copy()
    best_val = current_val

    for t in range(iters):
        T = start_temp * (end_temp / start_temp) ** (t / (iters - 1))
        step_size = 0.05 * (1 - (t/iters)**0.5) + 0.001
        
        idx = rng.randrange(n_pts)
        trial_pts = pts.copy()
        if rng.random() < 0.05:
            trial_pts[idx] = _random_point()
        else:
            delta_x = rng.uniform(-step_size, step_size)
            delta_y = rng.uniform(-step_size, step_size)
            trial_pts[idx] += np.array([delta_x, delta_y])
            trial_pts[idx] = _project_to_simplex(trial_pts[idx])

        trial_areas = areas_arr.copy()
        for triplet_idx in point_to_triplets[idx]:
            i, j, k = triplets[triplet_idx]
            trial_areas[triplet_idx] = _area(trial_pts[i], trial_pts[j], trial_pts[k])
        trial_val = np.min(trial_areas)
        
        delta = trial_val - current_val
        if delta > 0 or rng.random() < math.exp(delta / max(T, 1e-12)):
            pts = trial_pts
            areas_arr = trial_areas
            current_val = trial_val
            if trial_val > best_val:
                best_val = trial_val
                best_pts = trial_pts.copy()

    return best_pts

def find_points(seed: int | None = 0) -> List[Tuple[float, float]]:
    pts = _anneal(n_pts=11, iters=30000, seed=seed)
    return [(float(f"{x:.8f}"), float(f"{y:.8f}")) for x, y in pts]

```

---

## Ancestor #2: Organism 61*

| Property | Value |
|----------|-------|
| **ID** | 61* |
| **Fitness** | 0.02254603 |
| **Best at Time** | 0.02254603 |
| **Parent ID** | 2 |
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
            if best == 0:
                return 0
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
    # Add vertices
    pts.extend([_A, _B, _C])
    # Add edge midpoints
    pts.append((_A + _B) / 2)
    pts.append((_A + _C) / 2)
    pts.append((_B + _C) / 2)
    # Add center points
    pts.append((_A + _B + _C) / 3)
    # Fill remaining with random points
    while len(pts) < n_pts:
        pts.append(_random_point())
    return np.array(pts)

def _anneal(n_pts: int = 11,
            iters: int = 10000,
            start_temp: float = 0.2,
            end_temp: float = 1e-8,
            seed: int | None = 0) -> np.ndarray:
    rng = random.Random(seed)
    pts = _initial_configuration(n_pts)
    best_pts = pts.copy()
    best_val = _min_triangle_area(best_pts)

    for t in range(iters):
        T = start_temp * (end_temp / start_temp) ** (t / (iters - 1))
        step_size = 0.05 * (1 - (t/iters)**0.5) + 0.001
        
        idx = rng.randrange(n_pts)
        trial = pts.copy()
        if rng.random() < 0.05:  # occasional large jump
            trial[idx] = _random_point()
        else:
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

def find_points(seed: int | None = 0) -> List[Tuple[float, float]]:
    pts = _anneal(n_pts=11, iters=10000, seed=seed)
    return [(float(f"{x:.8f}"), float(f"{y:.8f}")) for x, y in pts]

```

---

## Ancestor #3: Organism 2*

| Property | Value |
|----------|-------|
| **ID** | 2* |
| **Fitness** | 0.01515107 |
| **Best at Time** | 0.01515107 |
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
            if best == 0:
                return 0
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

def _anneal(n_pts: int = 11,
            iters: int = 5000,
            start_temp: float = 0.1,
            end_temp: float = 1e-6,
            seed: int | None = 0) -> np.ndarray:
    rng = random.Random(seed)
    pts = np.array([_random_point() for _ in range(n_pts)])
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

def find_points(seed: int | None = 0) -> List[Tuple[float, float]]:
    pts = _anneal(n_pts=11, iters=5000, seed=seed)
    return [(float(f"{x:.8f}"), float(f"{y:.8f}")) for x, y in pts]

```

---

## Ancestor #4: Organism 1*

| Property | Value |
|----------|-------|
| **ID** | 1* |
| **Fitness** | 0.00388948 |
| **Best at Time** | 0.00388948 |
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
