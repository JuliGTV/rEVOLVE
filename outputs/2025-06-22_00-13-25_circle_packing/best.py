import numpy as np
from scipy.optimize import minimize
import itertools

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


centres, radii, sum = run_packing()


def verify_circles(circles: np.ndarray):
  """Checks that the circles are disjoint and lie inside a unit square.

    Args:
      circles: A numpy array of shape (num_circles, 3), where each row is
        of the form (x, y, radius), specifying a circle.

    Raises:
      AssertionError if the circles are not disjoint or do not lie inside the
      unit square.
  """
  # Check pairwise disjointness.
  for circle1, circle2 in itertools.combinations(circles, 2):
    center_distance = np.sqrt((circle1[0] - circle2[0])**2 + (circle1[1] - circle2[1])**2)
    radii_sum = circle1[2] + circle2[2]
    assert center_distance >= radii_sum, f"Circles are NOT disjoint: {circle1} and {circle2}."

  # Check all circles lie inside the unit square [0,1]x[0,1].
  for circle in circles:
    assert (0 <= min(circle[0], circle[1]) - circle[2] and max(circle[0],circle[1]) + circle[2] <= 1), f"Circle {circle} is NOT fully inside the unit square."


# def append_column(matrix: np.ndarray, column: np.ndarray) -> np.ndarray:
#     """
#     Appends a 1D array as a new column to a 2D matrix.

#     Parameters:
#     - matrix (np.ndarray): A 2D numpy array of shape (n, m)
#     - column (np.ndarray): A 1D numpy array of shape (n,)

#     Returns:
#     - np.ndarray: A new array of shape (n, m+1)
#     """
#     if matrix.shape[0] != column.shape[0]:
#         raise ValueError("Number of rows in matrix must match the size of the column")

#     # Reshape column to (n, 1)
#     column = column.reshape(-1, 1)

#     # Append the column
#     return np.hstack((matrix, column))

# data = append_column(centres, radii)

# print(data)

# #@title My data: Verification
# print(f"My data has {len(data)} circles.")
# verify_circles(data)
# print(f"The circles are disjoint and lie inside the unit square.")
# sum_radii = np.sum(data[:, 2])
# print(f"My data sum of radii: {sum_radii}")

# def validate_packing(centers: np.ndarray, radii: np.ndarray) -> bool:
#     """
#     Validate that circles don't overlap and are inside the unit square.
    
#     Args:
#         centers: np.array of shape (n, 2) with (x, y) coordinates
#         radii: np.array of shape (n) with radius of each circle
    
#     Returns:
#         True if valid, False otherwise
#     """
#     n = centers.shape[0]
    
#     # Check if circles are inside the unit square
#     for i in range(n):
#         x, y = centers[i]
#         r = radii[i]
#         if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
#             return False
    
#     # Check for overlaps
#     for i in range(n):
#         for j in range(i + 1, n):
#             dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
#             if dist < radii[i] + radii[j] - 1e-6:  # Allow for tiny numerical errors
#                 return False
    
#     return True


# print(centres)
# print(radii)
# print(sum)

# print(validate_packing(centres, radii))




data2 = np.array([
    [0.11077901, 0.11077901, 0.11077901],
    [0.0846395, 0.9153605, 0.0846395],
    [0.88884382, 0.11115618, 0.11115618], 
    [0.91507374, 0.91507374, 0.08492626],
    [0.31311581, 0.09239155, 0.09239155],
    [0.49942837, 0.09392734, 0.09392734],
    [0.68594302, 0.09259209, 0.09259209],
    [0.29460949, 0.8697789, 0.1302211],
    [0.49728445, 0.92113963, 0.07886037],
    [0.70230953, 0.86674143, 0.13325857],
    [0.09573233, 0.31674147, 0.09573233],
    [0.10306052, 0.5153992, 0.10306052],
    [0.10679014, 0.72521672, 0.10679014],
    [0.90384867, 0.31791996, 0.09615133],
    [0.89653277, 0.51740442, 0.10346723],
    [0.89481744, 0.72604716, 0.10518256],
    [0.23971053, 0.23632643, 0.06918068],
    [0.40335878, 0.25758295, 0.09584233],
    [0.59521973, 0.25795056, 0.09601898],
    [0.7593524, 0.23704114, 0.06944019],
    [0.27162985, 0.4023652, 0.09989835],
    [0.49866808, 0.47003658, 0.13701043],
    [0.72690571, 0.4039573, 0.10060037],
    [0.29474606, 0.61307645, 0.11207709],
    [0.49553176, 0.72465738, 0.11762969],
    [0.7026096, 0.61833416, 0.11514888]
])


#@title My data: Verification
# try:
#     verify_circles(data2)
# except AssertionError as e:
#     print("not disjoint")
#     print(e)
# sum_radii = np.sum(data2[:, 2])
# print(f"My data sum of radii: {sum_radii}")


data3 = data2.copy() - np.array([0,0,0.00000005])

print(f"My data has {len(data3)} circles.")
try:
    verify_circles(data3)
except AssertionError as e:
    print("not disjoint")
    print(e)
sum_radii = np.sum(data3[:, 2])
print(f"My data sum of radii: {sum_radii}")


# c1 = [0.31311581, 0.09239155, 0.09239155]
# c2 = [0.49942837, 0.09392734, 0.09392734]
# center_distance = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
# print(center_distance)
# print(center_distance - c2[2])
