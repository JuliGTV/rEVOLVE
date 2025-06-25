"""
Circle packing problem specification for the simple evolutionary system.
"""

from src.specification import ProblemSpecification, Hyperparameters
from src.population import Organism
from src.evaluation import Evaluation
from .evaluation import evaluate_circle_packing


# Initial solution - simplified version of the OpenEvolve initial program
INITIAL_SOLUTION = '''import numpy as np

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
'''


# System prompt for the LLM to understand the circle packing problem
SYSTEM_PROMPT = """You are an expert mathematician and programmer specializing in circle packing problems and computational geometry.

Your task is to improve Python code that packs exactly 26 circles into a unit square (0,0) to (1,1) to maximize the sum of their radii.

CONSTRAINTS:
- All circles must be entirely within the unit square
- No circles may overlap (distance between centers >= sum of radii)
- Must return exactly 26 circles
- Code must define a function called run_packing() that returns (centers, radii, sum_radii)
- centers: numpy array of shape (26, 2) with (x, y) coordinates
- radii: numpy array of shape (26,) with radius of each circle
- sum_radii: sum of all radii (float)

OPTIMIZATION GOAL:
- The target is to achieve a sum of radii of 2.635 (AlphaEvolve benchmark)
- Higher sum of radii = better fitness

KEY GEOMETRIC INSIGHTS:
- Similar radius circles often form regular patterns, while varied radii allow better space utilization
- Perfect symmetry may not yield the optimal packing due to edge effects
- Circle packings often follow hexagonal patterns in dense regions however purely hexagonal patterns are not optimal due to edge effects
- Circles can be arranged in layers, shells, or grid patterns (but other patterns may be better)
- Variable radius circles allow better space utilization than uniform radii
- Mathematical optimization techniques (scipy.optimize) can be very effective
- Consider different placement strategies: constructive, grid-based, optimization-based or hybrid
- Break through plateaus by trying fundamentally different approaches especially when asked for large changes
- The densest known circle packings often use a hybrid approach
- The optimization routine is critically important - simple physics-based models with carefully tuned parameters
- Consider strategic placement of circles at square corners and edges
- Adjusting the pattern to place larger circles at the center and smaller at the edges
- The math literature suggests special arrangements for specific values of n

CODING REQUIREMENTS:
- Import numpy as np (available)
- You may use scipy if needed (but remember to import it)
- Focus on the run_packing() function implementation
- Ensure all circles are valid (non-negative radii, proper placement)
- The code will be executed safely in a sandbox environment

Improve the existing solution by trying different geometric arrangements, optimization techniques, or hybrid approaches."""


def get_circle_packing_spec() -> ProblemSpecification:
    """
    Get the circle packing problem specification for the simple evolutionary system.
    
    Returns:
        ProblemSpecification configured for circle packing optimization
    """
    
    # Create initial population with the base solution
    # Evaluate the initial solution properly instead of using dummy value
    initial_evaluation = evaluate_circle_packing(INITIAL_SOLUTION)
    starting_population = [
        Organism(solution=INITIAL_SOLUTION, evaluation=initial_evaluation)
    ]
    
    # Configure hyperparameters for circle packing evolution
    hyperparameters = Hyperparameters(
        exploration_rate=0,      # Higher exploration for this complex problem
        elitism_rate=1,          # Keep good solutions
        max_steps=4000,             # Run for many iterations
        target_fitness=2.636,      # AlphaEvolve benchmark
        reason=True                # Enable reasoning for complex problem
    )
    
    return ProblemSpecification(
        name="circle_packing",
        systemprompt=SYSTEM_PROMPT,
        evaluator=evaluate_circle_packing,
        starting_population=starting_population,
        hyperparameters=hyperparameters
    )