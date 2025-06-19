from src.checkpoint_utils import get_checkpoint_info, checkpoint_exists
from src.evolve2 import Evolver2
from example_problems.circle_packing.spec import get_circle_packing_spec

def test_checkpoint_loading():
    """Test loading from checkpoint functionality"""

    print("=== Checkpoint Loading Test ===\n")

    # Check if checkpoint exists
    problem_name = "circle_packing"
    if checkpoint_exists(problem_name, "evolver2"):
        print(f"✓ Checkpoint found for {problem_name} (Evolver2)")

        # Get checkpoint info
        info = get_checkpoint_info(problem_name, "evolver2")
        if info:
            print(f"  - Problem: {info['problem_name']}")
            print(f"  - Step: {info['step']}")
            print(f"  - Population size: {info['population_size']}")
            print(f"  - Best fitness: {info['best_fitness']}")
            print(f"  - Timestamp: {info['timestamp']}")

        print("\n=== Testing Evolver2 Initialization with Checkpoint ===")

        # Initialize Evolver2 - it should automatically load from checkpoint
        spec = get_circle_packing_spec()

        print(f"Creating Evolver2 for '{spec.name}'...")
        evolver = Evolver2(spec)

        print(f"✓ Evolver2 initialized successfully")
        print(f"  - Starting from step: {evolver.current_step}")
        print(f"  - Population size: {len(evolver.population.get_population())}")
        print(f"  - Current best fitness: {evolver.population.get_best().evaluation.fitness}")

        # Optionally run a few more steps to test continuation
        print(f"\n=== Testing Evolution Continuation ===")
        print("Running 5 more evolution steps...")

        # Set a small max_steps just for testing
        evolver.max_steps = evolver.current_step + 5

        try:
            final_population = evolver.evolve()
            print(f"✓ Evolution completed successfully")
            print(f"  - Final step: {evolver.current_step}")
            print(f"  - Final best fitness: {final_population.get_best().evaluation.fitness}")
            print(f"  - Final report at: {evolver.report()}")
        except Exception as e:
            print(f"✗ Evolution failed: {str(e)}")

    else:
        print(f"✗ No checkpoint found for {problem_name} (Evolver2)")
        print("You can create a checkpoint by running an evolution that gets interrupted")

if __name__ == "__main__":
    test_checkpoint_loading()