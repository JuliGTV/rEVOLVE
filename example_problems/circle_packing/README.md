# Circle Packing with Simple Evolutionary System

This directory contains a simplified version of the circle packing experiment that runs on the rEVOLVE simple evolutionary system, adapted from the more complex OpenEvolve implementation in `circle_packing_openevolve/`.

## Problem Description

The circle packing problem involves placing 26 non-overlapping circles inside a unit square to maximize the sum of their radii. This is a classic computational geometry optimization problem.

**Constraints:**
- All circles must be entirely within the unit square [0,1] × [0,1]
- No circles may overlap
- Must pack exactly 26 circles
- Goal: maximize the sum of all circle radii

**Target:** The AlphaEvolve paper achieved a sum of radii of 2.635 for n=26.

## Files

- `spec.py` - Problem specification for the evolutionary system
- `evaluation.py` - Evaluation function that safely runs and validates solutions
- `run_circle_packing.py` - Main script to run the experiment
- `README.md` - This file

## How to Run

### Basic Usage

```bash
cd example_problems/circle_packing
python run_circle_packing.py
```

### Using the Example Runner

You can also use the general example runner:

```bash
cd example_problems
python run_example.py circle_packing
```

## System Architecture

This implementation uses the simple rEVOLVE evolutionary system with the following components:

### Problem Specification (`spec.py`)

- **Initial Solution**: Simple ring-based arrangement of circles
- **System Prompt**: Detailed instructions for the LLM on circle packing optimization
- **Hyperparameters**: 
  - Exploration rate: 0.3 (higher exploration for complex problem)
  - Elitism rate: 0.2 (preserve good solutions)
  - Max steps: 100
  - Target fitness: 2.635

### Evaluation (`evaluation.py`)

- **Safety**: Runs solutions in isolated subprocess with timeout
- **Validation**: Checks circle placement constraints and overlaps
- **Fitness**: Sum of radii (higher is better)
- **Error Handling**: Graceful handling of invalid solutions

### Evolution Process

1. Starts with a simple ring-based initial solution
2. Uses GPT-4o to mutate solutions based on geometric insights
3. Evaluates each candidate solution for validity and fitness
4. Selects best solutions using elitism and exploration
5. Continues until target fitness or max steps reached

## Key Differences from OpenEvolve Version

| Aspect | OpenEvolve | Simple System |
|--------|------------|---------------|
| **Complexity** | Multi-phase, complex configuration | Single-phase, simple setup |
| **Population** | 60-70 organisms, 4-5 islands | Single population |
| **Evaluation** | File-based with cascade evaluation | String-based with subprocess |
| **Configuration** | YAML configs, multiple models | Python specification |
| **Interface** | Command-line with checkpoints | Direct Python execution |

## Expected Results

The simple system should be able to discover good circle packing strategies, potentially including:

1. **Initial patterns**: Ring-based arrangements (sum ≈ 1.0)
2. **Grid patterns**: Regular grid with optimization (sum ≈ 2.0-2.2)
3. **Optimized arrangements**: Mathematical optimization approaches (sum ≈ 2.4-2.6)

## Visualization

The system automatically visualizes the best solution found, showing:
- Unit square boundary
- Circles with different colors and sizes
- Circle numbering for reference
- Sum of radii and target ratio

## Extending the Experiment

To modify the experiment:

1. **Adjust hyperparameters** in `spec.py`
2. **Modify the system prompt** to guide the LLM differently  
3. **Change evaluation criteria** in `evaluation.py`
4. **Add multiple initial solutions** for diversity

## Comparison with Complex System

This simplified version demonstrates that:

- Complex multi-phase evolution can be approximated with simpler approaches
- LLM-based mutation is effective for geometric optimization problems
- String-based code evolution is feasible for mathematical problems
- The simple system can achieve competitive results with less complexity

The trade-off is potentially slower convergence and less sophisticated exploration strategies compared to the full OpenEvolve system.