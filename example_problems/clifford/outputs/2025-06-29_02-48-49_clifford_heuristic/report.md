# Evolution Report

## Problem Information
- **Problem Name**: clifford_heuristic
- **Timestamp**: 2025-06-29_02-48-50

## Hyperparameters
- **Exploration Rate**: 0.1
- **Elitism Rate**: 0.3
- **Max Steps**: 100
- **Target Fitness**: 70.0
- **Reason**: True

## Evolver Configuration
- **Max Concurrent**: 7
- **Model Mix**: {
  "deepseek:deepseek-reasoner": 0.1,
  "deepseek:deepseek-chat": 0.9
}
- **Big Changes Rate**: 0.4
- **Best Model**: deepseek:deepseek-reasoner
- **Max Children Per Organism**: 15
- **Checkpoint Dir**: evolution_results/checkpoints
- **Population Path**: None

## Population Statistics
- **Number of Organisms**: 96
- **Best Fitness Score**: 31.93333333333333
- **Average Fitness Score**: -5.0415
- **Number of Best-So-Far Organisms**: 2

## Best-So-Far Organisms Summary
These organisms were the best fitness when they were created:

| ID | Fitness | Improvement |
|----|---------|-------------|
| 2 | 4.83333333 | +4.83333333 |
| 23 | 31.93333333 | +27.10000000 |

## Fitness Progression
![Fitness Progression](fitness_progression.png)

## Population Visualization
![Population Visualization](population_visualization.gv.png)

## Ancestry Analysis
![Ancestry Graph](ancestry_graph.png)

For detailed ancestry analysis of the best organism, see [best_ancestry.md](best_ancestry.md).

## Best Solution
```

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

## Additional Data from Best Solution
```json
{
  "score": "31.933333",
  "validity": "valid",
  "execution_method": "single_subprocess",
  "function_name": "heuristic"
}
```

## Creation Information for Best Solution
```json
{
  "model": "deepseek:deepseek-chat",
  "change_type": "SMALL ITERATIVE IMPROVEMENT",
  "step": 26,
  "is_reasoning": true,
  "big_changes_rate": 0.4,
  "child_number": 15
}
```

## Files in this Report
- `population_visualization.gv` / `population_visualization.gv.png` - Visual representation of the population
- `fitness_progression.png` - Plot showing fitness improvement over generations  
- `ancestry_graph.png` - Visualization of best organisms' ancestry relationships
- `best_ancestry.md` - Detailed ancestry analysis of the fittest organism
- `population.json` / `population.pkl` - Serialized population data
- `report.md` - This comprehensive report file

## Configuration Reproducibility

To reproduce this evolution run exactly, use the following configuration:

### Problem Specification
```python
from src.specification import get_clifford_heuristic_spec

spec = get_clifford_heuristic_spec()
```

### Evolver Configuration  
```python
evolver_config = {
  "checkpoint_dir": "evolution_results/checkpoints",
  "max_concurrent": 7,
  "model_mix": {
    "deepseek:deepseek-reasoner": 0.1,
    "deepseek:deepseek-chat": 0.9
  },
  "big_changes_rate": 0.4,
  "best_model": "deepseek:deepseek-reasoner",
  "max_children_per_organism": 15,
  "population_path": null
}
```

### Full Reproduction Script
```python
from src.evolve import AsyncEvolver

# Get specification and config
spec = get_clifford_heuristic_spec()
evolver_config = {
  "checkpoint_dir": "evolution_results/checkpoints",
  "max_concurrent": 7,
  "model_mix": {
    "deepseek:deepseek-reasoner": 0.1,
    "deepseek:deepseek-chat": 0.9
  },
  "big_changes_rate": 0.4,
  "best_model": "deepseek:deepseek-reasoner",
  "max_children_per_organism": 15,
  "population_path": null
}

# Create evolver
evolver = AsyncEvolver(
    specification=spec,
    **evolver_config
)

# Run evolution
population = await evolver.evolve()

# Generate report
from src.reporting import EvolutionReporter
reporter = EvolutionReporter(population, spec, evolver_config)
report_dir = reporter.generate_report()
```
