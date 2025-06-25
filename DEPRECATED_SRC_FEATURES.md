# Deprecated src/ Features Documentation

This document preserves the unique features and implementations from the `src/` directory before consolidation with `src_async/`. These features represent significant algorithmic innovations that should be considered for future re-implementation.

## Overview

The `src/` directory contained three distinct evolution algorithms (`evolve.py`, `evolve2.py`, `evolve3.py`) with progressively more sophisticated features. When migrating to async-only architecture, several unique capabilities were lost.

## Unique Evolution Algorithms

### Evolver2 - Stochastic Model Selection (`src/evolve2.py`)

**Key Innovation:** Dynamic model selection based on iteration timing and fitness performance.

#### Model Selection Strategy:
```python
# Special iterations (1/100 chance each):
# 1. Large changes to best solution using o4-mini
# 2. Iterative small changes until no improvement

# Normal iterations with probabilistic selection:
# - 1/100: openai:o4-mini-2025-04-16
# - ~14%:  openai:gpt-4.1  
# - Rest:  openai:gpt-4.1-mini
```

#### Adaptive Change Strategy:
- **Below-average organisms:** 60% probability of large changes
- **Above-average organisms:** 20% probability of large changes
- Dynamic adaptation based on `fitness < population_average`

#### Iterative Small Changes Feature:
```python
# Continuously apply small improvements until convergence
while True:
    new_organism = generate(prompt, openai:o4-mini, reason=False)
    if new_organism.fitness <= current.fitness:
        break
    current = new_organism
    improvement_count += 1
```

**Lost Capability:** This iterative refinement mechanism that continues until no further improvement is unique and not present in async version.

### Evolver3 - Advanced Metadata Tracking (`src/evolve3.py`)

**Key Innovation:** Comprehensive creation metadata and child limitation strategy.

#### Enhanced Model Selection:
```python
# Different model probabilities:
# - 1%:  openai:o3
# - 30%: deepseek:deepseek-reasoner  
# - Rest: deepseek:deepseek-chat
```

#### Child Limitation Strategy:
```python
max_children_per_organism = 10  # Configurable limit
# Skip organisms that have reached child limit
# Prevents over-exploitation of single solutions
```

#### Creation Metadata Tracking:
```python
creation_info = {
    "model": model_used,
    "change_type": "big" | "small", 
    "iteration_type": "normal" | "best_large" | "iterative_small",
    "step": current_step,
    "big_changes": big_changes_probability,
    "child_number": child_count_for_parent
}
```

**Lost Capability:** Detailed provenance tracking for each organism's creation, enabling sophisticated analysis of evolutionary patterns.

## Population Management Differences

### Selection Weighting Variance:
- **src/**: `fitness ** 3 + 1` - Moderate selection pressure
- **src_async/**: `fitness ** 4 + 1` - Higher selection pressure

**Impact:** Different diversity vs exploitation trade-offs.

### Population Loading:
- **src_async/**: Has `Population.from_pickle()` for resuming from existing populations
- **src/**: Limited to checkpoint-based resumption only

## Timing-Based Special Iterations

### Evolver2 Timing Strategy:
```python
# Every 100 iterations:
if step % 100 == 0:
    # Large changes to best solution
elif (step + 50) % 100 == 0:  
    # Iterative small changes until convergence
```

### Evolver3 Timing Strategy:
```python
tenth = max_steps // 10
# More sophisticated interval calculations
# Different special operation triggers
```

**Lost Capability:** Fine-grained control over when different evolutionary strategies are applied.

## Model Integration Features

### Reasoning Model Detection:
- **src/**: Simple detection `"o4" in model`
- **src_async/**: Comprehensive `"o4" in model or "reason" in model`

### Model Configuration:
- **src/**: Hardcoded per-evolver model selection logic
- **src_async/**: Configurable `model_mix` dictionary

**Migration Note:** The hardcoded strategies contained years of empirical tuning that may be lost in generic configuration.

## Checkpoint System Specialization

### Unique Checkpoint Naming:
```python
# Evolver-specific checkpoint files:
f"{name}_evolver2_checkpoint.pkl"
f"{name}_evolver3_checkpoint.pkl" 
f"{name}_async_checkpoint.pkl"
```

**Lost Capability:** Ability to run different evolver types simultaneously with separate checkpoints.

## Reporting and Analysis

### Evolver2 Reporting:
- Detailed stochastic behavior documentation
- Iterative improvement statistics
- Model selection frequency analysis

### Evolver3 Reporting:
- Creation metadata in JSON serialization
- Child count distributions
- Provenance tracking for best solutions

### JSON Serialization Differences:
- **src/**: Try JSON first, fallback to pickle on failure
- **src_async/**: Always create both JSON and pickle files
- **Evolver3**: Includes `creation_info` field in JSON

## Recommendations for Future Implementation

### High Priority Features to Port:
1. **Iterative small changes mechanism** from Evolver2
2. **Child limitation strategy** from Evolver3  
3. **Creation metadata tracking** from Evolver3
4. **Adaptive change probability** based on fitness performance
5. **Sophisticated timing-based special operations**

### Medium Priority Features:
1. **Multiple model selection strategies** as configurable options
2. **Different weighting schemes** for selection pressure
3. **Evolver-specific checkpoint naming** for parallel runs
4. **Enhanced reporting** with algorithm-specific metadata

### Configuration Suggestions:
```python
# Proposed async configuration to preserve src/ features:
EvolutionConfig(
    model_selection_strategy="stochastic|adaptive|weighted",
    iterative_refinement=True,
    max_children_per_organism=10,
    track_creation_metadata=True,
    selection_weighting_power=3,  # Configurable: 3 or 4
    special_iteration_intervals=[100, 200],  # Flexible timing
    adaptive_change_probability=True
)
```

## Historical Context

The `src/` directory represents approximately 2+ years of evolutionary algorithm development with:
- **3 distinct algorithmic approaches**
- **Empirically tuned hyperparameters**
- **Novel strategies for model selection**
- **Sophisticated population management**
- **Advanced metadata tracking capabilities**

This knowledge should not be lost and represents significant research investment in evolutionary programming with LLMs.

## Files Affected by Removal

When `src/` is deleted, these unique implementations will be lost:
- `src/evolve2.py` - 200+ lines of stochastic evolution logic
- `src/evolve3.py` - 250+ lines of advanced metadata tracking
- Unique population management strategies
- Specialized checkpoint handling
- Custom reporting mechanisms

**Total LOC at risk:** ~1000+ lines of researched evolutionary algorithms.