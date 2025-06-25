# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

rEVOLVE is an AI-powered evolutionary programming framework that uses LLMs to evolve code solutions. It implements genetic algorithms where LLMs serve as mutation operators, generating improved versions of existing solutions through iterative evolution.

## Core Architecture

### Key Components
- **`ProblemSpecification`** (`src/specification.py`): Defines problems with system prompts, evaluators, starting populations, and hyperparameters
- **`Population`** (`src/population.py`): Manages solution organisms with selection strategies (exploration/elitism/weighted)  
- **`Evolver/Evolver2/Evolver3`** (`src/evolve*.py`): Evolution engines with different capabilities and strategies
- **`Organism`** (`src/population.py`): Individual solutions with fitness scores and parent lineage tracking
- **Problem-specific evaluators**: Domain-specific fitness functions and constraint validation

### Evolution Flow
1. Initialize population with starting solutions
2. Select parent organism using exploration/elitism/weighted strategies  
3. Generate LLM prompt with system context + parent solution
4. LLM produces mutated solution via `generate()` function
5. Evaluate solution with problem-specific evaluator
6. Add to population with parent tracking and fitness score
7. Save checkpoints periodically for resumption

### Evolver Variants
- **Evolver**: Basic single-threaded evolution
- **Evolver2**: Enhanced with multiple model support and sophisticated reporting
- **Evolver3**: Advanced with big/small change control and population loading
- **AsyncEvolver** (`src_async/`): Concurrent LLM calls for high-throughput evolution

## Development Commands

### Running Experiments
```bash
# Run specific example problems
cd example_problems/circle_packing && python run_circle_packing.py
cd example_problems/guess_the_votes && python ../../run_example.py guess_the_votes

# Run with async evolver for better performance
cd example_problems/circle_packing && python run_async.py
```

### Environment Setup
```bash
# Install dependencies with UV
uv install

# Set up environment variables in .env
LOGFIRE_TOKEN=your_token_here
```

### Working with Checkpoints
```bash
# Experiments automatically save to checkpoints/ directory
# Resume interrupted runs by rerunning the same command
# Checkpoints include population state and evolution history
```

## Problem Development Pattern

### Creating New Problems
1. **Define Problem Specification**: Create `spec.py` with `ProblemSpecification`
2. **Implement Evaluator**: Create evaluation function that returns `Evaluation` object
3. **Set System Prompt**: Provide detailed instructions for LLM on problem domain
4. **Configure Hyperparameters**: Set exploration rate, elitism, max steps, target fitness
5. **Create Runner Script**: Script to initialize and run evolution

### Example Problem Structure
```python
# In spec.py
def get_problem_spec() -> ProblemSpecification:
    return ProblemSpecification(
        name="problem_name",
        systemprompt=SYSTEM_PROMPT,  # Detailed LLM instructions
        evaluator=evaluate_solution,  # Your evaluation function
        starting_population=[initial_organism],
        hyperparameters=Hyperparameters(
            exploration_rate=0.2,    # Random selection for diversity
            elitism_rate=0.1,        # Always select best
            max_steps=100,
            target_fitness=target_value
        )
    )
```

## Architecture Insights

### Selection Strategies
The population uses a three-tier selection approach:
- **Exploration** (20% default): Random selection for genetic diversity
- **Elitism** (10% default): Always select best performer
- **Weighted Selection** (70% default): Fitness-weighted with cubic scaling

### Safety & Robustness
- **Sandboxed Execution**: Problem evaluators run generated code in subprocesses with timeouts
- **Constraint Validation**: Domain-specific validation (e.g., circle overlap detection, test suite compliance)
- **Error Handling**: Graceful failure handling with detailed error reporting in evaluations

### Visualization & Reporting
Evolution runs automatically generate:
- **Population visualizations**: Ancestry graphs with fitness-based coloring via graphviz
- **Fitness progression plots**: Track improvement over generations  
- **Comprehensive reports**: Markdown reports with embedded visualizations
- **Best solution extraction**: Optimal solutions saved as separate files

### Model Integration
Uses `pydantic-ai` for LLM integration supporting:
- Multiple model providers (GPT, DeepSeek, etc.)
- Model mixing with configurable probabilities  
- Structured generation with retry logic
- Cost tracking and performance monitoring

## Key Files and Locations

### Core Framework
- `src/evolve*.py`: Evolution engines with different strategies
- `src/population.py`: Population management and selection logic
- `src/mutate.py`: LLM interface for solution generation
- `src/prompt.py`: Prompt generation for LLM context

### Example Problems  
- `example_problems/circle_packing/`: Geometric optimization (maximize circle radii sum)
- `example_problems/guess_the_votes/`: Code golf (minimize character count)
- Each problem includes `spec.py`, `evaluation.py`, and runner scripts

### Outputs and Checkpoints
- `outputs/`: Evolution run results with visualizations and reports
- `checkpoints/`: Automatic state saving for experiment resumption
- Generated files include population graphs, fitness plots, and best solutions

## Development Notes

### Hyperparameter Tuning
- **High exploration** (0.3+) for complex/novel problems requiring diverse approaches
- **High elitism** (0.2+) for refinement phases or when good solutions exist
- **Target fitness** enables early stopping when goals are achieved
- **Reasoning mode** (`reason=True`) provides LLM explanations for solution changes

### Performance Considerations
- Use `AsyncEvolver` for I/O-bound workloads with multiple LLM calls
- Checkpoint frequency balances resumption capability vs. I/O overhead  
- Population size affects diversity vs. computational cost
- Model selection impacts both quality and API costs

### Debugging Evolution
- Check `outputs/*/report.md` for detailed run summaries
- Use population visualizations to identify convergence patterns
- Monitor Logfire logs for detailed execution traces
- Examine checkpoint files for population state inspection