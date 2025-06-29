# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

rEVOLVE is an AI-powered evolutionary programming framework that uses LLMs to evolve code solutions. It implements genetic algorithms where LLMs serve as mutation operators, generating improved versions of existing solutions through iterative evolution.

## Core Architecture

### Key Components
- **`ProblemSpecification`** (`src/specification.py`): Defines problems with system prompts, evaluators, starting populations, and hyperparameters
- **`Population`** (`src/population.py`): Manages solution organisms with selection strategies (exploration/elitism/weighted)  
- **`AsyncEvolver`** (`src/evolve.py`): Main evolution engine with concurrent LLM calls for high-throughput evolution
- **`Organism`** (`src/population.py`): Individual solutions with fitness scores and parent lineage tracking
- **Problem-specific evaluators**: Domain-specific fitness functions and constraint validation

### Evolution Flow
1. Initialize population with starting solutions
2. Select parent organism using exploration/elitism/weighted strategies  
3. Generate LLM prompt with system context + parent solution
4. LLM produces mutated solution via `generate_async()` function
5. Evaluate solution with problem-specific evaluator
6. Add to population with parent tracking and fitness score
7. Save checkpoints periodically for resumption

### Async-Only Architecture
- **AsyncEvolver** (`src/evolve.py`): The single evolution engine with concurrent LLM calls
- **Configurable model selection**: Support for multiple model providers with probability mixing
- **Advanced features**: Big/small change control, population loading, creation metadata tracking
- **Deprecated features**: Legacy sync evolvers documented in `DEPRECATED_SRC_FEATURES.md`

## Development Commands

### Running Experiments
```bash
# Run specific example problems (async-only)
cd example_problems/circle_packing && python run_async.py
cd example_problems/guess_the_votes && python ../../src/run_example_async.py guess_the_votes

# All experiments now use AsyncEvolver by default
```

### Git Workflow
```bash
# IMPORTANT: Always commit changes when appropriate
# Commit after completing logical units of work such as:
# - Adding new features or modules
# - Fixing bugs or issues  
# - Refactoring or cleanup tasks
# - Updating documentation
# - Any changes that represent a cohesive improvement

# Check status and review changes before committing
git status
git diff

# Add and commit with descriptive messages
git add <files>
git commit -m "Descriptive commit message"
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
The new `EvolutionReporter` class (`src/reporting.py`) generates comprehensive reports:
- **Population visualizations**: Ancestry graphs with fitness-based coloring via graphviz
- **Fitness progression plots**: Track improvement over generations  
- **Ancestry analysis**: Complete lineage tracing of best organisms with solution code
- **Best-so-far tracking**: Summary tables of organisms that achieved new fitness records
- **Configuration reproducibility**: Full evolver settings saved for exact experiment replication
- **Comprehensive markdown reports**: All visualizations and analysis embedded
- **Multiple formats**: Both JSON and pickle serialization of population data

### Model Integration
Uses `pydantic-ai` for LLM integration supporting:
- Multiple model providers (GPT, DeepSeek, etc.)
- Model mixing with configurable probabilities  
- Structured generation with retry logic
- Cost tracking and performance monitoring

## Key Files and Locations

### Core Framework
- `src/evolve.py`: AsyncEvolver - main concurrent evolution engine
- `src/population.py`: Population management and selection logic
- `src/mutate.py`: LLM interface for solution generation
- `src/prompt.py`: Prompt generation for LLM context
- `src/reporting.py`: EvolutionReporter for comprehensive analysis and visualization

### Example Problems  
- `example_problems/circle_packing/`: Geometric optimization (maximize circle radii sum)
- `example_problems/guess_the_votes/`: Code golf (minimize character count)
- `example_problems/clifford/`: Reference materials for quantum circuit synthesis (upcoming)
- Each problem includes `spec.py`, `evaluation.py`, and runner scripts

### Outputs and Checkpoints
- `outputs/`: Evolution run results with comprehensive reports and visualizations
- `checkpoints/`: Automatic state saving for experiment resumption
- Generated files include:
  - `report.md`: Comprehensive markdown with all analysis and configuration
  - `ancestry_graph.png`: Visualization of best organisms' evolutionary relationships  
  - `best_ancestry.md`: Detailed lineage analysis with solution code
  - `fitness_progression.png`: Performance over time
  - `population_visualization.png`: Population structure graph
  - `population.json`/`population.pkl`: Serialized population data

## Development Notes

### Hyperparameter Tuning
- **High exploration** (0.3+) for complex/novel problems requiring diverse approaches
- **High elitism** (0.2+) for refinement phases or when good solutions exist
- **Target fitness** enables early stopping when goals are achieved
- **Reasoning mode** (`reason=True`) provides LLM explanations for solution changes

### Performance Considerations
- **Concurrency control**: Adjust `max_concurrent` parameter for LLM API rate limits
- **Model mixing**: Configure `model_mix` dictionary for optimal cost/performance balance
- **Checkpoint frequency**: Balances resumption capability vs. I/O overhead  
- **Population management**: Size affects diversity vs. computational cost
- **Model selection**: Impacts both solution quality and API costs

### Debugging Evolution
- Check `outputs/*/report.md` for detailed run summaries with full configuration
- Use `ancestry_graph.png` to visualize evolutionary relationships between best organisms
- Review `best_ancestry.md` for complete lineage analysis of optimal solutions
- Monitor Logfire logs for detailed execution traces
- Examine checkpoint files for population state inspection
- Debug scripts and utilities available in `jdebug/` directory
- Analysis notebooks and tools in `analysis/` directory
- Use the comprehensive reporting system for reproducible experiment analysis

### Repository Organization
- **Root directories**:
  - `src/`: Core async evolution framework
  - `example_problems/`: Problem implementations and reference materials
  - `outputs/`: All experiment results consolidated here
  - `checkpoints/`: Experiment state for resumption
  - `jdebug/`: Debug scripts and development utilities
  - `analysis/`: Jupyter notebooks and analysis tools
  - `DEPRECATED_SRC_FEATURES.md`: Documentation of legacy evolution algorithms

## Testing and Validation

### No Traditional Test Suite
rEVOLVE does not use conventional unit testing frameworks. Instead, validation occurs through:
- **Problem-specific evaluators**: Each problem implements its own validation logic in `evaluation.py`
- **Sandboxed execution**: Solutions are tested in isolated subprocesses with timeouts
- **Sample solution validation**: Reference implementations in `sample_solutions/` directories
- **Manual testing**: Individual problem test files (e.g., `example_problems/guess_the_votes/test.py`)

### Validation Commands
```bash
# Test specific problem solutions
cd example_problems/guess_the_votes && python test.py

# Validate solutions work by running evaluators directly
python -c "from example_problems.circle_packing.evaluation import evaluate_solution; print(evaluate_solution('your_code_here'))"
```

## Important Development Context

### Async-Only Architecture
- **All evolution is async**: `AsyncEvolver` in `src/evolve.py` is the only supported evolution engine
- **Deprecated sync methods**: Legacy sync evolvers documented in `DEPRECATED_SRC_FEATURES.md` should not be used
- **Concurrent execution**: Framework designed for high-throughput parallel LLM calls

### Environment Dependencies
- **UV package manager**: Use `uv install` instead of pip for dependency management
- **Python 3.13+**: Required for async features and type annotations
- **Logfire integration**: Optional telemetry for evolution monitoring via `LOGFIRE_TOKEN` environment variable