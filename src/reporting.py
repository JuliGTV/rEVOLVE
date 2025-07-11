"""
Evolution reporting module for generating comprehensive reports.

This module extracts the reporting functionality from AsyncEvolver and enhances it
with configuration reproducibility and analysis graphics/tables from analyse_evolution.ipynb.
"""

import os
import json
import pickle
import matplotlib.pyplot as plt
import graphviz
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logfire

from src.population import Population, Organism
from src.specification import ProblemSpecification


class EvolutionReporter:
    """Generate comprehensive evolution reports with analysis and visualizations."""
    
    def __init__(self, 
                 population: Population,
                 specification: ProblemSpecification,
                 evolver_config: Dict[str, Any],
                 outputs_dir: str = "outputs"):
        """
        Initialize the evolution reporter.
        
        Args:
            population: Final population from evolution
            specification: Problem specification used
            evolver_config: AsyncEvolver configuration parameters
            outputs_dir: Directory to create reports in
        """
        self.population = population
        self.specification = specification
        self.evolver_config = evolver_config
        self.outputs_dir = outputs_dir
        
    def generate_report(self) -> str:
        """
        Generate comprehensive evolution report with all analysis.
        
        Returns:
            Path to the generated report directory
        """
        # Create report directory
        report_dir = self._create_report_directory()
        
        # Generate all report components
        self._create_population_visualization(report_dir)
        self._create_fitness_progression_plot(report_dir)
        self._serialize_population_data(report_dir)
        self._create_ancestry_graph(report_dir)
        self._create_ancestry_analysis(report_dir)
        self._create_markdown_report(report_dir)
        
        logfire.info(f"Comprehensive report generated in directory: {report_dir}")
        return report_dir
    
    def _create_report_directory(self) -> str:
        """Create timestamped report directory."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        problem_name = self.specification.name.replace(" ", "_").replace("(", "").replace(")", "")
        dir_name = f"{timestamp}_{problem_name}"
        
        os.makedirs(self.outputs_dir, exist_ok=True)
        full_dir_path = os.path.join(self.outputs_dir, dir_name)
        os.makedirs(full_dir_path, exist_ok=True)
        
        return full_dir_path
    
    def _create_population_visualization(self, report_dir: str):
        """Create population visualization using existing method."""
        viz_path = os.path.join(report_dir, "population_visualization")
        self.population.visualize_population(filename=viz_path, view=False)
    
    def _create_fitness_progression_plot(self, report_dir: str):
        """Create fitness progression plot."""
        organisms = self.population.get_population()
        organisms_sorted = sorted(organisms, key=lambda x: x.id)
        
        # Track best fitness at each step
        best_fitness_progression = []
        current_best = float('-inf')
        
        for org in organisms_sorted:
            if org.evaluation.fitness > current_best:
                current_best = org.evaluation.fitness
            best_fitness_progression.append(current_best)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(best_fitness_progression) + 1), best_fitness_progression, 'b-', linewidth=2)
        plt.xlabel('Generation (Organism ID)')
        plt.ylabel('Best Fitness Score')
        plt.title('Fitness Progression Over Time')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fitness_plot_path = os.path.join(report_dir, "fitness_progression.png")
        plt.savefig(fitness_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _serialize_population_data(self, report_dir: str):
        """Serialize population data in both JSON and pickle formats."""
        organisms = self.population.get_population()
        
        # Always create pickle file
        with open(os.path.join(report_dir, "population.pkl"), "wb") as f:
            pickle.dump(organisms, f)
        
        # Try to serialize as JSON with all fields
        try:
            population_data = []
            for org in organisms:
                org_data = {
                    "id": org.id,
                    "parent_id": org.parent_id,
                    "solution": org.solution,
                    "evaluation": {
                        "fitness": org.evaluation.fitness,
                        "additional_data": org.evaluation.additional_data
                    },
                    "children": org.children,
                    "creation_info": org.creation_info if hasattr(org, 'creation_info') else None
                }
                population_data.append(org_data)
            
            with open(os.path.join(report_dir, "population.json"), "w") as f:
                json.dump(population_data, f, indent=2)
                
        except Exception as e:
            logfire.warning(f"JSON serialization failed: {e}, but pickle was created successfully")
    
    def _find_best_so_far_organisms(self) -> List[Organism]:
        """Find all organisms that were the best fitness when they were created."""
        all_orgs_sorted = sorted(self.population.get_population(), key=lambda org: org.id)
        
        best_so_far_orgs = []
        current_best_fitness = 0.0
        
        for org in all_orgs_sorted:
            if org.evaluation.fitness > current_best_fitness:
                best_so_far_orgs.append(org)
                current_best_fitness = org.evaluation.fitness
        
        return best_so_far_orgs
    
    def _trace_ancestry(self, organism: Organism) -> List[Dict[str, Any]]:
        """Trace the ancestry of an organism back to its origins."""
        org_map = {org.id: org for org in self.population.get_population()}
        
        ancestry = []
        current_id = organism.id
        
        while current_id is not None:
            if current_id in org_map:
                current_org = org_map[current_id]
                ancestry.append({
                    'id': current_org.id,
                    'fitness': current_org.evaluation.fitness,
                    'parent_id': current_org.parent_id,
                    'organism': current_org
                })
                current_id = current_org.parent_id
            else:
                break
        
        return ancestry
    
    def _create_ancestry_graph(self, report_dir: str):
        """Create graphviz visualization of best organisms' ancestries."""
        best_orgs = self._find_best_so_far_organisms()
        
        # Get all ancestors of best-so-far organisms
        all_ancestors = set()
        for org in best_orgs:
            ancestry = self._trace_ancestry(org)
            for ancestor in ancestry:
                all_ancestors.add(ancestor['id'])
        
        # Create directed graph
        dot = graphviz.Digraph(comment='Evolution Ancestry')
        dot.attr(rankdir='TB')  # Top to bottom layout
        dot.attr('node', shape='box', style='rounded,filled')
        
        org_map = {org.id: org for org in self.population.get_population()}
        best_org_ids = set(org.id for org in best_orgs)
        
        # Add nodes
        for org_id in all_ancestors:
            if org_id in org_map:
                org = org_map[org_id]
                fitness = org.evaluation.fitness
                
                # Different colors for best-so-far vs ancestors
                if org_id in best_org_ids:
                    color = 'lightblue'
                    label = f"ID: {org_id}*\\nFitness: {fitness:.6f}"
                else:
                    color = 'lightgray'
                    label = f"ID: {org_id}\\nFitness: {fitness:.6f}"
                
                dot.node(str(org_id), label, fillcolor=color)
        
        # Add edges (parent -> child relationships)
        for org_id in all_ancestors:
            if org_id in org_map:
                org = org_map[org_id]
                if org.parent_id is not None and org.parent_id in all_ancestors:
                    dot.edge(str(org.parent_id), str(org_id))
        
        # Save the graph
        graph_path = os.path.join(report_dir, "ancestry_graph")
        dot.render(graph_path, format='png', cleanup=True)
    
    def _create_ancestry_analysis(self, report_dir: str):
        """Create detailed ancestry analysis of the best organism."""
        best_organism = self.population.get_best()
        ancestry = self._trace_ancestry(best_organism)
        
        # Calculate best fitness at each point in time
        all_orgs_sorted = sorted(self.population.get_population(), key=lambda org: org.id)
        
        ancestry_with_best = []
        for ancestor in ancestry:
            # Find the best fitness among all organisms with ID <= ancestor's ID
            best_at_time = max(
                (org.evaluation.fitness for org in all_orgs_sorted if org.id <= ancestor['id']),
                default=ancestor['fitness']
            )
            
            # Check if this ancestor was the best when it was created
            best_before = max(
                (org.evaluation.fitness for org in all_orgs_sorted if org.id < ancestor['id']),
                default=0.0
            )
            was_best = ancestor['fitness'] > best_before
            
            ancestry_with_best.append({
                **ancestor,
                'best_at_time': best_at_time,
                'was_best_when_created': was_best
            })
        
        # Create markdown documentation
        md_content = []
        md_content.append("# Best Organism Ancestry Analysis")
        md_content.append("")
        md_content.append(f"This document traces the complete ancestry of the fittest organism (ID: {best_organism.id}) with fitness {best_organism.evaluation.fitness:.8f}.")
        md_content.append("")
        md_content.append("Each section shows an ancestor in the lineage, from the fittest organism back to the original ancestor.")
        md_content.append("Organisms marked with * were the best fitness when they were created.")
        md_content.append("")
        md_content.append("---")
        md_content.append("")
        
        # Document each ancestor
        for i, ancestor in enumerate(ancestry_with_best):
            was_best_marker = "*" if ancestor['was_best_when_created'] else ""
            md_content.append(f"## Ancestor #{i+1}: Organism {ancestor['id']}{was_best_marker}")
            md_content.append("")
            
            md_content.append("| Property | Value |")
            md_content.append("|----------|-------|")
            md_content.append(f"| **ID** | {ancestor['id']}{was_best_marker} |")
            md_content.append(f"| **Fitness** | {ancestor['fitness']:.8f} |")
            md_content.append(f"| **Best at Time** | {ancestor['best_at_time']:.8f} |")
            parent_display = ancestor['parent_id'] if ancestor['parent_id'] is not None else 'None'
            md_content.append(f"| **Parent ID** | {parent_display} |")
            md_content.append(f"| **Was Best When Created** | {'Yes' if ancestor['was_best_when_created'] else 'No'} |")
            md_content.append("")
            
            # Add solution code
            md_content.append("### Solution Code")
            md_content.append("")
            md_content.append("```python")
            md_content.append(ancestor['organism'].solution)
            md_content.append("```")
            md_content.append("")
            md_content.append("---")
            md_content.append("")
        
        # Write to file
        ancestry_file = os.path.join(report_dir, "best_ancestry.md")
        with open(ancestry_file, "w") as f:
            f.write("\n".join(md_content))
    
    def _create_markdown_report(self, report_dir: str):
        """Create comprehensive markdown report."""
        best_organism = self.population.get_best()
        num_organisms = len(self.population.get_population())
        avg_fitness = self.population.calculate_average_fitness()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Get hyperparameters and evolver config
        hyperparams = self.specification.hyperparameters
        
        # Find best-so-far organisms for summary
        best_orgs = self._find_best_so_far_organisms()
        
        markdown_content = f"""# Evolution Report

## Problem Information
- **Problem Name**: {self.specification.name}
- **Timestamp**: {timestamp}

## Hyperparameters
- **Exploration Rate**: {hyperparams.exploration_rate}
- **Elitism Rate**: {hyperparams.elitism_rate}
- **Max Steps**: {hyperparams.max_steps}
- **Target Fitness**: {hyperparams.target_fitness if hyperparams.target_fitness is not None else 'None'}
- **Reason**: {hyperparams.reason}

## Evolver Configuration
- **Max Concurrent**: {self.evolver_config.get('max_concurrent', 'N/A')}
- **Model Mix**: {json.dumps(self.evolver_config.get('model_mix', {}), indent=2)}
- **Big Changes Rate**: {self.evolver_config.get('big_changes_rate', 'N/A')}
- **Best Model**: {self.evolver_config.get('best_model', 'N/A')}
- **Max Children Per Organism**: {self.evolver_config.get('max_children_per_organism', 'N/A')}
- **Checkpoint Dir**: {self.evolver_config.get('checkpoint_dir', 'N/A')}
- **Population Path**: {self.evolver_config.get('population_path', 'None')}

## Population Statistics
- **Number of Organisms**: {num_organisms}
- **Best Fitness Score**: {best_organism.evaluation.fitness}
- **Average Fitness Score**: {avg_fitness:.4f}
- **Number of Best-So-Far Organisms**: {len(best_orgs)}

## Best-So-Far Organisms Summary
These organisms were the best fitness when they were created:

| ID | Fitness | Improvement |
|----|---------|-------------|"""

        prev_fitness = 0.0
        for org in best_orgs:
            improvement = org.evaluation.fitness - prev_fitness
            markdown_content += f"\n| {org.id} | {org.evaluation.fitness:.8f} | +{improvement:.8f} |"
            prev_fitness = org.evaluation.fitness

        markdown_content += f"""

## Fitness Progression
![Fitness Progression](fitness_progression.png)

## Population Visualization
![Population Visualization](population_visualization.gv.png)

## Ancestry Analysis
![Ancestry Graph](ancestry_graph.png)

For detailed ancestry analysis of the best organism, see [best_ancestry.md](best_ancestry.md).

## Best Solution
```
{best_organism.solution}
```

## Additional Data from Best Solution
```json
{json.dumps(best_organism.evaluation.additional_data, indent=2)}
```

## Creation Information for Best Solution
```json
{json.dumps(getattr(best_organism, 'creation_info', 'Not available'), indent=2)}
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
from {self.specification.__module__} import get_{self.specification.name}_spec

spec = get_{self.specification.name}_spec()
```

### Evolver Configuration  
```python
evolver_config = {json.dumps(self.evolver_config, indent=2)}
```

### Full Reproduction Script
```python
from src.evolve import AsyncEvolver

# Get specification and config
spec = get_{self.specification.name}_spec()
evolver_config = {json.dumps(self.evolver_config, indent=2)}

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
"""
        
        with open(os.path.join(report_dir, "report.md"), "w") as f:
            f.write(markdown_content)