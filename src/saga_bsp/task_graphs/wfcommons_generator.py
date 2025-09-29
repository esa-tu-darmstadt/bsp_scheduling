"""
WfCommons-based task graph generation.
"""

import json
import logging
import random
from typing import Tuple, Optional, List

import networkx as nx
import numpy as np
from wfcommons.wfchef.recipes import (
    BlastRecipe, BwaRecipe, CyclesRecipe, EpigenomicsRecipe,
    GenomeRecipe, MontageRecipe, SeismologyRecipe, SoykbRecipe,
    SrasearchRecipe
)
from wfcommons import WorkflowGenerator

from .base import TaskGraphGenerator, TaskGraphMetadata

logger = logging.getLogger(__name__)


class WfCommonsTaskGraphGenerator(TaskGraphGenerator):
    """Generator for WfCommons-based task graphs."""

    # Available WfCommons recipes
    RECIPES = {
        'blast': BlastRecipe,
        'bwa': BwaRecipe,
        'cycles': CyclesRecipe,
        'epigenomics': EpigenomicsRecipe,
        'genome': GenomeRecipe,
        'montage': MontageRecipe,
        'seismology': SeismologyRecipe,
        'soykb': SoykbRecipe,
        'srasearch': SrasearchRecipe,
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(cache_dir=cache_dir)
        # Get min task counts for each recipe
        self.TASK_COUNTS = {}
        for name, recipe_cls in self.RECIPES.items():
            recipe = recipe_cls()
            microstructures_dir = recipe.this_dir / 'microstructures'
            summary = json.loads((microstructures_dir / 'summary.json').read_text())
            base_graphs = summary['base_graphs']
            base_graph_orders = [base_graphs[g]['order'] for g in base_graphs]
            self.TASK_COUNTS[name] = list(set(base_graph_orders))

    def generate_task_graph(self, recipe_name: str, task_count: Optional[int] = None) -> Tuple[nx.DiGraph, TaskGraphMetadata]:
        """Generate a single task graph using specified WfCommons recipe.

        Args:
            recipe_name: Name of the WfCommons recipe
            task_count: Number of tasks in the workflow (None for default)

        Returns:
            Tuple of (task_graph, metadata)
        """
        if recipe_name not in self.RECIPES:
            raise ValueError(f"Unknown recipe: {recipe_name}. Available: {list(self.RECIPES.keys())}")

        recipe_class = self.RECIPES[recipe_name]

        # Use default task count if not specified
        if task_count is None:
            _, _, default_count = self.TASK_COUNT_RANGES[recipe_name]
            task_count = default_count

        # Generate workflow using WfCommons
        recipe = recipe_class.from_num_tasks(task_count)
        generator = WorkflowGenerator(recipe)
        workflow = generator.build_workflow()

        # Copy workflow (which is already a NetworkX DiGraph)
        task_graph = workflow.copy()

        # Set node weights (task execution times)
        for node in task_graph.nodes():
            node_data = task_graph.nodes[node]
            task_obj = node_data.get('task')
            if task_obj and hasattr(task_obj, 'runtime'):
                runtime = task_obj.runtime
                if runtime is None or runtime <= 0:
                    runtime = 1.0
            else:
                runtime = 1.0
            task_graph.nodes[node]['weight'] = runtime

        # Set edge weights (communication/data transfer sizes)
        for src, dst in task_graph.edges():
            dst_task_data = task_graph.nodes[dst]
            dst_task_obj = dst_task_data.get('task')

            # Calculate total input data size for this task
            total_size = 0
            if dst_task_obj and hasattr(dst_task_obj, 'input_files'):
                for input_file in dst_task_obj.input_files:
                    file_size = getattr(input_file, 'size', 0)
                    if file_size and file_size > 0:
                        total_size += file_size

            # Use average per-edge size
            incoming_edges = list(task_graph.predecessors(dst))
            if len(incoming_edges) > 0:
                avg_size = total_size / len(incoming_edges)
            else:
                avg_size = total_size

            # Ensure minimum weight
            if avg_size <= 0:
                avg_size = 1.0

            task_graph.edges[src, dst]['weight'] = avg_size

        # Calculate task graph statistics
        task_weights = [task_graph.nodes[node].get('weight', 1.0) for node in task_graph.nodes()]
        edge_weights = [task_graph.edges[edge].get('weight', 1.0) for edge in task_graph.edges()]

        avg_task_weight = np.mean(task_weights) if task_weights else 1.0
        avg_edge_weight = np.mean(edge_weights) if edge_weights else 1.0

        # Create metadata
        metadata = TaskGraphMetadata(
            source_type='wfcommons',
            source_name=recipe_name,
            task_count=len(task_graph.nodes()),
            edge_count=len(task_graph.edges()),
            avg_task_weight=avg_task_weight,
            avg_edge_weight=avg_edge_weight,
            additional_info={
                'requested_task_count': task_count
            }
        )

        return task_graph, metadata

    def generate_dataset(self, recipe_name: str, variations: int = 5,
                        task_counts: Optional[List[int]] = None,
                        overwrite_cache: bool = False) -> Tuple[List[nx.DiGraph], List[TaskGraphMetadata]]:
        """Generate a dataset of task graphs for a recipe.

        Args:
            recipe_name: Name of the WfCommons recipe
            variations: Number of variations to generate
            task_counts: List of task counts to use (None for recipe defaults)
            overwrite_cache: Whether to overwrite cached data

        Returns:
            Tuple of (task_graphs, metadata_list)
        """
        cache_key = f"wfcommons_{recipe_name}_{variations}"
        if task_counts:
            cache_key += f"_tasks_{'-'.join(map(str, task_counts))}"

        # Check cache first
        if not overwrite_cache:
            cached_data = self.load_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"Loaded cached WfCommons dataset for {recipe_name}: {len(cached_data[0])} graphs")
                return cached_data

        logger.info(f"Generating WfCommons dataset for {recipe_name}: {variations} variations")

        task_graphs = []
        metadata_list = []

        # Use default task count if not specified
        if task_counts is None:
            _, _, default_count = self.TASK_COUNT_RANGES[recipe_name]
            task_counts = [default_count]

        variation_count = 0
        total_variations = len(task_counts) * variations

        for task_count in task_counts:
            for i in range(variations):
                try:
                    # Set deterministic seed for reproducibility
                    seed = hash(f"{recipe_name}_{task_count}_{i}") % (2**32)
                    random.seed(seed)
                    np.random.seed(seed)

                    task_graph, metadata = self.generate_task_graph(recipe_name, task_count)
                    task_graphs.append(task_graph)
                    metadata_list.append(metadata)

                    variation_count += 1
                    if variation_count % 5 == 0:
                        logger.info(f"Generated {variation_count}/{total_variations} variations")

                except Exception as e:
                    logger.warning(f"Failed to generate variation {i+1} for {recipe_name} "
                                 f"with {task_count} tasks: {e}")

        # Reset random seeds
        random.seed(None)
        np.random.seed(None)

        # Save to cache
        if task_graphs and self.cache_dir:
            self.save_to_cache(cache_key, task_graphs, metadata_list)
            logger.info(f"Cached {len(task_graphs)} task graphs for {recipe_name}")

        return task_graphs, metadata_list

    def list_available_recipes(self) -> List[str]:
        """Get list of available WfCommons recipes."""
        return list(self.RECIPES.keys())