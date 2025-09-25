"""
Dataset generation using WfCommons with caching and metadata.

This module generates task graphs using different WfCommons recipes,
with caching support and metadata extraction for CCR adaptation.
"""

import json
import logging
import pathlib
import pickle
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import networkx as nx
import numpy as np
from wfcommons.wfchef.recipes import (
    BlastRecipe, BwaRecipe, CyclesRecipe, EpigenomicsRecipe,
    GenomeRecipe, MontageRecipe, SeismologyRecipe, SoykbRecipe,
    SrasearchRecipe
)
from wfcommons import WorkflowGenerator
from saga.schedulers.data.random import gen_random_networks
from saga_bsp.schedule import BSPHardware

from hardware_ipu import IPUHardware

logger = logging.getLogger(__name__)

@dataclass
class TaskGraphMetadata:
    """Metadata for cached task graphs."""
    recipe_name: str
    num_tasks: int
    avg_task_weight: float
    avg_edge_weight: float
    task_count: int
    edge_count: int
    target_ccr: float
    sync_time: float

class DatasetGenerator:
    """Generator for WfCommons-based datasets with caching."""

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

    # Task count distributions for each recipe (min, max)
    # These ranges are set above the base graph sizes for each recipe
    TASK_COUNT_RANGES = {
        'blast': (200, 500),        # Base graph ~183
        'bwa': (150, 400),          # Base graph ~106
        'cycles': (100, 500),       # Base graph ~69
        'epigenomics': (100, 400),  # Base graph varies
        'genome': (100, 400),       # Base graph varies
        'montage': (100, 400),      # Base graph varies
        'seismology': (100, 400),   # Base graph varies
        'soykb': (100, 400),        # Base graph varies
        'srasearch': (100, 400),    # Base graph varies
    }

    def get_sync_time(self, avg_task_weight: float, avg_compute_speed: float) -> float:
        base_sync_ratio = avg_task_weight / avg_compute_speed
        sync_time_multiplier = np.random.uniform(0.01, 0.05)
        return base_sync_ratio * sync_time_multiplier
    
    def get_num_tiles(self):
        return int(np.random.uniform(16, 512))

    def __init__(self, cache_dir: pathlib.Path, num_variations: int = 50):
        """Initialize the dataset generator.

        Args:
            cache_dir: Directory to store cached datasets
            num_variations: Number of variations per recipe
        """
        self.cache_dir = cache_dir
        self.num_variations = num_variations
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_task_graph(self, recipe_name: str, task_count: int) -> Tuple[nx.DiGraph, BSPHardware, TaskGraphMetadata]:
        """Generate a single task graph using specified recipe.

        Args:
            recipe_name: Name of the WfCommons recipe
            task_count: Number of tasks in the workflow

        Returns:
            Tuple of (scaled_task_graph, bsp_hardware, metadata)
        """
        if recipe_name not in self.RECIPES:
            raise ValueError(f"Unknown recipe: {recipe_name}")

        recipe_class = self.RECIPES[recipe_name]
        recipe = recipe_class.from_num_tasks(task_count)
        generator = WorkflowGenerator(recipe)

        # Generate workflow using WfCommons
        workflow = generator.build_workflow()

        # The workflow object IS a NetworkX DiGraph
        # Just copy it and ensure proper node/edge weights
        task_graph = workflow.copy()

        # Set node weights (task execution times)
        for node in task_graph.nodes():
            node_data = task_graph.nodes[node]
            # WfCommons stores the Task object in node data
            task_obj = node_data.get('task')
            if task_obj and hasattr(task_obj, 'runtime'):
                runtime = task_obj.runtime
                if runtime is None or runtime <= 0:
                    runtime = 1.0
            else:
                runtime = 1.0
            task_graph.nodes[node]['weight'] = runtime

        # Set edge weights (communication/data transfer sizes)
        # For WfCommons, edges represent data dependencies
        # We'll use the sum of input file sizes for the destination task
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

            # Use average per-edge size (divide by number of incoming edges)
            incoming_edges = list(task_graph.predecessors(dst))
            if len(incoming_edges) > 0:
                avg_size = total_size / len(incoming_edges)
            else:
                avg_size = total_size

            # Ensure minimum weight
            if avg_size <= 0:
                avg_size = 1.0

            task_graph.edges[src, dst]['weight'] = avg_size

        # Calculate original task and edge weights
        task_weights = [task_graph.nodes[node].get('weight', 1.0) for node in task_graph.nodes()]
        original_edge_weights = [task_graph.edges[edge].get('weight', 1.0) for edge in task_graph.edges()]

        avg_task_weight = np.mean(task_weights) if task_weights else 1.0
        original_avg_edge_weight = np.mean(original_edge_weights) if original_edge_weights else 1.0

        # Generate random CCR using log-uniform distribution between 0.1 and 5
        target_ccr = np.exp(np.random.uniform(np.log(0.1), np.log(5.0)))

        # Scale edge weights to achieve target CCR
        # CCR = avg_edge_weight / avg_task_weight
        # target_ccr = new_avg_edge_weight / avg_task_weight
        # new_avg_edge_weight = target_ccr * avg_task_weight
        scaling_factor = (target_ccr * avg_task_weight) / original_avg_edge_weight if original_avg_edge_weight > 0 else 1.0

        # Apply scaling to all edge weights
        for u, v in task_graph.edges():
            original_weight = task_graph.edges[u, v].get('weight', 1.0)
            task_graph.edges[u, v]['weight'] = original_weight * scaling_factor

        # Calculate new average edge weight after scaling
        scaled_edge_weights = [task_graph.edges[edge].get('weight', 1.0) for edge in task_graph.edges()]
        avg_edge_weight = np.mean(scaled_edge_weights) if scaled_edge_weights else 1.0

        # Generate BSP hardware with random sync time
        # saga_networks = gen_random_networks(num=1, num_nodes=64)
        # saga_network = saga_networks[0]
        
        bsp_hardware = IPUHardware(num_tiles=self.get_num_tiles(), sync_time=0.0)  # Placeholder sync_time, will be set below
        network = bsp_hardware.network

        # Calculate network compute speed (average node processing speed)
        network_node_weights = [network.nodes[node].get('weight', 1.0) for node in network.nodes()]
        avg_compute_speed = np.mean(network_node_weights) if network_node_weights else 1.0

        bsp_hardware.sync_time = self.get_sync_time(avg_task_weight, avg_compute_speed)

        # Create metadata
        metadata = TaskGraphMetadata(
            recipe_name=recipe_name,
            num_tasks=task_count,
            avg_task_weight=avg_task_weight,
            avg_edge_weight=avg_edge_weight,
            task_count=len(task_graph.nodes()),
            edge_count=len(task_graph.edges()),
            target_ccr=target_ccr,
            sync_time=bsp_hardware.sync_time
        )

        return task_graph, bsp_hardware, metadata

    def get_cached_dataset_path(self, recipe_name: str) -> pathlib.Path:
        """Get path for cached dataset file."""
        return self.cache_dir / f"{recipe_name}_cached.pkl"

    def get_cached_hardware_path(self, recipe_name: str) -> pathlib.Path:
        """Get path for cached BSP hardware file."""
        return self.cache_dir / f"{recipe_name}_hardware.pkl"

    def get_metadata_path(self, recipe_name: str) -> pathlib.Path:
        """Get path for dataset metadata file."""
        return self.cache_dir / f"{recipe_name}_metadata.json"

    def load_cached_dataset(self, recipe_name: str) -> Optional[Tuple[List[nx.DiGraph], List[BSPHardware], List[TaskGraphMetadata]]]:
        """Load cached dataset if it exists.

        Args:
            recipe_name: Name of the recipe

        Returns:
            Tuple of (task_graphs, bsp_hardware_list, metadata_list) or None if not cached
        """
        cache_path = self.get_cached_dataset_path(recipe_name)
        hardware_path = self.get_cached_hardware_path(recipe_name)
        metadata_path = self.get_metadata_path(recipe_name)

        if not cache_path.exists() or not hardware_path.exists() or not metadata_path.exists():
            return None

        try:
            # Load task graphs
            with open(cache_path, 'rb') as f:
                task_graphs = pickle.load(f)

            # Load BSP hardware
            with open(hardware_path, 'rb') as f:
                bsp_hardware_list = pickle.load(f)

            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata_dicts = json.load(f)
                metadata_list = [TaskGraphMetadata(**md) for md in metadata_dicts]

            logger.info(f"Loaded cached dataset for {recipe_name}: {len(task_graphs)} variations")
            return task_graphs, bsp_hardware_list, metadata_list

        except Exception as e:
            logger.warning(f"Failed to load cached dataset for {recipe_name}: {e}")
            return None

    def save_dataset(self, recipe_name: str, task_graphs: List[nx.DiGraph],
                     bsp_hardware_list: List[BSPHardware], metadata_list: List[TaskGraphMetadata]) -> None:
        """Save dataset to cache.

        Args:
            recipe_name: Name of the recipe
            task_graphs: List of generated task graphs
            bsp_hardware_list: List of BSP hardware configurations
            metadata_list: List of metadata for each task graph
        """
        cache_path = self.get_cached_dataset_path(recipe_name)
        hardware_path = self.get_cached_hardware_path(recipe_name)
        metadata_path = self.get_metadata_path(recipe_name)

        # Save task graphs
        with open(cache_path, 'wb') as f:
            pickle.dump(task_graphs, f)

        # Save BSP hardware
        with open(hardware_path, 'wb') as f:
            pickle.dump(bsp_hardware_list, f)

        # Save metadata as JSON
        metadata_dicts = [
            {
                'recipe_name': md.recipe_name,
                'num_tasks': md.num_tasks,
                'avg_task_weight': md.avg_task_weight,
                'avg_edge_weight': md.avg_edge_weight,
                'task_count': md.task_count,
                'edge_count': md.edge_count,
                'target_ccr': md.target_ccr,
                'sync_time': md.sync_time
            }
            for md in metadata_list
        ]

        with open(metadata_path, 'w') as f:
            json.dump(metadata_dicts, f, indent=2)

        logger.info(f"Cached dataset for {recipe_name}: {len(task_graphs)} variations")

    def generate_recipe_dataset(self, recipe_name: str, overwrite: bool = False) -> Tuple[List[nx.DiGraph], List[BSPHardware], List[TaskGraphMetadata]]:
        """Generate dataset for a specific recipe.

        Args:
            recipe_name: Name of the WfCommons recipe
            overwrite: Whether to overwrite cached data

        Returns:
            Tuple of (task_graphs, bsp_hardware_list, metadata_list)
        """
        # Check cache first
        if not overwrite:
            cached_data = self.load_cached_dataset(recipe_name)
            if cached_data is not None:
                return cached_data

        logger.info(f"Generating dataset for {recipe_name}...")

        # Get task count range for this recipe
        min_tasks, max_tasks = self.TASK_COUNT_RANGES[recipe_name]

        task_graphs = []
        bsp_hardware_list = []
        metadata_list = []

        # Generate variations
        for i in range(self.num_variations):
            # Random task count for this variation
            task_count = random.randint(min_tasks, max_tasks)

            try:
                task_graph, bsp_hardware, metadata = self.generate_task_graph(recipe_name, task_count)
                task_graphs.append(task_graph)
                bsp_hardware_list.append(bsp_hardware)
                metadata_list.append(metadata)

                if (i + 1) % 10 == 0:
                    logger.info(f"Generated {i + 1}/{self.num_variations} variations for {recipe_name}")

            except Exception as e:
                logger.warning(f"Failed to generate variation {i + 1} for {recipe_name}: {e}")
                continue

        # Save to cache
        if task_graphs:
            self.save_dataset(recipe_name, task_graphs, bsp_hardware_list, metadata_list)

        return task_graphs, bsp_hardware_list, metadata_list

    def generate_all_datasets(self, overwrite: bool = False) -> Dict[str, Tuple[List[nx.DiGraph], List[BSPHardware], List[TaskGraphMetadata]]]:
        """Generate datasets for all recipes.

        Args:
            overwrite: Whether to overwrite cached data

        Returns:
            Dictionary mapping recipe names to (task_graphs, bsp_hardware_list, metadata_list)
        """
        datasets = {}

        for recipe_name in self.RECIPES.keys():
            try:
                task_graphs, bsp_hardware_list, metadata_list = self.generate_recipe_dataset(recipe_name, overwrite)
                datasets[recipe_name] = (task_graphs, bsp_hardware_list, metadata_list)
                logger.info(f"Dataset {recipe_name}: {len(task_graphs)} variations generated")

            except Exception as e:
                logger.error(f"Failed to generate dataset for {recipe_name}: {e}")

        return datasets

    def adapt_task_graph_for_ccr(self, task_graph: nx.DiGraph, metadata: TaskGraphMetadata,
                                 target_ccr: float) -> nx.DiGraph:
        """Adapt task graph edge weights to achieve target CCR.

        Args:
            task_graph: Original task graph
            metadata: Task graph metadata
            target_ccr: Target communication-to-computation ratio

        Returns:
            Task graph with adapted edge weights
        """
        adapted_graph = task_graph.copy()

        # Calculate scaling factor for edge weights
        # CCR = (avg_edge_weight / avg_task_weight)
        # target_ccr = (new_avg_edge_weight / avg_task_weight)
        # scaling_factor = target_ccr * avg_task_weight / avg_edge_weight

        if metadata.avg_edge_weight > 0:
            scaling_factor = target_ccr * metadata.avg_task_weight / metadata.avg_edge_weight

            # Scale all edge weights
            for u, v in adapted_graph.edges():
                original_weight = adapted_graph.edges[u, v].get('weight', 1.0)
                adapted_graph.edges[u, v]['weight'] = original_weight * scaling_factor

        return adapted_graph