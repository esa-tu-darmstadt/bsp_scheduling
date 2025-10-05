"""
Dataset generation using WfCommons and SPNs with caching and metadata.

This module generates task graphs using different sources:
- WfCommons workflow recipes (with optional CCR adjustment)
- Sum-Product Networks (SPNs) (with natural bit/cycle weights)

Datasets are stored as single pickle files containing both task graph and hardware.
"""

import logging
import pathlib
import pickle
import random
from typing import Callable, Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import networkx as nx
import numpy as np
from saga_bsp.misc.saga_scheduler_wrapper import preprocess_task_graph
from saga_bsp.schedule import BSPHardware
from saga_bsp.hardware import IPUHardware
from saga_bsp.task_graphs import (
    WfCommonsTaskGraphGenerator, SPNTaskGraphGenerator,
    calculate_ccr, adjust_task_graph_to_ccr
)
from saga.schedulers.data.random import gen_in_trees, gen_out_trees, gen_parallel_chains


logger = logging.getLogger(__name__)

@dataclass
class DatasetItem:
    """Single dataset item containing task graph, hardware, and metadata."""
    task_graph: nx.DiGraph
    hardware: BSPHardware
    metadata: Dict[str, Any]

@dataclass
class DatasetMetadata:
    """Metadata for an entire dataset."""
    source_type: str  # 'wfcommons' or 'spn'
    source_name: str  # recipe name or SPN filename
    dataset_name: str  # display name for visualizations
    num_variations: int
    tile_counts: List[int]
    generated_timestamp: str
    additional_info: Optional[Dict[str, Any]] = None

def generate_wfcommons_dataset(cache_dir: pathlib.Path, recipe_name: str,
                               get_task_count: Optional[Callable[[int, List[int]], int]] = None,
                               tile_counts: List[int] = [4, 16, 32, 92],
                               get_variations_per_tile: Callable[[List[int]], int] = None,
                               overwrite_cache: bool = False) -> Tuple[List[DatasetItem], str]:
    """Generate datasets for WfCommons workflows.

    Args:
        cache_dir: Directory to store cached datasets
        recipe_name: WfCommons recipe name
        get_task_count: Function to determine task count of a variation given the iteration index and a list of task counts.
        tile_counts: List of tile counts to generate for
        get_variations_per_tile: Function to determine number of variations per tile count (given tile_counts)
        overwrite_cache: Whether to overwrite existing cache

    Returns:
        Tuple of (dataset_items, dataset_display_name)
    """
    
    # Default: use all available task counts for the recipe
    if get_task_count is None:
        get_task_count = lambda idx, task_counts: task_counts[idx % len(tile_counts)]
        
    # Default: up to 5 iterations per tile count
    if get_variations_per_tile is None:
        get_variations_per_tile = lambda tile_counts: max(5, len(tile_counts))

    wf_generator = WfCommonsTaskGraphGenerator(cache_dir=cache_dir)
    dataset_display_name = recipe_name  # Use recipe name as display name

    # Create cache key
    cache_key = f"wfcommons_{recipe_name.replace('.', '_').replace('/', '_')}"

    cache_path = cache_dir / f"{cache_key}_dataset.pkl"

    # Check cache first
    if not overwrite_cache and cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                dataset_items, _ = pickle.load(f)
            logger.info(f"Loaded cached WfCommons dataset {cache_key}: {len(dataset_items)} items")
            return dataset_items, dataset_display_name
        except Exception as e:
            logger.warning(f"Failed to load cached dataset {cache_key}: {e}")

    logger.info(f"Generating WfCommons dataset for {recipe_name}...")

    dataset_items = []
    variations_per_tile = get_variations_per_tile(tile_counts)
    total_variations = len(tile_counts) * variations_per_tile
    variation_count = 0

    for tile_count in tile_counts:
        for variation_idx in range(variations_per_tile):
            variation_count += 1

            try:
                
                # Set deterministic seed for reproducibility
                seed = hash(f"wfcommons_{recipe_name}_{tile_count}_{variation_idx}") % (2**32)
                random.seed(seed)
                np.random.seed(seed)

                task_count = get_task_count(variation_idx, wf_generator.TASK_COUNTS[recipe_name])

                # Generate base task graph
                task_graph, base_metadata = wf_generator.generate_task_graph(
                    recipe_name, task_count=task_count
                )

                bsp_hardware = IPUHardware(num_tiles=tile_count, sync_time=0)

                # Sync time = 1% of average computation time
                bsp_hardware.sync_time = base_metadata.avg_task_weight / bsp_hardware.avg_computation_speed * 0.01

                # Apply random CCR adjustment
                target_ccr = 0.1
                adjust_task_graph_to_ccr(task_graph, bsp_hardware.network, target_ccr)

                # Calculate final CCR for metadata
                final_ccr = calculate_ccr(task_graph, bsp_hardware.network)

                # Create metadata
                metadata = {
                    'source_type': 'wfcommons',
                    'source_name': recipe_name,
                    'dataset_name': dataset_display_name,
                    'task_count': base_metadata.task_count,
                    'edge_count': base_metadata.edge_count,
                    'avg_task_weight': base_metadata.avg_task_weight,
                    'avg_edge_weight': base_metadata.avg_edge_weight,
                    'num_tiles': tile_count,
                    'sync_time': bsp_hardware.sync_time,
                    'target_ccr': target_ccr,
                    'actual_ccr': final_ccr,
                    'base_metadata': base_metadata.additional_info
                }

                dataset_items.append(DatasetItem(task_graph=task_graph, hardware=bsp_hardware, metadata=metadata))

                if variation_count % 5 == 0:
                    logger.info(f"Generated {variation_count}/{total_variations} WfCommons variations")

            except Exception as e:
                logger.warning(f"Failed to generate WfCommons variation {variation_count}: {e}")
                continue

    # Reset random seeds
    random.seed(None)
    np.random.seed(None)

    # Save to cache
    if dataset_items:
        dataset_metadata = DatasetMetadata(
            source_type='wfcommons',
            source_name=recipe_name,
            dataset_name=dataset_display_name,
            num_variations=len(dataset_items),
            tile_counts=tile_counts,
            generated_timestamp=str(np.datetime64('now')),
            additional_info={'task_count': task_count} if task_count else None
        )

        with open(cache_path, 'wb') as f:
            pickle.dump((dataset_items, dataset_metadata), f)
        logger.info(f"Cached WfCommons dataset {cache_key}: {len(dataset_items)} items")

    return dataset_items, dataset_display_name


def generate_spn_dataset(cache_dir: pathlib.Path, spn_filename: str,
                         spn_data_dir: Optional[pathlib.Path] = None,
                         tile_counts: List[int] = [4, 16, 32, 92, 184],
                         overwrite_cache: bool = False) -> Tuple[List[DatasetItem], str]:
    """Generate datasets for Sum-Product Networks.

    Args:
        cache_dir: Directory to store cached datasets
        spn_filename: SPN filename
        spn_data_dir: Directory containing SPN data files
        tile_counts: List of tile counts to generate for
        overwrite_cache: Whether to overwrite existing cache

    Returns:
        Tuple of (dataset_items, dataset_display_name)
    """
    # Set up SPN data directory
    if spn_data_dir is None:
        spn_data_dir = pathlib.Path(__file__).parent.parent.parent.parent / "data" / "spn"

    if not spn_data_dir.exists():
        raise ValueError(f"SPN data directory not found: {spn_data_dir}")

    spn_generator = SPNTaskGraphGenerator(cache_dir=cache_dir, spn_schema_path=spn_data_dir / "spflow.capnp")
    dataset_display_name = spn_filename.split('_')[0]  # Use filename until first underscore

    # Create cache key
    cache_key = f"spn_{spn_filename.replace('.', '_').replace('/', '_')}"
    cache_path = cache_dir / f"{cache_key}_dataset.pkl"

    # Check cache first
    if not overwrite_cache and cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                dataset_items, _ = pickle.load(f)
            logger.info(f"Loaded cached SPN dataset {cache_key}: {len(dataset_items)} items")
            return dataset_items, dataset_display_name
        except Exception as e:
            logger.warning(f"Failed to load cached dataset {cache_key}: {e}")

    logger.info(f"Generating SPN dataset for {spn_filename}...")

    dataset_items = []
    variation_count = 0

    for tile_count in tile_counts:
        variation_count += 1

        try:
            # Set deterministic seed for reproducibility
            seed = hash(f"spn_{spn_filename}_{tile_count}") % (2**32)
            random.seed(seed)
            np.random.seed(seed)

            # Generate SPN task graph with natural bit/cycle weights
            spn_path = spn_data_dir / spn_filename
            task_graph, base_metadata = spn_generator.generate_task_graph(str(spn_path))

            # Create IPU hardware with sync time = 100 cycles
            bsp_hardware = IPUHardware(num_tiles=tile_count, sync_time=100.0)

            # Calculate natural CCR for metadata
            final_ccr = calculate_ccr(task_graph, bsp_hardware.network)

            # Create metadata
            metadata = {
                'source_type': 'spn',
                'source_name': spn_filename,
                'dataset_name': dataset_display_name,
                'task_count': base_metadata.task_count,
                'edge_count': base_metadata.edge_count,
                'avg_task_weight': base_metadata.avg_task_weight,
                'avg_edge_weight': base_metadata.avg_edge_weight,
                'num_tiles': tile_count,
                'sync_time': bsp_hardware.sync_time,
                'target_ccr': None,  # No CCR adjustment for SPNs
                'actual_ccr': final_ccr,
                'base_metadata': base_metadata.additional_info
            }

            dataset_items.append(DatasetItem(task_graph=task_graph, hardware=bsp_hardware, metadata=metadata))

            if variation_count % 5 == 0:
                logger.info(f"Generated {variation_count}/{len(tile_counts)} SPN variations")

        except Exception as e:
            logger.warning(f"Failed to generate SPN variation {variation_count}: {e}")
            continue

    # Reset random seeds
    random.seed(None)
    np.random.seed(None)

    # Save to cache
    if dataset_items:
        dataset_metadata = DatasetMetadata(
            source_type='spn',
            source_name=spn_filename,
            dataset_name=dataset_display_name,
            num_variations=len(dataset_items),
            tile_counts=tile_counts,
            generated_timestamp=str(np.datetime64('now'))
        )

        with open(cache_path, 'wb') as f:
            pickle.dump((dataset_items, dataset_metadata), f)
        logger.info(f"Cached SPN dataset {cache_key}: {len(dataset_items)} items")

    return dataset_items, dataset_display_name


def generate_wfcommons_datasets(cache_dir: pathlib.Path,
                               get_task_count: Optional[Callable[[int], int]] = None,
                               tile_counts: List[int] = [2, 4, 16, 32, 92, 184],
                               get_variations_per_tile: Callable[[List[int]], int] = None,
                               overwrite_cache: bool = False) -> Dict[str, Tuple[List[DatasetItem], str]]:
    """Generate datasets for all available WfCommons recipes.

    Args:
        cache_dir: Directory to store cached datasets
        get_task_count: Function to determine task count of a variation given min_tasks.
        tile_counts: List of tile counts to generate for
        get_variations_per_tile: Function to determine number of variations per tile count (given tile_counts)
        overwrite_cache: Whether to overwrite existing cache

    Returns:
        Dictionary mapping dataset keys to (dataset_items, dataset_display_name)
    """
    wf_generator = WfCommonsTaskGraphGenerator(cache_dir=cache_dir)
    available_recipes = wf_generator.list_available_recipes()

    datasets = {}
    for recipe_name in available_recipes:
        try:
            dataset_items, display_name = generate_wfcommons_dataset(
                cache_dir=cache_dir,
                recipe_name=recipe_name,
                get_task_count=get_task_count,
                tile_counts=tile_counts,
                get_variations_per_tile=get_variations_per_tile,
                overwrite_cache=overwrite_cache
            )
            datasets[f"wfcommons_{recipe_name}"] = (dataset_items, display_name)
            logger.info(f"Generated WfCommons dataset: {recipe_name} ({len(dataset_items)} items)")
        except Exception as e:
            logger.error(f"Failed to generate WfCommons dataset {recipe_name}: {e}")

    return datasets


def generate_spn_datasets(cache_dir: pathlib.Path,
                         spn_data_dir: Optional[pathlib.Path] = None,
                         tile_counts: List[int] = [2, 4, 16, 32, 92, 184],
                         overwrite_cache: bool = False) -> Dict[str, Tuple[List[DatasetItem], str]]:
    """Generate datasets for all available SPN files.

    Args:
        cache_dir: Directory to store cached datasets
        spn_data_dir: Directory containing SPN data files
        tile_counts: List of tile counts to generate for
        overwrite_cache: Whether to overwrite existing cache

    Returns:
        Dictionary mapping dataset keys to (dataset_items, dataset_display_name)
    """
    # Set up SPN data directory
    if spn_data_dir is None:
        spn_data_dir = pathlib.Path(__file__).parent.parent.parent.parent / "data" / "spn"

    if not spn_data_dir.exists():
        logger.warning(f"SPN data directory not found: {spn_data_dir}")
        return {}

    spn_generator = SPNTaskGraphGenerator(cache_dir=cache_dir, spn_schema_path=spn_data_dir / "spflow.capnp")
    available_spns = spn_generator.list_available_spns(spn_data_dir)

    datasets = {}
    for spn_filename in available_spns:
        try:
            dataset_items, display_name = generate_spn_dataset(
                cache_dir=cache_dir,
                spn_filename=spn_filename,
                spn_data_dir=spn_data_dir,
                tile_counts=tile_counts,
                overwrite_cache=overwrite_cache
            )
            datasets[f"spn_{spn_filename.replace('.', '_')}"] = (dataset_items, display_name)
            logger.info(f"Generated SPN dataset: {spn_filename} ({len(dataset_items)} items)")
        except Exception as e:
            logger.error(f"Failed to generate SPN dataset {spn_filename}: {e}")

    return datasets


def generate_primitives_dataset(cache_dir: pathlib.Path, graph_type: str,
                               tile_counts: List[int] = [2, 4, 16, 32, 92, 184],
                               variations_per_tile: int = 5,
                               overwrite_cache: bool = False) -> Tuple[List[DatasetItem], str]:
    """Generate dataset for a primitive graph type using SAGA.

    Args:
        cache_dir: Directory to store cached datasets
        graph_type: Type of graph ('in_tree', 'out_tree', 'parallel_chains')
        tile_counts: List of tile counts to generate for
        variations_per_tile: Number of variations per tile count
        overwrite_cache: Whether to overwrite existing cache

    Returns:
        Tuple of (dataset_items, dataset_display_name)
    """
    dataset_display_name = f"{graph_type}"

    # Create cache key (no config string, all variations in one file)
    cache_key = f"primitives_{graph_type}"
    cache_path = cache_dir / f"{cache_key}_dataset.pkl"

    # Check cache first
    if not overwrite_cache and cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                dataset_items, _ = pickle.load(f)
            logger.info(f"Loaded cached primitives dataset {cache_key}: {len(dataset_items)} items")
            return dataset_items, dataset_display_name
        except Exception as e:
            logger.warning(f"Failed to load cached dataset {cache_key}: {e}")

    logger.info(f"Generating primitives dataset for {graph_type}...")

    dataset_items = []
    total_variations = len(tile_counts) * variations_per_tile
    variation_count = 0

    for tile_count in tile_counts:
        for variation_idx in range(variations_per_tile):
            variation_count += 1

            try:
                # Set deterministic seed for reproducibility
                seed = hash(f"primitives_{graph_type}_{tile_count}_{variation_idx}") % (2**32)
                random.seed(seed)
                np.random.seed(seed)

                # Generate random configuration for this variation (like WfCommons does)
                if graph_type == 'in_tree':
                    num_levels = np.random.randint(3, 6)
                    branching_factor = np.random.randint(3, 6)
                    task_graphs = gen_in_trees(1, num_levels, branching_factor,
                                               lambda task_id: np.random.uniform(1.0, 10.0),
                                               lambda src_id, dst_id: np.random.uniform(0.5, 5.0))
                    config_str = f"{num_levels}_{branching_factor}"
                elif graph_type == 'out_tree':
                    num_levels = np.random.randint(3, 6)
                    branching_factor = np.random.randint(3, 6)
                    task_graphs = gen_out_trees(1, num_levels, branching_factor,
                                                lambda task_id: np.random.uniform(1.0, 10.0),
                                                lambda src_id, dst_id: np.random.uniform(0.5, 5.0))
                    config_str = f"{num_levels}_{branching_factor}"
                elif graph_type == 'parallel_chains':
                    num_chains = np.random.randint(5, 50)
                    chain_length = np.random.randint(5, 50)
                    task_graphs = gen_parallel_chains(1, num_chains, chain_length,
                                                      lambda task_id: np.random.uniform(1.0, 10.0),
                                                      lambda src_id, dst_id: np.random.uniform(0.5, 5.0))
                    config_str = f"{num_chains}_{chain_length}"
                else:
                    raise ValueError(f"Unsupported graph type: {graph_type}")

                task_graph = task_graphs[0]
                task_graph, _ = preprocess_task_graph(task_graph)

                bsp_hardware = IPUHardware(num_tiles=tile_count, sync_time=0)

                # Calculate average task weight for sync time calculation
                avg_task_weight = np.mean([task_graph.nodes[node]['weight'] for node in task_graph.nodes()])
                bsp_hardware.sync_time = avg_task_weight / bsp_hardware.avg_computation_speed * 0.01

                # Apply random CCR adjustment
                target_ccr = 0.1
                adjust_task_graph_to_ccr(task_graph, bsp_hardware.network, target_ccr)

                # Calculate final CCR for metadata
                final_ccr = calculate_ccr(task_graph, bsp_hardware.network)

                # Calculate metadata
                avg_edge_weight = np.mean([task_graph.edges[edge]['weight'] for edge in task_graph.edges()]) if task_graph.edges() else 0

                # Create metadata
                metadata = {
                    'source_type': 'primitives',
                    'source_name': graph_type,
                    'dataset_name': dataset_display_name,
                    'task_count': len(task_graph.nodes()),
                    'edge_count': len(task_graph.edges()),
                    'avg_task_weight': avg_task_weight,
                    'avg_edge_weight': avg_edge_weight,
                    'num_tiles': tile_count,
                    'sync_time': bsp_hardware.sync_time,
                    'target_ccr': target_ccr,
                    'actual_ccr': final_ccr,
                    'graph_type': graph_type,
                    'config': config_str,
                    'base_metadata': None
                }

                dataset_items.append(DatasetItem(task_graph=task_graph, hardware=bsp_hardware, metadata=metadata))

                if variation_count % 5 == 0:
                    logger.info(f"Generated {variation_count}/{total_variations} {graph_type} variations")

            except Exception as e:
                logger.warning(f"Failed to generate {graph_type} variation {variation_count}: {e}")
                continue

    # Reset random seeds
    random.seed(None)
    np.random.seed(None)

    # Save to cache
    if dataset_items:
        dataset_metadata = DatasetMetadata(
            source_type='primitives',
            source_name=graph_type,
            dataset_name=dataset_display_name,
            num_variations=len(dataset_items),
            tile_counts=tile_counts,
            generated_timestamp=str(np.datetime64('now')),
            additional_info={'graph_type': graph_type}
        )

        with open(cache_path, 'wb') as f:
            pickle.dump((dataset_items, dataset_metadata), f)
        logger.info(f"Cached primitives dataset {cache_key}: {len(dataset_items)} items")

    return dataset_items, dataset_display_name


def generate_primitives_datasets(cache_dir: pathlib.Path,
                                tile_counts: List[int] = [2, 4, 16, 32, 92, 184],
                                variations_per_tile: int = 10,
                                overwrite_cache: bool = False) -> Dict[str, Tuple[List[DatasetItem], str]]:
    """Generate datasets for all primitive graph types.

    Args:
        cache_dir: Directory to store cached datasets
        tile_counts: List of tile counts to generate for
        variations_per_tile: Number of variations per tile count
        overwrite_cache: Whether to overwrite existing cache

    Returns:
        Dictionary mapping dataset keys to (dataset_items, dataset_display_name)
    """
    datasets = {}
    primitive_types = ['in_tree', 'out_tree', 'parallel_chains']

    for graph_type in primitive_types:
        try:
            dataset_items, display_name = generate_primitives_dataset(
                cache_dir=cache_dir,
                graph_type=graph_type,
                tile_counts=tile_counts,
                variations_per_tile=variations_per_tile,
                overwrite_cache=overwrite_cache
            )
            datasets[f"primitives_{display_name}"] = (dataset_items, display_name)
            logger.info(f"Generated {graph_type} primitives dataset ({len(dataset_items)} items)")
        except Exception as e:
            logger.error(f"Failed to generate {graph_type} primitives dataset: {e}")

    return datasets


# Reusable dataset parsing functions
def load_dataset(dataset_path: pathlib.Path) -> Tuple[List[DatasetItem], DatasetMetadata]:
    """Reusable function to load a dataset from a pickle file.

    Args:
        dataset_path: Path to the dataset pickle file

    Returns:
        Tuple of (dataset_items, dataset_metadata)

    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If dataset file is corrupted
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    try:
        with open(dataset_path, 'rb') as f:
            dataset_items, dataset_metadata = pickle.load(f)

        # Validate loaded data
        if not isinstance(dataset_items, list) or not isinstance(dataset_metadata, DatasetMetadata):
            raise ValueError("Invalid dataset format")

        logger.info(f"Loaded dataset: {len(dataset_items)} items, source: {dataset_metadata.source_type}:{dataset_metadata.source_name}")
        return dataset_items, dataset_metadata

    except Exception as e:
        raise ValueError(f"Failed to load dataset from {dataset_path}: {e}")


def find_datasets(cache_dir: pathlib.Path, source_type: Optional[str] = None) -> List[pathlib.Path]:
    """Find available dataset files in cache directory.

    Args:
        cache_dir: Directory to search for datasets
        source_type: Filter by source type ('wfcommons' or 'spn'), None for all

    Returns:
        List of dataset file paths
    """
    if not cache_dir.exists():
        return []

    pattern = "*_dataset.pkl"
    if source_type:
        pattern = f"{source_type}_*_dataset.pkl"

    return list(cache_dir.glob(pattern))


def parse_dataset_for_experiments(dataset_items: List[DatasetItem]) -> Tuple[List[nx.DiGraph], List[BSPHardware], List[Dict]]:
    """Parse dataset items into separate lists for experiments.

    This is a compatibility function for existing benchmark code.

    Args:
        dataset_items: List of dataset items

    Returns:
        Tuple of (task_graphs, hardware_list, metadata_list)
    """
    task_graphs = [item.task_graph for item in dataset_items]
    hardware_list = [item.hardware for item in dataset_items]
    metadata_list = [item.metadata for item in dataset_items]

    return task_graphs, hardware_list, metadata_list


def get_dataset_statistics(dataset_items: List[DatasetItem]) -> Dict[str, Any]:
    """Get statistics for a dataset.

    Args:
        dataset_items: List of dataset items

    Returns:
        Dictionary with dataset statistics
    """
    if not dataset_items:
        return {'num_items': 0}

    # Calculate statistics
    task_counts = [item.metadata['task_count'] for item in dataset_items]
    edge_counts = [item.metadata['edge_count'] for item in dataset_items]
    tile_counts = [item.metadata['num_tiles'] for item in dataset_items]
    ccrs = [item.metadata.get('actual_ccr', 0) for item in dataset_items]

    stats = {
        'num_items': len(dataset_items),
        'source_type': dataset_items[0].metadata['source_type'],
        'source_name': dataset_items[0].metadata['source_name'],
        'task_count_range': (min(task_counts), max(task_counts)),
        'edge_count_range': (min(edge_counts), max(edge_counts)),
        'tile_count_range': (min(tile_counts), max(tile_counts)),
        'ccr_range': (min(ccrs), max(ccrs)),
        'avg_ccr': np.mean(ccrs),
    }

    return stats
