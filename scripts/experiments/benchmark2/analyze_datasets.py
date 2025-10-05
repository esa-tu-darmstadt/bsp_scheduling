#!/usr/bin/env python3
"""
Analyze existing datasets and output task count ranges.

This script scans all cached dataset pickle files and displays statistics
about the number of tasks per dataset type.
"""

import pathlib
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))

from scripts.experiments.benchmark2.dataset_generator import DatasetItem, DatasetMetadata


def load_dataset(dataset_path: pathlib.Path) -> Tuple[List[DatasetItem], DatasetMetadata]:
    """Load a dataset from pickle file."""
    with open(dataset_path, 'rb') as f:
        dataset_items, dataset_metadata = pickle.load(f)
    return dataset_items, dataset_metadata


def analyze_datasets(cache_dir: pathlib.Path):
    """Analyze all datasets in cache directory and print statistics."""

    # Find all dataset files
    dataset_files = sorted(cache_dir.glob("*_dataset.pkl"))

    if not dataset_files:
        print(f"No dataset files found in {cache_dir}")
        return

    print(f"Found {len(dataset_files)} datasets in {cache_dir}\n")
    print("=" * 80)

    # Group datasets by source type and name (merged)
    datasets_by_type = defaultdict(lambda: defaultdict(lambda: {
        'task_counts': [],
        'tile_counts': set(),
        'num_items': 0,
        'files': [],
        'items_per_tile': defaultdict(int)  # Track variations per tile count
    }))

    for dataset_path in dataset_files:
        try:
            dataset_items, metadata = load_dataset(dataset_path)

            # Extract task counts from all items
            task_counts = [item.metadata['task_count'] for item in dataset_items]

            # Merge by display name (this groups primitives by type)
            display_name = metadata.dataset_name
            datasets_by_type[metadata.source_type][display_name]['task_counts'].extend(task_counts)
            datasets_by_type[metadata.source_type][display_name]['tile_counts'].update(metadata.tile_counts)
            datasets_by_type[metadata.source_type][display_name]['num_items'] += len(dataset_items)
            datasets_by_type[metadata.source_type][display_name]['files'].append(dataset_path.name)

            # Count items per tile configuration
            for item in dataset_items:
                num_tiles = item.metadata['num_tiles']
                datasets_by_type[metadata.source_type][display_name]['items_per_tile'][num_tiles] += 1

        except Exception as e:
            print(f"Error loading {dataset_path.name}: {e}")

    # Print summary by source type
    for source_type in sorted(datasets_by_type.keys()):
        print(f"\n{source_type.upper()} Datasets:")
        print("-" * 80)

        # Sort by minimum task count
        datasets = sorted(datasets_by_type[source_type].items(),
                         key=lambda x: min(x[1]['task_counts']))

        for display_name, data in datasets:
            task_count_min = min(data['task_counts'])
            task_count_max = max(data['task_counts'])
            tile_counts_sorted = sorted(data['tile_counts'])
            tile_range = f"{min(tile_counts_sorted)}-{max(tile_counts_sorted)}"
            num_variants = len(data['files'])

            # Format items per tile: tile_count:num_variations
            items_per_tile_str = ", ".join(f"{tile}:{count}"
                                           for tile, count in sorted(data['items_per_tile'].items()))

            print(f"  {display_name:30} | Tasks: {task_count_min:6} - {task_count_max:6} | "
                  f"Tiles: {tile_range:8} | Items: {data['num_items']:3} | Variants: {num_variants:2}")
            print(f"  {' '*30} | Variations per tile: {items_per_tile_str}")

    print("\n" + "=" * 80)

    # Overall summary
    all_task_counts = []
    total_datasets = 0
    total_items = 0

    for source_type, datasets in datasets_by_type.items():
        for display_name, data in datasets.items():
            all_task_counts.extend(data['task_counts'])
            total_datasets += 1
            total_items += data['num_items']

    if all_task_counts:
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total datasets: {total_datasets}")
        print(f"  Total dataset items: {total_items}")
        print(f"  Task count range (across all): {min(all_task_counts)} - {max(all_task_counts)}")

        # Per-type summary
        print(f"\n  By source type:")
        for source_type in sorted(datasets_by_type.keys()):
            datasets = datasets_by_type[source_type]
            type_task_counts = []
            for display_name, data in datasets.items():
                type_task_counts.extend(data['task_counts'])
            print(f"    {source_type:15} | Datasets: {len(datasets):3} | Task range: {min(type_task_counts):6} - {max(type_task_counts):6}")


if __name__ == "__main__":
    # Default cache directory
    cache_dir = pathlib.Path(__file__).parent / "data"

    # Allow override via command line
    if len(sys.argv) > 1:
        cache_dir = pathlib.Path(sys.argv[1])

    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}")
        sys.exit(1)

    analyze_datasets(cache_dir)
