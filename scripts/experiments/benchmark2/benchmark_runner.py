"""
Benchmark runner for BSP scheduling experiments.

This module handles running benchmarks on datasets with different schedulers,
managing results storage and providing progress tracking.
"""

import logging
import pathlib
import pickle
from typing import Dict, List, Optional, Any
import pandas as pd
from joblib import Parallel, delayed

from dataset_generator import DatasetItem, load_dataset, find_datasets
from schedule_visualizer import save_schedule_visualization, should_save_visualization

logger = logging.getLogger(__name__)


def run_single_scheduler_task(scheduler_name, scheduler, dataset_item, task_graph_idx,
                             dataset_name=None, visualization_dir=None):
    """Run a single scheduler on a single dataset item."""
    logger.debug(f"Running scheduler {scheduler_name} on task graph {task_graph_idx}")
    try:
        # All schedulers now use the unified interface
        result = scheduler.schedule(dataset_item.hardware, dataset_item.task_graph)
        makespan = result['makespan']

        # Save schedule visualization for first task graph of each dataset/scheduler
        if (visualization_dir is not None and dataset_name is not None and
            should_save_visualization(task_graph_idx, dataset_name, scheduler_name)):
            save_schedule_visualization(
                schedule_result=result,
                scheduler_name=scheduler_name,
                dataset_name=dataset_name,
                task_graph_idx=task_graph_idx,
                output_dir=visualization_dir
            )

        return {
            'scheduler_name': scheduler_name,
            'makespan': makespan,
            'task_graph_idx': task_graph_idx,
            'target_ccr': dataset_item.metadata.get('target_ccr'),
            'actual_ccr': dataset_item.metadata.get('actual_ccr'),
            'sync_time': dataset_item.metadata.get('sync_time'),
            'num_tiles': dataset_item.metadata.get('num_tiles'),
            'source_type': dataset_item.metadata.get('source_type')
        }

    except Exception as e:
        logger.warning(f"Scheduler {scheduler_name} failed on task graph {task_graph_idx}: {e}")
        raise e




class BenchmarkRunner:
    """Runner for BSP scheduling benchmarks."""

    def __init__(self, schedulers: Dict[str, Any]):
        """Initialize benchmark runner.

        Args:
            schedulers: Dictionary of scheduler instances
        """
        self.schedulers = schedulers

    def run_dataset_benchmark(self, dataset_name: str, dataset_items: List[DatasetItem],
                             resultsdir: pathlib.Path, num_jobs: int = 1, overwrite: bool = False,
                             max_instances: int = 5, visualization_dir: Optional[pathlib.Path] = None) -> None:
        """Run benchmark on a specific dataset.

        Args:
            dataset_name: Name of the dataset
            dataset_items: List of dataset items (task graph + hardware + metadata)
            resultsdir: Directory to store results
            num_jobs: Number of parallel jobs
            overwrite: Whether to overwrite existing results
            max_instances: Maximum number of dataset items to use for benchmarking
            visualization_dir: Directory to save schedule visualizations (optional)
        """
        savepath = resultsdir / f"{dataset_name}.csv"
        if savepath.exists() and not overwrite:
            logger.info(f"Results for {dataset_name} already exist. Skipping.")
            return

        # Limit number of dataset items for benchmarking
        if max_instances > 0 and len(dataset_items) > max_instances:
            dataset_items = dataset_items[:max_instances]
            logger.info(f"Limited to {max_instances} instances for {dataset_name}")

        logger.info(f"Running benchmark on {dataset_name} with {len(dataset_items)} dataset items")

        # Create all scheduler-dataset_item combinations for parallel execution
        tasks = []
        for i, dataset_item in enumerate(dataset_items):
            for scheduler_name, scheduler in self.schedulers.items():
                tasks.append((scheduler_name, scheduler, dataset_item, i, dataset_name, visualization_dir))

        logger.info(f"Running {len(tasks)} scheduler-task combinations in parallel with {num_jobs} jobs")

        # Execute all combinations in parallel
        parallel_results = Parallel(n_jobs=num_jobs, verbose=1)(
            delayed(run_single_scheduler_task)(
                scheduler_name, scheduler, dataset_item, task_graph_idx, dataset_name, visualization_dir
            ) for scheduler_name, scheduler, dataset_item, task_graph_idx, dataset_name, visualization_dir in tasks
        )

        # Filter out None results (failed executions)
        valid_results = [r for r in parallel_results if r is not None]

        # Group results by task graph and calculate ratios
        results = []
        for task_graph_idx in range(len(dataset_items)):
            # Get results for this task graph
            task_results = [r for r in valid_results if r['task_graph_idx'] == task_graph_idx]

            if not task_results:
                continue

            # Calculate ratios relative to best scheduler for this task graph
            makespans = {r['scheduler_name']: r['makespan'] for r in task_results}
            best_makespan = min(makespans.values())

            for result in task_results:
                ratio = result['makespan'] / best_makespan
                results.append({
                    'scheduler': result['scheduler_name'],
                    'makespan': result['makespan'],
                    'makespan_ratio': ratio,
                    'task_graph_idx': result['task_graph_idx'],
                    'target_ccr': result.get('target_ccr'),
                    'actual_ccr': result.get('actual_ccr'),
                    'sync_time': result.get('sync_time'),
                    'num_tiles': result.get('num_tiles'),
                    'source_type': result.get('source_type')
                })

        # Convert results to DataFrame
        import pandas as pd
        df_comp = pd.DataFrame(results)

        # Save results
        savepath.parent.mkdir(exist_ok=True, parents=True)
        df_comp.to_csv(savepath)
        logger.info(f"Saved results to {savepath}")

    def run_all_benchmarks(self, datadir: pathlib.Path, resultsdir: pathlib.Path,
                          dataset_names: Optional[List[str]] = None, num_jobs: int = 1,
                          overwrite: bool = False, max_instances: int = 5,
                          visualization_dir: Optional[pathlib.Path] = None) -> None:
        """Run benchmarks on all datasets.

        Args:
            datadir: Directory containing cached datasets
            resultsdir: Directory to store results
            dataset_names: Optional list of specific datasets to benchmark
            num_jobs: Number of parallel jobs
            overwrite: Whether to overwrite existing results
            max_instances: Maximum number of instances per dataset (0 = no limit)
            visualization_dir: Directory to save schedule visualizations (optional)
        """
        # Find all available dataset files
        dataset_files = find_datasets(datadir)

        if not dataset_files:
            logger.warning("No dataset files found in {datadir}")
            return

        # Extract dataset names from file names
        available_datasets = []
        for dataset_file in dataset_files:
            # Remove "_dataset.pkl" suffix to get dataset name
            dataset_key = dataset_file.stem.replace("_dataset", "")
            available_datasets.append((dataset_key, dataset_file))

        # Filter datasets if specific names requested
        if dataset_names:
            filtered_datasets = []
            for requested_name in dataset_names:
                found = False
                for dataset_key, dataset_file in available_datasets:
                    if requested_name in dataset_key:
                        filtered_datasets.append((dataset_key, dataset_file))
                        found = True
                        break
                if not found:
                    logger.warning(f"Dataset {requested_name} not found")
            available_datasets = filtered_datasets

        logger.info(f"Running benchmarks on {len(available_datasets)} datasets: {[name for name, _ in available_datasets]}")

        # Run benchmarks for each dataset
        for dataset_key, dataset_file in available_datasets:
            try:
                # Load dataset
                dataset_items, dataset_metadata = load_dataset(dataset_file)

                if not dataset_items:
                    logger.warning(f"No items in dataset {dataset_key}, skipping")
                    continue

                # Use display name from metadata if available, fallback to dataset_key
                display_name = getattr(dataset_metadata, 'dataset_name', dataset_key)

                # Run benchmark
                self.run_dataset_benchmark(
                    dataset_name=display_name,
                    dataset_items=dataset_items,
                    resultsdir=resultsdir,
                    num_jobs=num_jobs,
                    overwrite=overwrite,
                    max_instances=max_instances,
                    visualization_dir=visualization_dir
                )

            except Exception as e:
                logger.error(f"Failed to run benchmark on {dataset_key}: {e}")

    def load_results(self, resultsdir: pathlib.Path, glob: str = "*.csv") -> pd.DataFrame:
        """Load benchmark results from CSV files.

        Args:
            resultsdir: Directory containing result files
            glob: Glob pattern for result files

        Returns:
            Combined DataFrame with all results
        """
        data = None
        for path in resultsdir.glob(glob):
            df_dataset = pd.read_csv(path, index_col=0)
            df_dataset["dataset"] = path.stem
            if data is None:
                data = df_dataset
            else:
                data = pd.concat([data, df_dataset], ignore_index=True)

        if data is None:
            return pd.DataFrame()

        # Clean up scheduler names (remove "Scheduler" suffix)
        if 'scheduler' in data.columns:
            data["scheduler"] = data["scheduler"].str.replace("Scheduler", "")

        return data

    def get_benchmark_summary(self, resultsdir: pathlib.Path) -> Dict[str, Any]:
        """Get summary statistics of benchmark results.

        Args:
            resultsdir: Directory containing result files

        Returns:
            Dictionary with summary statistics
        """
        data = self.load_results(resultsdir)

        if data.empty:
            return {"error": "No data found"}

        summary = {
            "total_experiments": len(data),
            "num_datasets": data["dataset"].nunique(),
            "num_schedulers": data["scheduler"].nunique(),
            "datasets": sorted(data["dataset"].unique()),
            "schedulers": sorted(data["scheduler"].unique()),
            "makespan_stats": {
                "min": data["makespan"].min(),
                "max": data["makespan"].max(),
                "mean": data["makespan"].mean(),
                "std": data["makespan"].std()
            }
        }

        if "makespan_ratio" in data.columns:
            summary["makespan_ratio_stats"] = {
                "min": data["makespan_ratio"].min(),
                "max": data["makespan_ratio"].max(),
                "mean": data["makespan_ratio"].mean(),
                "std": data["makespan_ratio"].std()
            }

        return summary