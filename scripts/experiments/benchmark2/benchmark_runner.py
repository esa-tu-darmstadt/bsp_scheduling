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

from dataset_generator import DatasetGenerator, TaskGraphMetadata
from schedule_visualizer import save_schedule_visualization, should_save_visualization

logger = logging.getLogger(__name__)


def run_single_scheduler_task(scheduler_name, scheduler, task_graph, bsp_hardware, metadata, task_graph_idx,
                             dataset_name=None, visualization_dir=None):
    """Run a single scheduler on a single task graph."""
    logger.debug(f"Running scheduler {scheduler_name} on task graph {task_graph_idx}")
    try:
        # All schedulers now use the unified interface
        result = scheduler.schedule(bsp_hardware, task_graph)
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
            'target_ccr': metadata.target_ccr,
            'sync_time': metadata.sync_time
        }

    except Exception as e:
        logger.warning(f"Scheduler {scheduler_name} failed on task graph {task_graph_idx}: {e}")
        raise e
        return None




class BenchmarkRunner:
    """Runner for BSP scheduling benchmarks."""

    def __init__(self, schedulers: Dict[str, Any]):
        """Initialize benchmark runner.

        Args:
            schedulers: Dictionary of scheduler instances
        """
        self.schedulers = schedulers

    def run_dataset_benchmark(self, dataset_name: str, task_graphs: List, bsp_hardware_list: List,
                             metadata_list: List[TaskGraphMetadata], resultsdir: pathlib.Path,
                             num_jobs: int = 1, overwrite: bool = False, max_instances: int = 5,
                             visualization_dir: Optional[pathlib.Path] = None) -> None:
        """Run benchmark on a specific dataset.

        Args:
            dataset_name: Name of the dataset
            task_graphs: List of task graphs
            bsp_hardware_list: List of BSP hardware configurations
            metadata_list: List of metadata for each task graph
            resultsdir: Directory to store results
            num_jobs: Number of parallel jobs
            overwrite: Whether to overwrite existing results
            max_instances: Maximum number of task graphs to use for benchmarking
            visualization_dir: Directory to save schedule visualizations (optional)
        """
        savepath = resultsdir / f"{dataset_name}.csv"
        if savepath.exists() and not overwrite:
            logger.info(f"Results for {dataset_name} already exist. Skipping.")
            return

        # Limit number of task graphs for benchmarking
        if max_instances > 0 and len(task_graphs) > max_instances:
            task_graphs = task_graphs[:max_instances]
            bsp_hardware_list = bsp_hardware_list[:max_instances]
            metadata_list = metadata_list[:max_instances]
            logger.info(f"Limited to {max_instances} instances for {dataset_name}")

        logger.info(f"Running benchmark on {dataset_name} with {len(task_graphs)} task graphs")

        # Create all scheduler-task_graph combinations for parallel execution
        tasks = []
        for i, (task_graph, bsp_hardware, metadata) in enumerate(zip(task_graphs, bsp_hardware_list, metadata_list)):
            for scheduler_name, scheduler in self.schedulers.items():
                tasks.append((scheduler_name, scheduler, task_graph, bsp_hardware, metadata, i, dataset_name, visualization_dir))

        logger.info(f"Running {len(tasks)} scheduler-task combinations in parallel with {num_jobs} jobs")

        # Execute all combinations in parallel
        parallel_results = Parallel(n_jobs=num_jobs, verbose=1)(
            delayed(run_single_scheduler_task)(
                scheduler_name, scheduler, task_graph, bsp_hardware, metadata, task_graph_idx, dataset_name, visualization_dir
            ) for scheduler_name, scheduler, task_graph, bsp_hardware, metadata, task_graph_idx, dataset_name, visualization_dir in tasks
        )

        # Filter out None results (failed executions)
        valid_results = [r for r in parallel_results if r is not None]

        # Group results by task graph and calculate ratios
        results = []
        for task_graph_idx in range(len(task_graphs)):
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
                    'target_ccr': result['target_ccr'],
                    'sync_time': result['sync_time']
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
        generator = DatasetGenerator(cache_dir=datadir)

        # Get available datasets
        available_recipes = list(generator.RECIPES.keys())
        if dataset_names:
            recipes_to_run = [name for name in dataset_names if name in available_recipes]
            if len(recipes_to_run) != len(dataset_names):
                missing = set(dataset_names) - set(recipes_to_run)
                logger.warning(f"Some requested datasets not found: {missing}")
        else:
            recipes_to_run = available_recipes

        logger.info(f"Running benchmarks on {len(recipes_to_run)} datasets: {recipes_to_run}")

        # Run benchmarks for each dataset
        for recipe_name in recipes_to_run:
            try:
                # Load cached dataset
                cached_data = generator.load_cached_dataset(recipe_name)
                if cached_data is None:
                    logger.warning(f"No cached data for {recipe_name}, skipping")
                    continue

                task_graphs, bsp_hardware_list, metadata_list = cached_data

                # Run benchmark
                self.run_dataset_benchmark(
                    dataset_name=recipe_name,
                    task_graphs=task_graphs,
                    bsp_hardware_list=bsp_hardware_list,
                    metadata_list=metadata_list,
                    resultsdir=resultsdir,
                    num_jobs=num_jobs,
                    overwrite=overwrite,
                    max_instances=max_instances,
                    visualization_dir=visualization_dir
                )

            except Exception as e:
                logger.error(f"Failed to run benchmark on {recipe_name}: {e}")

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