"""
Benchmark runner for BSP scheduling experiments.

This module handles running benchmarks on datasets with different schedulers,
managing results storage and providing progress tracking.
"""

import logging
import pathlib
import time
from typing import Dict, List, Optional, Any
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn
from rich.console import Console

from dataset_generator import DatasetItem, load_dataset, find_datasets
from schedule_visualizer import save_schedule_visualization, should_save_visualization

logger = logging.getLogger(__name__)


def run_single_scheduler_task(scheduler_name, scheduler, dataset_item, task_graph_idx,
                             dataset_name=None, visualization_dir=None):
    """Run a single scheduler on a single dataset item."""
    logger.debug(f"Running scheduler {scheduler_name} on task graph {task_graph_idx}")
    try:
        # All schedulers now use the unified interface
        start_time = time.perf_counter()
        result = scheduler.schedule(dataset_item.hardware, dataset_item.task_graph)
        end_time = time.perf_counter()

        scheduler_runtime_s = end_time - start_time
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
            'scheduler_runtime_s': scheduler_runtime_s,
            'task_graph_idx': task_graph_idx,
            'dataset_name': dataset_name,
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

    def __init__(self, schedulers: Dict[str, Any], console: Console = None):
        """Initialize benchmark runner.

        Args:
            schedulers: Dictionary of scheduler instances
            console: Rich console instance to use (optional, creates new if None)
        """
        self.schedulers = schedulers
        self.console = console if console is not None else Console()

    def _run_tasks_with_progress(self, tasks: List, num_jobs: int, dataset_name: str = "Unknown", progress_context=None) -> List:
        """Run tasks with Rich progress tracking."""
        # Group tasks by scheduler for better progress visualization
        scheduler_tasks = {}
        for task in tasks:
            scheduler_name = task[0]
            if scheduler_name not in scheduler_tasks:
                scheduler_tasks[scheduler_name] = []
            scheduler_tasks[scheduler_name].append(task)

        total_tasks = len(tasks)
        results = []

        # Use existing progress context or create new one
        if progress_context is not None:
            progress = progress_context
            should_close_progress = False
        else:
            # Clear console to prevent artifacts only when creating new progress context
            self.console.clear()
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}", justify="left"),
                BarColumn(bar_width=None),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                TimeElapsedColumn(),
                console=self.console,
                transient=False,
                refresh_per_second=4,  # Slower refresh to work better with logging
                expand=True,  # Allow logging messages to appear above progress bars
                get_time=lambda: __import__('time').time()
            )
            progress.start()
            should_close_progress = True

        # Create overall progress task with dataset info
        overall_task = progress.add_task(
            f"Dataset: {dataset_name} - Overall Progress",
            total=total_tasks
        )

        # Create progress tasks for each scheduler
        scheduler_progress_tasks = {}
        for scheduler_name, scheduler_task_list in scheduler_tasks.items():
            task_id = progress.add_task(
                f"{scheduler_name}",
                total=len(scheduler_task_list)
            )
            scheduler_progress_tasks[scheduler_name] = task_id

        if num_jobs == 1:
            # Sequential execution for easier debugging
            for task in tasks:
                scheduler_name = task[0]
                try:
                    result = run_single_scheduler_task(*task)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Task failed: {e}")
                    results.append(None)

                progress.advance(scheduler_progress_tasks[scheduler_name])
                progress.advance(overall_task)
        else:
            # Parallel execution using ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=num_jobs) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(run_single_scheduler_task, *task): task
                    for task in tasks
                }

                # Process completed tasks
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    scheduler_name = task[0]

                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"Task failed: {e}")
                        results.append(None)

                    progress.advance(scheduler_progress_tasks[scheduler_name])
                    progress.advance(overall_task)

        # Remove all tasks for this dataset (cleanup)
        progress.remove_task(overall_task)
        for scheduler_name, task_id in scheduler_progress_tasks.items():
            progress.remove_task(task_id)

        # Close progress context if we created it
        if should_close_progress:
            progress.stop()

        return results

    def run_dataset_benchmark(self, dataset_name: str, dataset_items: List[DatasetItem],
                             resultsdir: pathlib.Path, num_jobs: int = 1, overwrite: bool = False,
                             max_instances: int = 5, visualization_dir: Optional[pathlib.Path] = None,
                             progress_context=None) -> None:
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
            logger.debug(f"Results for {dataset_name} already exist. Skipping.")
            return

        # Limit number of dataset items for benchmarking
        if max_instances > 0 and len(dataset_items) > max_instances:
            dataset_items = dataset_items[:max_instances]
            logger.debug(f"Limited to {max_instances} instances for {dataset_name}")

        logger.debug(f"Running benchmark on {dataset_name} with {len(dataset_items)} dataset items")

        # Create all scheduler-dataset_item combinations for parallel execution
        tasks = []
        for i, dataset_item in enumerate(dataset_items):
            for scheduler_name, scheduler in self.schedulers.items():
                tasks.append((scheduler_name, scheduler, dataset_item, i, dataset_name, visualization_dir))

        logger.debug(f"Running {len(tasks)} scheduler-task combinations in parallel with {num_jobs} jobs")

        # Execute all combinations in parallel with Rich progress tracking
        parallel_results = self._run_tasks_with_progress(tasks, num_jobs, dataset_name, progress_context)

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
                    'scheduler_runtime_s': result['scheduler_runtime_s'],
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

        This method parallelizes across ALL datasets and schedulers globally,
        rather than per-dataset. This ensures better CPU utilization when some
        schedulers are faster than others.

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

        # Clear console before starting
        self.console.clear()
        self.console.print(f"[bold green]Starting benchmark on {len(available_datasets)} datasets")
        self.console.print(f"[cyan]Datasets: {[name for name, _ in available_datasets]}")
        self.console.print()

        # Step 1: Load all datasets and collect all tasks globally
        logger.info("Loading all datasets...")
        all_tasks = []  # List of (scheduler_name, scheduler, dataset_item, task_graph_idx, dataset_name, visualization_dir)
        dataset_info = {}  # dataset_name -> (savepath, num_items) for result saving

        for dataset_key, dataset_file in available_datasets:
            savepath = resultsdir / f"{dataset_key}.csv"
            if savepath.exists() and not overwrite:
                logger.debug(f"Results for {dataset_key} already exist. Skipping.")
                continue

            try:
                dataset_items, dataset_metadata = load_dataset(dataset_file)

                if not dataset_items:
                    logger.warning(f"No items in dataset {dataset_key}, skipping")
                    continue

                # Limit number of dataset items for benchmarking
                if max_instances > 0 and len(dataset_items) > max_instances:
                    dataset_items = dataset_items[:max_instances]
                    logger.debug(f"Limited to {max_instances} instances for {dataset_key}")

                # Use display name from metadata if available, fallback to dataset_key
                display_name = getattr(dataset_metadata, 'dataset_name', dataset_key)

                # Store dataset info for later result saving
                dataset_info[display_name] = {
                    'savepath': savepath,
                    'num_items': len(dataset_items)
                }

                # Create tasks for all scheduler-dataset_item combinations
                for i, dataset_item in enumerate(dataset_items):
                    for scheduler_name, scheduler in self.schedulers.items():
                        all_tasks.append((scheduler_name, scheduler, dataset_item, i, display_name, visualization_dir))

            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_key}: {e}")

        if not all_tasks:
            logger.warning("No tasks to run")
            return

        logger.info(f"Collected {len(all_tasks)} total tasks across {len(dataset_info)} datasets")

        # Step 2: Run all tasks in parallel globally
        all_results = self._run_all_tasks_globally(all_tasks, num_jobs, dataset_info)

        # Step 3: Group results by dataset and save
        self._save_results_by_dataset(all_results, dataset_info, resultsdir)

        # Final completion message
        self.console.print()
        self.console.print(f"[bold green]Benchmark completed! Processed {len(dataset_info)} datasets with {len(all_tasks)} total tasks.")
        self.console.print()

    def _run_all_tasks_globally(self, all_tasks: List, num_jobs: int, dataset_info: Dict) -> List:
        """Run all tasks globally with progress tracking per scheduler."""
        # Group tasks by scheduler for progress tracking
        scheduler_names = sorted(set(task[0] for task in all_tasks))
        total_tasks = len(all_tasks)
        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}", justify="left"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
            refresh_per_second=4,
            expand=True
        ) as progress:

            # Overall progress
            overall_task = progress.add_task("Overall Progress", total=total_tasks)

            # Per-scheduler progress
            scheduler_progress = {}
            for scheduler_name in scheduler_names:
                count = sum(1 for t in all_tasks if t[0] == scheduler_name)
                scheduler_progress[scheduler_name] = progress.add_task(
                    f"  {scheduler_name}",
                    total=count
                )

            if num_jobs == 1:
                # Sequential execution for debugging
                for task in all_tasks:
                    scheduler_name = task[0]
                    try:
                        result = run_single_scheduler_task(*task)
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"Task failed: {e}")
                        results.append(None)

                    progress.advance(scheduler_progress[scheduler_name])
                    progress.advance(overall_task)
            else:
                # Parallel execution across ALL tasks globally
                with ProcessPoolExecutor(max_workers=num_jobs) as executor:
                    future_to_task = {
                        executor.submit(run_single_scheduler_task, *task): task
                        for task in all_tasks
                    }

                    for future in as_completed(future_to_task):
                        task = future_to_task[future]
                        scheduler_name = task[0]

                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            logger.warning(f"Task failed: {e}")
                            results.append(None)

                        progress.advance(scheduler_progress[scheduler_name])
                        progress.advance(overall_task)

        return results

    def _save_results_by_dataset(self, all_results: List, dataset_info: Dict, resultsdir: pathlib.Path) -> None:
        """Group results by dataset and save to CSV files."""
        # Filter out None results
        valid_results = [r for r in all_results if r is not None]

        # Group results by dataset
        results_by_dataset = {}
        for result in valid_results:
            dataset_name = result.get('dataset_name')
            if dataset_name not in results_by_dataset:
                results_by_dataset[dataset_name] = []
            results_by_dataset[dataset_name].append(result)

        # Save results for each dataset
        for dataset_name, info in dataset_info.items():
            dataset_results = results_by_dataset.get(dataset_name, [])

            if not dataset_results:
                logger.warning(f"No results for dataset {dataset_name}")
                continue

            # Calculate ratios relative to best scheduler for each task graph
            final_results = []
            task_graph_indices = set(r['task_graph_idx'] for r in dataset_results)

            for task_graph_idx in task_graph_indices:
                task_results = [r for r in dataset_results if r['task_graph_idx'] == task_graph_idx]

                if not task_results:
                    continue

                makespans = {r['scheduler_name']: r['makespan'] for r in task_results}
                best_makespan = min(makespans.values())

                for result in task_results:
                    ratio = result['makespan'] / best_makespan
                    final_results.append({
                        'scheduler': result['scheduler_name'],
                        'makespan': result['makespan'],
                        'makespan_ratio': ratio,
                        'scheduler_runtime_s': result['scheduler_runtime_s'],
                        'task_graph_idx': result['task_graph_idx'],
                        'target_ccr': result.get('target_ccr'),
                        'actual_ccr': result.get('actual_ccr'),
                        'sync_time': result.get('sync_time'),
                        'num_tiles': result.get('num_tiles'),
                        'source_type': result.get('source_type')
                    })

            # Save to CSV
            df_comp = pd.DataFrame(final_results)
            savepath = info['savepath']
            savepath.parent.mkdir(exist_ok=True, parents=True)
            df_comp.to_csv(savepath)
            logger.info(f"Saved results to {savepath}")

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