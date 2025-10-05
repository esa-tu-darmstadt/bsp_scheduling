#!/usr/bin/env python3
"""
Benchmark2: Independent BSP scheduling benchmark using WfCommons

This benchmark script replaces the saga-dependent benchmark with a completely
independent implementation using WfCommons for workflow generation.

Features:
- Direct WfCommons integration for workflow generation
- IPU-specific BSP hardware modeling
- Task graph caching with metadata
- Box plot and heatmap visualizations
- Scheduler ordering and renaming support
- Special BusyCommHeft delay model visualization
"""

import logging
import pathlib
import argparse
import shutil
from typing import List, Optional

from rich.console import Console
from rich.logging import RichHandler

from dataset_generator import (
    generate_wfcommons_datasets, generate_spn_datasets, generate_primitives_datasets
)
from schedulers import get_bsp_schedulers
from benchmark_runner import BenchmarkRunner
from visualizations import BoxPlotVisualizer, HeatmapVisualizer, ScheduleComparisonVisualizer
from solver_statistics import write_solver_statistics

thisdir = pathlib.Path(__file__).parent.resolve()

def main():
    """Main benchmark execution function."""
    parser = argparse.ArgumentParser(description='BSP Scheduling Benchmark using WfCommons')
    parser.add_argument('--datadir', type=pathlib.Path, default=thisdir / "data",
                       help='Directory to store generated datasets')
    parser.add_argument('--resultsdir', type=pathlib.Path, default=thisdir / "results",
                       help='Directory to store benchmark results')
    parser.add_argument('--visdir', type=pathlib.Path, default=thisdir / "visualizations",
                       help='Directory to store visualization output')
    parser.add_argument('--num-jobs', type=int, default=1,
                       help='Number of parallel jobs for benchmarking')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing results')
    parser.add_argument('--skip-generation', action='store_true',
                       help='Skip dataset generation (use existing)')
    parser.add_argument('--skip-benchmarking', action='store_true',
                       help='Skip benchmarking (use existing results)')
    parser.add_argument('--datasets', nargs='*',
                       help='Specific datasets to benchmark (default: all)')
    parser.add_argument('--schedulers', nargs='*',
                       help='Specific schedulers to benchmark (default: all)')
    parser.add_argument('--max-instances', type=int, default=5,
                       help='Maximum instances per dataset for benchmarking (default: 5, 0 = no limit)')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')

    # Schedule comparison options
    parser.add_argument('--schedule-comparison-dataset', type=str, default=None,
                       help='Specific dataset for schedule comparison (default: all)')
    parser.add_argument('--schedule-comparison-task', type=int, default=-1,
                       help='Task instance index for schedule comparison (default: disable)')
    
    parser.add_argument('--clean-datasets', action='store_true',
                       help='Clean up existing datasets')
    parser.add_argument('--clean-results', action='store_true',
                       help='Clean up existing results')
    parser.add_argument('--clean-visualizations', action='store_true',
                       help='Clean up existing output files')

    args = parser.parse_args()

    # Setup Rich logging for consistent console output
    console = Console()

    # Configure root logger to catch ALL logging with Rich
    root_logger = logging.getLogger()

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add Rich handler to root logger
    rich_handler = RichHandler(console=console, show_time=True, show_path=False)
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(rich_handler)
    root_logger.setLevel(logging.INFO)

    logger = logging.getLogger(__name__)

    if args.clean_datasets:
        args.clean_results = True

    if args.clean_results:
        args.clean_visualizations = True

    if args.verbose:
        root_logger.setLevel(logging.DEBUG)

    # Create directories
    args.datadir.mkdir(parents=True, exist_ok=True)
    args.resultsdir.mkdir(parents=True, exist_ok=True)
    args.visdir.mkdir(parents=True, exist_ok=True)
    
    if args.clean_datasets:
        logger.info("Cleaning up existing datasets...")
        shutil.rmtree(args.datadir)

    if args.clean_results:
        logger.info("Cleaning up existing results...")
        shutil.rmtree(args.resultsdir)

    if args.clean_visualizations:
        logger.info("Cleaning up existing visualizations files...")
        shutil.rmtree(args.visdir)

    # Initialize components
    schedulers = get_bsp_schedulers(scheduler_names=args.schedulers)
    runner = BenchmarkRunner(schedulers=schedulers, console=console)

    # Step 1: Generate datasets
    if not args.skip_generation:
        logger.info("Generating datasets...")

        # Generate WfCommons datasets
        logger.info("Generating WfCommons datasets...")
        wfcommons_datasets = generate_wfcommons_datasets(
            cache_dir=args.datadir,
            get_task_count=lambda idx, task_counts: task_counts[idx % len(task_counts)],
            get_variations_per_tile=lambda tile_counts: max(5, len(tile_counts)),
            overwrite_cache=args.overwrite
        )
        logger.info(f"Generated {len(wfcommons_datasets)} WfCommons datasets")

        # Generate SPN datasets
        logger.info("Generating SPN datasets...")
        spn_datasets = generate_spn_datasets(
            cache_dir=args.datadir,
            overwrite_cache=args.overwrite
        )
        logger.info(f"Generated {len(spn_datasets)} SPN datasets")

        # Generate primitives datasets
        logger.info("Generating primitives datasets...")
        primitives_datasets = generate_primitives_datasets(
            cache_dir=args.datadir,
            overwrite_cache=args.overwrite
        )
        logger.info(f"Generated {len(primitives_datasets)} primitives datasets")

    # Step 2: Run benchmarks
    if not args.skip_benchmarking:
        logger.info("Running benchmarks...")
        schedules_viz_dir = args.visdir / "schedules"
        runner.run_all_benchmarks(
            datadir=args.datadir,
            resultsdir=args.resultsdir,
            dataset_names=args.datasets,
            num_jobs=args.num_jobs,
            overwrite=args.overwrite,
            max_instances=args.max_instances,
            visualization_dir=schedules_viz_dir
        )

    # Step 3: Generate visualizations
    logger.info("Generating visualizations...")

    # Box plot visualization
    box_visualizer = BoxPlotVisualizer()
    box_visualizer.generate_plots(
        resultsdir=args.resultsdir,
        outputdir=args.visdir / "boxplots"
    )

    # Heatmap visualization with special BusyCommHeft handling
    heatmap_visualizer = HeatmapVisualizer()
    heatmap_visualizer.generate_heatmaps(
        resultsdir=args.resultsdir,
        outputdir=args.visdir / "heatmaps"
    )

    if args.schedule_comparison_task >= 0:
        # Schedule comparison visualization
        schedule_comparison_visualizer = ScheduleComparisonVisualizer()
        schedule_comparison_visualizer.generate_schedule_comparisons(
            schedules_dir=schedules_viz_dir,
            outputdir=args.visdir / "schedule_comparisons",
            dataset_name=args.schedule_comparison_dataset,
            task_idx=args.schedule_comparison_task
        )

    # Step 4: Write solver statistics
    stats_out = args.resultsdir / "solver_statistics.txt"
    logger.info(f"Writing solver statistics to {stats_out}")
    write_solver_statistics(resultsdir=args.resultsdir, outpath=stats_out)

    logger.info("Benchmark complete!")

if __name__ == "__main__":
    main()
