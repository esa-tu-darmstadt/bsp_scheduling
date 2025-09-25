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

from dataset_generator import DatasetGenerator
from hardware_ipu import IPUHardware
from schedulers import get_bsp_schedulers
from benchmark_runner import BenchmarkRunner
from visualizations import BoxPlotVisualizer, HeatmapVisualizer
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
    
    parser.add_argument('--clean-datasets', action='store_true',
                       help='Clean up existing datasets')
    parser.add_argument('--clean-results', action='store_true',
                       help='Clean up existing results')
    parser.add_argument('--clean-visualizations', action='store_true',
                       help='Clean up existing output files')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

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
    dataset_generator = DatasetGenerator(cache_dir=args.datadir)
    schedulers = get_bsp_schedulers(scheduler_names=args.schedulers)
    runner = BenchmarkRunner(schedulers=schedulers)

    # Step 1: Generate datasets
    if not args.skip_generation:
        logger.info("Generating datasets...")
        dataset_generator.generate_all_datasets(overwrite=args.overwrite)

    # Step 2: Run benchmarks
    if not args.skip_benchmarking:
        logger.info("Running benchmarks...")
        runner.run_all_benchmarks(
            datadir=args.datadir,
            resultsdir=args.resultsdir,
            dataset_names=args.datasets,
            num_jobs=args.num_jobs,
            overwrite=args.overwrite,
            max_instances=args.max_instances
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

    # Step 4: Write solver statistics
    stats_out = args.resultsdir / "solver_statistics.txt"
    logger.info(f"Writing solver statistics to {stats_out}")
    write_solver_statistics(resultsdir=args.resultsdir, outpath=stats_out)

    logger.info("Benchmark complete!")

if __name__ == "__main__":
    main()
