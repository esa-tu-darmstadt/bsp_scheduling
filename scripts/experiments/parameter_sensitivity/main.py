"""
Parameter Sensitivity Analysis for BSP Scheduling.

This script analyzes how various parameters (CCR, sync time, tile count)
influence BSP scheduling quality across different schedulers.

Output: Line plots with error bands showing relative makespan ratio
as a function of each parameter.
"""

import argparse
import copy
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn
from saga.schedulers.data.random import gen_in_trees, gen_out_trees, gen_parallel_chains

# Import saga_bsp components
from saga_bsp.hardware.graphcore import IPUHardware
from saga_bsp.task_graphs.ccr_adjustment import (
    adjust_task_graph_to_ccr,
    calculate_ccr,
    calculate_sync_time,
)
from saga_bsp.misc.saga_scheduler_wrapper import preprocess_task_graph

# Import scheduler utilities from benchmark2
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from scripts.experiments.benchmark2.schedulers import (
    UnifiedSchedulerWrapper,
    get_scheduler_display_name,
    SCHEDULER_RENAMES,
)
from saga_bsp.misc.heft_busy_communication import HeftBusyCommScheduler
from saga_bsp.schedulers import FillInSplitBSPScheduler, AsyncToBSPScheduler
from saga_bsp.schedulers.delaymodel import HeftScheduler
import saga_bsp as bsp

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Default parameter values
CCR_VALUES = [0.1, 0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0]
# Sync time as multiple of avg computation time (e.g., 1.0 = sync takes as long as avg task)
# Values chosen to show meaningful sensitivity range
SYNC_PCT_VALUES = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
TILE_COUNT_VALUES = [2, 4, 8, 16, 24, 32]

# Default fixed values when sweeping other parameters
DEFAULT_CCR = 0.1
DEFAULT_SYNC_PCT = 1.0  # sync time = 10x avg computation time (reasonable BSP overhead)
DEFAULT_TILES = 8


def generate_primitive_graphs(
    graph_type: str,
    num_variations: int,
    num_levels: int = 4,
    branching_factor: int = 3,
    num_chains: int = 4,
    chain_length: int = 5,
) -> List[nx.DiGraph]:
    """Generate primitive task graphs using SAGA generators.

    Args:
        graph_type: One of 'in_tree', 'out_tree', 'parallel_chains'
        num_variations: Number of graph variations to generate
        num_levels: Number of levels for tree graphs
        branching_factor: Branching factor for tree graphs
        num_chains: Number of chains for parallel_chains
        chain_length: Length of each chain for parallel_chains

    Returns:
        List of task graphs with random weights
    """
    graphs = []

    for _ in range(num_variations):
        branching_factor = np.random.randint(3, 6)
        num_levels = np.random.randint(3, 6)
        num_chains = np.random.randint(5, 10)
        chain_length = np.random.randint(5, 10)
    
        # Random weight functions
        def get_task_weight(task_id):
            return np.random.uniform(1.0, 10.0)

        def get_edge_weight(src_id, dst_id):
            return np.random.uniform(0.5, 5.0)

        if graph_type == 'in_tree':
            generated = gen_in_trees(
                1, num_levels, branching_factor,
                get_task_weight, get_edge_weight
            )
        elif graph_type == 'out_tree':
            generated = gen_out_trees(
                1, num_levels, branching_factor,
                get_task_weight, get_edge_weight
            )
        elif graph_type == 'parallel_chains':
            generated = gen_parallel_chains(
                1, num_chains, chain_length,
                get_task_weight, get_edge_weight
            )
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")

        # Preprocess to remove SAGA's entry/exit nodes
        for g in generated:
            processed_graph, _ = preprocess_task_graph(g)
            graphs.append(processed_graph)

    logger.info(f"Generated {len(graphs)} {graph_type} graphs")
    return graphs


def create_schedulers() -> Dict[str, UnifiedSchedulerWrapper]:
    """Create representative subset of schedulers for comparison.

    Returns:
        Dictionary mapping scheduler names to wrapped schedulers
    """
    schedulers = {}

    # 1. HEFT with busy communication (delay model baseline)
    schedulers["HeftBusyCommScheduler"] = UnifiedSchedulerWrapper(
        HeftBusyCommScheduler()
    )

    # 2. HEFT converted to BSP using Earliest-Finishing-Next strategy
    schedulers["HEFT-BSP-EarliestNext"] = UnifiedSchedulerWrapper(
        AsyncToBSPScheduler(
            async_scheduler=HeftScheduler(),
            strategy="earliest-finishing-next"
        )
    )

    # 3. Native BSP - BALS with upward rank
    schedulers["FillInSplitBSPScheduler-HEFT"] = UnifiedSchedulerWrapper(
        FillInSplitBSPScheduler(priority_mode='heft')
    )

    # 4. BALS with upward rank + Superstep Elimination
    schedulers["FillInSplitBSPScheduler-HEFT-Merge"] = UnifiedSchedulerWrapper(
        FillInSplitBSPScheduler(priority_mode='heft', optimize_merging=True)
    )

    logger.info(f"Created {len(schedulers)} schedulers: {list(schedulers.keys())}")
    return schedulers


def create_hardware(num_tiles: int, sync_time: float) -> IPUHardware:
    """Create IPU hardware with specified configuration.

    Args:
        num_tiles: Number of tiles/processors
        sync_time: Synchronization time in nanoseconds

    Returns:
        IPUHardware instance
    """
    return IPUHardware(num_tiles=num_tiles, sync_time=sync_time)


def compute_sync_time_for_graph(
    task_graph: nx.DiGraph,
    num_tiles: int,
    sync_ratio: float
) -> float:
    """Compute sync time for a task graph as a fraction of avg computation time.

    This creates a temporary hardware to get the network, then uses
    calculate_sync_time() to compute the proper sync time.

    Args:
        task_graph: The task graph
        num_tiles: Number of tiles for the hardware
        sync_ratio: Sync time as multiple of avg computation time
                   (e.g., 1.0 = sync takes as long as avg task runtime)

    Returns:
        Sync time in nanoseconds (as expected by IPUHardware)
    """
    # Create temporary hardware to get the network topology
    temp_hardware = IPUHardware(num_tiles=num_tiles, sync_time=0.0)

    # calculate_sync_time returns seconds, convert to nanoseconds for IPUHardware
    sync_time_seconds = calculate_sync_time(task_graph, temp_hardware.network, sync_ratio)
    sync_time_ns = sync_time_seconds * 1e9
    return sync_time_ns


def run_single_experiment(
    task_graph: nx.DiGraph,
    hardware: IPUHardware,
    schedulers: Dict[str, UnifiedSchedulerWrapper]
) -> Dict[str, float]:
    """Run all schedulers on a single task graph and return makespans.

    Args:
        task_graph: The task graph to schedule
        hardware: The BSP hardware configuration
        schedulers: Dictionary of schedulers to run

    Returns:
        Dictionary mapping scheduler names to makespans
    """
    makespans = {}

    for name, scheduler in schedulers.items():
        try:
            result = scheduler.schedule(hardware, task_graph)
            makespans[name] = result['makespan']
        except Exception as e:
            logger.warning(f"Scheduler {name} failed: {e}")
            makespans[name] = float('inf')

    return makespans


def run_experiment_worker(task_graph, num_tiles, sync_ratio, target_ccr, var_idx, param_name, param_value):
    """Worker function for parallel execution of a single experiment.

    Creates schedulers fresh in each worker process to avoid pickling issues.
    """
    # Create schedulers in worker (can't pickle them)
    schedulers = create_schedulers_internal()

    # Compute sync_time and create hardware
    sync_time = compute_sync_time_for_graph(task_graph, num_tiles, sync_ratio)
    hardware = create_hardware(num_tiles, sync_time)

    # Adjust graph to target CCR
    graph = copy.deepcopy(task_graph)
    adjust_task_graph_to_ccr(graph, hardware.network, target_ccr)

    # Run schedulers
    makespans = run_single_experiment(graph, hardware, schedulers)

    # Calculate relative makespan ratio
    best_makespan = min(makespans.values())

    results = []
    for sched_name, makespan in makespans.items():
        ratio = makespan / best_makespan if best_makespan > 0 else float('inf')
        results.append({
            param_name: param_value,
            'scheduler': sched_name,
            'makespan_ratio': ratio,
            'variation_idx': var_idx,
        })

    return results


def create_schedulers_internal():
    """Create schedulers without logging (for use in worker processes)."""
    schedulers = {}
    schedulers["HeftBusyCommScheduler"] = UnifiedSchedulerWrapper(HeftBusyCommScheduler())
    schedulers["HEFT-BSP-EarliestNext"] = UnifiedSchedulerWrapper(
        AsyncToBSPScheduler(async_scheduler=HeftScheduler(), strategy="earliest-finishing-next"))
    schedulers["FillInSplitBSPScheduler-HEFT"] = UnifiedSchedulerWrapper(
        FillInSplitBSPScheduler(priority_mode='heft'))
    schedulers["FillInSplitBSPScheduler-HEFT-Merge"] = UnifiedSchedulerWrapper(
        FillInSplitBSPScheduler(priority_mode='heft', optimize_merging=True))
    return schedulers


def run_parallel_sweep(
    base_graphs: List[nx.DiGraph],
    param_name: str,
    param_values: List,
    default_tiles: int,
    default_sync_pct: float,
    default_ccr: float,
    num_jobs: int,
) -> pd.DataFrame:
    """Run a parameter sweep in parallel.

    Args:
        base_graphs: List of base task graphs
        param_name: Name of parameter being swept ('ccr', 'sync_pct', or 'tiles')
        param_values: Values to sweep
        default_tiles: Default tile count
        default_sync_pct: Default sync time ratio
        default_ccr: Default CCR value
        num_jobs: Number of parallel workers

    Returns:
        DataFrame with results
    """
    # Build list of tasks
    tasks = []
    for param_value in param_values:
        for var_idx, base_graph in enumerate(base_graphs):
            # Determine actual parameters based on which one we're sweeping
            if param_name == 'ccr':
                num_tiles, sync_ratio, target_ccr = default_tiles, default_sync_pct, param_value
            elif param_name == 'sync_pct':
                num_tiles, sync_ratio, target_ccr = default_tiles, param_value, default_ccr
            elif param_name == 'tiles':
                num_tiles, sync_ratio, target_ccr = param_value, default_sync_pct, default_ccr
            else:
                raise ValueError(f"Unknown param_name: {param_name}")

            tasks.append((base_graph, num_tiles, sync_ratio, target_ccr, var_idx, param_name, param_value))

    total_tasks = len(tasks)
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task_progress = progress.add_task(f"Sweeping {param_name}", total=total_tasks)

        with ProcessPoolExecutor(max_workers=num_jobs) as executor:
            futures = {executor.submit(run_experiment_worker, *task): task for task in tasks}

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.extend(result)
                except Exception as e:
                    logger.warning(f"Task failed: {e}")
                progress.advance(task_progress)

    return pd.DataFrame(results)


def plot_sensitivity(
    ccr_results: pd.DataFrame,
    sync_results: pd.DataFrame,
    tile_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create 3-row figure with line plots and error bands.

    Args:
        ccr_results: DataFrame from sweep_ccr
        sync_results: DataFrame from sweep_sync_time
        tile_results: DataFrame from sweep_tiles
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Color palette for schedulers
    colors = plt.cm.tab10.colors
    scheduler_colors = {}

    def plot_sweep(ax, df: pd.DataFrame, x_col: str, title: str, xlabel: str):
        """Helper to plot a single sweep."""
        schedulers = df['scheduler'].unique()

        for i, sched in enumerate(schedulers):
            if sched not in scheduler_colors:
                scheduler_colors[sched] = colors[len(scheduler_colors) % len(colors)]
            color = scheduler_colors[sched]

            sched_data = df[df['scheduler'] == sched]

            # Group by x value and calculate mean/std
            grouped = sched_data.groupby(x_col)['makespan_ratio']
            means = grouped.mean()
            stds = grouped.std()

            x_vals = means.index.values
            y_means = means.values
            y_stds = stds.values

            # Plot line with error band
            display_name = get_scheduler_display_name(sched)
            ax.plot(x_vals, y_means, '-o', color=color, label=display_name, linewidth=2, markersize=6)
            ax.fill_between(x_vals, y_means - y_stds, y_means + y_stds,
                           color=color, alpha=0.2)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Relative Makespan Ratio', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0.95)  # Start slightly below 1.0

    # Plot CCR sensitivity
    plot_sweep(axes[0], ccr_results, 'ccr',
               'CCR Sensitivity', 'CCR (Communication-to-Computation Ratio)')
    axes[0].set_xscale('log')

    # Plot sync time sensitivity
    plot_sweep(axes[1], sync_results, 'sync_pct',
               'Sync Time Sensitivity', 'Sync Time (multiple of avg task runtime)')

    # Plot tile count sensitivity
    plot_sweep(axes[2], tile_results, 'tiles',
               'Tile Count Sensitivity', 'Number of Tiles')

    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    logger.info(f"Saved figure to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Parameter sensitivity analysis for BSP scheduling'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path('./results/sensitivity'),
        help='Directory to save results and plots'
    )
    parser.add_argument(
        '--num-variations', type=int, default=20,
        help='Number of graph variations per parameter point'
    )
    parser.add_argument(
        '--primitive-type', type=str, default='in_tree',
        choices=['in_tree', 'out_tree', 'parallel_chains'],
        help='Type of primitive graphs to generate'
    )
    parser.add_argument(
        '--num-jobs', '-j', type=int, default=os.cpu_count(),
        help='Number of parallel workers (default: number of CPUs)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("Parameter Sensitivity Analysis for BSP Scheduling")
    logger.info("=" * 60)

    # 1. Generate base graphs
    logger.info(f"\n1. Generating {args.num_variations} {args.primitive_type} graphs...")
    base_graphs = generate_primitive_graphs(
        args.primitive_type,
        args.num_variations
    )

    # 2. Show scheduler info
    logger.info("\n2. Schedulers: HEFT, HEFT+EFN, BALS Upward, BALS Upw.+Elim.")
    logger.info(f"   Using {args.num_jobs} parallel workers")

    # 3. Run parameter sweeps in parallel
    logger.info("\n3. Running parameter sweeps...")

    ccr_results = run_parallel_sweep(
        base_graphs, 'ccr', CCR_VALUES,
        DEFAULT_TILES, DEFAULT_SYNC_PCT, DEFAULT_CCR, args.num_jobs
    )

    sync_results = run_parallel_sweep(
        base_graphs, 'sync_pct', SYNC_PCT_VALUES,
        DEFAULT_TILES, DEFAULT_SYNC_PCT, DEFAULT_CCR, args.num_jobs
    )

    tile_results = run_parallel_sweep(
        base_graphs, 'tiles', TILE_COUNT_VALUES,
        DEFAULT_TILES, DEFAULT_SYNC_PCT, DEFAULT_CCR, args.num_jobs
    )

    # 4. Save raw results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ccr_results.to_csv(args.output_dir / 'ccr_sweep.csv', index=False)
    sync_results.to_csv(args.output_dir / 'sync_sweep.csv', index=False)
    tile_results.to_csv(args.output_dir / 'tile_sweep.csv', index=False)
    logger.info(f"\nSaved raw results to {args.output_dir}")

    # 5. Generate plots
    logger.info("\n4. Generating plots...")
    plot_sensitivity(
        ccr_results, sync_results, tile_results,
        args.output_dir / 'parameter_sensitivity.png'
    )

    logger.info("\nDone!")


if __name__ == '__main__':
    main()
