"""Analysis module for BSP benchmarking results."""

import logging
import pathlib
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# Import SAGA's gradient_heatmap
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent / "saga" / "src"))
from saga.utils.draw import gradient_heatmap

# Import prepare module for loading datasets
from prepare import load_dataset

# Dataset order for consistent plotting
DATASET_ORDER = [
    "in_trees", "out_trees", "chains",
    "blast", "bwa", "cycles", "epigenomics", 
    "genome", "montage", "seismology", "soykb", "srasearch",
    "etl", "predict", "stats", "train",
]

SCHEDULER_RENAMES = {
    "FillInSplitBSP": "FillOrSplit",
    "HEFT-BSP-Eager": "HEFT + Eager",
    "HEFT-BSP-EarliestNext": "HEFT + EarliestNext",
    "HeftBusyCommScheduler": "Async HEFT (Busy Comm)",
}

SCHEDULER_ORDER = [
    "FillInSplitBSP",
    "HEFT-BSP-Eager",
    "HEFT-BSP-EarliestNext"
    "HeftBusyCommScheduler",
]

# Note: Scheduler names are now generated cleanly in the BSP library,
# so no renaming is needed


def load_bsp_data(resultsdir: pathlib.Path, glob: str = None) -> pd.DataFrame:
    """Load BSP benchmarking results from CSV files."""
    data = None
    glob = glob or "*_bsp_sync*.csv"
    
    for path in resultsdir.glob(glob):
        df_dataset = pd.read_csv(path, index_col=0)
        df_dataset["dataset"] = path.stem.replace("_bsp_sync0.0", "")  # Remove sync time suffix
        if data is None:
            data = df_dataset
        else:
            data = pd.concat([data, df_dataset], ignore_index=True)
    
    if data is None:
        return pd.DataFrame()
    
    return data


def clean_scheduler_names(data: pd.DataFrame) -> pd.DataFrame:
    """Clean up scheduler names for better readability in plots."""
    data = data.copy()
    
    # Scheduler names are now generated cleanly in the BSP library,
    # so minimal cleaning is needed
    
    return data


def create_scheduler_comparison_plot(data: pd.DataFrame, outputdir: pathlib.Path) -> None:
    """Create a plot comparing all schedulers."""
    
    if data.empty:
        logging.warning("No data found for scheduler comparison")
        return
    
    # Use makespan_ratio if makespan is not available
    y_column = "makespan" if "makespan" in data.columns else "makespan_ratio"
    y_label = "Makespan" if y_column == "makespan" else "Makespan Ratio"
    
    rc_context_opts = {'text.usetex': True, 'font.size': 10}
    with plt.rc_context(rc=rc_context_opts):
    
        # Simple boxplot of all schedulers
        plt.figure(figsize=(7.16/2, 4))
        sns.boxplot(
            data=data,
            x="scheduler",
            y=y_column,
            palette="Set2"
        )
        plt.xlabel("Scheduler")
        plt.ylabel(y_label)
        plt.grid(visible=True, axis='y', linestyle='--', alpha=0.7)
        plt.ylim(1, 5)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(outputdir.joinpath("bsp_scheduler_comparison.pdf"), dpi=300, bbox_inches='tight')
        plt.savefig(outputdir.joinpath("bsp_scheduler_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()


def create_makespan_heatmap(data: pd.DataFrame, outputdir: pathlib.Path, title: str = None, upper_threshold: float = 5.0) -> None:
    """Create a heatmap showing makespan ratios across datasets and schedulers."""
    
    if data.empty:
        logging.info("No data found. Skipping.")
        return

    ax = gradient_heatmap(
        data,
        x="scheduler",
        y="dataset",
        color="makespan_ratio",
        cmap="coolwarm",
        upper_threshold=upper_threshold,
        title=title,
        x_label="Scheduler",
        y_label="Dataset",
        color_label="Maximum Makespan Ratio",
        figsize=(7.16, 5),
        font_size=10
    )
    ax.get_figure().savefig(
        outputdir.joinpath("bsp_makespan_heatmap.pdf"),
        dpi=300,
        bbox_inches='tight'
    )
    ax.get_figure().savefig(
        outputdir.joinpath("bsp_makespan_heatmap.png"),
        dpi=300,
        bbox_inches='tight'
    )
    # Also save as PGF for LaTeX
    ax.get_figure().savefig(
        outputdir.joinpath("bsp_makespan_heatmap.pgf"),
        bbox_inches='tight'
    )


def calculate_ccr(network: nx.Graph, task_graph: nx.DiGraph) -> float:
    """Calculate Communication-to-Computation Ratio (CCR) for a task graph.

    CCR = (average communication cost) / (average computation cost)

    Takes heterogeneity into account by using average speeds.

    Args:
        network: Network graph with processor speeds
        task_graph: Task DAG with computation and communication weights

    Returns:
        CCR value (higher = more communication-intensive)
    """
    # Calculate average computation cost
    avg_processor_speed = np.mean([network.nodes[p]['weight'] for p in network.nodes])
    total_computation_weight = sum(task_graph.nodes[t]['weight'] for t in task_graph.nodes)
    avg_computation_cost = total_computation_weight / (len(task_graph.nodes) * avg_processor_speed)

    # Calculate average communication cost
    if task_graph.number_of_edges() == 0:
        return 0.0  # No communication

    avg_link_speed = np.mean([network.edges[e]['weight'] for e in network.edges])
    total_communication_weight = sum(task_graph.edges[e]['weight'] for e in task_graph.edges)
    avg_communication_cost = total_communication_weight / (len(task_graph.edges) * avg_link_speed)

    # Avoid division by zero
    if avg_computation_cost == 0:
        return float('inf') if avg_communication_cost > 0 else 0.0

    return avg_communication_cost / avg_computation_cost


def add_ccr_to_data(data: pd.DataFrame, datadir: pathlib.Path) -> pd.DataFrame:
    """Add CCR values to the benchmarking data.

    Args:
        data: DataFrame with benchmarking results
        datadir: Directory containing dataset files

    Returns:
        DataFrame with added CCR column
    """
    data = data.copy()
    ccr_values = {}

    # Calculate CCR for each unique dataset
    for dataset_name in data['dataset'].unique():
        try:
            dataset = load_dataset(datadir, dataset_name)
            # Use first instance to calculate CCR (assuming all instances have similar characteristics)
            network, task_graph = dataset[0]
            ccr = calculate_ccr(network, task_graph)
            ccr_values[dataset_name] = ccr
            logging.info(f"Dataset {dataset_name}: CCR = {ccr:.15f}")
        except Exception as e:
            logging.warning(f"Could not calculate CCR for dataset {dataset_name}: {e}")
            ccr_values[dataset_name] = np.nan

    # Add CCR column to data
    data['ccr'] = data['dataset'].map(ccr_values)
    return data


def create_ccr_scatter_plot(data: pd.DataFrame, outputdir: pathlib.Path) -> None:
    """Create density and violin plots of scheduler performance vs CCR.

    Args:
        data: DataFrame with benchmarking results and CCR values
        outputdir: Output directory for plots
    """
    if 'ccr' not in data.columns:
        logging.warning("No CCR data available for scatter plot")
        return

    # Filter out any rows with NaN CCR
    plot_data = data[data['ccr'].notna()].copy()

    if plot_data.empty:
        logging.warning("No valid CCR data for plotting")
        return

    # Use makespan_ratio for performance metric
    y_column = "makespan_ratio" if "makespan_ratio" in plot_data.columns else "makespan"

    # Add log CCR for easier binning
    plot_data['log_ccr'] = np.log10(plot_data['ccr'])

    rc_context_opts = {'text.usetex': True, 'font.size': 10}

    # Create hexbin density plot
    with plt.rc_context(rc=rc_context_opts):
        fig, axes = plt.subplots(2, 1, figsize=(7.16, 8))

        # Get unique schedulers and assign colors
        schedulers = plot_data['scheduler'].unique()
        colors = sns.color_palette("Set2", n_colors=len(schedulers))
        scheduler_colors = dict(zip(schedulers, colors))

        # Plot 1: Hexbin density plot for all data
        ax = axes[0]
        hexbin = ax.hexbin(
            plot_data['ccr'],
            plot_data[y_column],
            xscale='log',
            gridsize=30,
            cmap='YlOrRd',
            mincnt=1
        )

        # Add colorbar
        plt.colorbar(hexbin, ax=ax, label='Number of Data Points')

        # Set log scale for x-axis
        ax.set_xscale('log')
        ax.set_xlabel('Communication-to-Computation Ratio (CCR)')
        ax.set_ylabel('Makespan Ratio' if y_column == 'makespan_ratio' else 'Makespan')
        ax.set_title('Density Plot: Scheduler Performance vs Task Graph Granularity')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim(1, 5)

        # Add CCR region annotations
        ccr_min, ccr_max = ax.get_xlim()
        y_pos = ax.get_ylim()[1] * 0.95

        regions = [
            (0.01, "Coarse"),
            (0.1, "Balanced"),
            (1.0, "Fine"),
            (10.0, "Ultra-fine")
        ]

        for threshold, label in regions:
            if ccr_min <= threshold <= ccr_max:
                ax.axvline(x=threshold, color='gray', linestyle='--', alpha=0.3)
                if threshold < ccr_max:
                    ax.text(threshold * 1.2, y_pos, label, fontsize=8, alpha=0.7)

        # Plot 2: Violin plot by CCR bins
        ax = axes[1]

        # Create CCR bins
        ccr_bins = [-3, -2, -1, 0, 1, 2]  # Log scale bins
        ccr_labels = ['[0.001,0.01)', '[0.01,0.1)', '[0.1,1)', '[1,10)', '[10,100)']
        plot_data['ccr_bin'] = pd.cut(plot_data['log_ccr'], bins=ccr_bins, labels=ccr_labels)

        # Create violin plot
        sns.violinplot(
            data=plot_data,
            x='ccr_bin',
            y=y_column,
            hue='scheduler',
            ax=ax,
            palette=scheduler_colors,
            split=False,
            inner='quartile'
        )

        ax.set_xlabel('CCR Range')
        ax.set_ylabel('Makespan Ratio' if y_column == 'makespan_ratio' else 'Makespan')
        ax.set_title('Performance Distribution by CCR Range')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(1, 5)
        ax.legend(title='Scheduler', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(outputdir.joinpath("bsp_ccr_density.pdf"), dpi=300, bbox_inches='tight')
        plt.savefig(outputdir.joinpath("bsp_ccr_density.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # Also create a plot with trend lines
    with plt.rc_context(rc=rc_context_opts):
        fig, ax = plt.subplots(figsize=(7.16, 5))

        for scheduler in schedulers:
            scheduler_data = plot_data[plot_data['scheduler'] == scheduler]

            # Plot scatter points
            ax.scatter(
                scheduler_data['ccr'],
                scheduler_data[y_column],
                label=scheduler,
                color=scheduler_colors[scheduler],
                alpha=0.6,
                s=30
            )

            # Add trend line (polynomial fit in log space)
            if len(scheduler_data) > 3:
                x_log = np.log10(scheduler_data['ccr'].values)
                y = scheduler_data[y_column].values

                # Fit polynomial
                z = np.polyfit(x_log, y, 2)
                p = np.poly1d(z)

                # Generate smooth curve
                x_smooth_log = np.linspace(x_log.min(), x_log.max(), 100)
                x_smooth = 10 ** x_smooth_log
                y_smooth = p(x_smooth_log)

                ax.plot(x_smooth, y_smooth, color=scheduler_colors[scheduler],
                       linestyle='-', linewidth=2, alpha=0.7)

        ax.set_xscale('log')
        ax.set_xlabel('Communication-to-Computation Ratio (CCR)')
        ax.set_ylabel('Makespan Ratio' if y_column == 'makespan_ratio' else 'Makespan')
        ax.set_title('Scheduler Performance Trends vs Task Graph Granularity')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        plt.savefig(outputdir.joinpath("bsp_ccr_trends.pdf"), dpi=300, bbox_inches='tight')
        plt.savefig(outputdir.joinpath("bsp_ccr_trends.png"), dpi=300, bbox_inches='tight')
        plt.close()

    logging.info("Created CCR scatter plots")


def create_summary_statistics(data: pd.DataFrame, outputdir: pathlib.Path) -> None:
    """Generate summary statistics for BSP schedulers."""
    
    # Check what columns are available
    available_columns = data.columns.tolist()
    logging.info(f"Available columns: {available_columns}")
    
    # Build aggregation dict based on available columns
    agg_dict = {"dataset": "count"}  # Always count datasets
    
    if "makespan" in data.columns:
        agg_dict["makespan"] = ["mean", "median", "std", "min", "max"]
    
    if "makespan_ratio" in data.columns:
        agg_dict["makespan_ratio"] = ["mean", "median", "std", "min", "max"]
    
    summary_stats = data.groupby("scheduler").agg(agg_dict).round(3)
    
    # Flatten column names
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
    summary_stats.rename(columns={"dataset_count": "num_datasets"}, inplace=True)
    
    # Save summary statistics
    summary_stats.to_csv(outputdir.joinpath("bsp_summary_statistics.csv"))
    
    # Create a readable summary report
    with open(outputdir.joinpath("bsp_analysis_report.txt"), "w") as f:
        f.write("BSP Scheduler Benchmarking Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total datasets evaluated: {data['dataset'].nunique()}\n")
        f.write(f"Total schedulers evaluated: {data['scheduler'].nunique()}\n")
        f.write(f"Total experiments: {len(data)}\n\n")
        
        f.write("Scheduler Performance Summary:\n")
        f.write("-" * 30 + "\n")
        
        for scheduler in summary_stats.index:
            f.write(f"\n{scheduler}:\n")
            
            # Write available metrics
            if "makespan_mean" in summary_stats.columns:
                f.write(f"  Mean makespan: {summary_stats.loc[scheduler, 'makespan_mean']:.3f}\n")
                f.write(f"  Median makespan: {summary_stats.loc[scheduler, 'makespan_median']:.3f}\n")
                f.write(f"  Std makespan: {summary_stats.loc[scheduler, 'makespan_std']:.3f}\n")
            
            if "makespan_ratio_mean" in summary_stats.columns:
                f.write(f"  Mean ratio: {summary_stats.loc[scheduler, 'makespan_ratio_mean']:.3f}\n")
                f.write(f"  Median ratio: {summary_stats.loc[scheduler, 'makespan_ratio_median']:.3f}\n")
                f.write(f"  Std ratio: {summary_stats.loc[scheduler, 'makespan_ratio_std']:.3f}\n")


def run_bsp_analysis(resultsdir: pathlib.Path,
                     outputdir: pathlib.Path,
                     glob: str = None,
                     title: str = "BSP Scheduler Benchmarking",
                     datadir: pathlib.Path = None) -> None:
    """Run complete analysis of BSP benchmarking results.

    Args:
        resultsdir: Directory containing result CSV files
        outputdir: Directory to save analysis outputs
        glob: Glob pattern for result files
        title: Title for plots
        datadir: Directory containing dataset files (for CCR calculation)
    """

    outputdir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading BSP benchmarking data...")
    data = load_bsp_data(resultsdir, glob)

    if data.empty:
        logging.warning("No BSP benchmarking data found. Skipping analysis.")
        return

    logging.info(f"Loaded {len(data)} results from {data['dataset'].nunique()} datasets")

    # Clean scheduler names
    data = clean_scheduler_names(data)

    # Add CCR analysis if datadir is provided
    if datadir is not None:
        logging.info("Calculating CCR values for datasets...")
        data = add_ccr_to_data(data, datadir)

        logging.info("Creating CCR scatter plot...")
        create_ccr_scatter_plot(data, outputdir)

    # Generate analysis outputs
    logging.info("Creating strategy comparison plot...")
    create_scheduler_comparison_plot(data, outputdir)

    logging.info("Creating makespan heatmap...")
    create_makespan_heatmap(data, outputdir)

    logging.info("Generating summary statistics...")
    create_summary_statistics(data, outputdir)

    logging.info(f"BSP analysis complete. Results saved to {outputdir}")


if __name__ == "__main__":
    # For standalone testing
    import sys

    if len(sys.argv) < 3:
        print("Usage: python postprocessing.py <results_dir> <output_dir> [data_dir]")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)

    resultsdir = pathlib.Path(sys.argv[1])
    outputdir = pathlib.Path(sys.argv[2])
    datadir = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else None

    run_bsp_analysis(resultsdir, outputdir, datadir=datadir)