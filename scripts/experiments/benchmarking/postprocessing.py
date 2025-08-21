"""Analysis module for BSP benchmarking results."""

import logging
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import SAGA's gradient_heatmap
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent / "saga" / "src"))
from saga.utils.draw import gradient_heatmap

# Dataset order for consistent plotting
DATASET_ORDER = [
    "in_trees", "out_trees", "chains",
    "blast", "bwa", "cycles", "epigenomics", 
    "genome", "montage", "seismology", "soykb", "srasearch",
    "etl", "predict", "stats", "train",
]

# Note: Scheduler names are now generated cleanly in the BSP library,
# so no renaming is needed


def load_bsp_data(resultsdir: pathlib.Path, glob: str = None) -> pd.DataFrame:
    """Load BSP benchmarking results from CSV files."""
    data = None
    glob = glob or "*_bsp_sync*.csv"
    
    for path in resultsdir.glob(glob):
        df_dataset = pd.read_csv(path, index_col=0)
        df_dataset["dataset"] = path.stem.replace("_bsp_sync1.0", "")  # Remove sync time suffix
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


def create_strategy_comparison_plot(data: pd.DataFrame, outputdir: pathlib.Path) -> None:
    """Create a plot comparing different BSP conversion strategies across all schedulers."""
    
    if data.empty:
        logging.warning("No data found for strategy comparison")
        return
    
    # Extract base scheduler and strategy from scheduler names
    # Expected format: "SCHEDULER-BSP-STRATEGY" or "SCHEDULER-BSP-STRATEGY-BF20%"
    def parse_scheduler_name(name):
        parts = name.split("-")
        if len(parts) >= 3 and parts[1] == "BSP":
            base_scheduler = parts[0]
            strategy = parts[2]
            backfill = parts[3] if len(parts) > 3 else None
            return base_scheduler, strategy, backfill
        return name, None, None
    
    # Parse scheduler names
    parsed_data = data.copy()
    parsed_info = parsed_data["scheduler"].apply(parse_scheduler_name)
    parsed_data["base_scheduler"] = [info[0] for info in parsed_info]
    parsed_data["strategy"] = [info[1] for info in parsed_info]
    parsed_data["backfill"] = [info[2] for info in parsed_info]
    
    # Filter out any schedulers that don't follow the expected naming pattern
    strategy_data = parsed_data[parsed_data["strategy"].notna()]
    
    if strategy_data.empty:
        logging.warning("No BSP strategy data found in expected format")
        return
    
    # Use makespan_ratio if makespan is not available
    y_column = "makespan" if "makespan" in strategy_data.columns else "makespan_ratio"
    y_label = "Makespan" if y_column == "makespan" else "Makespan Ratio"
    
    # Create a combined strategy+backfill column for better visualization
    strategy_data["strategy_full"] = strategy_data.apply(
        lambda row: f"{row['strategy']}" + (f"-{row['backfill']}" if row['backfill'] else ""), 
        axis=1
    )
    
    # Create subplot for each base scheduler
    base_schedulers = sorted(strategy_data["base_scheduler"].unique())
    n_schedulers = len(base_schedulers)
    
    if n_schedulers == 1:
        # Single scheduler - simple box plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=strategy_data,
            x="strategy_full",
            y=y_column,
            palette="Set2"
        )
        plt.title(f"BSP Strategy Comparison - {base_schedulers[0]}")
        plt.xlabel("BSP Strategy")
        plt.ylabel(y_label)
        plt.xticks(rotation=45, ha='right')
    else:
        # Multiple schedulers - grouped plot
        plt.figure(figsize=(14, 8))
        sns.boxplot(
            data=strategy_data,
            x="strategy_full", 
            y=y_column,
            hue="base_scheduler",
            palette="Set2"
        )
        plt.title("BSP Strategy Comparison Across Schedulers")
        plt.xlabel("BSP Strategy") 
        plt.ylabel(y_label)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Base Scheduler", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    plt.savefig(outputdir.joinpath("bsp_strategy_comparison.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(outputdir.joinpath("bsp_strategy_comparison.png"), dpi=300, bbox_inches='tight')
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
        color_label="Maximum Makespan Ratio"
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

    
    # if "makespan_ratio" not in data.columns:
    #     logging.warning("No makespan_ratio column found. Computing relative to minimum.")
    #     # Compute makespan ratio relative to minimum per dataset
    #     data_with_ratio = data.copy()
    #     min_makespan_per_dataset = data.groupby("dataset")["makespan"].min()
    #     data_with_ratio["makespan_ratio"] = data.apply(
    #         lambda row: row["makespan"] / min_makespan_per_dataset[row["dataset"]], 
    #         axis=1
    #     )
    # else:
    #     data_with_ratio = data.copy()
    
    # # Handle duplicate entries by taking the mean
    # data_agg = data_with_ratio.groupby(["dataset", "scheduler"])["makespan_ratio"].mean().reset_index()
    
    # # Create pivot table for heatmap
    # pivot_data = data_agg.pivot(
    #     index="dataset", 
    #     columns="scheduler", 
    #     values="makespan_ratio"
    # )
    
    # # Reorder datasets if possible
    # available_datasets = [d for d in DATASET_ORDER if d in pivot_data.index]
    # if available_datasets:
    #     pivot_data = pivot_data.reindex(available_datasets)
    
    # plt.figure(figsize=(14, 8))
    # sns.heatmap(
    #     pivot_data, 
    #     annot=True, 
    #     fmt='.2f', 
    #     cmap='RdYlBu_r',
    #     center=1.0,  # Center colormap at 1.0 (optimal)
    #     cbar_kws={'label': 'Makespan Ratio'}
    # )
    
    # plt.title("BSP Scheduler Performance Heatmap")
    # plt.xlabel("BSP Scheduler")
    # plt.ylabel("Dataset")
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    
    # plt.savefig(outputdir.joinpath("bsp_makespan_heatmap.pdf"), dpi=300, bbox_inches='tight')
    # plt.savefig(outputdir.joinpath("bsp_makespan_heatmap.png"), dpi=300, bbox_inches='tight')
    # plt.close()


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
                     title: str = "BSP Scheduler Benchmarking") -> None:
    """Run complete analysis of BSP benchmarking results."""
    
    outputdir.mkdir(parents=True, exist_ok=True)
    
    logging.info("Loading BSP benchmarking data...")
    data = load_bsp_data(resultsdir, glob)
    
    if data.empty:
        logging.warning("No BSP benchmarking data found. Skipping analysis.")
        return
    
    logging.info(f"Loaded {len(data)} results from {data['dataset'].nunique()} datasets")
    
    # Clean scheduler names
    data = clean_scheduler_names(data)
    
    # Generate analysis outputs
    logging.info("Creating strategy comparison plot...")
    create_strategy_comparison_plot(data, outputdir)
    
    logging.info("Creating makespan heatmap...")  
    create_makespan_heatmap(data, outputdir)
    
    logging.info("Generating summary statistics...")
    create_summary_statistics(data, outputdir)
    
    logging.info(f"BSP analysis complete. Results saved to {outputdir}")


if __name__ == "__main__":
    # For standalone testing
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python post_bsp_benchmarking.py <results_dir> <output_dir>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    
    resultsdir = pathlib.Path(sys.argv[1])
    outputdir = pathlib.Path(sys.argv[2])
    
    run_bsp_analysis(resultsdir, outputdir)