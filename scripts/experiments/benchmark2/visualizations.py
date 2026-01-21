"""
Visualization components for BSP scheduling benchmarks.

This module implements box plot and heatmap visualizations with support
for scheduler reordering and special handling of delay model schedulers.
"""

import logging
import pathlib
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from schedulers import get_scheduler_display_name, get_ordered_scheduler_names, is_delay_model_scheduler


def propagate_timeouts_per_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Mark all results for a (scheduler, dataset) pair as timed out if any instance timed out.

    This is done for fairer comparison: if a scheduler times out on some instances,
    it likely struggled with the harder cases in that dataset. Without this propagation,
    problematic cases would be masked by the timeout, making the scheduler appear
    better than it actually is (since only the "easy" instances would contribute to
    its average).

    Args:
        data: DataFrame with benchmark results containing 'timed_out' column

    Returns:
        DataFrame with propagated timeout flags
    """
    if 'timed_out' not in data.columns:
        return data

    data = data.copy()

    # Find (scheduler, dataset) pairs with any timeout
    timeout_pairs = data[data['timed_out'] == True].groupby(['scheduler', 'dataset']).size().reset_index()[['scheduler', 'dataset']]

    if len(timeout_pairs) > 0:
        # Create a set of (scheduler, dataset) tuples for fast lookup
        timeout_set = set(zip(timeout_pairs['scheduler'], timeout_pairs['dataset']))

        # Mark all rows for these pairs as timed out
        data['timed_out'] = data.apply(
            lambda row: True if (row['scheduler'], row['dataset']) in timeout_set else row['timed_out'],
            axis=1
        )

    return data


def format_value_4char(value: float) -> str:
    """Format a value to fit in ~4 characters.

    Examples: 1.02, 9.99, 10.1, 99.9, 100, 999
    """
    if value < 10:
        return f'{value:.2f}'
    elif value < 100:
        return f'{value:.1f}'
    else:
        return f'{value:.0f}'

logger = logging.getLogger(__name__)


def organize_datasets_by_type(data: pd.DataFrame) -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
    """Organize datasets by their source type for grouped visualization.

    Args:
        data: DataFrame with benchmark results containing 'source_type' column

    Returns:
        Tuple of (ordered_datasets, group_boundaries)
        where group_boundaries maps group_name to (start_idx, end_idx)
    """
    if 'source_type' not in data.columns:
        # No source type info, return original sorted order
        datasets = sorted(data['dataset'].unique())
        return datasets, {}

    # Get dataset to source_type mapping
    dataset_types = data.groupby('dataset')['source_type'].first().to_dict()

    # Group datasets by type
    dataset_groups = {
        'primitives': [],
        'wfcommons': [],
        'spn': []
    }

    for dataset, source_type in dataset_types.items():
        if source_type in dataset_groups:
            dataset_groups[source_type].append(dataset)

    # Sort datasets within each group
    for group in dataset_groups.values():
        group.sort()

    # Create ordered list and boundaries
    ordered_datasets = []
    group_boundaries = {}

    # Order: primitives, wfcommons, spn
    group_order = ['primitives', 'wfcommons', 'spn']
    group_labels = {'primitives': 'Primitives', 'wfcommons': 'WfCommons', 'spn': 'SPNs'}

    current_idx = 0
    for group_name in group_order:
        group_datasets = dataset_groups[group_name]
        if group_datasets:
            start_idx = current_idx
            ordered_datasets.extend(group_datasets)
            end_idx = current_idx + len(group_datasets)
            group_boundaries[group_labels[group_name]] = (start_idx, end_idx)
            current_idx = end_idx

    return ordered_datasets, group_boundaries

class BoxPlotVisualizer:
    """Box plot visualization for benchmark results."""

    def __init__(self):
        """Initialize box plot visualizer."""
        pass

    def generate_plots(self, resultsdir: pathlib.Path, outputdir: pathlib.Path) -> None:
        """Generate box plots for all datasets.

        Args:
            resultsdir: Directory containing benchmark results
            outputdir: Directory to save plots
        """
        outputdir.mkdir(parents=True, exist_ok=True)

        # Load data
        data = self._load_data(resultsdir)
        if data.empty:
            logger.warning("No data found for box plots")
            return

        # Clean and prepare data
        data = self._prepare_data(data)

        # # Generate plots by dataset (using grouped order)
        # datasets, _ = organize_datasets_by_type(data)
        # for dataset in datasets:
        #     self._create_dataset_boxplot(data, dataset, outputdir)

        # Generate combined plot
        self._create_combined_boxplot(data, outputdir)

        # Generate aggregated plot (all datasets and variations)
        self._create_aggregated_boxplot(data, outputdir)

        logger.info(f"Box plots saved to {outputdir}")

    def _load_data(self, resultsdir: pathlib.Path) -> pd.DataFrame:
        """Load data from result files."""
        data = []
        for path in resultsdir.glob("*.csv"):
            df_dataset = pd.read_csv(path, index_col=0)
            df_dataset["dataset"] = path.stem
            data.append(df_dataset)

        if not data:
            return pd.DataFrame()

        return pd.concat(data, ignore_index=True)

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for visualization."""
        # Apply display names
        data["scheduler_display"] = data["scheduler"].apply(get_scheduler_display_name)

        # Order schedulers
        scheduler_names = data["scheduler"].unique()
        ordered_names = get_ordered_scheduler_names(scheduler_names)
        data["scheduler_order"] = data["scheduler"].map({name: i for i, name in enumerate(ordered_names)})

        return data

    def _create_dataset_boxplot(self, data: pd.DataFrame, dataset: str, outputdir: pathlib.Path) -> None:
        """Create box plot for a specific dataset."""
        dataset_data = data[data['dataset'] == dataset].copy()

        if dataset_data.empty:
            return

        plt.figure(figsize=(12, 8))

        # Sort by scheduler order
        dataset_data = dataset_data.sort_values('scheduler_order')

        # Create box plot
        sns.boxplot(data=dataset_data, x='scheduler_display', y='makespan_ratio')

        plt.title(f'Makespan Distribution - {dataset.title()}', fontsize=14)
        plt.xlabel('Scheduler', fontsize=12)
        plt.ylabel('Makespan Ratio', fontsize=12)
        plt.xticks(rotation=45, ha='right')

        # Add special marking for delay model scheduler
        ax = plt.gca()
        scheduler_names = dataset_data['scheduler'].unique()
        for i, scheduler in enumerate(dataset_data['scheduler_display'].unique()):
            # Find original scheduler name
            orig_name = dataset_data[dataset_data['scheduler_display'] == scheduler]['scheduler'].iloc[0]
            if is_delay_model_scheduler(orig_name):
                # Add red border or different color
                ax.patches[i].set_edgecolor('red')
                ax.patches[i].set_linewidth(2)

        plt.tight_layout()
        plt.savefig(outputdir / f'boxplot_{dataset}.png', dpi=300, bbox_inches='tight')
        plt.savefig(outputdir / f'boxplot_{dataset}.pdf', bbox_inches='tight')
        plt.close()

    def _create_combined_boxplot(self, data: pd.DataFrame, outputdir: pathlib.Path) -> None:
        """Create combined box plot for all datasets.

        Timed-out results are excluded from box plots, with timeout counts shown
        as red triangle markers at the top of each subplot.
        """
        plt.figure(figsize=(16, 10))

        # Sort by scheduler order
        data = data.sort_values('scheduler_order')

        # Create subplot for each dataset (using grouped order)
        datasets, _ = organize_datasets_by_type(data)
        n_datasets = len(datasets)
        cols = min(3, n_datasets)
        rows = (n_datasets + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()

        # Check for timeout column
        has_timeout_col = 'timed_out' in data.columns

        for i, dataset in enumerate(datasets):
            dataset_data = data[data['dataset'] == dataset].copy()

            ax = axes[i] if i < len(axes) else plt.subplot(rows, cols, i+1)

            # Handle timeouts
            if has_timeout_col:
                # Calculate timeout counts per scheduler
                timeout_counts = dataset_data.groupby('scheduler_display')['timed_out'].sum().to_dict()
                # Filter out timed-out results
                plot_data = dataset_data[~dataset_data['timed_out']].copy()
            else:
                timeout_counts = {}
                plot_data = dataset_data.copy()

            # Filter out NaN makespan_ratio values
            plot_data = plot_data[plot_data['makespan_ratio'].notna()]

            sns.boxplot(data=plot_data, x='scheduler_display', y='makespan_ratio', ax=ax, log_scale=10)

            ax.set_title(f'{dataset.title()}', fontsize=12)
            ax.set_xlabel('')
            ax.set_ylabel('Makespan Ratio' if i % cols == 0 else '')
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylim(1, 10)

            # Mark delay model schedulers
            scheduler_names = dataset_data['scheduler'].unique()
            for j, scheduler in enumerate(plot_data['scheduler_display'].unique()):
                orig_name = plot_data[plot_data['scheduler_display'] == scheduler]['scheduler'].iloc[0]
                if is_delay_model_scheduler(orig_name):
                    if len(ax.patches) > j:
                        ax.patches[j].set_edgecolor('red')
                        ax.patches[j].set_linewidth(2)

            # Add timeout markers at the top
            if has_timeout_col and any(timeout_counts.values()):
                ordered_display = [get_scheduler_display_name(name) for name in get_ordered_scheduler_names(dataset_data['scheduler'].unique())]
                y_max = ax.get_ylim()[1]
                for j, scheduler_display in enumerate(ordered_display):
                    timeout_n = timeout_counts.get(scheduler_display, 0)
                    if timeout_n > 0:
                        ax.plot(j, y_max * 0.9, marker='v', color='red', markersize=6, zorder=10)
                        ax.text(j, y_max * 0.95, f'{timeout_n}', ha='center', va='bottom',
                               fontsize=6, color='red', fontweight='bold')

        # Hide unused subplots
        for i in range(n_datasets, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(outputdir / 'boxplot_combined.png', dpi=300, bbox_inches='tight')
        plt.savefig(outputdir / 'boxplot_combined.pdf', bbox_inches='tight')
        plt.close()

    def _create_aggregated_boxplot(self, data: pd.DataFrame, outputdir: pathlib.Path) -> None:
        """Create single aggregated box plot comparing all schedulers across all datasets.

        Timed-out results are excluded from box plots, with timeout counts shown as
        red triangle markers at the top of the plot.
        """
        plt.figure(figsize=(7.16/2, 4.5))

        # Sort by scheduler order
        data = data.sort_values('scheduler_order')

        # Check for timeout column and filter data
        has_timeout_col = 'timed_out' in data.columns
        if has_timeout_col:
            # Calculate timeout counts per scheduler before filtering
            timeout_counts = data.groupby('scheduler_display')['timed_out'].sum().to_dict()
            total_counts = data.groupby('scheduler_display').size().to_dict()
            # Filter out timed-out results for box plot
            plot_data = data[~data['timed_out']].copy()
        else:
            timeout_counts = {}
            total_counts = {}
            plot_data = data.copy()

        # Filter out NaN makespan_ratio values
        plot_data = plot_data[plot_data['makespan_ratio'].notna()]

        # Create box plot with filtered data
        ax = sns.boxplot(data=plot_data, x='scheduler_display', y='makespan_ratio', palette='Set2')

        plt.xlabel('Scheduler', fontsize=10)
        plt.ylabel('Makespan Ratio', fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(1, 5)  # Fixed scale from 1 to 5
        plt.grid(visible=True, axis='y', linestyle='--', alpha=0.7)

        # Add delay model separator if applicable
        scheduler_names = data['scheduler'].unique()
        delay_scheduler_count = sum(1 for name in scheduler_names if is_delay_model_scheduler(name))
        if delay_scheduler_count > 0:
            plt.axvline(x=delay_scheduler_count - 0.5, color='black', linewidth=2, alpha=0.8)

        # Add timeout markers at the top of the plot
        if has_timeout_col and any(timeout_counts.values()):
            # Get x-tick labels in order
            ordered_display = [get_scheduler_display_name(name) for name in get_ordered_scheduler_names(data['scheduler'].unique())]
            y_max = plt.ylim()[1]

            for i, scheduler_display in enumerate(ordered_display):
                timeout_n = timeout_counts.get(scheduler_display, 0)
                if timeout_n > 0:
                    total_n = total_counts.get(scheduler_display, 0)
                    # Add red triangle marker at top
                    ax.plot(i, y_max * 0.95, marker='v', color='red', markersize=8, zorder=10)
                    # Add timeout count text
                    ax.text(i, y_max * 0.98, f'{timeout_n}', ha='center', va='bottom',
                           fontsize=7, color='red', fontweight='bold')

        plt.tight_layout()
        plt.savefig(outputdir / 'boxplot_aggregated.png', dpi=300, bbox_inches='tight')
        plt.savefig(outputdir / 'boxplot_aggregated.pdf', bbox_inches='tight')
        plt.close()


class HeatmapVisualizer:
    """Heatmap visualization for benchmark results."""

    def __init__(self):
        """Initialize heatmap visualizer."""
        pass

    def generate_heatmaps(self, resultsdir: pathlib.Path, outputdir: pathlib.Path) -> None:
        """Generate heatmaps with special delay model visualization.

        Args:
            resultsdir: Directory containing benchmark results
            outputdir: Directory to save heatmaps
        """
        outputdir.mkdir(parents=True, exist_ok=True)

        # Load data
        data = self._load_data(resultsdir)
        if data.empty:
            logger.warning("No data found for heatmaps")
            return

        # Prepare data
        data = self._prepare_data(data)

        self._create_makespan_ratio_heatmap(data, outputdir)

        logger.info(f"Heatmaps saved to {outputdir}")

    def _load_data(self, resultsdir: pathlib.Path) -> pd.DataFrame:
        """Load data from result files."""
        data = []
        for path in resultsdir.glob("*.csv"):
            df_dataset = pd.read_csv(path, index_col=0)
            df_dataset["dataset"] = path.stem
            data.append(df_dataset)

        if not data:
            return pd.DataFrame()

        return pd.concat(data, ignore_index=True)

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for heatmap visualization."""
        # Propagate timeouts: if any instance timed out, mark all for that scheduler/dataset
        data = propagate_timeouts_per_dataset(data)

        # Apply display names
        data["scheduler_display"] = data["scheduler"].apply(get_scheduler_display_name)

        # Order schedulers
        scheduler_names = data["scheduler"].unique()
        ordered_names = get_ordered_scheduler_names(scheduler_names)

        return data

    def _create_makespan_ratio_heatmap(self, data: pd.DataFrame, outputdir: pathlib.Path,
                                     upper_threshold: float = 4.0, figsize: tuple = (7.16, 6), label_textsize = 8, value_textsize = 7) -> None:
        """Create heatmap showing all data variations as gradients within cells using imshow.

        Cells with timeouts are shown with hatched patterns. If all instances timed out,
        the cell is fully hatched with gray. If partial timeouts, shows the gradient of
        successful runs with a hatched overlay and timeout count annotation.
        """
        import numpy as np
        from matplotlib.patches import Rectangle

        # Order columns (schedulers)
        scheduler_names = data['scheduler'].unique()
        ordered_names = get_ordered_scheduler_names(scheduler_names)
        ordered_display_names = [get_scheduler_display_name(name) for name in ordered_names]

        # Get datasets organized by type
        datasets, group_boundaries = organize_datasets_by_type(data)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Remove axes spines and grid
        ax.set_frame_on(False)
        ax.grid(False)

        # Create colormap
        cmap = plt.cm.coolwarm

        # Determine global min/max for consistent coloring
        global_min = min(1.0, data['makespan_ratio'].min())
        global_max = min(upper_threshold, data['makespan_ratio'].max())

        # Create the heatmap using imshow approach (similar to SAGA)
        for i, dataset in enumerate(datasets):
            for j, scheduler_display in enumerate(ordered_display_names):       
                # Get all values for this scheduler-dataset combination
                mask = (data['dataset'] == dataset) & (data['scheduler_display'] == scheduler_display)
                cell_data = data[mask]

                if len(cell_data) == 0:
                    # Add white cell if no data
                    rect = Rectangle((j, len(datasets) - 1 - i), 1, 1,
                                   linewidth=1, edgecolor='black', facecolor='white')
                    ax.add_patch(rect)
                    continue

                # Check for timeouts
                has_timeout_col = 'timed_out' in cell_data.columns
                if has_timeout_col:
                    timeout_count = cell_data['timed_out'].sum()
                    total_count = len(cell_data)
                else:
                    timeout_count = 0
                    total_count = len(cell_data)

                # Get non-timed-out values (filter out NaN makespan_ratio)
                valid_mask = cell_data['makespan_ratio'].notna()
                if has_timeout_col:
                    valid_mask = valid_mask & ~cell_data['timed_out']
                values = cell_data[valid_mask]['makespan_ratio'].values

                # Calculate cell position
                cell_x = j
                cell_y = len(datasets) - 1 - i  # Flip y-axis for proper ordering

                if len(values) == 0:
                    # All instances timed out - show fully hatched gray cell (no text)
                    rect = Rectangle((cell_x, cell_y), 1, 1,
                                   linewidth=1, edgecolor='black', facecolor='lightgray',
                                   hatch='///', alpha=0.8)
                    ax.add_patch(rect)
                    continue

                # Store original values for mean calculation
                original_values = values.copy()

                # Cap values at threshold for coloring only
                values = np.clip(values, a_min=None, a_max=upper_threshold)

                # Sort values for gradient
                sorted_values = np.sort(values)

                # Create smooth gradient using imshow (like SAGA)
                if len(sorted_values) > 1:
                    gradient = sorted_values.reshape(1, -1)
                    im = ax.imshow(
                        gradient,
                        cmap=cmap,
                        aspect='auto',
                        extent=[cell_x, cell_x + 1, cell_y, cell_y + 1],
                        vmin=global_min,
                        vmax=global_max
                    )
                else:
                    # Single value - use imshow for consistency
                    gradient = sorted_values.reshape(1, -1)
                    im = ax.imshow(
                        gradient,
                        cmap=cmap,
                        aspect='auto',
                        extent=[cell_x, cell_x + 1, cell_y, cell_y + 1],
                        vmin=global_min,
                        vmax=global_max
                    )

                # Add cell border (with hatching if partial timeouts)
                if timeout_count > 0:
                    # Partial timeouts - add hatched overlay
                    rect = Rectangle((cell_x, cell_y), 1, 1,
                                   linewidth=1, edgecolor='black', facecolor='none',
                                   hatch='///', alpha=0.3)
                    ax.add_patch(rect)
                else:
                    rect = Rectangle((cell_x, cell_y), 1, 1,
                                   linewidth=1, edgecolor='black', facecolor='none')
                    ax.add_patch(rect)

                # Add mean value as text (using original uncapped values)
                mean_value = np.mean(original_values)
                if timeout_count > 0:
                    # Show mean with timeout count
                    label_text = f'{format_value_4char(mean_value)}\n⏱{timeout_count}/{total_count}'
                    ax.text(cell_x + 0.5, cell_y + 0.5, label_text,
                           ha='center', va='center', fontsize=value_textsize, fontweight='bold',
                           color='white' if mean_value > (global_min + global_max) / 2 else 'black')
                else:
                    ax.text(cell_x + 0.5, cell_y + 0.5, format_value_4char(mean_value),
                           ha='center', va='center', fontsize=value_textsize, fontweight='bold',
                           color='white' if mean_value > (global_min + global_max) / 2 else 'black')

        # Set up axes
        ax.set_xlim(0, len(ordered_display_names))
        ax.set_ylim(0, len(datasets))

        # Set ticks and labels
        ax.set_xticks(np.arange(len(ordered_display_names)) + 0.5)
        ax.set_xticklabels(ordered_display_names, rotation=45, ha='right', fontsize=label_textsize)
        ax.set_yticks(np.arange(len(datasets)) + 0.5)
        ax.set_yticklabels(reversed(datasets), fontsize=label_textsize)

        # Add grid lines
        for i in range(len(ordered_display_names) + 1):
            ax.axvline(i, color='black', linewidth=1)
        for i in range(len(datasets) + 1):
            ax.axhline(i, color='black', linewidth=1)

        # Add visual separator for delay model scheduler
        delay_scheduler_count = sum(1 for name in ordered_names if is_delay_model_scheduler(name))
        if delay_scheduler_count > 0:
            # Add vertical line after delay model schedulers
            ax.axvline(x=delay_scheduler_count, color='black', linewidth=4, alpha=1)
            # Add text annotation
            # ax.text(delay_scheduler_count/2, -0.3, 'Delay Model',
            #        ha='center', va='top', color='black', fontweight='bold', fontsize=10)
            # ax.text(delay_scheduler_count + (len(ordered_display_names) - delay_scheduler_count)/2, -0.3, 'BSP Models',
            #        ha='center', va='top', color='black', fontweight='bold', fontsize=10)

        # Add horizontal colorbar at bottom to save horizontal space
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=global_min, vmax=global_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.6, pad=0.22, aspect=30)
        cbar.set_label('Makespan Ratio', fontsize=label_textsize+1)

        # Add visual separators between dataset groups
        if group_boundaries:
            for group_name, (start_idx, end_idx) in group_boundaries.items():
                # Add horizontal line between groups (except before first group)
                if start_idx > 0:
                    ax.axhline(y=len(datasets) - start_idx, color='black', linewidth=4)

                # Add group label on the right side (rotated 90 degrees)
                group_center_y = len(datasets) - (start_idx + end_idx) / 2
                ax.text(len(ordered_display_names) + 0.1, group_center_y, group_name,
                       ha='left', va='center', fontsize=label_textsize+1, fontweight='bold',
                       color='black', rotation=90)

        # Title and labels
        plt.xlabel('Scheduler', fontsize=label_textsize+1)
        plt.ylabel('Dataset', fontsize=label_textsize+1)

        plt.tight_layout()
        plt.savefig(outputdir / 'heatmap_makespan_ratio.png', dpi=300, bbox_inches='tight')
        plt.savefig(outputdir / 'heatmap_makespan_ratio.pdf', bbox_inches='tight')
        plt.close()


class ScheduleComparisonVisualizer:
    """Schedule comparison visualization for detailed scheduler analysis."""

    def __init__(self):
        """Initialize schedule comparison visualizer."""
        pass

    def generate_schedule_comparisons(self, schedules_dir: pathlib.Path, outputdir: pathlib.Path,
                                    dataset_name: str = None, task_idx: int = 0) -> None:
        """Generate schedule comparison plots for a specific task instance.

        Args:
            schedules_dir: Directory containing saved schedule visualizations
            outputdir: Directory to save comparison plots
            dataset_name: Specific dataset to visualize (None for all)
            task_idx: Task graph index to visualize
        """
        outputdir.mkdir(parents=True, exist_ok=True)

        # Find available schedule files
        schedule_files = self._find_schedule_files(schedules_dir, dataset_name, task_idx)

        if not schedule_files:
            logger.warning(f"No schedule files found for dataset {dataset_name}, task {task_idx}")
            return

        # Group by dataset
        datasets = {}
        for file_path, file_dataset, file_task_idx, scheduler in schedule_files:
            if file_dataset not in datasets:
                datasets[file_dataset] = {}
            datasets[file_dataset][scheduler] = file_path

        # Generate comparison plots for each dataset
        for dataset, scheduler_files in datasets.items():
            if dataset_name is None or dataset == dataset_name:
                self._create_dataset_comparison(dataset, scheduler_files, task_idx, outputdir)

        logger.info(f"Schedule comparison plots saved to {outputdir}")

    def _find_schedule_files(self, schedules_dir: pathlib.Path, dataset_name: str, task_idx: int) -> List[Tuple]:
        """Find schedule visualization files matching criteria."""
        schedule_files = []

        if not schedules_dir.exists():
            return schedule_files

        # Look for PKL files in nested structure: {dataset}/task{task_idx}/*.pkl
        # Pattern: {dataset_name}_{scheduler_name}_task{task_idx}.pkl
        if dataset_name is not None:
            # Search specific dataset
            dataset_dirs = [schedules_dir / dataset_name]
        else:
            # Search all dataset directories
            dataset_dirs = [d for d in schedules_dir.iterdir() if d.is_dir()]

        for dataset_dir in dataset_dirs:
            if not dataset_dir.is_dir():
                continue

            file_dataset = dataset_dir.name
            task_dir = dataset_dir / f"task{task_idx}"

            if not task_dir.exists():
                continue

            # Find PKL files in the task directory
            for file_path in task_dir.glob("*.pkl"):
                # Expected pattern: {dataset_name}_{scheduler_name}_task{task_idx}.pkl
                parts = file_path.stem.split('_')
                if len(parts) >= 3 and parts[-1] == f"task{task_idx}":
                    # Extract scheduler name (everything between dataset and task)
                    dataset_parts = file_dataset.split('_')
                    # Find where dataset name ends and scheduler begins
                    scheduler_start = len(dataset_parts)
                    scheduler_parts = parts[scheduler_start:-1]  # Exclude the "task{X}" part
                    scheduler = '_'.join(scheduler_parts)

                    if scheduler:  # Make sure we found a valid scheduler name
                        schedule_files.append((file_path, file_dataset, task_idx, scheduler))

        return schedule_files

    def _create_dataset_comparison(self, dataset: str, scheduler_files: Dict[str, pathlib.Path],
                                 task_idx: int, outputdir: pathlib.Path) -> None:
        """Create comparison plot for all schedulers on a dataset task."""
        from schedulers import get_ordered_scheduler_names, get_scheduler_display_name
        import pickle

        # Load schedule results
        schedule_results = {}
        for scheduler, file_path in scheduler_files.items():
            try:
                with open(file_path, 'rb') as f:
                    schedule_data = pickle.load(f)
                    schedule_results[scheduler] = schedule_data
            except Exception as e:
                logger.warning(f"Failed to load schedule for {scheduler}: {e}")

        if not schedule_results:
            logger.warning(f"No valid schedule results for {dataset}")
            return

        # Order schedulers consistently
        available_schedulers = list(schedule_results.keys())
        ordered_schedulers = get_ordered_scheduler_names(available_schedulers)

        # Calculate consistent x-limit for all schedulers
        # Get maximum makespan from all schedulers
        max_xlim = 0
        for scheduler in ordered_schedulers:
            schedule_data = schedule_results[scheduler]
            makespan = schedule_data.get('makespan', 0)
            max_xlim = max(max_xlim, makespan)

        # Add 5% padding
        consistent_xlim = max_xlim * 1.05 if max_xlim > 0 else None

        # Calculate layout: 3 columns per row
        n_schedulers = len(ordered_schedulers)
        n_cols = 3
        n_rows = (n_schedulers + n_cols - 1) // n_cols

        # Create figure with tight layout for paper
        # Use same width as heatmap (7.16 inches)
        fig_width = 7.16
        fig_height = n_rows * 2.0  # 2 inches per row
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

        # Handle single row case
        if n_rows == 1:
            axes = axes.reshape(1, -1) if n_cols > 1 else [[axes]]
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Plot each scheduler
        for idx, scheduler in enumerate(ordered_schedulers):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row][col]

            schedule_data = schedule_results[scheduler]
            scheduler_display = get_scheduler_display_name(scheduler)

            self._plot_single_schedule(ax, schedule_data, scheduler_display, xlim=consistent_xlim)

        # Hide unused subplots
        for idx in range(n_schedulers, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row][col].set_visible(False)

        # Adjust layout for paper figure
        plt.tight_layout(pad=0.3)

        # Save figure
        filename = f"schedule_comparison_{dataset}_task{task_idx}"
        plt.savefig(outputdir / f"{filename}.png", dpi=300, bbox_inches='tight')
        plt.savefig(outputdir / f"{filename}.pdf", bbox_inches='tight')
        plt.close()

        logger.info(f"Created schedule comparison for {dataset}, task {task_idx}")

    def _plot_single_schedule(self, ax: plt.Axes, schedule_data: Dict, scheduler_name: str, xlim: Optional[float] = None) -> None:
        """Plot a single schedule on the given axis.

        Args:
            ax: Matplotlib axis to plot on
            schedule_data: Schedule data dictionary
            scheduler_name: Display name for the scheduler
            xlim: Optional x-axis limit to use for consistent scaling across plots
        """
        try:
            if 'schedule_type' in schedule_data and schedule_data['schedule_type'] == 'bsp':
                # BSP schedule
                from saga_bsp.utils.visualization import draw_bsp_gantt
                bsp_schedule = schedule_data['schedule']

                draw_bsp_gantt(
                    bsp_schedule=bsp_schedule,
                    axis=ax,
                    show_phases=True,
                    show_task_names=False,  # Hide task names for space
                    figsize=None,  # Don't create new figure
                    legend_loc=None,  # No legend for space
                    font_size=8,  # Smaller font
                    tick_font_size=7
                )

            elif 'schedule_type' in schedule_data and schedule_data['schedule_type'] == 'busy_comm':
                # Busy communication schedule
                from saga_bsp.utils.visualization import draw_busy_comm_gantt
                async_schedule = schedule_data['schedule']

                draw_busy_comm_gantt(
                    schedule=async_schedule,
                    axis=ax,
                    figsize=None,  # Don't create new figure
                    legend_loc=None,  # No legend for space
                    font_size=8,  # Smaller font
                    tick_font_size=7,
                    draw_task_labels=False  # Hide task labels for space
                )
            else:
                # Generic async schedule (fallback)
                ax.text(0.5, 0.5, f'Schedule\n{scheduler_name}',
                       ha='center', va='center', transform=ax.transAxes)

            # Set title with scheduler name
            ax.set_title(scheduler_name, fontsize=9, pad=2)

            # Set consistent x-axis limit if provided
            if xlim is not None:
                ax.set_xlim(0, xlim)

            # Optimize axis for space
            ax.tick_params(labelsize=6)
            ax.set_xlabel('Time', fontsize=7)

            # Remove y-axis label and replace processor names with integers (1 to n)
            ax.set_ylabel('Processor')

            # Get current y-axis limits to determine number of processors
            ylim = ax.get_ylim()
            n_processors = int(ylim[1] - ylim[0]) + 1

            # Set integer tick labels (1 to n_processors)
            current_yticks = ax.get_yticks()
            if len(current_yticks) > 0:
                # Map from 0-indexed positions to 1-indexed labels
                integer_labels = [str(int(tick) + 1) if 0 <= tick < n_processors else '' for tick in current_yticks]
                ax.set_yticklabels(integer_labels)

        except Exception as e:
            logger.warning(f"Failed to plot schedule for {scheduler_name}: {e}")
            ax.text(0.5, 0.5, f'Error\n{scheduler_name}',
                   ha='center', va='center', transform=ax.transAxes)