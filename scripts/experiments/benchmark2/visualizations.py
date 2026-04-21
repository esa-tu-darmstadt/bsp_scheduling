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

from schedulers import (
    get_scheduler_display_name, get_ordered_scheduler_names, is_delay_model_scheduler,
    organize_schedulers_by_group, SCHEDULER_GROUP_LABELS, get_scheduler_group
)



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
                        # Triangle pointing up to indicate results beyond visible range
                        ax.plot(j, y_max * 0.96, marker='^', color='red', markersize=8, zorder=10)
                        # Timeout count below the triangle
                        ax.text(j, y_max * 0.92, f'{timeout_n}', ha='center', va='top',
                            fontsize=8, color='red', fontweight='bold')

        # Hide unused subplots
        for i in range(n_datasets, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(outputdir / 'boxplot_combined.png', dpi=300, bbox_inches='tight')
        plt.savefig(outputdir / 'boxplot_combined.pdf', bbox_inches='tight')
        plt.close()

    def _create_aggregated_boxplot(self, data: pd.DataFrame, outputdir: pathlib.Path, label_textsize = 8) -> None:
        """Create single aggregated box plot comparing all schedulers across all datasets.

        Timed-out results are excluded from box plots, with timeout counts shown as
        red triangle markers at the top of the plot.
        """
        plt.figure(figsize=(7.16, 4))

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

        plt.xlabel('Scheduler', fontsize=label_textsize+1)
        plt.ylabel('Makespan Ratio', fontsize=label_textsize+1)
        plt.xticks(rotation=45, ha='right', fontsize=label_textsize)
        plt.yticks(fontsize=label_textsize)
        plt.ylim(1, 5)  # Fixed scale from 1 to 5
        plt.grid(visible=True, axis='y', linestyle='--', alpha=0.7)

        # Add scheduler group separators and labels
        scheduler_names = data['scheduler'].unique()
        _, scheduler_group_boundaries = organize_schedulers_by_group(scheduler_names)
        if scheduler_group_boundaries:
            y_min, y_max = plt.ylim()
            # Position for bracket and labels at top
            bracket_y = y_max + (y_max - y_min) * 0.05
            label_y = y_max + (y_max - y_min) * 0.08
            tick_height = (y_max - y_min) * 0.03

            for group_label, (start_idx, end_idx) in scheduler_group_boundaries.items():
                # Add vertical line between groups (except before first group)
                if start_idx > 0:
                    plt.axvline(x=start_idx - 0.5, color='black', linewidth=3)

                # Draw bracket: horizontal line with end ticks
                bracket_left = start_idx - 0.4
                bracket_right = end_idx - 0.6
                group_center_x = (start_idx + end_idx) / 2 - 0.5

                # Horizontal line
                ax.plot([bracket_left, bracket_right], [bracket_y, bracket_y],
                       color='black', linewidth=1.5, clip_on=False)
                # Left tick
                ax.plot([bracket_left, bracket_left], [bracket_y - tick_height, bracket_y],
                       color='black', linewidth=1.5, clip_on=False)
                # Right tick
                ax.plot([bracket_right, bracket_right], [bracket_y - tick_height, bracket_y],
                       color='black', linewidth=1.5, clip_on=False)

                # Add group label above bracket
                ax.text(group_center_x, label_y, group_label,
                       ha='center', va='bottom', fontsize=8, fontweight='bold',
                       color='black', clip_on=False)

        # Add timeout markers at the top of the plot
        if has_timeout_col and any(timeout_counts.values()):
            # Get x-tick labels in order
            ordered_display = [get_scheduler_display_name(name) for name in get_ordered_scheduler_names(data['scheduler'].unique())]
            y_max = plt.ylim()[1]

            for i, scheduler_display in enumerate(ordered_display):
                timeout_n = timeout_counts.get(scheduler_display, 0)
                if timeout_n > 0:
                    # Triangle pointing up to indicate results beyond visible range
                    ax.plot(i, y_max * 0.96, marker='^', color='red', markersize=8, zorder=10)
                    # Timeout count below the triangle
                    ax.text(i, y_max * 0.92, f'{timeout_n}', ha='center', va='top',
                           fontsize=8, color='red', fontweight='bold')

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

        # Timeout area styling
        timeout_facecolor = 'lightgray'
        timeout_hatch = '///'
        timeout_alpha = 0.8

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
                    # All instances timed out - show fully hatched cell (no text)
                    rect = Rectangle((cell_x, cell_y), 1, 1,
                                   linewidth=1, edgecolor='black', facecolor=timeout_facecolor,
                                   hatch=timeout_hatch, alpha=timeout_alpha)
                    ax.add_patch(rect)
                    continue

                # Store original values for mean calculation
                original_values = values.copy()

                # Cap values at threshold for coloring only
                values = np.clip(values, a_min=None, a_max=upper_threshold)

                # Sort values for gradient
                sorted_values = np.sort(values)

                # Calculate proportions for split cell visualization
                n_valid = len(values)
                valid_fraction = n_valid / total_count if total_count > 0 else 1.0
                timeout_fraction = 1.0 - valid_fraction

                # Create smooth gradient using imshow - only in the valid (left) portion
                gradient_extent_right = cell_x + valid_fraction
                if len(sorted_values) > 1:
                    gradient = sorted_values.reshape(1, -1)
                    im = ax.imshow(
                        gradient,
                        cmap=cmap,
                        aspect='auto',
                        extent=[cell_x, gradient_extent_right, cell_y, cell_y + 1],
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
                        extent=[cell_x, gradient_extent_right, cell_y, cell_y + 1],
                        vmin=global_min,
                        vmax=global_max
                    )

                # Add hatched rectangle for timeout portion (right side)
                if timeout_count > 0:
                    timeout_rect = Rectangle((gradient_extent_right, cell_y), timeout_fraction, 1,
                                           linewidth=0, edgecolor='none', facecolor=timeout_facecolor,
                                           hatch=timeout_hatch, alpha=timeout_alpha)
                    ax.add_patch(timeout_rect)

                # Add cell border
                rect = Rectangle((cell_x, cell_y), 1, 1,
                               linewidth=1, edgecolor='black', facecolor='none')
                ax.add_patch(rect)

                # Add mean value as text only if no timeouts (using original uncapped values)
                if timeout_count == 0:
                    mean_value = np.mean(original_values)
                    ax.text(cell_x + 0.5, cell_y + 0.5, format_value_4char(mean_value),
                           ha='center', va='center', fontsize=value_textsize, fontweight='bold',
                           color='white' if mean_value > (global_min + global_max) * 0.7 else 'black')

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

        # Get scheduler group boundaries and add separators with labels
        _, scheduler_group_boundaries = organize_schedulers_by_group(scheduler_names)
        if scheduler_group_boundaries:
            # Position for bracket and labels at top
            bracket_y = len(datasets) + 0.3
            label_y = len(datasets) + 0.7
            tick_height = 0.15

            for group_label, (start_idx, end_idx) in scheduler_group_boundaries.items():
                # Add vertical line between groups (except before first group)
                if start_idx > 0:
                    ax.axvline(x=start_idx, color='black', linewidth=4)

                # Draw bracket: horizontal line with end ticks
                bracket_left = start_idx + 0.1
                bracket_right = end_idx - 0.1
                group_center_x = (start_idx + end_idx) / 2

                # Horizontal line
                ax.plot([bracket_left, bracket_right], [bracket_y, bracket_y],
                       color='black', linewidth=1.5, clip_on=False)
                # Left tick
                ax.plot([bracket_left, bracket_left], [bracket_y - tick_height, bracket_y],
                       color='black', linewidth=1.5, clip_on=False)
                # Right tick
                ax.plot([bracket_right, bracket_right], [bracket_y - tick_height, bracket_y],
                       color='black', linewidth=1.5, clip_on=False)

                # Add group label above bracket
                ax.text(group_center_x, label_y, group_label,
                       ha='center', va='bottom', fontsize=label_textsize, fontweight='bold',
                       color='black', clip_on=False)

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
        # plt.xlabel('Scheduler', fontsize=label_textsize+1)
        plt.ylabel('Dataset', fontsize=label_textsize+1)

        plt.tight_layout()
        plt.savefig(outputdir / 'heatmap_makespan_ratio.png', dpi=300, bbox_inches='tight')
        plt.savefig(outputdir / 'heatmap_makespan_ratio.pdf', bbox_inches='tight')
        plt.close()


class LatexTableGenerator:
    """LaTeX table generator for benchmark summary statistics."""

    def __init__(self):
        """Initialize LaTeX table generator."""
        pass

    def generate_table(self, resultsdir: pathlib.Path, outputdir: pathlib.Path,
                       highlight_scheduler: Optional[str] = None) -> None:
        """Generate LaTeX summary table with makespan ratio and runtime statistics.

        Args:
            resultsdir: Directory containing benchmark results
            outputdir: Directory to save the LaTeX file
            highlight_scheduler: Optional scheduler name to highlight in bold
        """
        outputdir.mkdir(parents=True, exist_ok=True)

        # Load data
        data = self._load_data(resultsdir)
        if data.empty:
            logger.warning("No data found for LaTeX table")
            return

        # Prepare data
        data = self._prepare_data(data)

        # Generate table
        latex_content = self._create_summary_table(data, highlight_scheduler)

        # Write to file
        output_path = outputdir / "summary_table.tex"
        with open(output_path, 'w') as f:
            f.write(latex_content)

        logger.info(f"LaTeX table saved to {output_path}")

        # Generate stats file with \newcommand definitions
        self._generate_stats_file(data, outputdir)

    def _generate_stats_file(self, data: pd.DataFrame, outputdir: pathlib.Path) -> None:
        """Generate LaTeX file with \\newcommand definitions for all statistics.

        Uses \\csname/\\endcsname pattern to allow scheduler keys with numbers.
        Access stats via: \\schedstat{scheduler-key}{StatName}

        Args:
            data: Prepared DataFrame with benchmark results
            outputdir: Directory to save the stats file
        """
        # Filter out timed-out results for statistics
        has_timeout_col = 'timed_out' in data.columns
        if has_timeout_col:
            valid_data = data[~data['timed_out']].copy()
        else:
            valid_data = data.copy()

        # Filter out NaN values
        valid_data = valid_data[valid_data['makespan_ratio'].notna()]

        # Get ordered scheduler names
        scheduler_names = data['scheduler'].unique()
        ordered_names = get_ordered_scheduler_names(scheduler_names)

        lines = []
        lines.append("% summary_stats.tex - Auto-generated scheduler statistics")
        lines.append("% Generated from benchmark results - DO NOT EDIT MANUALLY")
        lines.append("%")
        lines.append("% Usage: \\schedstat{scheduler-key}{StatName}")
        lines.append("% Example: \\schedstat{HEFT-BSP-Eager}{MeanRatio} -> 1.50")
        lines.append("% Example: \\schedstat{HDaggScheduler-0.01}{MeanOverhead} -> 525")
        lines.append("")

        # ============================================================
        # LOOKUP MACRO
        # ============================================================
        lines.append("% " + "=" * 60)
        lines.append("% LOOKUP MACRO")
        lines.append("% " + "=" * 60)
        lines.append("\\newcommand{\\schedstat}[2]{\\csname sched@#1@#2\\endcsname}")
        lines.append("")

        # ============================================================
        # BENCHMARK METADATA
        # ============================================================
        lines.append("% " + "=" * 60)
        lines.append("% BENCHMARK METADATA")
        lines.append("% " + "=" * 60)
        # Count unique problems (dataset + variation combinations) per scheduler
        # Use the first scheduler to count since all should have same problems
        first_scheduler = ordered_names[0] if ordered_names else None
        if first_scheduler:
            num_problems = len(data[data['scheduler'] == first_scheduler])
        else:
            num_problems = len(data['scheduler'].unique())
        num_schedulers = len(ordered_names)
        num_datasets = data['dataset'].nunique() if 'dataset' in data.columns else 0
        lines.append(f"\\newcommand{{\\benchNumProblems}}{{{num_problems}}}")
        lines.append(f"\\newcommand{{\\benchNumSchedulers}}{{{num_schedulers}}}")
        lines.append(f"\\newcommand{{\\benchNumDatasets}}{{{num_datasets}}}")
        lines.append("")

        # ============================================================
        # PER-SCHEDULER STATISTICS
        # ============================================================
        lines.append("% " + "=" * 60)
        lines.append("% PER-SCHEDULER STATISTICS")
        lines.append("% " + "=" * 60)
        lines.append("")

        # Helper to generate csname command
        def csname_cmd(key: str, stat: str, value: str) -> str:
            return f"\\expandafter\\newcommand\\csname sched@{key}@{stat}\\endcsname{{{value}}}"

        # Track stats for best-of calculations
        all_stats = []

        for scheduler in ordered_names:
            scheduler_data = valid_data[valid_data['scheduler'] == scheduler]

            if len(scheduler_data) == 0:
                continue

            display_name = get_scheduler_display_name(scheduler)

            # Makespan ratio statistics
            mean_ratio = scheduler_data['makespan_ratio'].mean()
            median_ratio = scheduler_data['makespan_ratio'].median()
            std_ratio = scheduler_data['makespan_ratio'].std()

            # Calculate overhead percentages: (ratio - 1) * 100
            mean_overhead = (mean_ratio - 1) * 100
            median_overhead = (median_ratio - 1) * 100

            # Count timeouts from original data
            original_scheduler_data = data[data['scheduler'] == scheduler]
            if 'timed_out' in original_scheduler_data.columns:
                timeout_count = int(original_scheduler_data['timed_out'].sum())
            else:
                timeout_count = 0

            # Store for best-of calculations
            all_stats.append({
                'scheduler': scheduler,
                'display_name': display_name,
                'mean_ratio': mean_ratio,
                'median_ratio': median_ratio,
                'std_ratio': std_ratio,
                'mean_overhead': mean_overhead,
                'median_overhead': median_overhead,
                'timeout_count': timeout_count,
                'group': get_scheduler_group(scheduler),
            })

            # Write per-scheduler commands using csname pattern
            lines.append(f"% --- {scheduler} ({display_name}) ---")
            lines.append(csname_cmd(scheduler, "MeanRatio", f"{mean_ratio:.2f}"))
            lines.append(csname_cmd(scheduler, "MedianRatio", f"{median_ratio:.2f}"))
            lines.append(csname_cmd(scheduler, "StdRatio", f"{std_ratio:.2f}"))
            lines.append(csname_cmd(scheduler, "MeanOverhead", f"{mean_overhead:.0f}"))
            lines.append(csname_cmd(scheduler, "MedianOverhead", f"{median_overhead:.0f}"))
            lines.append(csname_cmd(scheduler, "Timeouts", f"{timeout_count}"))
            lines.append("")

        # ============================================================
        # BEST-OF STATISTICS (simple commands for convenience)
        # ============================================================
        lines.append("% " + "=" * 60)
        lines.append("% BEST-OF STATISTICS")
        lines.append("% " + "=" * 60)
        lines.append("")

        # Best overall (excluding baseline/delay-model schedulers)
        bsp_stats = [s for s in all_stats if not is_delay_model_scheduler(s['scheduler'])]
        if bsp_stats:
            best_overall = min(bsp_stats, key=lambda x: x['mean_ratio'])
            lines.append("% === Best Overall (excluding async baseline) ===")
            lines.append(f"\\newcommand{{\\schedBestOverallKey}}{{{best_overall['scheduler']}}}")
            lines.append(f"\\newcommand{{\\schedBestOverallName}}{{{best_overall['display_name']}}}")
            lines.append(f"\\newcommand{{\\schedBestOverallMeanRatio}}{{{best_overall['mean_ratio']:.2f}}}")
            lines.append(f"\\newcommand{{\\schedBestOverallMedianRatio}}{{{best_overall['median_ratio']:.2f}}}")
            lines.append(f"\\newcommand{{\\schedBestOverallMeanOverhead}}{{{best_overall['mean_overhead']:.0f}}}")
            lines.append(f"\\newcommand{{\\schedBestOverallMedianOverhead}}{{{best_overall['median_overhead']:.0f}}}")
            lines.append("")

        # Best per group
        for group_key, group_label in [('baseline', 'Baseline'), ('proposed', 'Proposed'), ('existing', 'Existing')]:
            group_stats = [s for s in all_stats if s['group'] == group_key]
            if group_stats:
                best_in_group = min(group_stats, key=lambda x: x['mean_ratio'])
                lines.append(f"% === Best {group_label} ===")
                lines.append(f"\\newcommand{{\\schedBest{group_label}Key}}{{{best_in_group['scheduler']}}}")
                lines.append(f"\\newcommand{{\\schedBest{group_label}Name}}{{{best_in_group['display_name']}}}")
                lines.append(f"\\newcommand{{\\schedBest{group_label}MeanRatio}}{{{best_in_group['mean_ratio']:.2f}}}")
                lines.append(f"\\newcommand{{\\schedBest{group_label}MedianRatio}}{{{best_in_group['median_ratio']:.2f}}}")
                lines.append(f"\\newcommand{{\\schedBest{group_label}MeanOverhead}}{{{best_in_group['mean_overhead']:.0f}}}")
                lines.append(f"\\newcommand{{\\schedBest{group_label}MedianOverhead}}{{{best_in_group['median_overhead']:.0f}}}")
                lines.append("")

        # ============================================================
        # COMPARISON STATISTICS
        # ============================================================
        lines.append("% " + "=" * 60)
        lines.append("% COMPARISON STATISTICS")
        lines.append("% " + "=" * 60)
        lines.append("")

        # Calculate improvement of our best vs existing best
        proposed_stats = [s for s in all_stats if s['group'] == 'proposed']
        existing_stats = [s for s in all_stats if s['group'] == 'existing']

        if proposed_stats and existing_stats:
            best_proposed = min(proposed_stats, key=lambda x: x['mean_ratio'])
            best_existing = min(existing_stats, key=lambda x: x['mean_ratio'])

            # Improvement: how much better is ours vs existing (as percentage)
            # Formula: (existing - ours) / existing * 100
            if best_existing['mean_ratio'] > 0:
                improvement = (best_existing['mean_ratio'] - best_proposed['mean_ratio']) / best_existing['mean_ratio'] * 100
                lines.append("% Improvement of our best vs existing best")
                lines.append(f"% Formula: (existing_ratio - our_ratio) / existing_ratio * 100")
                lines.append(f"\\newcommand{{\\schedProposedVsExistingImprovement}}{{{improvement:.0f}}}")
                lines.append("")

            # Ratio comparison: how many times better
            if best_proposed['mean_ratio'] > 0:
                ratio_factor = best_existing['mean_ratio'] / best_proposed['mean_ratio']
                lines.append("% How many times better our best is vs existing best")
                lines.append(f"\\newcommand{{\\schedProposedVsExistingFactor}}{{{ratio_factor:.1f}}}")
                lines.append("")

        # Write to file
        output_path = outputdir / "summary_stats.tex"
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        logger.info(f"LaTeX stats file saved to {output_path}")

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
        """Prepare data for table generation."""

        # Apply display names
        data["scheduler_display"] = data["scheduler"].apply(get_scheduler_display_name)

        # Order schedulers
        scheduler_names = data["scheduler"].unique()
        ordered_names = get_ordered_scheduler_names(scheduler_names)
        data["scheduler_order"] = data["scheduler"].map({name: i for i, name in enumerate(ordered_names)})

        return data

    def _create_summary_table(self, data: pd.DataFrame, highlight_scheduler: Optional[str] = None) -> str:
        """Create LaTeX table content with summary statistics.

        Args:
            data: Prepared DataFrame with benchmark results
            highlight_scheduler: Optional scheduler name (original, not display) to highlight.
                               If None, automatically highlights the scheduler with the smallest
                               mean makespan ratio (excluding delay model schedulers).

        Returns:
            LaTeX table content as string
        """
        # Filter out timed-out results for statistics
        has_timeout_col = 'timed_out' in data.columns
        if has_timeout_col:
            valid_data = data[~data['timed_out']].copy()
        else:
            valid_data = data.copy()

        # Filter out NaN values
        valid_data = valid_data[valid_data['makespan_ratio'].notna()]

        # Get ordered scheduler names
        scheduler_names = data['scheduler'].unique()
        ordered_names = get_ordered_scheduler_names(scheduler_names)

        # Calculate statistics per scheduler
        stats = []
        for scheduler in ordered_names:
            scheduler_data = valid_data[valid_data['scheduler'] == scheduler]

            if len(scheduler_data) == 0:
                continue

            display_name = get_scheduler_display_name(scheduler)

            # Makespan ratio statistics
            makespan_mean = scheduler_data['makespan_ratio'].mean()
            makespan_median = scheduler_data['makespan_ratio'].median()
            makespan_std = scheduler_data['makespan_ratio'].std()

            # Runtime statistics (if available)
            if 'scheduler_runtime_s' in scheduler_data.columns:
                runtime_mean = scheduler_data['scheduler_runtime_s'].mean()
                runtime_median = scheduler_data['scheduler_runtime_s'].median()
                runtime_std = scheduler_data['scheduler_runtime_s'].std()
            else:
                runtime_mean = runtime_median = runtime_std = float('nan')

            # Count timeouts from original data (before filtering)
            original_scheduler_data = data[data['scheduler'] == scheduler]
            if 'timed_out' in original_scheduler_data.columns:
                timeout_count = original_scheduler_data['timed_out'].sum()
            else:
                timeout_count = 0

            stats.append({
                'scheduler': scheduler,
                'display_name': display_name,
                'makespan_mean': makespan_mean,
                'makespan_median': makespan_median,
                'makespan_std': makespan_std,
                'runtime_mean': runtime_mean,
                'runtime_median': runtime_median,
                'runtime_std': runtime_std,
                'timeout_count': int(timeout_count),
            })

        # Auto-detect best scheduler if not specified (exclude delay model schedulers)
        if highlight_scheduler is None:
            best_mean = float('inf')
            for stat in stats:
                if not is_delay_model_scheduler(stat['scheduler']):
                    if stat['makespan_mean'] < best_mean:
                        best_mean = stat['makespan_mean']
                        highlight_scheduler = stat['scheduler']

        # Build LaTeX table
        lines = []
        lines.append(r"\begin{tabular}{l|ccc|ccc|c}")
        lines.append(r"\multirow{2}{*}{\textbf{Algorithm}} & \multicolumn{3}{c|}{\textbf{Makespan Ratio}} & \multicolumn{3}{c|}{\textbf{Sched. Runtime (s)}} & Time- \\")
        lines.append(r"\cline{2-4}\cline{5-7}")
        lines.append(r" & Mean & Median & Dev. & Mean & Median & Dev. & outs \\")
        lines.append(r"\Xhline{2\arrayrulewidth}")

        # Track current group for adding thick lines between groups
        current_group = None

        for i, stat in enumerate(stats):
            scheduler = stat['scheduler']
            scheduler_group = get_scheduler_group(scheduler)

            # Add thick line when transitioning to a new group (except for the first row)
            if current_group is not None and scheduler_group != current_group:
                # Replace the previous \hline with a thick line
                if lines[-1] == r"\hline":
                    lines[-1] = r"\Xhline{2\arrayrulewidth}"

            current_group = scheduler_group

            display_name = self._escape_latex(stat['display_name'])
            display_name = self._format_name_multiline(display_name)

            # Format values
            makespan_mean = self._format_stat(stat['makespan_mean'])
            makespan_median = self._format_stat(stat['makespan_median'])
            makespan_std = self._format_stat(stat['makespan_std'])
            runtime_mean = self._format_stat(stat['runtime_mean'])
            runtime_median = self._format_stat(stat['runtime_median'])
            runtime_std = self._format_stat(stat['runtime_std'])
            timeout_count = stat['timeout_count']

            # Check if this scheduler should be highlighted
            is_highlighted = (highlight_scheduler is not None and scheduler == highlight_scheduler)

            # Format timeout cell with red background if > 0
            if timeout_count > 0:
                timeout_cell = f"\\cellcolor{{red!20}}{timeout_count}"
            else:
                timeout_cell = str(timeout_count)

            if is_highlighted:
                # Bold the display name and green background for makespan values
                row = f"\\textbf{{{display_name}}} & \\cellcolor{{green!20}}\\textbf{{{makespan_mean}}} & \\cellcolor{{green!20}}\\textbf{{{makespan_median}}} & \\cellcolor{{green!20}}\\textbf{{{makespan_std}}} & {runtime_mean} & {runtime_median} & {runtime_std} & {timeout_cell} \\\\"
            else:
                row = f"{display_name} & {makespan_mean} & {makespan_median} & {makespan_std} & {runtime_mean} & {runtime_median} & {runtime_std} & {timeout_cell} \\\\"

            lines.append(row)

            # Add hline after each row except the last
            if i < len(stats) - 1:
                lines.append(r"\hline")

        lines.append(r"\end{tabular}")

        return "\n".join(lines)

    def _format_stat(self, value: float) -> str:
        """Format a statistic value for display."""
        if np.isnan(value):
            return "--"
        elif value < 10:
            return f"{value:.2f}"
        elif value < 100:
            return f"{value:.1f}"
        else:
            return f"{value:.0f}"

    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters in text."""
        # Order matters: escape backslash first
        replacements = [
            ('\\', r'\textbackslash{}'),
            ('%', r'\%'),
            ('&', r'\&'),
            ('#', r'\#'),
            ('_', r'\_'),
            ('{', r'\{'),
            ('}', r'\}'),
            ('~', r'\textasciitilde{}'),
            ('^', r'\textasciicircum{}'),
        ]
        for char, escaped in replacements:
            text = text.replace(char, escaped)
        return text

    def _format_name_multiline(self, name: str, max_length: int = 20) -> str:
        """Format a scheduler name, splitting into two lines if too long.

        Uses \\makecell for multi-line cell content.

        Args:
            name: The scheduler display name (already LaTeX-escaped)
            max_length: Maximum length before splitting

        Returns:
            Formatted name, possibly wrapped in \\makecell
        """
        if len(name) <= max_length:
            return name

        # Try to split at a space, + or similar delimiter near the middle
        split_chars = [' + ', ' ', '+']
        best_split = None
        best_balance = float('inf')

        for char in split_chars:
            idx = name.find(char)
            while idx != -1:
                # Calculate how balanced this split is
                balance = abs(idx - (len(name) - idx))
                if balance < best_balance:
                    best_balance = balance
                    best_split = (idx, len(char))
                idx = name.find(char, idx + 1)

        if best_split is not None:
            idx, char_len = best_split
            line1 = name[:idx + char_len].strip()
            line2 = name[idx + char_len:].strip()
            return f"\\makecell[l]{{{line1} \\\\ {line2}}}"

        # No good split point found, just return as-is
        return name


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
                from bsp_scheduling.utils.visualization import draw_bsp_gantt
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
                from bsp_scheduling.utils.visualization import draw_busy_comm_gantt
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