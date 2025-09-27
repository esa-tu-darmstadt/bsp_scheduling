"""
Visualization components for BSP scheduling benchmarks.

This module implements box plot and heatmap visualizations with support
for scheduler reordering and special handling of delay model schedulers.
"""

import logging
import pathlib
from typing import List, Optional, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from schedulers import get_scheduler_display_name, get_ordered_scheduler_names, is_delay_model_scheduler

logger = logging.getLogger(__name__)

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

        # Generate plots by dataset
        datasets = data['dataset'].unique()
        for dataset in datasets:
            self._create_dataset_boxplot(data, dataset, outputdir)

        # Generate combined plot
        self._create_combined_boxplot(data, outputdir)

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
        # Clean scheduler names
        data["scheduler"] = data["scheduler"].str.replace("Scheduler", "")

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
        """Create combined box plot for all datasets."""
        plt.figure(figsize=(16, 10))

        # Sort by scheduler order
        data = data.sort_values('scheduler_order')

        # Create subplot for each dataset
        datasets = sorted(data['dataset'].unique())
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

        for i, dataset in enumerate(datasets):
            dataset_data = data[data['dataset'] == dataset].copy()

            ax = axes[i] if i < len(axes) else plt.subplot(rows, cols, i+1)

            sns.boxplot(data=dataset_data, x='scheduler_display', y='makespan_ratio', ax=ax, log_scale=10)

            ax.set_title(f'{dataset.title()}', fontsize=12)
            ax.set_xlabel('')
            ax.set_ylabel('Makespan Ratio' if i % cols == 0 else '')
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylim(1, 10)

            # Mark delay model schedulers
            scheduler_names = dataset_data['scheduler'].unique()
            for j, scheduler in enumerate(dataset_data['scheduler_display'].unique()):
                orig_name = dataset_data[dataset_data['scheduler_display'] == scheduler]['scheduler'].iloc[0]
                if is_delay_model_scheduler(orig_name):
                    if len(ax.patches) > j:
                        ax.patches[j].set_edgecolor('red')
                        ax.patches[j].set_linewidth(2)

        # Hide unused subplots
        for i in range(n_datasets, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(outputdir / 'boxplot_combined.png', dpi=300, bbox_inches='tight')
        plt.savefig(outputdir / 'boxplot_combined.pdf', bbox_inches='tight')
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
        # Clean scheduler names
        data["scheduler"] = data["scheduler"].str.replace("Scheduler", "")

        # Apply display names
        data["scheduler_display"] = data["scheduler"].apply(get_scheduler_display_name)

        # Order schedulers
        scheduler_names = data["scheduler"].unique()
        ordered_names = get_ordered_scheduler_names(scheduler_names)

        return data

    def _create_makespan_ratio_heatmap(self, data: pd.DataFrame, outputdir: pathlib.Path,
                                     upper_threshold: float = 5.0, figsize: tuple = (7.16, 5)) -> None:
        """Create heatmap showing all data variations as gradients within cells."""
        import numpy as np
        from matplotlib.patches import Rectangle

        # Order columns (schedulers)
        scheduler_names = data['scheduler'].unique()
        ordered_names = get_ordered_scheduler_names(scheduler_names)
        ordered_display_names = [get_scheduler_display_name(name) for name in ordered_names]

        # Get datasets
        datasets = sorted(data['dataset'].unique())

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

        # Create the heatmap by drawing each cell manually
        for i, dataset in enumerate(datasets):
            for j, scheduler_display in enumerate(ordered_display_names):
                # Get all values for this scheduler-dataset combination
                mask = (data['dataset'] == dataset) & (data['scheduler_display'] == scheduler_display)
                values = data[mask]['makespan_ratio'].values

                if len(values) == 0:
                    continue

                # Store original values for mean calculation
                original_values = values.copy()

                # Cap values at threshold for coloring only
                values = np.clip(values, a_min=None, a_max=upper_threshold)

                # Calculate cell position
                cell_x = j
                cell_y = len(datasets) - 1 - i  # Flip y-axis for proper ordering

                if len(values) == 1:
                    # Single value - fill entire cell
                    color = cmap((values[0] - global_min) / (global_max - global_min))
                    rect = Rectangle((cell_x, cell_y), 1, 1, facecolor=color, edgecolor='none', linewidth=0)
                    ax.add_patch(rect)
                else:
                    # Multiple values - create gradient within cell
                    sorted_values = np.sort(values)

                    # Divide cell into segments based on number of values
                    n_segments = len(values)
                    segment_width = 1.0 / n_segments

                    for k, value in enumerate(sorted_values):
                        color = cmap((value - global_min) / (global_max - global_min))
                        segment_x = cell_x + k * segment_width
                        rect = Rectangle((segment_x, cell_y), segment_width, 1,
                                       facecolor=color, edgecolor='none', linewidth=0)
                        ax.add_patch(rect)

                # Add mean value as text (using original uncapped values)
                mean_value = np.mean(original_values)
                ax.text(cell_x + 0.5, cell_y + 0.5, f'{mean_value:.2f}',
                       ha='center', va='center', fontsize=9, fontweight='bold',
                       color='white' if mean_value > (global_min + global_max) / 2 else 'black')

        # Set up axes
        ax.set_xlim(0, len(ordered_display_names))
        ax.set_ylim(0, len(datasets))

        # Set ticks and labels
        ax.set_xticks(np.arange(len(ordered_display_names)) + 0.5)
        ax.set_xticklabels(ordered_display_names, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(datasets)) + 0.5)
        ax.set_yticklabels(reversed(datasets))

        # Add grid lines
        for i in range(len(ordered_display_names) + 1):
            ax.axvline(i, color='black', linewidth=1)
        for i in range(len(datasets) + 1):
            ax.axhline(i, color='black', linewidth=1)

        # Add visual separator for delay model scheduler
        delay_scheduler_count = sum(1 for name in ordered_names if is_delay_model_scheduler(name))
        if delay_scheduler_count > 0:
            # Add vertical line after delay model schedulers
            ax.axvline(x=delay_scheduler_count, color='black', linewidth=4, alpha=0.8)
            # Add text annotation
            ax.text(delay_scheduler_count/2, -0.3, 'Delay Model',
                   ha='center', va='top', color='black', fontweight='bold', fontsize=10)
            ax.text(delay_scheduler_count + (len(ordered_display_names) - delay_scheduler_count)/2, -0.3, 'BSP Models',
                   ha='center', va='top', color='black', fontweight='bold', fontsize=10)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=global_min, vmax=global_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label(f'Makespan Ratio (capped at {upper_threshold})', fontsize=10)

        # Title and labels
        plt.xlabel('Scheduler', fontsize=12)
        plt.ylabel('Dataset', fontsize=12)

        plt.tight_layout()
        plt.savefig(outputdir / 'heatmap_makespan_ratio.png', dpi=300, bbox_inches='tight')
        plt.savefig(outputdir / 'heatmap_makespan_ratio.pdf', bbox_inches='tight')
        plt.close()