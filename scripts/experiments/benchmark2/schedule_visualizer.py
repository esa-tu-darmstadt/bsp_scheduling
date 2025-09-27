"""
Schedule visualization module for benchmark2.

This module provides functions to visualize and save schedule plots for both
BSP schedules and delay model (async) schedules.
"""

import logging
import pathlib
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt

# Import visualization functions from saga_bsp
from saga_bsp.utils.visualization import draw_bsp_gantt, draw_busy_comm_gantt
from saga_bsp.schedule import BSPSchedule

logger = logging.getLogger(__name__)


def save_schedule_visualization(schedule_result: Dict[str, Any],
                               scheduler_name: str,
                               dataset_name: str,
                               task_graph_idx: int,
                               output_dir: pathlib.Path) -> None:
    """Save schedule visualization to PNG file.

    Args:
        schedule_result: Result dictionary from scheduler containing 'schedule'
        scheduler_name: Name of the scheduler used
        dataset_name: Name of the dataset
        task_graph_idx: Index of the task graph
        output_dir: Directory to save the visualization
    """
    try:
        schedule = schedule_result['schedule']

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename
        folder = output_dir / dataset_name / f"task{task_graph_idx}"
        folder.mkdir(parents=True, exist_ok=True)
        filename = f"{dataset_name}_{scheduler_name}_task{task_graph_idx}.png"
        filepath = folder / filename

        # Create figure
        plt.figure(figsize=(12, 6))

        # Determine schedule type and visualize accordingly
        if isinstance(schedule, BSPSchedule):
            # BSP Schedule - use draw_bsp_gantt
            ax = draw_bsp_gantt(
                bsp_schedule=schedule,
                show_phases=True,
                title=f"{scheduler_name} - {dataset_name} Task {task_graph_idx}",
                show_task_names=False,
                figsize=(12, 6)
            )
            logger.debug(f"Drew BSP Gantt chart for {scheduler_name}")

        elif isinstance(schedule, dict):
            # Delay model schedule - use draw_busy_comm_gantt
            ax = draw_busy_comm_gantt(
                schedule=schedule,
                title=f"{scheduler_name} - {dataset_name} Task {task_graph_idx}",
                figsize=(12, 6),
                draw_task_labels=False
            )
            logger.debug(f"Drew busy comm Gantt chart for {scheduler_name}")

        else:
            logger.warning(f"Unknown schedule type {type(schedule)} for {scheduler_name}")
            return

        # Save the figure
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved schedule visualization to {filepath}")

    except Exception as e:
        logger.error(f"Failed to save schedule visualization for {scheduler_name}: {e}")


def should_save_visualization(task_graph_idx: int, dataset_name: str,
                            scheduler_name: str) -> bool:
    """Determine if visualization should be saved for this combination.

    Currently saves only the first task graph (task_graph_idx == 0) of each dataset/scheduler.

    Args:
        task_graph_idx: Index of the task graph
        dataset_name: Name of the dataset
        scheduler_name: Name of the scheduler

    Returns:
        True if visualization should be saved
    """
    # Save only the first task graph for each dataset/scheduler combination
    return True
    return task_graph_idx == 1 or task_graph_idx == 4