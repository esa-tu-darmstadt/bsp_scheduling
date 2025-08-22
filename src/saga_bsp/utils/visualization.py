from typing import Dict, Hashable, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from matplotlib import rc_context
from ..schedule import BSPSchedule


def draw_bsp_gantt(bsp_schedule: BSPSchedule, 
                   show_phases: bool = True,
                   use_latex: bool = False,
                   font_size: int = 16,
                   tick_font_size: int = 14,
                   figsize: Tuple[int, int] = (12, 6),
                   axis: Optional[plt.Axes] = None) -> plt.Axes:
    """Draw BSP schedule with superstep boundaries and phases.
    
    Args:
        bsp_schedule: The BSP schedule to visualize
        show_phases: Whether to show sync/exchange/compute phases
        use_latex: Whether to use LaTeX formatting
        font_size: Font size for labels
        tick_font_size: Font size for tick labels  
        figsize: Figure size
        axis: Existing axis to plot on
        
    Returns:
        The matplotlib axis with the plot
    """
    rc_context_opts = {'text.usetex': use_latex}
    with rc_context(rc=rc_context_opts):
        
        if axis is None:
            _, axis = plt.subplots(figsize=figsize)
            
        # Get all processors from the schedule
        all_processors = set()
        for superstep in bsp_schedule.supersteps:
            all_processors.update(superstep.tasks.keys())
        all_processors = sorted(list(all_processors))
        
        if not all_processors:
            axis.text(0.5, 0.5, 'Empty Schedule', ha='center', va='center', transform=axis.transAxes)
            return axis
        
        # Color scheme for different phases
        colors = {
            'sync': '#FF6B6B',      # Red
            'exchange': '#4ECDC4',   # Teal
            'compute': '#45B7D1',    # Blue
            'task': '#FFFFFF'        # White for task boxes
        }
        
        y_positions = {proc: i for i, proc in enumerate(all_processors)}
        max_time = bsp_schedule.makespan
        
        # Draw each superstep
        for superstep in bsp_schedule.supersteps:
            start_time = superstep.start_time
            
            if show_phases:
                # Draw synchronization phase
                if superstep.sync_time > 0:
                    for proc in all_processors:
                        rect = patches.Rectangle(
                            (start_time, y_positions[proc] - 0.4),
                            superstep.sync_time, 0.8,
                            facecolor=colors['sync'], 
                            edgecolor='black',
                            alpha=0.7,
                            linewidth=0.5
                        )
                        axis.add_patch(rect)
                
                # Draw exchange phase  
                if superstep.exchange_time > 0:
                    exchange_start = start_time + superstep.sync_time
                    for proc in all_processors:
                        rect = patches.Rectangle(
                            (exchange_start, y_positions[proc] - 0.4),
                            superstep.exchange_time, 0.8,
                            facecolor=colors['exchange'],
                            edgecolor='black', 
                            alpha=0.7,
                            linewidth=0.5
                        )
                        axis.add_patch(rect)
            
            # Draw computation phase with individual tasks
            compute_start = superstep.compute_phase_start
            
            for proc, tasks in superstep.tasks.items():
                y_pos = y_positions[proc]
                
                # Draw background compute phase
                if show_phases and superstep.compute_time > 0:
                    rect = patches.Rectangle(
                        (compute_start, y_pos - 0.4),
                        superstep.compute_time, 0.8,
                        facecolor=colors['compute'],
                        edgecolor='black',
                        alpha=0.3,
                        linewidth=0.5
                    )
                    axis.add_patch(rect)
                
                # Draw individual tasks
                for task in tasks:
                    task_start = compute_start + task.rel_start
                    task_duration = task.duration
                    
                    # Task rectangle
                    rect = patches.Rectangle(
                        (task_start, y_pos - 0.35),
                        task_duration, 0.7,
                        facecolor=colors['task'],
                        edgecolor='black',
                        linewidth=1.0
                    )
                    axis.add_patch(rect)
                    
                    # Task label
                    if task_duration > max_time * 0.02:  # Only show label if task is wide enough
                        axis.text(
                            task_start + task_duration / 2, y_pos,
                            task.node, ha='center', va='center',
                            fontsize=font_size - 2, weight='bold'
                        )
            
            # Draw superstep boundary line
            if superstep.index < len(bsp_schedule.supersteps) - 1:
                axis.axvline(x=superstep.end_time, color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        # Formatting
        axis.set_yticks(list(y_positions.values()))
        axis.set_yticklabels(list(y_positions.keys()))
        axis.set_xlabel('Time', fontsize=font_size)
        axis.set_ylabel('Processors', fontsize=font_size)
        axis.tick_params(axis='both', which='major', labelsize=tick_font_size)
        axis.grid(True, which='both', linestyle=':', alpha=0.3)
        axis.set_axisbelow(True)
        axis.set_xlim(0, max_time * 1.05)
        axis.set_ylim(-0.5, len(all_processors) - 0.5)
        
        # Add legend if showing phases
        if show_phases:
            legend_elements = [
                patches.Patch(color=colors['sync'], alpha=0.7, label='Synchronization'),
                patches.Patch(color=colors['exchange'], alpha=0.7, label='Exchange'),
                patches.Patch(color=colors['compute'], alpha=0.3, label='Computation'),
                patches.Patch(facecolor=colors['task'], edgecolor='black', label='Tasks')
            ]
            axis.legend(handles=legend_elements, loc='upper right', fontsize=font_size-2)
        
        plt.tight_layout()
        return axis


def draw_superstep_breakdown(bsp_schedule: BSPSchedule,
                            figsize: Tuple[int, int] = (10, 6),
                            font_size: int = 14) -> plt.Figure:
    """Visualize compute/sync/exchange time per superstep.
    
    Args:
        bsp_schedule: The BSP schedule to analyze
        figsize: Figure size
        font_size: Font size for labels
        
    Returns:
        The matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    if not bsp_schedule.supersteps:
        ax1.text(0.5, 0.5, 'No supersteps', ha='center', va='center', transform=ax1.transAxes)
        ax2.text(0.5, 0.5, 'No supersteps', ha='center', va='center', transform=ax2.transAxes)
        return fig
    
    superstep_indices = list(range(len(bsp_schedule.supersteps)))
    sync_times = [ss.sync_time for ss in bsp_schedule.supersteps]
    exchange_times = [ss.exchange_time for ss in bsp_schedule.supersteps]
    compute_times = [ss.compute_time for ss in bsp_schedule.supersteps]
    total_times = [ss.total_time for ss in bsp_schedule.supersteps]
    
    # Stacked bar chart
    width = 0.6
    ax1.bar(superstep_indices, sync_times, width, label='Sync', color='#FF6B6B', alpha=0.8)
    ax1.bar(superstep_indices, exchange_times, width, bottom=sync_times, label='Exchange', color='#4ECDC4', alpha=0.8)
    ax1.bar(superstep_indices, compute_times, width, bottom=np.array(sync_times) + np.array(exchange_times), 
           label='Compute', color='#45B7D1', alpha=0.8)
    
    ax1.set_ylabel('Time', fontsize=font_size)
    ax1.set_title('Superstep Time Breakdown', fontsize=font_size + 2)
    ax1.legend(fontsize=font_size - 2)
    ax1.grid(True, alpha=0.3)
    
    # Individual time components
    x = np.array(superstep_indices)
    ax2.plot(x, sync_times, 'o-', label='Sync', color='#FF6B6B', linewidth=2, markersize=6)
    ax2.plot(x, exchange_times, 's-', label='Exchange', color='#4ECDC4', linewidth=2, markersize=6)
    ax2.plot(x, compute_times, '^-', label='Compute', color='#45B7D1', linewidth=2, markersize=6)
    ax2.plot(x, total_times, 'd-', label='Total', color='#333333', linewidth=2, markersize=6)
    
    ax2.set_xlabel('Superstep', fontsize=font_size)
    ax2.set_ylabel('Time', fontsize=font_size)
    ax2.set_title('Time Components by Superstep', fontsize=font_size + 2)
    ax2.legend(fontsize=font_size - 2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def draw_tile_activity(bsp_schedule: BSPSchedule, 
                      tile_subset: Optional[List[str]] = None,
                      figsize: Tuple[int, int] = (12, 8),
                      font_size: int = 12) -> plt.Axes:
    """Visualize activity across tiles (similar to Popvision).
    
    Shows a heatmap-like visualization of processor utilization over time.
    
    Args:
        bsp_schedule: The BSP schedule to visualize
        tile_subset: Subset of processors to show (None for all)
        figsize: Figure size
        font_size: Font size for labels
        
    Returns:
        The matplotlib axis
    """
    fig, axis = plt.subplots(figsize=figsize)
    
    if not bsp_schedule.supersteps:
        axis.text(0.5, 0.5, 'No supersteps', ha='center', va='center', transform=axis.transAxes)
        return axis
        
    # Get processors to show
    all_processors = set()
    for superstep in bsp_schedule.supersteps:
        all_processors.update(superstep.tasks.keys())
    
    if tile_subset:
        processors = [p for p in tile_subset if p in all_processors]
    else:
        processors = sorted(list(all_processors))
    
    if not processors:
        axis.text(0.5, 0.5, 'No processors found', ha='center', va='center', transform=axis.transAxes)
        return axis
    
    # Create activity matrix
    time_resolution = 100  # Number of time bins
    max_time = bsp_schedule.makespan
    time_bins = np.linspace(0, max_time, time_resolution)
    activity_matrix = np.zeros((len(processors), time_resolution))
    
    # Fill activity matrix
    for superstep in bsp_schedule.supersteps:
        for i, proc in enumerate(processors):
            if proc in superstep.tasks:
                # Mark computation phase as active
                compute_start = superstep.compute_phase_start
                compute_end = superstep.end_time
                
                start_bin = int((compute_start / max_time) * (time_resolution - 1))
                end_bin = int((compute_end / max_time) * (time_resolution - 1))
                
                # Set activity level based on number of tasks
                num_tasks = len(superstep.tasks[proc])
                activity_level = min(1.0, num_tasks / 5.0)  # Normalize to [0,1]
                
                activity_matrix[i, start_bin:end_bin+1] = activity_level
    
    # Create heatmap
    im = axis.imshow(activity_matrix, cmap='YlOrRd', aspect='auto', 
                    extent=[0, max_time, len(processors)-0.5, -0.5])
    
    # Formatting
    axis.set_yticks(range(len(processors)))
    axis.set_yticklabels(processors)
    axis.set_xlabel('Time', fontsize=font_size)
    axis.set_ylabel('Processors', fontsize=font_size)
    axis.set_title('Processor Activity Heatmap', fontsize=font_size + 2)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axis)
    cbar.set_label('Activity Level', fontsize=font_size)
    
    # Add superstep boundaries
    for superstep in bsp_schedule.supersteps:
        if superstep.index < len(bsp_schedule.supersteps) - 1:
            axis.axvline(x=superstep.end_time, color='white', linestyle='--', alpha=0.8, linewidth=1)
    
    plt.tight_layout()
    return axis

def print_bsp_schedule(schedule, title = "BSP Schedule"):
    """Print schedule details"""
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Makespan: {schedule.makespan:.2f}")
    print(f"Number of supersteps: {schedule.num_supersteps}")
    
    for i, superstep in enumerate(schedule.supersteps):
        print(f"\nSuperstep {i} (start: {superstep.start_time:.2f}, end: {superstep.end_time:.2f}):")
        print(f"  Sync time: {superstep.sync_time:.2f}")
        print(f"  Exchange time: {superstep.exchange_time:.2f}")
        print(f"  Compute time: {superstep.compute_time:.2f}")
        print(f"  Total time: {superstep.total_time:.2f}")
        
        for proc, tasks in superstep.tasks.items():
            if tasks:
                task_info = ", ".join([f"{t.node}({t.duration:.1f})" for t in tasks])
                print(f"    {proc}: {task_info}")