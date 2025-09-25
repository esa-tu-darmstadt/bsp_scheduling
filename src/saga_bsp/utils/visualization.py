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
                   legend_loc: Optional[str] = 'upper right',
                   title: Optional[str] = None,
                   y_label: Optional[str] = 'Processors',
                   axis: Optional[plt.Axes] = None) -> plt.Axes:
    """Draw BSP schedule with superstep boundaries and phases.
    
    Args:
        bsp_schedule: The BSP schedule to visualize
        show_phases: Whether to show sync/exchange/compute phases
        use_latex: Whether to use LaTeX formatting
        font_size: Font size for labels
        tick_font_size: Font size for tick labels  
        figsize: Figure size
        legend_loc: Location for the legend (None for no legend)
        title: Optional title for the plot
        y_label: Label for the y-axis (None for no label)
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
                
                # Draw exchange phase (per-processor exchange times)
                if superstep.max_exchange_time > 0:
                    for proc in all_processors:
                        # Get processor-specific exchange time
                        proc_exchange_time = superstep.exchange_time(proc) if proc in superstep.tasks else 0.0
                        if proc_exchange_time > 0:
                            exchange_start = start_time + superstep.sync_time
                            rect = patches.Rectangle(
                                (exchange_start, y_positions[proc] - 0.4),
                                proc_exchange_time, 0.8,
                                facecolor=colors['exchange'],
                                edgecolor='black',
                                alpha=0.7,
                                linewidth=0.5
                            )
                            axis.add_patch(rect)
            
            # Draw computation phase with individual tasks
            for proc, tasks in superstep.tasks.items():
                y_pos = y_positions[proc]

                # Get processor-specific compute phase start
                proc_compute_start = superstep.compute_phase_start(proc)
                proc_compute_time = superstep.compute_time(proc)

                # Draw background compute phase
                if show_phases and proc_compute_time > 0:
                    rect = patches.Rectangle(
                        (proc_compute_start, y_pos - 0.4),
                        proc_compute_time, 0.8,
                        facecolor=colors['compute'],
                        edgecolor='black',
                        alpha=0.3,
                        linewidth=0.5
                    )
                    axis.add_patch(rect)

                # Draw individual tasks
                for task in tasks:
                    task_start = proc_compute_start + task.rel_start
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
        if y_label:
            axis.set_ylabel(y_label, fontsize=font_size)
        axis.tick_params(axis='both', which='major', labelsize=tick_font_size)
        axis.grid(True, which='both', linestyle=':', alpha=0.3)
        axis.set_axisbelow(True)
        axis.set_xlim(0, max_time * 1.05)
        axis.set_ylim(-0.5, len(all_processors) - 0.5)
        if title:
            axis.set_title(title, fontsize=font_size + 2)
        
        # Add legend if showing phases
        if show_phases:
            legend_elements = [
                patches.Patch(color=colors['sync'], alpha=0.7, label='Synchronization'),
                patches.Patch(color=colors['exchange'], alpha=0.7, label='Exchange'),
                patches.Patch(color=colors['compute'], alpha=0.3, label='Computation'),
                patches.Patch(facecolor=colors['task'], edgecolor='black', label='Tasks')
            ]
            if legend_loc is not None:
                axis.legend(handles=legend_elements, loc=legend_loc, fontsize=font_size-1, framealpha=1.0)

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
    exchange_times = [ss.max_exchange_time for ss in bsp_schedule.supersteps]
    compute_times = [ss.max_compute_time for ss in bsp_schedule.supersteps]
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
                compute_start = superstep.compute_phase_start(proc)
                # Compute end is the start plus the processor's compute time
                compute_end = compute_start + superstep.compute_time(proc)
                
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

def print_bsp_schedule(schedule, title = "BSP Schedule", show_per_processor_comm=False):
    """Print schedule details

    Args:
        schedule: The BSP schedule to print
        title: Title for the output
        show_per_processor_comm: If True, show per-processor communication breakdown
    """
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Makespan: {schedule.makespan:.2f}")
    print(f"Number of supersteps: {schedule.num_supersteps}")

    for i, superstep in enumerate(schedule.supersteps):
        print(f"\nSuperstep {i} (start: {superstep.start_time:.2f}, end: {superstep.end_time:.2f}):")
        print(f"  Sync time: {superstep.sync_time:.2f}")
        print(f"  Max exchange time: {superstep.max_exchange_time:.2f}")
        print(f"  Max compute time: {superstep.max_compute_time:.2f}")
        print(f"  Total time: {superstep.total_time:.2f}")

        if show_per_processor_comm and superstep.edges_to_communicate:
            print(f"  Communication edges:")
            for source_proc, dest_proc, source_task, dest_task, comm_time in superstep.edges_to_communicate:
                print(f"    {source_task}@{source_proc} -> {dest_task}@{dest_proc}: {comm_time:.2f}")

        for proc, tasks in superstep.tasks.items():
            if tasks:
                task_info = ", ".join([f"{t.node}({t.duration:.1f})" for t in tasks])
                if show_per_processor_comm:
                    send_time = superstep.send_time(proc)
                    recv_time = superstep.receive_time(proc)
                    exch_time = superstep.exchange_time(proc)
                    comp_time = superstep.compute_time(proc)
                    print(f"    {proc}: tasks=[{task_info}], send={send_time:.2f}, recv={recv_time:.2f}, exch={exch_time:.2f}, comp={comp_time:.2f}")
                else:
                    print(f"    {proc}: {task_info}")


def draw_processor_comm_breakdown(bsp_schedule: BSPSchedule,
                                 superstep_idx: Optional[int] = None,
                                 figsize: Tuple[int, int] = (12, 8),
                                 font_size: int = 12) -> plt.Figure:
    """Visualize per-processor send/receive/exchange times for supersteps.

    Args:
        bsp_schedule: The BSP schedule to analyze
        superstep_idx: If provided, show only this superstep. Otherwise show all.
        figsize: Figure size
        font_size: Font size for labels

    Returns:
        The matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Determine which supersteps to show
    if superstep_idx is not None:
        if 0 <= superstep_idx < len(bsp_schedule.supersteps):
            supersteps = [bsp_schedule.supersteps[superstep_idx]]
            indices = [superstep_idx]
        else:
            raise ValueError(f"Invalid superstep index: {superstep_idx}")
    else:
        supersteps = bsp_schedule.supersteps
        indices = list(range(len(supersteps)))

    if not supersteps:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No supersteps', ha='center', va='center', transform=ax.transAxes)
        return fig

    # Collect all processors
    all_processors = set()
    for ss in supersteps:
        all_processors.update(ss.tasks.keys())
    processors = sorted(list(all_processors))

    if not processors:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No processors', ha='center', va='center', transform=ax.transAxes)
        return fig

    # Create subplots: one for send times, one for receive times, one for exchange times
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Prepare data
    send_data = np.zeros((len(processors), len(supersteps)))
    recv_data = np.zeros((len(processors), len(supersteps)))
    exch_data = np.zeros((len(processors), len(supersteps)))

    for j, ss in enumerate(supersteps):
        for i, proc in enumerate(processors):
            if proc in ss.tasks:
                send_data[i, j] = ss.send_time(proc)
                recv_data[i, j] = ss.receive_time(proc)
                exch_data[i, j] = ss.exchange_time(proc)

    # Create heatmaps
    data_sets = [
        (send_data, 'Send Times', axes[0]),
        (recv_data, 'Receive Times', axes[1]),
        (exch_data, 'Exchange Times (max of send/receive)', axes[2])
    ]

    for data, title, ax in data_sets:
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
        ax.set_title(title, fontsize=font_size + 2)
        ax.set_yticks(range(len(processors)))
        ax.set_yticklabels(processors, fontsize=font_size)

        # Add text annotations
        for i in range(len(processors)):
            for j in range(len(supersteps)):
                if data[i, j] > 0:
                    ax.text(j, i, f'{data[i, j]:.1f}',
                           ha="center", va="center", color="black",
                           fontsize=font_size - 2)

        # Add colorbar
        plt.colorbar(im, ax=ax)

    # Set x-axis labels only on bottom plot
    axes[-1].set_xticks(range(len(indices)))
    axes[-1].set_xticklabels([f'SS{idx}' for idx in indices], fontsize=font_size)
    axes[-1].set_xlabel('Superstep', fontsize=font_size)

    # Set y-axis label
    axes[1].set_ylabel('Processors', fontsize=font_size)

    plt.suptitle('Per-Processor Communication Breakdown', fontsize=font_size + 4)
    plt.tight_layout()

    return fig


def draw_busy_comm_gantt(schedule: Dict[Hashable, List],
                         use_latex: bool = False,
                         font_size: int = 16,
                         tick_font_size: int = 14,
                         figsize: Tuple[int, int] = (12, 6),
                         title: Optional[str] = None,
                         legend_loc: Optional[str] = 'upper right',
                         axis: Optional[plt.Axes] = None,
                         draw_task_labels: bool = True) -> plt.Axes:
    """Draw Gantt chart for an asynchronous schedule with busy communication.
    
    Shows communication time (teal) and computation time (white) for each task.
    Tasks with comm_time attribute will be visualized as two blocks:
    - Teal block for communication (busy waiting/receiving data)
    - White block for computation (actual task execution)
    
    Args:
        schedule: Schedule dict mapping processors to task lists
        use_latex: Whether to use LaTeX formatting
        font_size: Font size for labels
        tick_font_size: Font size for tick labels
        figsize: Figure size
        title: Optional title for the plot
        legend_loc: Location for the legend (None for no legend)
        axis: Existing axis to plot on
        draw_task_labels: Whether to draw task labels
        
    Returns:
        The matplotlib axis with the plot
    """
    rc_context_opts = {'text.usetex': use_latex}
    with rc_context(rc=rc_context_opts):
        
        if axis is None:
            _, axis = plt.subplots(figsize=figsize)
            
        # Get all processors
        processors = sorted(schedule.keys())
        
        if not processors:
            axis.text(0.5, 0.5, 'Empty Schedule', ha='center', va='center', transform=axis.transAxes)
            return axis
        
        # Create y-position for each processor
        y_pos = {proc: i for i, proc in enumerate(processors)}
        
        # Draw tasks
        for proc, tasks in schedule.items():
            y = y_pos[proc]
            
            for task in tasks:
                # Check if task has comm_time attribute
                if hasattr(task, 'comm_time') and task.comm_time > 0:
                    # Draw communication block (teal - same as exchange phase)
                    comm_start = task.start
                    comm_duration = task.comm_time
                    rect_comm = patches.Rectangle(
                        (comm_start, y - 0.4), comm_duration, 0.8,
                        linewidth=0.5, edgecolor='black', facecolor='#4ECDC4', alpha=0.7
                    )
                    axis.add_patch(rect_comm)
                    
                    # Add label for communicated predecessors if available
                    if hasattr(task, 'comm_predecessors') and task.comm_predecessors and comm_duration > 0:
                        # Create label like "T1,T2" for the predecessors being received
                        pred_label = ','.join(task.comm_predecessors)
                        axis.text(
                            comm_start + comm_duration / 2, y,
                            pred_label, ha='center', va='center',
                            color='black', fontsize=font_size - 4, style='italic'
                        )
                    
                    # Draw computation block (white with black border - same as tasks in draw_bsp_gantt)
                    comp_start = task.start + task.comm_time
                    comp_duration = task.end - comp_start
                    rect_comp = patches.Rectangle(
                        (comp_start, y - 0.4), comp_duration, 0.8,
                        linewidth=1.0, edgecolor='black', facecolor='#FFFFFF'
                    )
                    axis.add_patch(rect_comp)
                    
                    # Add task label in the computation block (black text, bold)
                    if draw_task_labels and comp_duration > 0:
                        axis.text(
                            comp_start + comp_duration / 2, y,
                            task.name, ha='center', va='center',
                            color='black', fontsize=font_size - 2, weight='bold'
                        )
                else:
                    # Draw single block for tasks without communication (white with black border)
                    rect = patches.Rectangle(
                        (task.start, y - 0.4), task.end - task.start, 0.8,
                        linewidth=1.0, edgecolor='black', facecolor='#FFFFFF'
                    )
                    axis.add_patch(rect)
                    
                    # Add task label (black text, bold)
                    if draw_task_labels and task.end - task.start > 0:
                        axis.text(
                            task.start + (task.end - task.start) / 2, y,
                            task.name, ha='center', va='center',
                            color='black', fontsize=font_size - 2, weight='bold'
                        )
        
        # Set axis properties
        axis.set_yticks(range(len(processors)))
        axis.set_yticklabels(processors)
        axis.set_ylim(-0.5, len(processors) - 0.5)
        
        # Set x-axis limits
        max_time = max(task.end for tasks in schedule.values() for task in tasks) if any(schedule.values()) else 1
        axis.set_xlim(0, max_time * 1.05)
        
        # Labels and title
        axis.set_xlabel('Time', fontsize=font_size)
        axis.set_ylabel('Processors', fontsize=font_size)
        if title:
            axis.set_title(title, fontsize=font_size + 2)
        
        # Grid and formatting
        axis.grid(True, alpha=0.3, axis='x')
        axis.tick_params(axis='both', which='major', labelsize=tick_font_size)
        
        # Add legend with consistent colors
        if legend_loc is not None:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#4ECDC4', edgecolor='black', alpha=0.7, label='Communication'),
                Patch(facecolor='#FFFFFF', edgecolor='black', label='Computation')
            ]
            axis.legend(handles=legend_elements, loc=legend_loc, fontsize=font_size - 2)
        
        plt.tight_layout()
        return axis