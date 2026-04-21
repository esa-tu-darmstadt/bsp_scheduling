"""
Superstep Elimination Optimization for BSP Schedules.

This module provides post-processing optimization that iteratively eliminates
supersteps to reduce synchronization overhead and improve makespan. The optimization
can be applied to any BSP schedule produced by any scheduler.

Key Features:
- Eliminates supersteps by redistributing their tasks to adjacent supersteps
- Protects supersteps containing sink tasks (tasks with no successors)
- Uses smallest-first heuristic for efficient elimination ordering
- Automatically repairs dependencies through task duplication when needed
- Performs actual elimination and keeps changes only if beneficial
"""

import logging
from typing import Optional
from ..schedule import BSPSchedule, Superstep, BSPTask

logger = logging.getLogger(__name__)


def optimize_superstep_elimination(schedule: BSPSchedule, verbose: bool = False) -> BSPSchedule:
    """Optimize a BSP schedule by iteratively eliminating supersteps.

    This optimization works by:
    1. Identifying eliminable supersteps (excluding those with sink tasks)
    2. Sorting supersteps by size (smallest first)
    3. Attempting to eliminate each superstep and calculating actual new makespan
    4. Keeping the elimination if it improves makespan, trying next superstep otherwise
    5. Repeating until no beneficial eliminations remain

    Args:
        schedule: BSP schedule to optimize
        verbose: Enable detailed logging of optimization process

    Returns:
        Optimized BSP schedule (new copy with eliminations applied)

    Example:
        >>> from bsp_scheduling.optimization import optimize_superstep_elimination
        >>> # After creating a schedule with any scheduler
        >>> schedule = scheduler.schedule(hardware, task_graph)
        >>> optimized = optimize_superstep_elimination(schedule, verbose=True)
        >>> print(f"Makespan improved from {schedule.makespan} to {optimized.makespan}")
    """
    # Start with a copy of the input schedule
    current_schedule = schedule.copy()
    
    total_eliminations = 0
    iteration = 0

    # Find all sink tasks (tasks with no successors) - compute once for efficiency
    sink_tasks = {node for node in current_schedule.task_graph.nodes()
                 if current_schedule.task_graph.out_degree(node) == 0}

    if verbose:
        initial_makespan = current_schedule.makespan
        initial_supersteps = len(current_schedule.supersteps)
        logger.info(f"\nStarting superstep elimination optimization:")
        logger.info(f"  Initial makespan: {initial_makespan:.2f}")
        logger.info(f"  Initial supersteps: {initial_supersteps}")
        logger.info(f"  Sink tasks (protected): {len(sink_tasks)}")

    while True:
        iteration += 1
        eliminated_in_this_iteration = False

        # Get eliminable superstep indices (exclude supersteps with sink tasks)
        eliminable_indices = []
        for idx in range(0, len(current_schedule.supersteps)):
            superstep = current_schedule.supersteps[idx]
            # Check if superstep contains any sink tasks
            contains_sink = False
            for proc, tasks in superstep.tasks.items():
                if any(task.node in sink_tasks for task in tasks):
                    contains_sink = True
                    break
            if not contains_sink:
                eliminable_indices.append(idx)

        if not eliminable_indices:
            if verbose:
                logger.info(f"  Iteration {iteration}: No eliminable supersteps remaining")
            break

        # Sort supersteps by size (smallest first) using their indices
        eliminable_indices.sort(key=lambda idx: current_schedule.supersteps[idx].total_time)

        if verbose:
            superstep_sizes = [(idx, current_schedule.supersteps[idx].total_time)
                             for idx in eliminable_indices]
            logger.info(f"  Iteration {iteration}: Checking supersteps by size: {superstep_sizes}")

        # Try eliminating each superstep in order (smallest first)
        for superstep_idx in eliminable_indices:
            current_makespan = current_schedule.makespan

            # Create a trial schedule with the elimination
            trial_schedule = current_schedule.copy()

            # Use batch mode to defer start_time propagation during repairs
            trial_schedule.begin_batch_update()
            try:
                # Perform elimination and repair on trial schedule
                _eliminate_and_repair_superstep(trial_schedule, superstep_idx)
            finally:
                trial_schedule.end_batch_update()

            # Calculate new makespan
            new_makespan = trial_schedule.makespan
            benefit = current_makespan - new_makespan

            if benefit > 0:
                # Beneficial elimination - keep it
                if verbose:
                    superstep_size = current_schedule.supersteps[superstep_idx].total_time
                    logger.info(f"    Eliminated superstep {superstep_idx} "
                             f"(size: {superstep_size:.2f}, benefit: {benefit:.2f})")
                current_schedule = trial_schedule
                total_eliminations += 1
                eliminated_in_this_iteration = True
                break  # Restart with fresh sorting
            else:
                # Not beneficial - discard trial and try next superstep
                if verbose:
                    logger.debug(f"    Superstep {superstep_idx} elimination not beneficial "
                               f"(benefit: {benefit:.2f}), trying next")

        # If no elimination was performed in this iteration, we're done
        if not eliminated_in_this_iteration:
            if verbose:
                logger.info(f"  Iteration {iteration}: No beneficial eliminations found, stopping optimization")
            break

    if verbose:
        final_makespan = current_schedule.makespan
        final_supersteps = len(current_schedule.supersteps)
        if total_eliminations > 0:
            total_reduction = initial_makespan - final_makespan
            logger.info(f"  Optimization complete:")
            logger.info(f"    Final makespan: {final_makespan:.2f}")
            logger.info(f"    Final supersteps: {final_supersteps}")
            logger.info(f"    Total makespan reduction: {total_reduction:.2f}")
            logger.info(f"    Total supersteps eliminated: {total_eliminations}")
        else:
            logger.info(f"  No beneficial eliminations found")

    return current_schedule


def _eliminate_and_repair_superstep(schedule: BSPSchedule, superstep_idx: int):
    """Eliminate a superstep and repair all subsequent supersteps by duplicating tasks as needed.

    Args:
        schedule: BSP schedule to modify
        superstep_idx: Index of the superstep to eliminate
    """
    # Remove the target superstep and update task mappings
    eliminated_superstep = schedule.supersteps[superstep_idx]

    # Remove tasks from task mapping
    for processor, tasks in eliminated_superstep.tasks.items():
        for task in tasks:
            if task.node in schedule.task_mapping and task in schedule.task_mapping[task.node]:
                schedule.task_mapping[task.node].remove(task)

    # Remove the superstep from the schedule
    schedule.supersteps.pop(superstep_idx)

    # Invalidate cached indices for all following supersteps
    for i in range(superstep_idx, len(schedule.supersteps)):
        if 'index' in schedule.supersteps[i].__dict__:
            del schedule.supersteps[i].__dict__['index']

    # Build initial task index for supersteps before superstep_idx (built once, updated incrementally)
    task_index = set()
    for i in range(superstep_idx):
        for proc, tasks in schedule.supersteps[i].tasks.items():
            for task in tasks:
                task_index.add(task.node)

    # Repair all subsequent supersteps (indices have shifted down after removal)
    for superstep in schedule.supersteps[superstep_idx:]:
        _repair_superstep_dependencies(schedule, superstep, task_index)
        # Add this superstep's tasks to the index for the next iteration
        for proc, tasks in superstep.tasks.items():
            for task in tasks:
                task_index.add(task.node)


def _build_task_location_index(schedule: BSPSchedule, up_to_superstep: Superstep) -> set:
    """Build an index of which tasks exist in supersteps before the given superstep.

    This enables O(1) lookups instead of O(n) scans through all earlier supersteps.

    Args:
        schedule: BSP schedule
        up_to_superstep: Build index for all supersteps before this one

    Returns:
        Set of task names that exist in earlier supersteps
    """
    task_set = set()

    for ss in schedule.supersteps:
        if ss is up_to_superstep:
            break
        for proc, tasks in ss.tasks.items():
            for task in tasks:
                task_set.add(task.node)

    return task_set


def _repair_superstep_dependencies(schedule: BSPSchedule, superstep: Superstep, task_index: set):
    """Repair dependencies in a superstep by duplicating missing predecessors.

    Args:
        schedule: BSP schedule being modified
        superstep: Superstep to repair
        task_index: Set of task names available in earlier supersteps (passed for efficiency)
    """

    # Repair each processor's tasks
    for processor in list(superstep.tasks.keys()):
        task_list = superstep.tasks[processor]

        # Repair each task in the processor (iterate by index since list may grow)
        task_idx = 0
        while task_idx < len(task_list):
            task = task_list[task_idx]
            # Use task.index instead of task_idx since the task may have moved after inserting predecessors
            _ensure_task_dependencies(schedule, task, superstep, processor, task.index, task_index)
            task_idx += 1


def _ensure_task_dependencies(schedule: BSPSchedule, task: BSPTask,
                             superstep: Superstep, processor: str, task_position: int,
                             task_index: set):
    """Ensure all dependencies of a task are available by duplicating missing predecessors.

    Args:
        schedule: BSP schedule being modified
        task: Task whose dependencies to check
        superstep: Superstep containing the task
        processor: Processor running the task
        task_position: Position of the task in processor's task list
        task_index: Set of task names available in earlier supersteps
    """
    for pred_name in schedule.task_graph.predecessors(task.node):
        if not _predecessor_available(schedule, pred_name, task, superstep, processor, task_position, task_index):
            logger.debug(f"Duplicating missing predecessor {pred_name} for task {task.node} at superstep {superstep.index}, processor {processor}, position {task_position}")
            # Recursively duplicate the missing predecessor
            _duplicate_task_recursively(schedule, pred_name, superstep, processor, task_position, task_index)


def _predecessor_available(schedule: BSPSchedule, pred_name: str, task: Optional[BSPTask],
                          current_superstep: Superstep, processor: str, task_position: int,
                          task_index: set) -> bool:
    """Check if a predecessor is available for a task.

    A predecessor is available if:
    1. It's in the same superstep on same processor before this task, OR
    2. It's in any earlier superstep

    Args:
        schedule: BSP schedule
        pred_name: Name of the predecessor task
        task: Task that needs the predecessor (unused, kept for compatibility)
        current_superstep: Superstep containing the task
        processor: Processor running the task
        task_position: Position of the task in processor's task list
        task_index: Set of task names available in earlier supersteps

    Returns:
        True if predecessor is available, False otherwise
    """
    # Check if predecessor is in an earlier superstep (O(1) lookup with index)
    if pred_name in task_index:
        return True

    # Check if predecessor is in same superstep on same processor before this task
    if processor in current_superstep.tasks:
        for idx, t in enumerate(current_superstep.tasks[processor]):
            if idx >= task_position:
                break
            if t.node == pred_name:
                return True

    return False


def _duplicate_task_recursively(schedule: BSPSchedule, task_name: str,
                               target_superstep: Superstep, target_processor: str,
                               before_position: int, task_index: set):
    """Recursively duplicate a task and its dependencies into a target superstep.

    Args:
        schedule: BSP schedule being modified
        task_name: Name of task to duplicate
        target_superstep: Superstep to duplicate into
        target_processor: Processor to duplicate onto
        before_position: Insert the task BEFORE this position (i.e., at this position,
                        pushing existing tasks back). The dependent task will be at a
                        later position after insertion.
        task_index: Set of task names available in earlier supersteps
    """
    # FIRST: Insert this task at the given position
    # This inserts the task at before_position, pushing subsequent tasks back
    schedule.schedule(task_name, target_processor, target_superstep, before_position)

    # THEN: Insert any missing dependencies at the same position
    # They will be inserted before this task, pushing it (and everything else) back
    for pred_name in schedule.task_graph.predecessors(task_name):
        if not _predecessor_available(schedule, pred_name, None, target_superstep,
                                     target_processor, before_position, task_index):
            # Recursively duplicate the missing predecessor at the same position
            # This will push our just-inserted task back, ensuring correct order
            _duplicate_task_recursively(schedule, pred_name, target_superstep,
                                       target_processor, before_position, task_index)
