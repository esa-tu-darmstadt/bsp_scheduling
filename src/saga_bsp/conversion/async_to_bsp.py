from typing import Dict, Hashable, List, Literal, Optional
import networkx as nx
from saga.scheduler import Task
from ..schedule import AsyncSchedule, BSPSchedule, BSPHardware


def convert_async_to_bsp(
    hardware: BSPHardware,
    task_graph: nx.DiGraph,
    async_schedule: Dict[Hashable, List[Task]],
    strategy: Literal["eager", "level-based", "earliest-finishing-next"] = "earliest-finishing-next",
    backfill_threshold_percent: float = None,
    verbose: bool = False,
    optimize_sa: bool = False,
    sa_max_iterations: int = 1000,
    sa_max_temp: float = 100.0,
    sa_min_temp: float = 0.1,
    sa_cooling_rate: float = 0.99,
) -> BSPSchedule:
    """Convert an async schedule to a BSP schedule using one of three strategies.
    
    Args:
        hardware: The BSP hardware defining the network and synchronization time
        task_graph: The task dependency graph  
        async_schedule: The async schedule as Dict[processor, List[Task]]
        strategy: Conversion strategy to use:
            - "eager": Schedule as many tasks as possible per superstep
            - "level-based": Schedule one task per processor per superstep
            - "earliest-finishing-next": Schedule earliest finishing task next
        backfill_threshold_percent: Only applies to "earliest-finishing-next" strategy.
            If specified, allows backfilling tasks into current superstep as long as 
            superstep finish time doesn't increase by more than this percentage (e.g., 0.05 = 5%).
        optimize_sa: If True, apply simulated annealing optimization after conversion
        sa_max_iterations: Maximum SA iterations (default: 1000)
        sa_max_temp: Maximum SA temperature (default: 100.0)
        sa_min_temp: Minimum SA temperature (default: 0.1)  
        sa_cooling_rate: SA cooling rate (default: 0.99)
        
    Returns:
        BSPSchedule: The converted BSP schedule, optionally optimized with SA
    """
    # Create wrapper for easier access
    async_sched = AsyncSchedule(async_schedule)
    
    # Create BSP schedule
    bsp_schedule = BSPSchedule(hardware, task_graph)
    
    # Create mapping from task name to processor and async task
    task_to_processor: Dict[str, Hashable] = {}
    task_to_async: Dict[str, Task] = {}
    
    for processor, tasks in async_sched.items():
        for task in tasks:
            task_to_processor[task.name] = processor
            task_to_async[task.name] = task
    
    # Use appropriate strategy to determine superstep mapping
    if strategy == "eager":
        superstep_mapping = _eager_strategy(
            task_graph, task_to_processor, async_sched
        )
    elif strategy == "level-based":
        superstep_mapping = _level_based_strategy(
            task_graph, task_to_processor, async_sched
        )
    elif strategy == "earliest-finishing-next":
        superstep_mapping = _earliest_finishing_next_strategy(
            task_graph, task_to_processor, async_sched, hardware, backfill_threshold_percent, verbose
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Create supersteps and schedule tasks
    num_supersteps = max(superstep_mapping.values()) + 1 if superstep_mapping else 0
    
    for _ in range(num_supersteps):
        bsp_schedule.add_superstep()
    
    # Schedule tasks into their assigned supersteps
    for task_name, superstep_idx in superstep_mapping.items():
        processor = task_to_processor[task_name]
        superstep = bsp_schedule.supersteps[superstep_idx]
        superstep.schedule_task(task_name, processor)
    
    # Apply simulated annealing optimization if requested
    if optimize_sa:
        try:
            from ..optimization.simulated_annealing_v2 import BSPSimulatedAnnealing
            
            if verbose:
                print(f"Initial BSP schedule makespan: {bsp_schedule.makespan:.2f}")
            
            # Create and run simulated annealing
            sa = BSPSimulatedAnnealing(
                max_iterations=sa_max_iterations,
                max_temp=sa_max_temp,
                min_temp=sa_min_temp,
                cooling_rate=sa_cooling_rate
            )
            
            optimized_schedule = sa.optimize(bsp_schedule)
            
            if verbose:
                print(f"Optimized BSP schedule makespan: {optimized_schedule.makespan:.2f}")
                improvement = bsp_schedule.makespan - optimized_schedule.makespan
                improvement_pct = improvement / bsp_schedule.makespan * 100 if bsp_schedule.makespan > 0 else 0
                print(f"Improvement: {improvement:.2f} ({improvement_pct:.1f}%)")
                sa.print_optimization_stats()
            
            return optimized_schedule
            
        except ImportError:
            if verbose:
                print("Warning: Simulated annealing optimization requested but module not available")
            return bsp_schedule
    
    return bsp_schedule


def _eager_strategy(
    task_graph: nx.DiGraph,
    task_to_processor: Dict[str, Hashable], 
    async_schedule: AsyncSchedule
) -> Dict[str, int]:
    """Eager strategy: Schedule as many tasks as possible in each superstep.
    
    Opens a new superstep and for each processor, schedules tasks in order
    as long as inter-processor dependencies can be satisfied.
    """
    superstep_mapping: Dict[str, int] = {}
    current_superstep = 0
    
    # Track which tasks have been scheduled
    scheduled_tasks = set()
    
    while len(scheduled_tasks) < len(task_graph.nodes):
        # Track if any task was scheduled in this superstep
        tasks_scheduled_this_step = False
        
        # For each processor, try to schedule tasks in async order
        for processor, tasks in async_schedule.items():
            for task in tasks:
                if task.name in scheduled_tasks:
                    continue
                    
                # Check if all inter-processor dependencies are satisfied
                can_schedule = True
                for pred in task_graph.predecessors(task.name):
                    pred_processor = task_to_processor[pred]
                    
                    if pred_processor != processor:
                        # Inter-processor dependency - must be in earlier superstep
                        if pred not in scheduled_tasks:
                            can_schedule = False
                            break
                        pred_superstep = superstep_mapping[pred]
                        if pred_superstep >= current_superstep:
                            can_schedule = False
                            break
                
                if can_schedule:
                    superstep_mapping[task.name] = current_superstep
                    scheduled_tasks.add(task.name)
                    tasks_scheduled_this_step = True
                else:
                    # Can't schedule this task, so stop for this processor
                    break
        
        if not tasks_scheduled_this_step:
            # No tasks could be scheduled, move to next superstep
            current_superstep += 1
        else:
            current_superstep += 1
    
    return superstep_mapping


def _level_based_strategy(
    task_graph: nx.DiGraph,
    task_to_processor: Dict[str, Hashable], 
    async_schedule: AsyncSchedule
) -> Dict[str, int]:
    """Level-based strategy: Schedule one task per processor per superstep.
    
    For each processor, schedule just one task (if possible) per superstep.
    Creates small supersteps with high synchronization overhead.
    """
    superstep_mapping: Dict[str, int] = {}
    current_superstep = 0
    
    # Track position in each processor's task list
    processor_positions = {proc: 0 for proc in async_schedule.keys()}
    
    while any(pos < len(tasks) for pos, tasks in 
              zip(processor_positions.values(), async_schedule.values())):
        
        # Try to schedule one task per processor in this superstep
        for processor, tasks in async_schedule.items():
            pos = processor_positions[processor]
            
            if pos < len(tasks):
                task = tasks[pos]
                
                # Check if dependencies are satisfied
                can_schedule = True
                for pred in task_graph.predecessors(task.name):
                    pred_processor = task_to_processor[pred]
                    
                    if pred_processor != processor:
                        # Inter-processor dependency - must be in earlier superstep
                        if pred not in superstep_mapping:
                            can_schedule = False
                            break
                        pred_superstep = superstep_mapping[pred]
                        if pred_superstep >= current_superstep:
                            can_schedule = False
                            break
                
                if can_schedule:
                    superstep_mapping[task.name] = current_superstep
                    processor_positions[processor] += 1
        
        current_superstep += 1
    
    return superstep_mapping


def _calculate_task_finish_time(
    task: str,
    task_finish_times: Dict[str, float],
    task_graph: nx.DiGraph,
    hardware: BSPHardware,
    task_to_processor: Dict[str, Hashable]
) -> float:
    """Calculate earliest possible finish time for a task."""
    processor = task_to_processor[task]
    
    # Calculate earliest start time based on predecessor finish times
    earliest_start = 0.0
    for pred in task_graph.predecessors(task):
        if pred in task_finish_times:
            earliest_start = max(earliest_start, task_finish_times[pred])
    
    # Calculate execution time and finish time
    task_weight = task_graph.nodes[task]['weight']
    proc_speed = hardware.network.nodes[processor]['weight']
    execution_time = task_weight / proc_speed
    
    return earliest_start + execution_time


def _backfill_superstep(
    ready_queue: List[str],
    superstep_mapping: Dict[str, int],
    task_finish_times: Dict[str, float],
    current_superstep: int,
    threshold_percent: float,
    task_graph: nx.DiGraph,
    hardware: BSPHardware,
    task_to_processor: Dict[str, Hashable],
    verbose: bool = False
) -> None:
    """Backfill tasks into current superstep, directly scheduling them."""
    if not ready_queue:
        return
    
    # Calculate current superstep finish time (max processor finish time in this superstep)
    processor_finish_times = {}
    for task, superstep in superstep_mapping.items():
        if superstep == current_superstep:
            processor = task_to_processor[task]
            task_finish = task_finish_times[task]
            processor_finish_times[processor] = max(
                processor_finish_times.get(processor, 0.0), task_finish
            )
    
    current_superstep_finish = max(processor_finish_times.values()) if processor_finish_times else 0.0
    max_allowed_finish = current_superstep_finish * (1.0 + threshold_percent)
    
    if verbose:
        print(f"  Backfill: current_superstep_finish = {current_superstep_finish}, max_allowed = {max_allowed_finish}")
    
    # Find and schedule backfill candidates
    tasks_to_remove = []
    
    for task in ready_queue:
        processor = task_to_processor[task]
        
        # Check if task can be scheduled in current superstep (no inter-processor deps)
        can_schedule = True
        for pred in task_graph.predecessors(task):
            pred_processor = task_to_processor[pred]
            if pred_processor != processor:
                if pred not in superstep_mapping or superstep_mapping[pred] >= current_superstep:
                    can_schedule = False
                    break
        
        if can_schedule:
            # Calculate what the processor finish time would be with this task
            current_proc_finish = processor_finish_times.get(processor, 0.0)
            task_weight = task_graph.nodes[task]['weight']
            proc_speed = hardware.network.nodes[processor]['weight']
            execution_time = task_weight / proc_speed
            new_proc_finish = current_proc_finish + execution_time
            
            # Check if adding this task would exceed the threshold
            new_superstep_finish = max(current_superstep_finish, new_proc_finish)
            
            if new_superstep_finish <= max_allowed_finish:
                if verbose:
                    print(f"  Backfilling {task} (proc {processor}): new_superstep_finish = {new_superstep_finish}")
                    
                # Schedule the task
                task_finish_time = _calculate_task_finish_time(
                    task, task_finish_times, task_graph, hardware, task_to_processor
                )
                superstep_mapping[task] = current_superstep
                task_finish_times[task] = task_finish_time
                tasks_to_remove.append(task)
                
                # Update processor finish times for next iteration
                processor_finish_times[processor] = new_proc_finish
            else:
                if verbose:
                    print(f"  Cannot backfill {task}: would increase superstep finish to {new_superstep_finish}")
    
    # Remove scheduled tasks from ready queue
    for task in tasks_to_remove:
        ready_queue.remove(task)
        
        # Add newly ready tasks to queue
        for successor in task_graph.successors(task):
            if successor not in superstep_mapping:
                all_deps_satisfied = all(pred in superstep_mapping 
                                       for pred in task_graph.predecessors(successor))
                if all_deps_satisfied and successor not in ready_queue:
                    ready_queue.append(successor)


def _earliest_finishing_next_strategy(
    task_graph: nx.DiGraph,
    task_to_processor: Dict[str, Hashable], 
    async_schedule: AsyncSchedule,
    hardware: BSPHardware,
    backfill_threshold_percent: float = None,
    verbose: bool = False
) -> Dict[str, int]:
    """Earliest-finishing-next strategy with optional backfilling.
    
    Args:
        backfill_threshold_percent: If specified, allows backfilling tasks into
            current superstep as long as superstep finish time doesn't increase
            by more than this percentage (e.g., 0.05 = 5%).
    """
    superstep_mapping: Dict[str, int] = {}
    task_finish_times: Dict[str, float] = {}
    current_superstep = 0
    
    # Initialize ready queue with tasks that have no dependencies
    ready_queue = [task for task in task_graph.nodes() 
                   if task_graph.in_degree(task) == 0]
    
    while ready_queue:
        if verbose:
            print(f"\n--- Superstep {current_superstep} ---")
            print(f"Ready queue: {ready_queue}")
            
        # Find task with earliest finish time among ready tasks
        best_task = None
        best_finish_time = float('inf')
        
        for task in ready_queue:
            finish_time = _calculate_task_finish_time(
                task, task_finish_times, task_graph, hardware, task_to_processor
            )
            if verbose:
                print(f"  {task} (proc {task_to_processor[task]}): finish_time = {finish_time}")
            if finish_time < best_finish_time:
                best_task = task
                best_finish_time = finish_time
        
        if best_task is None:
            break
            
        # Check if best_task can be scheduled in current superstep
        can_schedule = True
        best_processor = task_to_processor[best_task]
        
        for pred in task_graph.predecessors(best_task):
            pred_processor = task_to_processor[pred]
            
            if pred_processor != best_processor:
                # Inter-processor dependency - predecessor must be in earlier superstep
                if pred not in superstep_mapping:
                    can_schedule = False
                    break
                pred_superstep = superstep_mapping[pred]
                if pred_superstep >= current_superstep:
                    can_schedule = False
                    break
        
        if can_schedule:
            if verbose:
                print(f"✓ Scheduling {best_task} in superstep {current_superstep} (finish: {best_finish_time})")
                
            # Schedule task in current superstep
            superstep_mapping[best_task] = current_superstep
            task_finish_times[best_task] = best_finish_time
            
            # Remove from ready queue
            ready_queue.remove(best_task)
            
            # Add newly ready tasks to queue
            for successor in task_graph.successors(best_task):
                if successor not in superstep_mapping:
                    # Check if all dependencies of successor are satisfied
                    all_deps_satisfied = all(pred in superstep_mapping 
                                           for pred in task_graph.predecessors(successor))
                    if all_deps_satisfied and successor not in ready_queue:
                        ready_queue.append(successor)
                        if verbose:
                            print(f"  Added {successor} to ready queue")
        else:
            if verbose:
                print(f"✗ Cannot schedule {best_task} in superstep {current_superstep} (inter-processor dependency)")
                
            # Try backfilling before starting new superstep
            if backfill_threshold_percent is not None:
                if verbose:
                    print(f"Attempting backfill with {backfill_threshold_percent*100}% threshold...")
                _backfill_superstep(
                    ready_queue, superstep_mapping, task_finish_times,
                    current_superstep, backfill_threshold_percent,
                    task_graph, hardware, task_to_processor, verbose
                )
            
            # Start new superstep
            current_superstep += 1
            if verbose:
                print(f"Starting new superstep {current_superstep}")
    
    return superstep_mapping