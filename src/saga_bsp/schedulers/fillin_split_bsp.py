"""Fill-in/Split BSP Scheduler with three-phase placement strategy.

This scheduler supports three priority modes:
- HEFT mode: Uses upward rank for task prioritization (default)
- CPOP mode: Uses upward + downward rank for task prioritization
- DS mode: Uses dominantsequence (static upward + dynamic downward rank)

Implements a three-phase placement strategy in strict priority order:
1. Fill-in: Try to place tasks in holes of existing supersteps
2. Append: If dependency-ready time is in last superstep, try appending
3. Split: Create new superstep or split existing one
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import logging
from queue import PriorityQueue
from .base import BSPScheduler
from .. import draw_bsp_gantt
from ..schedule import BSPSchedule, BSPHardware, Superstep, BSPTask
import networkx as nx
from .delaymodel.priorities import upward_rank, cpop_ranks, calculate_dynamic_downward_rank

logger = logging.getLogger(__name__)


@dataclass(order=True)
class PrioritizedTask:
    """Task with priority for queue ordering."""
    priority: float
    task: str
    
    def __post_init__(self):
        # Make priority negative so higher ranks are processed first
        self.priority = -self.priority


class FillInSplitBSPScheduler(BSPScheduler):
    """Fill-in/Split BSP scheduler with three-phase placement strategy.

    This scheduler:
    1. Computes task priority using HEFT (upward rank) or CPOP (upward + downward rank)
    2. Processes tasks in priority order
    3. For each task, tries strategies in order:
       - Phase 1: Fill-in (find holes in existing supersteps)
       - Phase 2: Append (if would split last superstep, try appending instead)
       - Phase 3: Split (create/split superstep at dependency-ready time)
    4. Uses first successful strategy (early exit)
    """
    
    def __init__(self, verbose: bool = False, draw_after_each_step: bool = False,
                 priority_mode: str = "heft", optimize_merging: bool = False):
        """Initialize the Fill-in/Split BSP scheduler.

        Args:
            verbose: Enable detailed output to console
            draw_after_each_step: Enable drawing Gantt chart after each scheduling step
            priority_mode: Priority calculation mode - "heft" (upward rank only), "cpop" (upward + downward rank),
                          or "ds" (dominantsequence: static upward + dynamic downward rank)
            optimize_merging: Enable second phase optimization that iteratively merges adjacent supersteps
                            for maximum makespan reduction
        """
        super().__init__()
        self.priority_mode = priority_mode.lower()
        self.name = "FillInSplit+" + self.priority_mode.capitalize()
        self.verbose = verbose
        self.draw_after_each_step = draw_after_each_step
        self.optimize_merging = optimize_merging

        if self.priority_mode not in ["heft", "cpop", "ds"]:
            raise ValueError(f"Invalid priority_mode: {priority_mode}. Must be 'heft', 'cpop', or 'ds'")

        # Initialize statistics
        self._reset_stats()

    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"  {message}")

    def _reset_stats(self):
        """Reset scheduler statistics."""
        self.stats = {
            'tasks_scheduled': 0,
            'fill_in_placements': 0,
            'append_placements': 0,
            'split_placements': 0,
            'supersteps_created': 0,
            'supersteps_split': 0,
            'supersteps_merged': 0,
            'optimization_merges': 0,
            'optimization_iterations': 0
        }
    
    def schedule(self, hardware: BSPHardware, task_graph: nx.DiGraph) -> BSPSchedule:
        """Schedule tasks using three-phase placement strategy.
        
        Args:
            hardware: BSP hardware configuration
            task_graph: Task dependency graph
            
        Returns:
            Optimized BSP schedule
        """
        self._reset_stats()
        
        # Initialize schedule
        schedule = BSPSchedule(hardware, task_graph)
        
        # Compute task priorities based on selected mode
        if self.priority_mode == "heft":
            rank = upward_rank(hardware.network, task_graph)
        elif self.priority_mode == "cpop":
            rank = cpop_ranks(hardware.network, task_graph)
        elif self.priority_mode == "ds":
            # For dominantsequence, we need static upward ranks + dynamic recalculation
            static_upward_ranks = upward_rank(hardware.network, task_graph)
            rank = {}  # Will be recalculated dynamically

        # Initialize priority queue and ready tasks set
        queue = PriorityQueue()
        ready_tasks = set()

        # Add tasks with no dependencies to queue
        for task in task_graph.nodes:
            if task_graph.in_degree(task) == 0:
                if self.priority_mode == "ds":
                    # For DS mode, calculate initial priority
                    dynamic_downward = calculate_dynamic_downward_rank(task, task_graph, schedule)
                    task_priority = static_upward_ranks[task] + dynamic_downward
                    rank[task] = task_priority
                else:
                    task_priority = rank[task]

                queue.put(PrioritizedTask(task_priority, task))
                ready_tasks.add(task)

                if self.verbose:
                    self._log(f"Task {task} added to initial queue with {self.priority_mode} rank {task_priority}")
        
        # Create initial superstep if needed
        if not schedule.supersteps:
            schedule.add_superstep()
            self.stats['supersteps_created'] = 1
        
        # Process tasks in priority order
        task_num = 0
        while not queue.empty():
            task_item = queue.get()
            task_name = task_item.task
            
            if self.verbose:
                self._log(f"\n#{task_num}: Processing task {task_name} with {self.priority_mode} rank {rank[task_name]}")
            
            # Try placement strategies in priority order
            placed = False
            
            # Phase 1: Try fill-in strategy
            processor, superstep = self._try_fill_in(task_name, schedule, hardware, task_graph)
            if processor is not None:
                schedule.schedule(task_name, processor, superstep)
                self.stats['fill_in_placements'] += 1
                placed = True
                if self.verbose:
                    self._log(f"  Placed using fill-in strategy on processor {processor}, "
                               f"superstep {superstep.index}")
            
            # Phase 2: Try append strategy (if dependency-ready time is in last superstep)
            if not placed and False:
                processor, superstep = self._try_append(task_name, schedule, hardware, task_graph)
                if processor is not None:
                    schedule.schedule(task_name, processor, superstep)
                    self.stats['append_placements'] += 1
                    placed = True
                    if self.verbose:
                        self._log(f"  Placed using append strategy on processor {processor}, "
                                   f"superstep {superstep.index}")
            
            # Phase 3: Use split strategy (always works)
            if not placed:
                processor, superstep = self._do_split(task_name, schedule, hardware, task_graph)
                # Schedule the task at the beginning of the new/split superstep
                schedule.schedule(task_name, processor, superstep, 0) 
                self.stats['split_placements'] += 1
                placed = True
                if self.verbose:
                    self._log(f"  Placed using split strategy on processor {processor}, "
                               f"superstep {superstep.index}")
            
            # FIXME: 
            if superstep.index > 0 and schedule.merge_supersteps(superstep.index-1) > 0:
                self.stats['supersteps_merged'] += 1
                self._log(f"  Merged supersteps {superstep.index-1} and {superstep.index} after placement")
            self.stats['tasks_scheduled'] += 1
            
            # Add newly ready successors to queue
            for successor in task_graph.successors(task_name):
                if all(schedule.task_scheduled(pred) for pred in task_graph.predecessors(successor)):
                    if successor not in ready_tasks:
                        ready_tasks.add(successor)

            # For DS mode, recalculate priorities for all ready tasks after placing a task
            if self.priority_mode == "ds":
                # Clear current queue and recalculate priorities
                new_queue = PriorityQueue()
                for ready_task in ready_tasks:
                    if not schedule.task_scheduled(ready_task):
                        dynamic_downward = calculate_dynamic_downward_rank(ready_task, task_graph, schedule)
                        task_priority = static_upward_ranks[ready_task] + dynamic_downward
                        rank[ready_task] = task_priority
                        new_queue.put(PrioritizedTask(task_priority, ready_task))

                        if self.verbose:
                            self._log(f"  Task {ready_task} priority recalculated: DS rank {task_priority}")
                queue = new_queue
            else:
                # For HEFT and CPOP modes, just add new ready tasks
                for successor in task_graph.successors(task_name):
                    if (all(schedule.task_scheduled(pred) for pred in task_graph.predecessors(successor)) and
                        not schedule.task_scheduled(successor)):
                        queue.put(PrioritizedTask(rank[successor], successor))
                        if self.verbose:
                            mode_str = {"heft": "HEFT", "cpop": "CPOP"}[self.priority_mode]
                            self._log(f"  Task {successor} became ready with {mode_str} rank {rank[successor]}")
            
            # Optionally draw Gantt chart after each step
            if self.draw_after_each_step:
                draw_bsp_gantt(schedule, title=f"After #{task_num}: Scheduling {task_name}")
            task_num += 1
        
        # Merge supersteps where possible
        merges = schedule.merge_supersteps()
        self.stats['supersteps_merged'] = merges
        if self.verbose and merges > 0:
            self._log(f"\nMerged {merges} superstep(s) to reduce synchronization overhead")

        # Second phase: Optimize by iteratively eliminating supersteps for maximum makespan reduction
        if self.optimize_merging:
            optimization_eliminations = self._optimize_superstep_elimination(schedule)
            self.stats['optimization_merges'] = optimization_eliminations
            if self.verbose and optimization_eliminations > 0:
                self._log(f"Optimization phase: eliminated {optimization_eliminations} superstep(s) to reduce makespan")

        # Validate the final schedule
        schedule.assert_valid()
        return schedule
    
    def _calculate_task_duration(self, task_name: str, processor: str,
                                hardware: BSPHardware, task_graph: nx.DiGraph) -> float:
        """Calculate computation duration for a task on a processor.
        
        Args:
            task_name: Task to calculate duration for
            processor: Target processor
            hardware: Hardware configuration
            task_graph: Task dependency graph
            
        Returns:
            Task duration (computation time only)
        """
        task_weight = task_graph.nodes[task_name]['weight']
        processor_speed = hardware.network.nodes[processor]['weight']
        return task_weight / processor_speed
    
    def _calculate_communication_time(self, task_name: str, processor: str,
                                     superstep: Superstep, schedule: BSPSchedule,
                                     hardware: BSPHardware, task_graph: nx.DiGraph) -> float:
        """Calculate communication time for predecessors not on same processor.
        
        Since we don't use task duplication, each predecessor has exactly one instance.
        Communication is needed if the predecessor is on a different processor and in
        a previous superstep.
        
        Args:
            task_name: Task to calculate communication for
            processor: Target processor
            superstep: Target superstep
            schedule: Current schedule
            hardware: Hardware configuration
            task_graph: Task dependency graph
            
        Returns:
            Total communication time needed
        """
        comm_time = 0.0
        
        for pred_name in task_graph.predecessors(task_name):
            if not schedule.task_scheduled(pred_name):
                continue  # Predecessor not scheduled yet
            
            # Get the single instance of the predecessor
            pred_instance = schedule.get_single_instance(pred_name)
            
            # Check if communication is needed
            # No communication needed if:
            # 1. Predecessor is on same processor AND in same superstep, OR
            # 2. Predecessor is on same processor AND in earlier superstep
            if pred_instance.proc == processor:
                # Same processor - no communication needed
                continue
            
            # Different processor - need communication if predecessor is in earlier superstep
            if pred_instance.superstep.index < superstep.index:
                # Calculate communication time
                edge_weight = task_graph.edges[(pred_name, task_name)]['weight']
                
                if hardware.network.has_edge(pred_instance.proc, processor):
                    network_speed = hardware.network.edges[pred_instance.proc, processor]['weight']
                    comm_time = comm_time + edge_weight / network_speed
                else:
                    # No network connection available
                    return float('inf')
        
        return comm_time
    
    def _try_fill_in(self, task_name: str, schedule: BSPSchedule,
                    hardware: BSPHardware, task_graph: nx.DiGraph) -> Tuple[Optional[str], Optional[Superstep]]:
        """Try to place task in a hole of an existing superstep.
        
        Args:
            task_name: Task to place
            schedule: Current schedule
            hardware: Hardware configuration
            task_graph: Task dependency graph
            
        Returns:
            Tuple of (processor, superstep) if successful, (None, None) otherwise
        """
        best_processor = None
        best_superstep = None
        best_finish_time = float('inf')
        
        # Check all supersteps for holes
        for superstep in schedule.supersteps:
            for processor in hardware.network.nodes:
                # Check if dependencies allow scheduling here
                if not schedule.can_be_scheduled_in(task_name, superstep, processor):
                    continue
                
                # Calculate task duration
                task_duration = self._calculate_task_duration(task_name, processor, hardware, task_graph)
                
                # Current end time of processor in this superstep
                proc_end_time = superstep.compute_phase_start(processor) + superstep.compute_time(processor)
                
                # Would the task fit in a hole?
                task_finish_time = proc_end_time + task_duration
                superstep_end_time = superstep.end_time
                
                # Check if there's a hole (task fits without extending superstep)
                if task_finish_time <= superstep_end_time * 1.0001:  # Small tolerance
                    if task_finish_time < best_finish_time:
                        best_finish_time = task_finish_time
                        best_processor = processor
                        best_superstep = superstep
                        
                        if self.verbose:
                            hole_size = superstep_end_time - proc_end_time
                            self._log(f"    Found fill-in opportunity: superstep {superstep.index}, "
                                       f"proc {processor}, hole_size={hole_size:.2f}")
        
        return best_processor, best_superstep
    
    def _try_append(self, task_name: str, schedule: BSPSchedule,
                   hardware: BSPHardware, task_graph: nx.DiGraph) -> Tuple[Optional[str], Optional[Superstep]]:
        """Try to append task to last superstep if dependency-ready time allows.
        
        Args:
            task_name: Task to place
            schedule: Current schedule
            hardware: Hardware configuration
            task_graph: Task dependency graph
            
        Returns:
            Tuple of (processor, superstep) if successful, (None, None) otherwise
        """
        if not schedule.supersteps:
            return None, None
        
        last_superstep = schedule.supersteps[-1]
        best_processor = None
        best_finish_time = float('inf')
        
        # Get dependency-ready time (same for all processors)
        dep_ready_time = self._get_dependency_ready_time(task_name, schedule, task_graph)
        
        if dep_ready_time == float('inf'):
            return None, None
        
        # Check which superstep this time falls into
        superstep_at_time = schedule.get_superstep_at_time(dep_ready_time)
        
        # Only proceed if dependency-ready time is in last superstep
        if superstep_at_time != last_superstep:
            return None, None
        
        # For each processor, try to append to last superstep
        for processor in hardware.network.nodes:
            # Check if we can schedule in last superstep
            if schedule.can_be_scheduled_in(task_name, last_superstep, processor):
                # Calculate task duration
                task_duration = self._calculate_task_duration(task_name, processor, hardware, task_graph)
                
                # Calculate finish time
                proc_end_time = last_superstep.compute_time(processor)
                task_finish_relative = proc_end_time + task_duration
                task_finish_absolute = last_superstep.compute_phase_start + task_finish_relative
                
                if task_finish_absolute < best_finish_time:
                    best_finish_time = task_finish_absolute
                    best_processor = processor
                    
                    if self.verbose:
                        self._log(f"    Found append opportunity: last superstep, "
                                   f"proc {processor}, finish={task_finish_absolute:.2f}")
        
        if best_processor is not None:
            return best_processor, last_superstep
        
        return None, None
    
    def _do_split(self, task_name: str, schedule: BSPSchedule,
                 hardware: BSPHardware, task_graph: nx.DiGraph) -> Tuple[str, Superstep]:
        """Place task using split strategy (create/split superstep).
        
        This method:
        1. Gets the dependency-ready time
        2. Creates/splits superstep at that time
        3. Evaluates all processors on this new superstep
        4. Places task on best processor
        
        Args:
            task_name: Task to place
            schedule: Current schedule
            hardware: Hardware configuration
            task_graph: Task dependency graph
            
        Returns:
            Tuple of (processor, superstep) for placement
        """
        # Step 1: Get the dependency-ready time (same for all processors)
        dep_ready_time = self._get_dependency_ready_time(task_name, schedule, task_graph)
        
        if dep_ready_time == float('inf'):
            raise ValueError(f"Could not find valid dependency-ready time for task {task_name}")
        
        if self.verbose:
            self._log(f"    Dependency-ready time: {dep_ready_time:.2f}")
        
        # Step 2: Create/split superstep at the dependency-ready time (only once!)
        original_count = len(schedule.supersteps)
        superstep = schedule.get_or_create_superstep_at_time(dep_ready_time)
        
        # Update statistics for superstep creation/splitting
        if len(schedule.supersteps) > original_count:
            if superstep.index == len(schedule.supersteps) - 1:
                self.stats['supersteps_created'] += 1
                if self.verbose:
                    self._log(f"    Created new superstep {superstep.index}")
            else:
                self.stats['supersteps_split'] += 1
                if self.verbose:
                    self._log(f"    Split to create superstep {superstep.index}")
        
        # Step 3: Evaluate each processor for best placement
        best_processor = None
        best_finish_time = float('inf')
        
        for processor in hardware.network.nodes:
            # Make sure we can schedule in this superstep on this processor
            if not schedule.can_be_scheduled_in(task_name, superstep, processor):
                raise ValueError(f"Cannot schedule task {task_name} on processor {processor} in superstep {superstep.index} after split")
            
            # Calculate task duration including communication
            task_duration = self._calculate_task_duration(task_name, processor, hardware, task_graph) + \
                            self._calculate_communication_time(task_name, processor, superstep, schedule, hardware, task_graph)
            
            # Calculate when task would approximately finish
            task_finish_absolute = superstep.compute_phase_start(processor) + superstep.compute_time(processor) + task_duration
            
            if task_finish_absolute < best_finish_time:
                best_finish_time = task_finish_absolute
                best_processor = processor
                
                if self.verbose:
                    self._log(f"    Processor {processor}: finish={task_finish_absolute:.2f}")
        
        if self.verbose:
            self._log(f"    Selected processor {best_processor} with finish time {best_finish_time:.2f}")
        
        return best_processor, superstep
    
    def _get_dependency_ready_time(self, task_name: str,
                                  schedule: BSPSchedule, task_graph: nx.DiGraph) -> float:
        """Calculate the exact time when all dependencies are ready for a task.
        
        Since we don't use task duplication, each predecessor has exactly one instance.
        The ready time is the latest end time among all predecessors.
        
        Args:
            task_name: Task to check
            schedule: Current schedule
            task_graph: Task dependency graph
            
        Returns:
            Exact time when all dependencies have completed
        """
        ready_time = 0.0
        
        for pred_name in task_graph.predecessors(task_name):
            if not schedule.task_scheduled(pred_name):
                return float('inf')
            
            # Get the single instance of the predecessor
            pred_instance = schedule.get_single_instance(pred_name)
            
            # Dependency is ready exactly when the predecessor task completes
            ready_time = max(ready_time, pred_instance.end)
        
        return ready_time

    def _calculate_elimination_benefit(self, schedule: BSPSchedule, superstep_idx: int) -> float:
        """Calculate the makespan reduction from eliminating a superstep.

        Args:
            schedule: Current BSP schedule
            superstep_idx: Index of the superstep to eliminate

        Returns:
            Makespan reduction (positive value means beneficial elimination, 0 or negative means no benefit)
        """
        # Don't eliminate first or last superstep
        if superstep_idx <= 0 or superstep_idx >= len(schedule.supersteps) - 1:
            return 0.0

        # Get current makespan
        original_makespan = schedule.makespan

        # Create a copy of the schedule to test the elimination
        test_schedule = schedule.copy()

        try:
            # Eliminate the superstep and repair dependencies
            self._eliminate_and_repair_superstep(test_schedule, superstep_idx)

            # Calculate new makespan
            new_makespan = test_schedule.makespan

            # Return the benefit (reduction in makespan)
            return original_makespan - new_makespan

        except Exception:
            # If elimination fails for any reason, return no benefit
            return 0.0

    def _eliminate_and_repair_superstep(self, schedule: BSPSchedule, superstep_idx: int):
        """Eliminate a superstep and repair all subsequent supersteps by duplicating tasks as needed.

        Args:
            schedule: BSP schedule to modify
            superstep_idx: Index of the superstep to eliminate
        """
        if superstep_idx <= 0 or superstep_idx >= len(schedule.supersteps):
            raise ValueError(f"Cannot eliminate superstep {superstep_idx}")

        # Remove the target superstep and update task mappings
        eliminated_superstep = schedule.supersteps[superstep_idx]

        # Remove tasks from task mapping
        for processor, tasks in eliminated_superstep.tasks.items():
            for task in tasks:
                if task in schedule.task_mapping[task.node]:
                    schedule.task_mapping[task.node].remove(task)

        # Remove the superstep from the schedule
        schedule.supersteps.pop(superstep_idx)

        # Invalidate cached indices for all following supersteps
        for i in range(superstep_idx, len(schedule.supersteps)):
            if 'index' in schedule.supersteps[i].__dict__:
                del schedule.supersteps[i].__dict__['index']

        # Repair all subsequent supersteps (indices have shifted down after removal)
        for superstep in schedule.supersteps[superstep_idx:]:
            self._repair_superstep_dependencies(schedule, superstep)

    def _repair_superstep_dependencies(self, schedule: BSPSchedule, superstep: Superstep):
        """Repair dependencies in a superstep by duplicating missing predecessors.

        Args:
            schedule: BSP schedule being modified
            superstep: Superstep to repair
        """
        # Repair each processor's tasks
        for processor in list(superstep.tasks.keys()):
            task_list = superstep.tasks[processor]

            # Repair each task in the processor (iterate by index since list may grow)
            task_idx = 0
            while task_idx < len(task_list):
                task = task_list[task_idx]
                self._ensure_task_dependencies(schedule, task, superstep, processor, task_idx)
                task_idx += 1

    def _ensure_task_dependencies(self, schedule: BSPSchedule, task: BSPTask,
                                 superstep: Superstep, processor: str, task_position: int):
        """Ensure all dependencies of a task are available by duplicating missing predecessors.

        Args:
            schedule: BSP schedule being modified
            task: Task whose dependencies to check
            superstep: Superstep containing the task
            processor: Processor running the task
            task_position: Position of the task in processor's task list
        """
        for pred_name in schedule.task_graph.predecessors(task.node):
            if not self._predecessor_available(schedule, pred_name, task, superstep, processor, task_position):
                # Recursively duplicate the missing predecessor
                self._duplicate_task_recursively(schedule, pred_name, superstep, processor, task_position)

    def _predecessor_available(self, schedule: BSPSchedule, pred_name: str, task: Optional[BSPTask],
                              current_superstep: Superstep, processor: str, task_position: int) -> bool:
        """Check if a predecessor is available for a task.

        A predecessor is available if:
        1. It's in the same superstep on same processor before this task, OR
        2. It's in any earlier superstep

        Args:
            schedule: BSP schedule
            pred_name: Name of the predecessor task
            task: Task that needs the predecessor
            current_superstep: Superstep containing the task
            processor: Processor running the task
            task_position: Position of the task in processor's task list

        Returns:
            True if predecessor is available, False otherwise
        """
        # Check 1: Same superstep, same processor, executed before this task
        for i, task_instance in enumerate(current_superstep.tasks[processor]):
            if task_instance.node == pred_name and i < task_position:
                return True

        # Check 2: Any earlier superstep
        for earlier_superstep in schedule.supersteps[:current_superstep.index]:
            for proc_tasks in earlier_superstep.tasks.values():
                for task_instance in proc_tasks:
                    if task_instance.node == pred_name:
                        return True

        return False

    def _duplicate_task_recursively(self, schedule: BSPSchedule, task_name: str,
                                   target_superstep: Superstep, target_processor: str,
                                   before_position: int):
        """Recursively duplicate a task and all its missing dependencies.

        Args:
            schedule: BSP schedule being modified
            task_name: Name of the task to duplicate
            target_superstep: Superstep to duplicate into
            target_processor: Processor to duplicate onto
            before_position: Position to insert the task (and its dependencies)
        """
        # First, recursively ensure all predecessors of this task are available
        insertion_position = before_position

        for pred_name in schedule.task_graph.predecessors(task_name):
            if not self._predecessor_available(schedule, pred_name, None, target_superstep,
                                             target_processor, insertion_position):
                # Recursively duplicate the predecessor
                self._duplicate_task_recursively(schedule, pred_name, target_superstep,
                                               target_processor, insertion_position)
                insertion_position += 1  # Adjust position as we insert dependencies

        # Now duplicate this task at the correct position
        schedule.schedule(task_name, target_processor, target_superstep, insertion_position)

    def _optimize_superstep_elimination(self, schedule: BSPSchedule) -> int:
        """Optimize schedule by iteratively eliminating supersteps for maximum makespan reduction.

        This method implements the elimination-based optimization that:
        1. Runs in a while loop until no beneficial eliminations are found
        2. In each iteration, finds the superstep with highest makespan reduction when eliminated
        3. Performs that elimination if it provides any benefit
        4. Continues until no more beneficial eliminations are possible

        Args:
            schedule: BSP schedule to optimize

        Returns:
            Total number of supersteps eliminated
        """
        total_eliminations = 0
        iteration = 0

        if self.verbose:
            initial_makespan = schedule.makespan
            initial_supersteps = len(schedule.supersteps)
            self._log(f"\nStarting elimination-based optimization phase:")
            self._log(f"  Initial makespan: {initial_makespan:.2f}")
            self._log(f"  Initial supersteps: {initial_supersteps}")

        while True:
            iteration += 1
            best_benefit = 0.0
            best_superstep_to_eliminate = None

            # Evaluate eliminating each superstep (except first and last)
            for i in range(1, len(schedule.supersteps) - 1):
                benefit = self._calculate_elimination_benefit(schedule, i)
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_superstep_to_eliminate = i

            # If no beneficial elimination found, stop optimization
            if best_superstep_to_eliminate is None or best_benefit <= 0:
                if self.verbose:
                    self._log(f"  Iteration {iteration}: No beneficial eliminations found, stopping optimization")
                break

            # Perform the best elimination
            if self.verbose:
                self._log(f"  Iteration {iteration}: Eliminating superstep {best_superstep_to_eliminate} "
                         f"(benefit: {best_benefit:.2f})")

            self._eliminate_and_repair_superstep(schedule, best_superstep_to_eliminate)
            total_eliminations += 1

        self.stats['optimization_iterations'] = iteration - 1

        if self.verbose and total_eliminations > 0:
            final_makespan = schedule.makespan
            final_supersteps = len(schedule.supersteps)
            total_reduction = initial_makespan - final_makespan
            self._log(f"  Optimization complete:")
            self._log(f"    Final makespan: {final_makespan:.2f}")
            self._log(f"    Final supersteps: {final_supersteps}")
            self._log(f"    Total makespan reduction: {total_reduction:.2f}")
            self._log(f"    Total supersteps eliminated: {total_eliminations}")

        return total_eliminations

    def print_stats(self):
        """Print scheduling statistics."""
        print("\n" + "="*50)
        mode_str = {"heft": "HEFT", "cpop": "CPOP", "ds": "DominantSequence"}[self.priority_mode]
        print(f"Fill-in/Split BSP Scheduler Statistics ({mode_str} priority mode)")
        print("="*50)
        print(f"Tasks scheduled:         {self.stats['tasks_scheduled']}")
        print(f"Placement strategies:")
        print(f"  - Fill-in:            {self.stats['fill_in_placements']}")
        print(f"  - Append:             {self.stats['append_placements']}")
        print(f"  - Split:              {self.stats['split_placements']}")
        print(f"Superstep operations:")
        print(f"  - Created:            {self.stats['supersteps_created']}")
        print(f"  - Split:              {self.stats['supersteps_split']}")
        print(f"  - Merged:             {self.stats['supersteps_merged']}")
        if self.optimize_merging:
            print(f"Optimization phase:")
            print(f"  - Iterations:         {self.stats['optimization_iterations']}")
            print(f"  - Eliminations:       {self.stats['optimization_merges']}")
        print("="*50)