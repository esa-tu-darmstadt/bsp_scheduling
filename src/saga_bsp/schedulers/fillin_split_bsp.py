"""Fill-in/Split BSP Scheduler with three-phase placement strategy.

This scheduler uses upward rank for task prioritization and implements a three-phase
placement strategy in strict priority order:
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
from ..schedule import BSPSchedule, BSPHardware, Superstep
import networkx as nx
from saga.schedulers.cpop import upward_rank

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
    1. Computes upward rank for all tasks
    2. Processes tasks in priority order
    3. For each task, tries strategies in order:
       - Phase 1: Fill-in (find holes in existing supersteps)
       - Phase 2: Append (if would split last superstep, try appending instead)
       - Phase 3: Split (create/split superstep at dependency-ready time)
    4. Uses first successful strategy (early exit)
    """
    
    def __init__(self, verbose: bool = False, draw_after_each_step: bool = False):
        """Initialize the Fill-in/Split BSP scheduler.
        
        Args:
            verbose: Enable detailed logging
            draw_after_each_step: Enable drawing Gantt chart after each scheduling step
        """
        super().__init__()
        self.name = "FillInSplitBSP"
        self.verbose = verbose
        self.draw_after_each_step = draw_after_each_step
        
        # Initialize statistics
        self._reset_stats()
        
        if verbose:
            logger.setLevel(logging.DEBUG)
    
    def _reset_stats(self):
        """Reset scheduler statistics."""
        self.stats = {
            'tasks_scheduled': 0,
            'fill_in_placements': 0,
            'append_placements': 0,
            'split_placements': 0,
            'supersteps_created': 0,
            'supersteps_split': 0,
            'supersteps_merged': 0
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
        
        # Compute task priorities using upward rank
        rank = upward_rank(hardware.network, task_graph)
        
        # Initialize priority queue
        queue = PriorityQueue()
        
        # Add tasks with no dependencies to queue
        for task in task_graph.nodes:
            if task_graph.in_degree(task) == 0:
                queue.put(PrioritizedTask(rank[task], task))
                if self.verbose:
                    logger.debug(f"Task {task} added to initial queue with rank {rank[task]}")
        
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
                logger.debug(f"\n#{task_num}: Processing task {task_name} with rank {rank[task_name]}")
            
            # Try placement strategies in priority order
            placed = False
            
            # Phase 1: Try fill-in strategy
            processor, superstep = self._try_fill_in(task_name, schedule, hardware, task_graph)
            if processor is not None:
                schedule.schedule(task_name, processor, superstep)
                self.stats['fill_in_placements'] += 1
                placed = True
                if self.verbose:
                    logger.debug(f"  Placed using fill-in strategy on processor {processor}, "
                               f"superstep {superstep.index}")
            
            # Phase 2: Try append strategy (if dependency-ready time is in last superstep)
            if not placed and False:
                processor, superstep = self._try_append(task_name, schedule, hardware, task_graph)
                if processor is not None:
                    schedule.schedule(task_name, processor, superstep)
                    self.stats['append_placements'] += 1
                    placed = True
                    if self.verbose:
                        logger.debug(f"  Placed using append strategy on processor {processor}, "
                                   f"superstep {superstep.index}")
            
            # Phase 3: Use split strategy (always works)
            if not placed:
                processor, superstep = self._do_split(task_name, schedule, hardware, task_graph)
                schedule.schedule(task_name, processor, superstep)
                self.stats['split_placements'] += 1
                placed = True
                if self.verbose:
                    logger.debug(f"  Placed using split strategy on processor {processor}, "
                               f"superstep {superstep.index}")
            
            self.stats['tasks_scheduled'] += 1
            
            # Add newly ready successors to queue
            for successor in task_graph.successors(task_name):
                if all(schedule.task_scheduled(pred) for pred in task_graph.predecessors(successor)):
                    queue.put(PrioritizedTask(rank[successor], successor))
                    if self.verbose:
                        logger.debug(f"  Task {successor} became ready with rank {rank[successor]}")
            
            # Optionally draw Gantt chart after each step
            if self.draw_after_each_step:
                draw_bsp_gantt(schedule, title=f"After #{task_num}: Scheduling {task_name}")
            task_num += 1
        
        # Merge supersteps where possible
        merges = schedule.merge_supersteps()
        self.stats['supersteps_merged'] = merges
        if self.verbose and merges > 0:
            logger.debug(f"\nMerged {merges} superstep(s) to reduce synchronization overhead")
        
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
            pred_instances = schedule.get_all_instances(pred_name)
            
            # Find best instance to communicate from
            need_comm = True
            for instance in pred_instances:
                # If predecessor is on same processor and available, no communication needed
                if instance.proc == processor:
                    if instance.superstep.index < superstep.index or \
                       (instance.superstep.index == superstep.index and instance.superstep == superstep):
                        need_comm = False
                        break
            
            if need_comm:
                # Find earliest instance from different processor
                best_instance = None
                for instance in pred_instances:
                    if instance.superstep.index < superstep.index:
                        if best_instance is None or instance.end < best_instance.end:
                            best_instance = instance
                
                if best_instance and best_instance.proc != processor:
                    edge_weight = task_graph.edges[(pred_name, task_name)]['weight']
                    if hardware.network.has_edge(best_instance.proc, processor):
                        network_speed = hardware.network.edges[best_instance.proc, processor]['weight']
                        comm_time = max(comm_time, edge_weight / network_speed)
        
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
                proc_end_time = superstep.compute_time_of_processor(processor)
                
                # Would the task fit in a hole?
                task_finish_time = proc_end_time + task_duration
                superstep_compute_time = superstep.compute_time
                
                # Check if there's a hole (task fits without extending superstep)
                if task_finish_time <= superstep_compute_time + 0.001:  # Small tolerance
                    # Calculate absolute finish time for comparison
                    abs_finish_time = superstep.compute_phase_start + task_finish_time
                    
                    if abs_finish_time < best_finish_time:
                        best_finish_time = abs_finish_time
                        best_processor = processor
                        best_superstep = superstep
                        
                        if self.verbose:
                            hole_size = superstep_compute_time - proc_end_time
                            logger.debug(f"    Found fill-in opportunity: superstep {superstep.index}, "
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
        
        # For each processor, check if we would split at last superstep
        for processor in hardware.network.nodes:
            # Calculate dependency-ready time
            dep_ready_time = self._get_dependency_ready_time(
                task_name, processor, schedule, task_graph
            )
            
            if dep_ready_time == float('inf'):
                continue
            
            # Check which superstep this time falls into
            superstep_at_time = schedule.get_superstep_at_time(dep_ready_time)
            
            # Only try append if dependency-ready time is in last superstep
            if superstep_at_time == last_superstep:
                # Check if we can schedule in last superstep
                if schedule.can_be_scheduled_in(task_name, last_superstep, processor):
                    # Calculate task duration
                    task_duration = self._calculate_task_duration(task_name, processor, hardware, task_graph)
                    
                    # Calculate finish time
                    proc_end_time = last_superstep.compute_time_of_processor(processor)
                    task_finish_relative = proc_end_time + task_duration
                    task_finish_absolute = last_superstep.compute_phase_start + task_finish_relative
                    
                    if task_finish_absolute < best_finish_time:
                        best_finish_time = task_finish_absolute
                        best_processor = processor
                        
                        if self.verbose:
                            logger.debug(f"    Found append opportunity: last superstep, "
                                       f"proc {processor}, finish={task_finish_absolute:.2f}")
        
        if best_processor is not None:
            return best_processor, last_superstep
        
        return None, None
    
    def _do_split(self, task_name: str, schedule: BSPSchedule,
                 hardware: BSPHardware, task_graph: nx.DiGraph) -> Tuple[str, Superstep]:
        """Place task using split strategy (create/split superstep).
        
        Args:
            task_name: Task to place
            schedule: Current schedule
            hardware: Hardware configuration
            task_graph: Task dependency graph
            
        Returns:
            Tuple of (processor, superstep) for placement
        """
        best_processor = None
        best_finish_time = float('inf')
        best_split_time = None
        
        # Evaluate each processor
        for processor in hardware.network.nodes:
            # Calculate dependency-ready time
            dep_ready_time = self._get_dependency_ready_time(
                task_name, processor, schedule, task_graph
            )
            
            if dep_ready_time == float('inf'):
                continue
            
            # Create test schedule to evaluate
            test_schedule = schedule.copy()
            test_superstep = test_schedule.get_or_create_superstep_at_time(dep_ready_time)
            
            # Schedule task to get timing
            test_task = test_schedule.schedule(task_name, processor, test_superstep)
            task_finish_time = test_task.end
            
            if task_finish_time < best_finish_time:
                best_finish_time = task_finish_time
                best_processor = processor
                best_split_time = dep_ready_time
                
                if self.verbose:
                    logger.debug(f"    Split option: proc {processor}, split_time={dep_ready_time:.2f}, "
                               f"finish={task_finish_time:.2f}")
        
        if best_processor is None:
            raise ValueError(f"Could not find valid processor for task {task_name}")
        
        # Apply the split
        original_count = len(schedule.supersteps)
        superstep = schedule.get_or_create_superstep_at_time(best_split_time)
        
        # Update statistics
        if len(schedule.supersteps) > original_count:
            if superstep.index == len(schedule.supersteps) - 1:
                self.stats['supersteps_created'] += 1
                if self.verbose:
                    logger.debug(f"    Created new superstep {superstep.index}")
            else:
                self.stats['supersteps_split'] += 1
                if self.verbose:
                    logger.debug(f"    Split to create superstep {superstep.index}")
        
        return best_processor, superstep
    
    def _get_dependency_ready_time(self, task_name: str, processor: str,
                                  schedule: BSPSchedule, task_graph: nx.DiGraph) -> float:
        """Calculate when all dependencies are ready for a task on a processor.
        
        Args:
            task_name: Task to check
            processor: Target processor
            schedule: Current schedule
            task_graph: Task dependency graph
            
        Returns:
            Earliest time when all dependencies are available
        """
        ready_time = 0.0
        
        for pred_name in task_graph.predecessors(task_name):
            pred_instances = schedule.get_all_instances(pred_name)
            if not pred_instances:
                return float('inf')
            
            # Find earliest available time for this dependency
            earliest_available = float('inf')
            for instance in pred_instances:
                if instance.proc == processor:
                    # Same processor - available right after completion
                    earliest_available = min(earliest_available, instance.end)
                else:
                    # Different processor - available after superstep ends
                    earliest_available = min(earliest_available, instance.superstep.end_time)
            
            ready_time = max(ready_time, earliest_available)
        
        return ready_time
    
    def print_stats(self):
        """Print scheduling statistics."""
        print("\n" + "="*50)
        print("Fill-in/Split BSP Scheduler Statistics")
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
        print("="*50)