"""List-based BSP Scheduler with advanced placement strategies.

This scheduler uses upward rank for task prioritization and evaluates two
placement strategies (append to existing superstep or split/create superstep) 
for each task to minimize makespan.
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


class ListBSPScheduler(BSPScheduler):
    """List-based BSP scheduler with advanced placement strategies.
    
    This scheduler:
    1. Computes upward rank for all tasks
    2. Processes tasks in priority order
    3. For each task, evaluates two strategies:
       - Append to existing superstep (try all valid supersteps)
       - Split/create superstep at dependency-ready time
    4. Chooses the strategy that minimizes makespan
    """
    
    def __init__(self, verbose: bool = False, draw_after_each_step: bool = False):
        """Initialize the List BSP scheduler.
        
        Args:
            verbose: Enable detailed logging
            draw_after_each_step: Enable drawing Gantt chart after each scheduling step
        """
        super().__init__()
        self.name = "ListBSP"
        self.verbose = verbose
        self.draw_after_each_step = draw_after_each_step
        
        # Initialize statistics
        self.stats = {
            'tasks_scheduled': 0,
            'append_evaluations': 0,
            'split_evaluations': 0,
            'append_chosen': 0,
            'split_chosen': 0,
            'supersteps_created': 0,
            'supersteps_split': 0,
            'supersteps_merged': 0,
            'total_evaluations': 0
        }

        if verbose:
            logger.setLevel(logging.DEBUG)
    
    def schedule(self, hardware: BSPHardware, task_graph: nx.DiGraph) -> BSPSchedule:
        """Schedule tasks using list scheduling with advanced placement strategies.
        
        Args:
            hardware: BSP hardware configuration
            task_graph: Task dependency graph
            
        Returns:
            Optimized BSP schedule
        """
        # Reset statistics for this scheduling run
        self.stats = {
            'tasks_scheduled': 0,
            'append_evaluations': 0,
            'split_evaluations': 0,
            'append_chosen': 0,
            'split_chosen': 0,
            'supersteps_created': 0,
            'supersteps_split': 0,
            'supersteps_merged': 0,
            'total_evaluations': 0
        }
        
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
        
        # Create initial superstep if empty
        if not schedule.supersteps:
            schedule.add_superstep()
            self.stats['supersteps_created'] = 1
        
        # Process tasks in priority order
        i = 0
        while not queue.empty():
            task_item = queue.get()
            task_name = task_item.task
            
            if self.verbose:
                logger.debug(f"\nProcessing task {task_name} with rank {rank[task_name]}")
            
            # Find best placement strategy
            best_processor, best_superstep, best_makespan = self._find_best_placement(
                task_name, schedule, hardware, task_graph
            )
            
            if self.verbose:
                logger.debug(f"Best placement: processor={best_processor}, "
                           f"superstep_idx={best_superstep.index}, makespan={best_makespan:.2f}")
            
            # Schedule the task in the chosen superstep
            schedule.schedule(task_name, best_processor, best_superstep)
            self.stats['tasks_scheduled'] += 1
            
            # Add newly ready successors to queue
            for successor in task_graph.successors(task_name):
                if all(schedule.task_scheduled(pred) for pred in task_graph.predecessors(successor)):
                    queue.put(PrioritizedTask(rank[successor], successor))
                    if self.verbose:
                        logger.debug(f"Task {successor} became ready with rank {rank[successor]}")
            
            # Optionally draw Gantt chart after each step
            if self.draw_after_each_step:
                draw_bsp_gantt(schedule, title=f"#{i} After scheduling {task_name}")
            i += 1

        # Merge supersteps where possible to reduce synchronization overhead
        merges = schedule.merge_supersteps()
        self.stats['supersteps_merged'] = merges
        if self.verbose and merges > 0:
            logger.debug(f"\nMerged {merges} superstep(s) to reduce synchronization overhead")
        
        # Validate the final schedule
        schedule.assert_valid()
        return schedule
    
    def _find_best_placement(self, task_name: str, schedule: BSPSchedule, 
                            hardware: BSPHardware, task_graph: nx.DiGraph) -> Tuple[str, Superstep, float]:
        """Find the best placement for a task.
        
        Evaluates both append and split strategies for all processors.
        
        Args:
            task_name: Task to schedule
            schedule: Current BSP schedule
            hardware: BSP hardware configuration
            task_graph: Task dependency graph
            
        Returns:
            Tuple of (best_processor, best_superstep, best_makespan)
        """
        best_makespan = float('inf')
        best_processor = None
        best_superstep = None
        best_split_time = None  # Track split time for deferred execution
        
        # Try each processor
        for processor in hardware.network.nodes:
            if self.verbose:
                logger.debug(f"  Evaluating processor {processor}")
            
            # Strategy 1: Append to existing superstep
            for superstep in schedule.supersteps:
                if schedule.can_be_scheduled_in(task_name, superstep, processor):
                    makespan = self._evaluate_append_strategy(
                        task_name, processor, superstep, schedule, hardware, task_graph
                    )
                    self.stats['append_evaluations'] += 1
                    self.stats['total_evaluations'] += 1
                    
                    if self.verbose:
                        logger.debug(f"    append on proc {processor} to superstep {superstep.index} would yield makespan={makespan:.2f}")
                    
                    if makespan < best_makespan:
                        best_makespan = makespan
                        best_processor = processor
                        best_superstep = superstep
                        best_split_time = None  # Not a split strategy
                        
                        if self.verbose:
                            logger.debug(f"    -- 🎉 New best with append!")
            
            # Strategy 2: Split/create superstep at dependency-ready time
            split_time, split_makespan = self._evaluate_split_strategy(
                task_name, processor, schedule, hardware, task_graph
            )
            if split_time is not None:
                self.stats['split_evaluations'] += 1
                self.stats['total_evaluations'] += 1
            
            if self.verbose:
                logger.debug(f"    split/create on proc {processor} at time {split_time:.2f} would yield makespan={split_makespan:.2f}")
            
            if split_time is not None and split_makespan < best_makespan:
                best_makespan = split_makespan
                best_processor = processor
                best_superstep = None  # Will be created when we apply the strategy
                best_split_time = split_time
                
                if self.verbose:
                    logger.debug(f"    -- 🎉 New best with split!")
        
        # If best strategy is split, create the superstep now
        if best_split_time is not None:
            original_count = len(schedule.supersteps)
            best_superstep = schedule.get_or_create_superstep_at_time(best_split_time)
            # Track if we created a new superstep or split an existing one
            if len(schedule.supersteps) > original_count:
                if best_superstep.index == len(schedule.supersteps) - 1:
                    self.stats['supersteps_created'] += 1
                else:
                    self.stats['supersteps_split'] += 1
            self.stats['split_chosen'] += 1
        else:
            self.stats['append_chosen'] += 1
        
        return best_processor, best_superstep, best_makespan
    
    def _evaluate_append_strategy(self, task_name: str, processor: str, superstep: Superstep,
                                 schedule: BSPSchedule, hardware: BSPHardware, 
                                 task_graph: nx.DiGraph) -> float:
        """Evaluate makespan if task is appended to existing superstep.
        
        Args:
            task_name: Task to schedule
            processor: Processor to use
            superstep: Superstep to append to
            schedule: Current schedule
            hardware: Hardware configuration
            task_graph: Task dependency graph
            
        Returns:
            Estimated makespan after scheduling
        """
        # Create a copy of the schedule to test
        test_schedule = schedule.copy()
        
        try:
            # Find the corresponding superstep in the test schedule
            test_superstep = test_schedule.supersteps[superstep.index]
            
            # Schedule the task
            test_schedule.schedule(task_name, processor, test_superstep)
            
            # Return the makespan
            return test_schedule.makespan
        except Exception as e:
            if self.verbose:
                logger.debug(f"    Append failed: {e}")
            return float('inf')
    
    def _evaluate_split_strategy(self, task_name: str, processor: str,
                                schedule: BSPSchedule, hardware: BSPHardware,
                                task_graph: nx.DiGraph) -> Tuple[Optional[float], float]:
        """Evaluate splitting/creating a superstep for the task.
        
        Finds when all dependencies are available and evaluates creating a superstep at that time.
        
        Args:
            task_name: Task to schedule
            processor: Processor to use
            schedule: Current schedule
            hardware: Hardware configuration
            task_graph: Task dependency graph
            
        Returns:
            Tuple of (split_time, makespan) or (None, inf) if not feasible
        """
        # Find when all dependencies are available
        dependency_ready_time = 0.0
        
        for pred_name in task_graph.predecessors(task_name):
            pred_instances = schedule.get_all_instances(pred_name)
            if not pred_instances:
                # Predecessor not scheduled yet
                return None, float('inf')
            
            # Find earliest instance of this predecessor
            earliest_instance = min(pred_instances, key=lambda x: x.end)
            dependency_ready_time = max(dependency_ready_time, earliest_instance.end)
        
        # Create a test schedule to evaluate the new makespan
        test_schedule = schedule.copy()
        
        # Use the helper method to get or create superstep at dependency_ready_time
        new_superstep = test_schedule.get_or_create_superstep_at_time(dependency_ready_time)
        
        # Make sure we can schedule the task here - this should always be true
        if not test_schedule.can_be_scheduled_in(task_name, new_superstep, processor):
            raise ValueError(f"Cannot schedule task {task_name} in new superstep {new_superstep.index} at time {dependency_ready_time}")

        # Schedule the task
        test_schedule.schedule(task_name, processor, new_superstep)
        
        # Return the split time and makespan (don't modify original schedule)
        return dependency_ready_time, test_schedule.makespan
    
    def print_stats(self):
        """Print scheduling statistics."""
        print("\n" + "="*50)
        print("ListBSP Scheduler Statistics")
        print("="*50)
        print(f"Tasks scheduled:        {self.stats['tasks_scheduled']}")
        print(f"Total evaluations:      {self.stats['total_evaluations']}")
        print(f"  - Append evaluations: {self.stats['append_evaluations']}")
        print(f"  - Split evaluations:  {self.stats['split_evaluations']}")
        print(f"Strategies chosen:")
        print(f"  - Append chosen:      {self.stats['append_chosen']}")
        print(f"  - Split chosen:       {self.stats['split_chosen']}")
        print(f"Superstep operations:")
        print(f"  - Created:            {self.stats['supersteps_created']}")
        print(f"  - Split:              {self.stats['supersteps_split']}")
        print(f"  - Merged:             {self.stats['supersteps_merged']}")
        print("="*50)
