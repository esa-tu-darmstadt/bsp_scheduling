from collections.abc import MutableMapping
from dataclasses import dataclass
from functools import cached_property, cache
import sys
from typing import Dict, Hashable, Iterator, List, Literal, Set, Optional, Tuple
from collections import defaultdict
import networkx as nx
from saga.scheduler import Task
import copy
import random
import statistics

@dataclass
class BSPHardware:
    """Encapsulates BSP hardware properties: network topology and synchronization costs."""
    
    network: nx.Graph
    sync_time: float
    
    def __post_init__(self):
        """Validate hardware configuration."""
        if self.sync_time < 0:
            raise ValueError("Sync time must be non-negative")
        if not isinstance(self.network, nx.Graph):
            raise TypeError("Network must be a NetworkX Graph")
        if len(self.network.nodes) == 0:
            raise ValueError("Network must have at least one node")
    
    @cached_property
    def avg_computation_speed(self) -> float:
        """Average computation speed across all processors (from node weights)"""
        node_weights = [self.network.nodes[node]['weight'] for node in self.network.nodes()]
        return statistics.mean(node_weights) if node_weights else 0
    
    @cached_property 
    def avg_communication_speed(self) -> float:
        """Average communication speed across all network links (edge weights)"""
        edge_weights = [self.network.edges[edge]['weight'] for edge in self.network.edges()]
        
        return statistics.mean(edge_weights) if edge_weights else 0


class BSPTask:
    """Extended task for BSP scheduling"""

    def __init__(self, node: str, proc: str, superstep: "Superstep"):
        self.node = node  # The node where the task is executed
        self.proc = proc  # The processor on which the task is executed
        self.superstep = superstep  # The superstep this task belongs to

    def invalidate_timings(self):
        """Invalidate all cached timings for this task"""
        if 'rel_start' in self.__dict__:
            del self.__dict__['rel_start']

    @property
    def schedule(self) -> "BSPSchedule":
        """BSP schedule this task belongs to"""
        return self.superstep.schedule

    @property
    def task_graph(self) -> nx.DiGraph:
        """Task graph for this task"""
        return self.superstep.task_graph

    @property
    def network(self) -> nx.Graph:
        """Network graph for this task"""
        return self.superstep.hardware.network

    @cached_property
    def speed(self) -> float:
        """Speed of the processor executing this task"""
        return self.network.nodes[self.proc]["weight"]

    @cached_property
    def cost(self) -> float:
        """Cost of executing this task"""
        return self.task_graph.nodes[self.node]["weight"]

    @cached_property
    def duration(self) -> float:
        """Duration of the task execution on the processor"""
        return self.cost / self.speed

    @property
    def index(self) -> int:
        """Index of this task in the processor's task list for the superstep"""
        return self.superstep.tasks[self.proc].index(self)

    @cached_property
    def rel_start(self) -> float:
        """Start time relative to the computation phase of the superstep. 
        Equals the end time of the previous task on the same processor."""
        index_ = self.index
        if index_ == 0:
            return 0.0
        # We could simply do the following, but recursion in python is very slow
        # return self.superstep.tasks[self.proc][self.index - 1].rel_end
        
        # Instead, lets calculate it iteratively
        sum_ = 0.0
        for task in self.superstep.tasks[self.proc][:index_]:
            sum_ += task.duration
        return sum_

    @property
    def rel_end(self) -> float:
        """End time relative to the computation phase of the superstep"""
        return self.rel_start + self.duration

    @property
    def start(self) -> float:
        """Absolute start time of the task"""
        return self.superstep.compute_phase_start + self.rel_start

    @property
    def end(self) -> float:
        """Absolute end time of the task"""
        return self.superstep.compute_phase_start + self.rel_end

    def __repr__(self):
        return f"BSPTask(node={self.node}, proc={self.proc}, rel_start={self.rel_start}, superstep_index={self.superstep.index})"


class Superstep:
    """Represents a single BSP superstep. The execution if a superstep consists of three phases in the given order:
    1. Synchronization
    2. Exchange
    3. Computation
    This class manages the scheduling of tasks across processors for a single superstep.
    """

    def __init__(self, schedule: "BSPSchedule"):
        self.schedule = schedule  # The BSP schedule this superstep belongs to
        self.tasks: Dict[str, List[BSPTask]] = defaultdict(
            list
        )  # Tasks scheduled on each processor

    def schedule_task(self, task: str, processor: str, position: Optional[int] = None) -> BSPTask:
        """Schedule a task at a specific position on a processor in this superstep.
        
        Args:
            task: Task name to schedule
            processor: Processor to schedule on
            position: Optional position in the processor's task list. If None, appends to end.
        """
        bsp_task = BSPTask(task, processor, self)
        
        if position is None:
            self.tasks[processor].append(bsp_task)
        else:
            self.tasks[processor].insert(position, bsp_task)
            
        # Update the task mapping - add to list of instances (defaultdict handles creation)
        self.schedule.task_mapping[task].append(bsp_task)
        
        # Invalidate timings of all tasks on the processor after this one in the superstep
        for i in range(bsp_task.index + 1, len(self.tasks[processor])):
            self.tasks[processor][i].invalidate_timings()
            
        # Invalidate superstep timings since we added a new task
        self.invalidate_timings()
        return bsp_task

    def invalidate_timings(self):
        """Invalidate all cached timings for this superstep"""
        self.compute_time_of_processor.cache_clear()
        self.exchange_time_of_processor.cache_clear()
        # Clear cached properties safely
        if 'compute_time' in self.__dict__:
            del self.__dict__['compute_time']
        if 'sync_time' in self.__dict__:
            del self.__dict__['sync_time']
        if 'exchange_time' in self.__dict__:
            del self.__dict__['exchange_time']

    @cached_property
    def index(self) -> int:
        """Index of this superstep in the schedule"""
        return self.schedule.supersteps.index(self)

    @property
    def hardware(self) -> BSPHardware:
        """BSP hardware for this superstep"""
        return self.schedule.hardware
    
    @property
    def network(self) -> nx.Graph:
        """Network graph for this superstep"""
        return self.schedule.hardware.network

    @property
    def task_graph(self) -> nx.DiGraph:
        """Task graph for this superstep"""
        return self.schedule.task_graph

    @property
    def total_time(self) -> float:
        """Total time of this superstep, including computation, synchronization, and exchange times."""
        return self.compute_time + self.sync_time + self.exchange_time

    @property
    def start_time(self) -> float:
        """Start time of this superstep, which is the end time of the previous superstep or 0 for the first one."""
        index_ = self.index
        if index_ == 0:
            return 0.0
        # We could simply do the following, but recursion in python is very slow
        # return self.schedule.supersteps[index_ - 1].end_time
        
        # Instead, lets calculate it iteratively
        return sum(
            superstep.total_time for superstep in self.schedule.supersteps[:index_]
        )

    @property
    def end_time(self) -> float:
        """End time of this superstep, which is the start time plus total time."""
        return self.start_time + self.total_time

    @cached_property
    def compute_time(self) -> float:
        """Total computation time. Calculated as the maximum of the compute times of all processors."""
        if not self.tasks:
            return 0.0
        return max(self.compute_time_of_processor(proc) for proc in self.tasks)

    @property
    def sync_time(self) -> float:
        """Total synchronization time."""
        # No sync time for the first superstep
        if self.index == 0:
            return 0.0
        # # If no tasks are scheduled, no sync time is needed (empty superstep)
        # if not self.tasks:
        #     return 0.0
        # # No sync is required if nothing to communicate
        # if self.exchange_time == 0.0:
        #     return 0.0
        return self.hardware.sync_time

    @cached_property
    def exchange_time(self) -> float:
        """Total exchange time. Calculated as the maximum of the exchange times of all processors."""
        if not self.tasks:
            return 0.0
        return max(self.exchange_time_of_processor(proc) for proc in self.tasks)

    @cache
    def compute_time_of_processor(self, processor: str) -> float:
        return (self.tasks[processor][-1].rel_end
                if self.tasks[processor] else 0.0)

    @cache
    def exchange_time_of_processor(self, processor: str) -> float:
        """Calculate exchange time for a processor in this superstep.
        
        Exchange time is the time needed to receive data from tasks in previous supersteps
        that are executed on different processors and whose outputs are needed by tasks
        in this superstep on this processor.
        
        Important: Data that was already communicated to this processor in an earlier
        superstep does not need to be communicated again.
        """
        # Track which data needs to be communicated to this processor
        # Key: (source_task, source_proc), Value: communication cost
        data_to_communicate = {}
        
        # Get all tasks on this processor in this superstep
        tasks_on_processor = self.tasks.get(processor, [])
        
        for task in tasks_on_processor:
            # For each task, find dependencies that require communication
            for pred_task_name in self.task_graph.predecessors(task.node):
                pred_instances = self.schedule.task_mapping[pred_task_name]
                
                # First check: Is the predecessor already on this processor in this or a previous superstep?
                local_instance = None
                for pred_instance in pred_instances:
                    if pred_instance.proc == processor and pred_instance.superstep.index <= self.index:
                        local_instance = pred_instance
                        break
                
                # If we have a local instance, no communication needed
                if local_instance:
                    continue
                
                # Otherwise, find first instance from a previous superstep for communication
                comm_instance = None
                for pred_instance in pred_instances:
                    if pred_instance.superstep.index < self.index:
                        comm_instance = pred_instance
                        break
                
                if comm_instance:
                    # Check if this data was already communicated to this processor in a previous superstep
                    data_already_available = False
                    
                    # Check all previous supersteps to see if this data was already communicated
                    for prev_superstep_idx in range(1, self.index):
                        prev_superstep = self.schedule.supersteps[prev_superstep_idx]
                        prev_tasks = prev_superstep.tasks.get(processor, [])
                        
                        # Check if any task in the previous superstep on this processor needed this data
                        for prev_task in prev_tasks:
                            if pred_task_name in self.task_graph.predecessors(prev_task.node):
                                # This processor already received this data in a previous superstep
                                data_already_available = True
                                break
                        
                        if data_already_available:
                            break
                    
                    # Only communicate if data is not already available
                    if not data_already_available:
                        # Calculate communication cost from this instance
                        comm_weight = self.task_graph.edges[(pred_task_name, task.node)]['weight']
                        
                        # Get network connection speed between processors
                        if self.network.has_edge(comm_instance.proc, processor):
                            network_speed = self.network.edges[comm_instance.proc, processor]['weight']
                        else:
                            raise ValueError(
                                f"Task wants to communicate between processors {comm_instance.proc} and {processor}, "
                                "but no connection exists in the network."
                            )
                        
                        # Exchange time = communication weight / network speed
                        exchange_time = comm_weight / network_speed
                        
                        # Track the maximum communication time for this data source
                        key = (pred_task_name, comm_instance.proc)
                        data_to_communicate[key] = max(data_to_communicate.get(key, 0), exchange_time)
        
        # Return the total exchange time for this processor
        return sum(data_to_communicate.values()) if data_to_communicate else 0.0

    @property
    def compute_phase_start(self) -> float:
        """Absolute start time of the computation phase of this superstep"""
        return self.start_time + self.sync_time + self.exchange_time


class BSPSchedule:
    """Complete BSP schedule representation"""

    def __init__(self, hardware: BSPHardware, task_graph: nx.DiGraph):
        self.hardware = hardware  # The BSP hardware configuration
        self.task_graph = task_graph  # The task graph for this schedule
        self.supersteps: List[Superstep] = []  # List of supersteps in this schedule
        self.task_mapping: defaultdict[str, List[BSPTask]] = defaultdict(
            list
        )  # Mapping of task names to lists of BSPTask objects (supports duplication)

    def add_superstep(self) -> Superstep:
        """Add a new superstep to the schedule and return it."""
        superstep = Superstep(self)
        self.supersteps.append(superstep)
        return superstep

    def __getitem__(self, task_name: str) -> List[BSPTask]:
        """Returns the list of BSPTask instances for a given task name.
        Returns empty list if task not found (due to defaultdict behavior).
        """
        return self.task_mapping[task_name]
    
    def get_primary_instance(self, task_name: str) -> Optional[BSPTask]:
        """Returns the first (primary) instance of a task, or None if not found.
        Useful for backward compatibility with single-instance assumptions.
        """
        instances = self.task_mapping[task_name]
        return instances[0] if instances else None
    
    def get_all_instances(self, task_name: str) -> List[BSPTask]:
        """Returns all instances of a task. Same as __getitem__ but more explicit."""
        return self.task_mapping[task_name]
    
    def get_single_instance(self, task_name: str) -> BSPTask:
        """Returns the single instance of a task.
        Used by schedulers that assume no task duplication.
        
        Args:
            task_name: Name of the task
            
        Returns:
            The single BSPTask instance
            
        Raises:
            ValueError: If task has no instances or multiple instances (duplication)
        """
        instances = self.task_mapping[task_name]
        if len(instances) == 0:
            raise ValueError(f"Task {task_name} has not been scheduled")
        if len(instances) > 1:
            raise ValueError(f"Task {task_name} has {len(instances)} instances (duplication not supported)")
        return instances[0]
    
    def task_scheduled(self, task_name: str) -> bool:
        """Check if a task has been scheduled in this schedule."""
        return task_name in self.task_mapping and len(self.task_mapping[task_name]) > 0
    
    def schedule(self, task: str, processor: str, superstep: Superstep, position: Optional[int] = None) -> BSPTask:
        """Schedule a task on a specific processor in a superstep."""
        bsp_task = superstep.schedule_task(task, processor, position)
        return bsp_task
    
    def unschedule(self, task: BSPTask) -> None:
        """Remove a specific task instance from the schedule."""
        if task.node in self.task_mapping and task in self.task_mapping[task.node]:
            self.task_mapping[task.node].remove(task)
            task.superstep.tasks[task.proc].remove(task)
            task.superstep.invalidate_timings()
        else:
            raise KeyError(f"Task instance {task} not found in schedule.")
    
    def get_superstep_at_time(self, time: float) -> Optional[Superstep]:
        """Get the superstep that contains the given absolute time.
        
        Args:
            time: The absolute time to check
            
        Returns:
            The superstep containing the given time, or None if the time is outside the schedule
        """
        if time < 0:
            return None
        
        for superstep in self.supersteps:
            if superstep.start_time <= time < superstep.end_time:
                return superstep
        
        # Check if time is exactly at the end of the last superstep
        if self.supersteps and time == self.supersteps[-1].end_time:
            return self.supersteps[-1]
        
        return None
    
    def get_or_create_superstep_at_time(self, time: float) -> Superstep:
        """Get or create a superstep that can accommodate a task starting at the given time.
        
        This method will:
        - If time is within an existing superstep, split it and return the new superstep
        - If time is at a superstep boundary, return the superstep that starts at that time
        - If time is beyond the schedule, create and return a new superstep at the end
        
        Args:
            time: The absolute time when a task needs to start
            
        Returns:
            A superstep where a task starting at 'time' can be scheduled
        """
        # Handle negative time
        if time < 0:
            time = 0

        # Find which superstep contains this time
        containing_superstep = self.get_superstep_at_time(time)
        
        if not containing_superstep:
            # Time is beyond current schedule - add new superstep at end
            return self.add_superstep()
        
        # Check if we're at the start of a superstep
        if abs(time - containing_superstep.start_time) < sys.float_info.epsilon:
            return containing_superstep
        
        # Check if we're at the end of a superstep (boundary between supersteps)
        if abs(time - containing_superstep.end_time) < sys.float_info.epsilon:
            # Check if there's a next superstep
            idx = containing_superstep.index
            if idx + 1 < len(self.supersteps):
                return self.supersteps[idx + 1]
            else:
                # At the end of last superstep, create new one
                return self.add_superstep()
        
        # We're in the middle of a superstep - try to split it
        if containing_superstep.start_time < time < containing_superstep.end_time:
            return self.split_superstep(containing_superstep, time)
        
        # Shouldn't reach here, but just in case
        return self.add_superstep()
    
    def split_superstep(self, superstep: Superstep, split_time: float, threshold_point: Literal["start", "midpoint"]) -> Superstep:
        """Split a superstep at a given absolute time, moving tasks that start after this time to a new superstep.
        Always returns a new superstep (which may be empty if no tasks are moved).

        Tasks that start at or after the split_time (in absolute time) will be moved to the new superstep.
        
        Args:
            superstep: The superstep to split
            split_time: The absolute time at which to split the superstep
            threshold_point: The strategy for deciding whether a task instance belongs to the old or new superstep.
            For "start", tasks that start at or after split_time will be moved to the new superstep.
            For "midpoint", tasks whose midpoint (start + duration/2) is at or after split_time will be moved.

        Returns:
            The new superstep (may be empty if no tasks were moved)

        Raises:
            ValueError: If the superstep is not part of this schedule or split_time is invalid
        """
        if superstep not in self.supersteps:
            raise ValueError("Superstep is not part of this schedule")
        
        # Check that split_time is within the superstep's time range
        if split_time <= superstep.start_time or split_time >= superstep.end_time:
            raise ValueError(f"Split time {split_time} must be within superstep time range [{superstep.start_time}, {superstep.end_time}]")
        
        # Collect task instances to move
        task_instances_to_move = []
        
        # Identify specific task instances that should be moved to the new superstep
        for processor, task_list in list(superstep.tasks.items()):
            for task_instance in list(task_list):  # Create a copy to iterate safely
                if threshold_point == "start":
                    # Check if this specific task instance starts at or after the split time (absolute)
                    if task_instance.start >= split_time:
                        task_instances_to_move.append(task_instance)
                elif threshold_point == "midpoint":
                    # Check if this specific task instance's midpoint is at or after the split time
                    if task_instance.start + task_instance.duration / 2 >= split_time:
                        task_instances_to_move.append(task_instance)

        # Always create a new superstep (even if empty)
        new_superstep = Superstep(self)
        
        # Find the index where to insert the new superstep (right after the current one)
        current_index = self.supersteps.index(superstep)
        self.supersteps.insert(current_index + 1, new_superstep)
    
        
        # Move each task instance to the new superstep
        for task_instance in task_instances_to_move:
            task_name = task_instance.node
            processor = task_instance.proc
            
            # Unschedule this specific instance from current superstep
            self.unschedule(task_instance)
            
            # Schedule a new instance in the new superstep
            self.schedule(task_name, processor, new_superstep)
        
        # Invalidate cached indices for all following supersteps
        for i in range(current_index + 1, len(self.supersteps)):
            if 'index' in self.supersteps[i].__dict__:
                del self.supersteps[i].__dict__['index']
        
        return new_superstep
    
    def can_be_scheduled_in(self, task_name: str, superstep: Superstep, processor: int) -> bool:
        """Check if a task can be scheduled in the given superstep on the processor.

        A task can be scheduled in the current superstep if at least one instance of all predecessors is either:
        1. Scheduled in the same superstep on the same processor (can pass data directly)
        2. Scheduled in previous supersteps (data is available after sync)
        
        This function only takes predecessors into account, not successors!
        Args:
            task_name: Name of the task to check
            superstep: The superstep to potentially schedule in
            processor: The processor index to schedule on
            
        Returns:
            True if the task can be scheduled, False otherwise
        """
        for pred_name in self.task_graph.predecessors(task_name):
            if not self.task_scheduled(pred_name):
                return False
            
            # Check if at least one instance of the predecessor satisfies the requirements
            pred_instances = self.get_all_instances(pred_name)
            valid_instance_found = False
            
            for pred_instance in pred_instances:
                pred_superstep_idx = pred_instance.superstep.index
                current_superstep_idx = superstep.index
                
                # Valid if predecessor is in an earlier superstep (data is available)
                if pred_superstep_idx < current_superstep_idx:
                    valid_instance_found = True
                    break
                # Valid if in same superstep on same processor (can pass data directly)
                elif pred_superstep_idx == current_superstep_idx and pred_instance.proc == processor:
                    valid_instance_found = True
                    break
            
            if not valid_instance_found:
                return False
        
        return True
        

    @property
    def makespan(self) -> float:
        """Total makespan of the schedule, which is the end time of the last superstep."""
        return self.supersteps[-1].end_time if self.supersteps else 0.0

    @property
    def num_supersteps(self) -> int:
        return len(self.supersteps)
    
    def copy(self) -> "BSPSchedule":
        """Create a deep copy of the schedule while preserving cached timing values.
        
        This method creates a structurally identical copy of the schedule while
        preserving expensive cached computations like rel_start, compute_time, etc.
        This significantly improves performance when schedulers frequently copy schedules.
        """
        # Create new schedule instance
        new_schedule = BSPSchedule(self.hardware, self.task_graph)
        
        # Copy supersteps while preserving caches
        for superstep in self.supersteps:
            new_superstep = new_schedule.add_superstep()
            
            # Copy tasks and build structure
            for processor, tasks in superstep.tasks.items():
                for task in tasks:
                    # Note: schedule() creates a new BSPTask, we'll copy cache after
                    new_task = new_schedule.schedule(task.node, processor, new_superstep)
                    
                    # Preserve rel_start cache from original task if it exists
                    if hasattr(task, '__dict__') and 'rel_start' in task.__dict__:
                        new_task.__dict__['rel_start'] = task.__dict__['rel_start']
            
            # Preserve superstep-level cached properties if they exist
            if hasattr(superstep, '__dict__'):
                # Preserve compute_time cache
                if 'compute_time' in superstep.__dict__:
                    new_superstep.__dict__['compute_time'] = superstep.__dict__['compute_time']
                # Preserve exchange_time cache  
                if 'exchange_time' in superstep.__dict__:
                    new_superstep.__dict__['exchange_time'] = superstep.__dict__['exchange_time']
                # Preserve sync_time cache (though it's a property, it might be cached in __dict__)
                if 'sync_time' in superstep.__dict__:
                    new_superstep.__dict__['sync_time'] = superstep.__dict__['sync_time']
        
        
        return new_schedule
    
    def is_valid(self) -> Tuple[bool, List[str]]:
        """Validate the BSP schedule and return (is_valid, list_of_errors)"""
        errors = []
        
        # Check that all tasks from task graph are scheduled
        scheduled_tasks = set(self.task_mapping.keys())
        graph_tasks = set(self.task_graph.nodes())
        
        if scheduled_tasks != graph_tasks:
            missing = graph_tasks - scheduled_tasks
            extra = scheduled_tasks - graph_tasks
            if missing:
                errors.append(f"Tasks missing from schedule: {missing}")
            if extra:
                errors.append(f"Extra tasks in schedule not in graph: {extra}")
        
        # Check precedence constraints for all task instances
        for task_name in self.task_mapping:
            task_instances = self.task_mapping[task_name]
            
            for task_instance in task_instances:
                current_superstep = task_instance.superstep.index
                
                # All predecessors must be in earlier supersteps or same superstep but with proper ordering
                for pred_name in self.task_graph.predecessors(task_name):
                    if pred_name in self.task_mapping:
                        pred_instances = self.task_mapping[pred_name]
                        
                        # Check if there's at least one valid predecessor instance
                        valid_predecessor = False
                        for pred_instance in pred_instances:
                            pred_superstep = pred_instance.superstep.index
                            
                            if pred_superstep < current_superstep:
                                # Predecessor in earlier superstep - always valid
                                valid_predecessor = True
                                break
                            elif pred_superstep == current_superstep and pred_instance.proc == task_instance.proc:
                                # Same superstep and same processor - check execution order
                                if pred_instance.index < task_instance.index:
                                    valid_predecessor = True
                                    break
                            # Note: Different processors in same superstep is INVALID in BSP
                            # as tasks in the same superstep must be independent
                        
                        if not valid_predecessor:
                            errors.append(f"Precedence violation: No valid predecessor {pred_name} found for {task_name} instance in superstep {current_superstep} on processor {task_instance.proc}")
        
        # Check that all processors exist in hardware network
        for superstep in self.supersteps:
            for processor in superstep.tasks:
                if processor not in self.hardware.network.nodes:
                    errors.append(f"Processor {processor} not found in hardware network")
        
        # Check that task mapping is consistent with superstep assignments
        for task_name, task_instances in self.task_mapping.items():
            for task_instance in task_instances:
                if task_instance not in task_instance.superstep.tasks[task_instance.proc]:
                    errors.append(f"Task mapping inconsistency: {task_name} instance not found in superstep {task_instance.superstep.index} processor {task_instance.proc}")
        
        # Check for duplicate task assignments
        all_scheduled_tasks = []
        for superstep in self.supersteps:
            for processor, tasks in superstep.tasks.items():
                for task in tasks:
                    all_scheduled_tasks.append((task.node, superstep.index, processor))
        
        seen = set()
        for task_info in all_scheduled_tasks:
            if task_info in seen:
                errors.append(f"Duplicate task assignment: {task_info}")
            seen.add(task_info)
        
        return len(errors) == 0, errors


    def assert_valid(self):
        """Assert that the schedule is valid. Raises ValueError if not."""
        is_valid, errors = self.is_valid()
        if not is_valid:
            raise ValueError(f"Invalid BSP schedule: {errors}")
    
    def can_merge_supersteps(self, first_idx: int, second_idx: int) -> bool:
        """Check if two consecutive supersteps can be merged without violating dependencies.
        
        Two supersteps can be merged if no task in the second superstep depends on 
        any task in the first superstep (except for same-processor dependencies).
        
        Args:
            first_idx: Index of the first superstep
            second_idx: Index of the second superstep (should be first_idx + 1)
            
        Returns:
            True if the supersteps can be merged, False otherwise
        """
        if second_idx != first_idx + 1:
            return False
        
        if first_idx >= len(self.supersteps) or second_idx >= len(self.supersteps):
            return False
            
        first_superstep = self.supersteps[first_idx]
        second_superstep = self.supersteps[second_idx]
        
        # Check all tasks in the second superstep
        for proc, tasks in second_superstep.tasks.items():
            for task in tasks:
                # Check dependencies of this task
                for pred_name in self.task_graph.predecessors(task.node):
                    # Find instances of the predecessor
                    pred_instances = self.get_all_instances(pred_name)
                    for pred_instance in pred_instances:
                        # If predecessor is in the first superstep
                        if pred_instance.superstep == first_superstep:
                            # Only allow merge if they're on the same processor
                            if pred_instance.proc != task.proc:
                                return False
        
        return True
    
    def merge_supersteps(self, superstep_idx: Optional[int] = None) -> int:
        """Merge consecutive supersteps when possible to reduce synchronization overhead.
        
        Args:
            superstep_idx: If provided, only try to merge this superstep with the next one.
                          If None, try to merge all consecutive supersteps.
                          
        Returns:
            Number of merges performed
        """
        merges_performed = 0
        
        if superstep_idx is not None:
            # Try to merge only the specified superstep with the next one
            if superstep_idx < len(self.supersteps) - 1:
                if self.can_merge_supersteps(superstep_idx, superstep_idx + 1):
                    self._perform_merge(superstep_idx, superstep_idx + 1)
                    merges_performed = 1
        else:
            # Try to merge all consecutive supersteps
            i = 0
            while i < len(self.supersteps) - 1:
                if self.can_merge_supersteps(i, i + 1):
                    self._perform_merge(i, i + 1)
                    merges_performed += 1
                    # Don't increment i since we just removed a superstep
                else:
                    i += 1
        
        return merges_performed
    
    def _perform_merge(self, first_idx: int, second_idx: int):
        """Actually perform the merge of two supersteps.
        
        Moves all tasks from the second superstep to the first superstep,
        maintaining their processor assignments.
        
        Args:
            first_idx: Index of the first superstep
            second_idx: Index of the second superstep
        """
        first_superstep = self.supersteps[first_idx]
        second_superstep = self.supersteps[second_idx]
        
        # Move all tasks from second to first superstep
        for proc, tasks in second_superstep.tasks.items():
            for task in list(tasks):  # Create a copy to iterate safely
                # Update the task's superstep reference
                task.superstep = first_superstep
                # Add to the first superstep's task list
                first_superstep.tasks[proc].append(task)
                # Invalidate cached timings
                task.invalidate_timings()
        
        # Remove the second superstep from the schedule
        self.supersteps.pop(second_idx)
        
        # Invalidate timings for the merged superstep
        first_superstep.invalidate_timings()
        
        # Invalidate cached indices for all supersteps after the merge
        for superstep in self.supersteps[first_idx:]:
            if 'index' in superstep.__dict__:
                del superstep.__dict__['index']
class AsyncSchedule(MutableMapping[Hashable, List[Task]]):
    """Encapsulates an asynchronous schedule as a mapping of processors to task lists.
    In SAGA, `Dict[Hashable, List[Task]]` is used to represent a (asynchronous) schedule.
    This class provides a more structured interface to access and manipulate an asynchronous schedule.
    """
    
    def __init__(self, schedule_dict: Optional[Dict[Hashable, List[Task]]] = None):
        self._schedule: Dict[Hashable, List[Task]] = schedule_dict or {}
    
    # Dict-like interface
    def __getitem__(self, processor: Hashable) -> List[Task]:
        return self._schedule[processor]
    
    def __setitem__(self, processor: Hashable, tasks: List[Task]):
        self._schedule[processor] = tasks
    
    def __delitem__(self, processor: Hashable):
        del self._schedule[processor]
    
    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._schedule)
    
    def __len__(self) -> int:
        return len(self._schedule)
    
    def __contains__(self, processor: object) -> bool:
        return processor in self._schedule