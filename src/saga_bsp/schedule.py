from collections.abc import MutableMapping
from dataclasses import dataclass
from functools import cached_property, cache
from typing import Dict, Hashable, Iterator, List, Set, Optional, Tuple
from collections import defaultdict
import networkx as nx
from saga.scheduler import Task
import copy
import random


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


class BSPTask:
    """Extended task for BSP scheduling"""

    def __init__(self, node: str, proc: str, superstep: "Superstep"):
        self.node = node  # The node where the task is executed
        self.proc = proc  # The processor on which the task is executed
        self.superstep = superstep  # The superstep this task belongs to

    def invalidate_timings(self):
        """Invalidate all cached timings for this task"""
        del self.rel_start

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

    @property
    def speed(self) -> float:
        """Speed of the processor executing this task"""
        return self.network.nodes[self.proc]["weight"]

    @property
    def cost(self) -> float:
        """Cost of executing this task"""
        return self.task_graph.nodes[self.node]["weight"]

    @property
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
        return (
            self.superstep.tasks[self.proc][self.index - 1].rel_end
            if self.index > 0
            else 0.0
        )

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

    @property
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
        if self.index == 0:
            return 0.0
        return self.schedule.supersteps[self.index - 1].end_time

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
        if not self.tasks:
            return 0.0
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
        """
        total_exchange_time = 0.0
        
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
                    total_exchange_time = max(total_exchange_time, exchange_time)
        
        return total_exchange_time

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
        

    @property
    def makespan(self) -> float:
        """Total makespan of the schedule, which is the end time of the last superstep."""
        return self.supersteps[-1].end_time if self.supersteps else 0.0

    @property
    def num_supersteps(self) -> int:
        return len(self.supersteps)
    
    def copy(self) -> "BSPSchedule":
        """Create a deep copy of the schedule."""
        new_schedule = BSPSchedule(self.hardware, self.task_graph)
        
        # Recreate the schedule structure properly
        # First create all supersteps
        for _ in range(len(self.supersteps)):
            new_schedule.add_superstep()
        
        # Then schedule all tasks in the correct order
        for superstep_idx, superstep in enumerate(self.supersteps):
            for processor, tasks in superstep.tasks.items():
                for task in tasks:
                    new_schedule.schedule(task.node, processor, new_schedule.supersteps[superstep_idx])
        
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