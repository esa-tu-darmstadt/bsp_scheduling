from dataclasses import dataclass, field
import logging
from queue import PriorityQueue
from .base import BSPScheduler
from ..schedule import BSPSchedule, BSPHardware, Superstep

import networkx as nx

from saga.schedulers.cpop import upward_rank
from saga.schedulers.heft import heft_rank_sort

logger = logging.getLogger(__name__)
@dataclass(order=True)
class PrioritizedTask:
    # Lower priority values are processed first
    priority: float
    task: str=field(compare=False)

class HeftBSPScheduler(BSPScheduler):
    """HEFT-based BSP Scheduler"""

    def __init__(self, verbose: bool = False):
        """Initialize the HEFT BSP scheduler."""
        super().__init__()
        self.name = "HeftBSP"
        
        if verbose:
            logger.setLevel(logging.DEBUG)
        

    def schedule(self, hardware, task_graph):
        network = hardware.network
        schedule = BSPSchedule(hardware, task_graph)
        # Determine the priority of each task
        rank = upward_rank(network, task_graph)
        
        queue = PriorityQueue()
        
        def _schedule_task(task_name: str, superstep: Superstep, processor: str):
            """Schedule the given task and add its successors to the queue if they became ready."""
            schedule.schedule(task_name, processor, superstep)
            for successor in task_graph.successors(task_name):
                if all(schedule.task_scheduled(pred) for pred in task_graph.predecessors(successor)):
                    queue.put(PrioritizedTask(-rank[successor], successor))

        # Initialize the ready queue with tasks that have no dependencies
        for task in task_graph.nodes:
            if task_graph.in_degree(task) == 0:
                queue.put(PrioritizedTask(-rank[task], task))
                logger.debug(f"Task {task} has no predecessors; added to initial queue with priority {rank[task]}")
        
        current_superstep = schedule.add_superstep()
        while not queue.empty():
            task_name = queue.get().task
            logger.debug(f"Processing task {task_name} with priority {rank[task_name]}")
            
            # Find the best processor for this task
            best_processor = None
            min_finish_time = float('inf')
            best_requires_sync = False
            
            for processor in network.nodes:
                requires_sync = not schedule.can_be_scheduled_in(task_name, current_superstep, processor)
                compute_time = task_graph.nodes[task_name]["weight"] / network.nodes[processor]["weight"]
                this_finishing_time = None
                logger.debug(f"-- Checking processor {processor}, requires_sync={requires_sync}, compute_time={compute_time}")
                
                if requires_sync:
                    # Scheduling the task on this processor would require creating a new superstep
                    comm_time = 0.0 # FIXME
                    this_finishing_time = current_superstep.end_time + compute_time + hardware.sync_time + comm_time
                else: 
                    processor_end = current_superstep.tasks[processor][-1].end if current_superstep.tasks[processor] else 0.0
                    this_finishing_time = processor_end + compute_time
                    
                if this_finishing_time < min_finish_time:
                    min_finish_time = this_finishing_time
                    best_processor = processor
                    best_requires_sync = requires_sync

            logger.debug(f"==> Best processor is {best_processor} with finish time {min_finish_time}, requires_sync={best_requires_sync}")

            if best_requires_sync:
                # If we need to sync, create a new superstep
                current_superstep = schedule.add_superstep()
            _schedule_task(task_name, current_superstep, best_processor)
    
        schedule.assert_valid()  # Ensure the schedule is valid
        return schedule
    
    #   def schedule(self, hardware, task_graph):
    #     network = hardware.network
    #     schedule = BSPSchedule(hardware, task_graph)
    #     # Determine the priority of each task
    #     rank = upward_rank(network, task_graph)
        
    #     queue = PriorityQueue()
        
    #     def _schedule_task(task_name: str, superstep: Superstep, processor: str):
    #         """Schedule the given task and add its successors to the queue if they became ready."""
    #         schedule.schedule(task_name, processor, superstep)
    #         for successor in task_graph.successors(task_name):
    #             if all(schedule.task_scheduled(pred) for pred in task_graph.predecessors(successor)):
    #                 queue.put(PrioritizedTask(-rank[successor], successor))

    #     # Initialize the ready queue with tasks that have no dependencies
    #     for task in task_graph.nodes:
    #         if task_graph.in_degree(task) == 0:
    #             queue.put(PrioritizedTask(-rank[task], task))
    #             logger.debug(f"Task {task} has no predecessors; added to initial queue with priority {rank[task]}")
        
    #     current_superstep = schedule.add_superstep()
    #     while not queue.empty():
    #         task_name = queue.get().task
    #         logger.debug(f"Processing task {task_name} with priority {rank[task_name]}")
            
    #         # Find the best processor for this task
    #         best_processor = None
    #         min_finish_time = float('inf')
    #         best_requires_sync = False
            
    #         for processor in network.nodes:
    #             requires_sync = not schedule.can_be_scheduled_in(task_name, current_superstep, processor)
    #             compute_time = task_graph.nodes[task_name]["weight"] / network.nodes[processor]["weight"]
    #             this_finishing_time = None
    #             logger.debug(f"-- Checking processor {processor}, requires_sync={requires_sync}, compute_time={compute_time}")
                
    #             if requires_sync:
    #                 # Scheduling the task on this processor would require creating a new superstep
    #                 comm_time = 0.0 # FIXME
    #                 this_finishing_time = current_superstep.end_time + compute_time + hardware.sync_time + comm_time
    #             else: 
    #                 processor_end = current_superstep.tasks[processor][-1].end if current_superstep.tasks[processor] else 0.0
    #                 this_finishing_time = processor_end + compute_time
                    
    #             if this_finishing_time < min_finish_time:
    #                 min_finish_time = this_finishing_time
    #                 best_processor = processor
    #                 best_requires_sync = requires_sync

    #         logger.debug(f"==> Best processor is {best_processor} with finish time {min_finish_time}, requires_sync={best_requires_sync}")

    #         if best_requires_sync:
    #             # If we need to sync, create a new superstep
    #             current_superstep = schedule.add_superstep()
    #         _schedule_task(task_name, current_superstep, best_processor)
    
    #     schedule.assert_valid()  # Ensure the schedule is valid
    #     return schedule