from typing import Dict, Hashable, List
from saga.scheduler import Task
from ..schedule import BSPSchedule


def convert_bsp_to_async(bsp_schedule: BSPSchedule) -> Dict[Hashable, List[Task]]:
    """Convert a BSP schedule to an async schedule compatible with SAGA.
    
    This function takes a BSPSchedule and converts it to the format expected
    by SAGA's async scheduling infrastructure: Dict[processor, List[Task]]
    where each Task has the correct start and end times from the BSP schedule.
    
    Args:
        bsp_schedule: The BSP schedule to convert
        
    Returns:
        Dict mapping processors to lists of Tasks with BSP timing information
    """
    async_schedule: Dict[Hashable, List[Task]] = {}
    
    # Get all processors from the hardware network
    for processor in bsp_schedule.hardware.network.nodes():
        async_schedule[processor] = []
    
    # Convert each BSP task to a SAGA Task
    for superstep in bsp_schedule.supersteps:
        for processor, bsp_tasks in superstep.tasks.items():
            for bsp_task in bsp_tasks:
                # Create SAGA Task with BSP timing information
                saga_task = Task(
                    node=processor,          # The processor executing the task
                    name=bsp_task.node,     # The task name from task graph
                    start=bsp_task.start,   # BSP absolute start time
                    end=bsp_task.end        # BSP absolute end time
                )
                async_schedule[processor].append(saga_task)
    
    # Sort tasks by start time on each processor
    for processor in async_schedule:
        async_schedule[processor].sort(key=lambda task: task.start or 0.0)
    
    return async_schedule