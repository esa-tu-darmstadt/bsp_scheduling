from typing import Dict, Hashable, List
import networkx as nx
from saga.scheduler import Scheduler, Task

from ..schedulers.base import BSPScheduler
from ..schedule import BSPHardware
from ..conversion import convert_bsp_to_async


class SagaSchedulerWrapper(Scheduler):
    """SAGA-compatible wrapper for BSP schedulers.
    
    This wrapper allows any BSP scheduler to be used within SAGA's
    infrastructure (benchmarking, simulated annealing, etc.) by converting
    BSP schedules back to the async format with proper BSP timing information.
    
    The wrapper bridges the gap between BSP scheduling algorithms and SAGA's
    async-focused analysis tools.
    
    Args:
        bsp_scheduler: Any BSP scheduler to wrap
        sync_time: BSP synchronization overhead time
    """
    
    def __init__(self, bsp_scheduler: BSPScheduler, sync_time: float = 1.0):
        self.bsp_scheduler = bsp_scheduler
        self.sync_time = sync_time
        
        # Use the BSP scheduler's name directly (already clean)
        self.name = bsp_scheduler.__name__
    
    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        """Schedule tasks using BSP model and convert to SAGA async format.
        
        Args:
            network: The network topology from SAGA
            task_graph: The task dependency graph
            
        Returns:
            Dict mapping processors to lists of Tasks with BSP timing
        """
        # Create BSP hardware from SAGA's network and our sync time
        bsp_hardware = BSPHardware(network=network, sync_time=self.sync_time)
        
        # Get BSP schedule
        bsp_schedule = self.bsp_scheduler.schedule(bsp_hardware, task_graph)
        
        # Convert to SAGA-compatible async format with BSP timing
        async_schedule = convert_bsp_to_async(bsp_schedule)
        
        return async_schedule