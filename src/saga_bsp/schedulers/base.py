from abc import ABC, abstractmethod
import networkx as nx
from ..schedule import BSPSchedule, BSPHardware


class BSPScheduler(ABC):
    """Abstract base class for BSP schedulers.
    
    BSP schedulers work directly with BSP hardware and produce BSP schedules
    with supersteps, synchronization, and communication phases.
    """
    
    @abstractmethod
    def schedule(self, hardware: BSPHardware, task_graph: nx.DiGraph) -> BSPSchedule:
        """Schedule tasks on BSP hardware.
        
        Args:
            hardware: BSP hardware configuration (network + sync time)
            task_graph: Task dependency graph
            
        Returns:
            BSPSchedule with supersteps and timing information
        """
        raise NotImplementedError
    
    @property 
    def __name__(self) -> str:
        """Get the name of the BSP scheduler."""
        if hasattr(self, "name"):
            return self.name
        return self.__class__.__name__