from typing import Literal
import networkx as nx
from saga.scheduler import Scheduler

from .base import BSPScheduler
from ..schedule import BSPSchedule, BSPHardware
from ..conversion import convert_async_to_bsp


class AsyncToBSPScheduler(BSPScheduler):
    """BSP scheduler that wraps a SAGA async scheduler.
    
    This scheduler takes any SAGA async scheduler and converts its output
    to a BSP schedule using one of the async-to-BSP conversion strategies.
    
    Args:
        async_scheduler: Any SAGA scheduler to wrap
        strategy: BSP conversion strategy
        backfill_threshold_percent: Optional backfilling threshold
    """
    
    def __init__(
        self,
        async_scheduler: Scheduler,
        strategy: Literal["eager", "level-based", "earliest-finishing-next"] = "earliest-finishing-next",
        backfill_threshold_percent: float = None
    ):
        self.async_scheduler = async_scheduler
        self.strategy = strategy
        self.backfill_threshold_percent = backfill_threshold_percent
        
        # Create clean descriptive name
        base_name = async_scheduler.__class__.__name__.replace("Scheduler", "").upper()
        
        # Map strategy to clean name
        strategy_map = {
            "eager": "Eager",
            "level-based": "Level", 
            "earliest-finishing-next": "EarliestNext"
        }
        
        strategy_name = strategy_map.get(strategy, strategy)
        
        # Build final name
        name_parts = [base_name, "BSP", strategy_name]
        if backfill_threshold_percent is not None:
            backfill_pct = int(backfill_threshold_percent * 100)
            name_parts.append(f"BF{backfill_pct}%")
        
        self.name = "-".join(name_parts)
    
    def schedule(self, hardware: BSPHardware, task_graph: nx.DiGraph) -> BSPSchedule:
        """Schedule tasks by converting async schedule to BSP.
        
        Args:
            hardware: BSP hardware configuration
            task_graph: Task dependency graph
            
        Returns:
            BSP schedule with supersteps
        """
        # Get async schedule using the wrapped scheduler
        async_schedule = self.async_scheduler.schedule(hardware.network, task_graph)
        
        # Convert to BSP schedule
        bsp_schedule = convert_async_to_bsp(
            hardware=hardware,
            task_graph=task_graph,
            async_schedule=async_schedule,
            strategy=self.strategy,
            backfill_threshold_percent=self.backfill_threshold_percent
        )
        
        return bsp_schedule