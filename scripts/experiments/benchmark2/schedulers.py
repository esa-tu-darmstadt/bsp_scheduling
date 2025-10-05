"""
Scheduler configuration for BSP benchmarking.

This module defines the schedulers currently enabled in the BSP benchmarking,
including the async HeftBusyCommScheduler and BSP schedulers.
"""

from typing import List, Dict, Optional
import logging

# Import optimized delay model schedulers (replaces slow SAGA schedulers)
from saga_bsp.misc.saga_scheduler_wrapper import preprocess_task_graph
from saga_bsp.schedulers.delaymodel import HeftScheduler

# Import BSP schedulers and utilities
from saga_bsp.misc.heft_busy_communication import HeftBusyCommScheduler
from saga_bsp.schedulers import FillInSplitBSPScheduler, BCSHScheduler
from saga_bsp import AsyncToBSPScheduler

logger = logging.getLogger(__name__)


class UnifiedSchedulerWrapper:
    """Unified wrapper for all scheduler types with consistent interface."""

    def __init__(self, scheduler):
        """Initialize unified scheduler wrapper.

        Args:
            scheduler: The underlying scheduler instance
        """
        self.scheduler = scheduler
        self.scheduler_type = self._detect_scheduler_type(scheduler)

    def _detect_scheduler_type(self, scheduler):
        """Detect scheduler type based on its superclass."""
        # Get the class hierarchy (MRO - Method Resolution Order)
        mro_names = [cls.__name__ for cls in type(scheduler).__mro__]

        # BSP schedulers inherit from BSPScheduler
        if 'BSPScheduler' in mro_names:
            return "bsp"
        # Async schedulers inherit from Scheduler (but not BSPScheduler)
        elif 'Scheduler' in mro_names:
            return "async"
        else:
            # Default to BSP if we can't determine
            return "bsp"

    def schedule(self, bsp_hardware, task_graph):
        """Schedule task graph on BSP hardware.

        Args:
            bsp_hardware: BSPHardware instance with network and sync_time
            task_graph: NetworkX DiGraph representing the task graph

        Returns:
            Dictionary with 'makespan' and 'schedule' keys
        """
        if self.scheduler_type == "async":
            # Async schedulers (like HeftBusyCommScheduler) use NetworkX graph
            network = bsp_hardware.network
            schedule = self.scheduler.schedule(network, task_graph)

            # HeftBusyCommScheduler returns a dict mapping processor keys to task schedules
            if isinstance(schedule, dict):
                makespan = self._calculate_makespan_from_async_schedule(schedule)
            elif hasattr(schedule, 'makespan'):
                makespan = schedule.makespan()
            else:
                raise ValueError(f"Async scheduler returned unexpected result type: {type(schedule)}")

            return {
                'makespan': makespan,
                'schedule': schedule
            }

        else:  # BSP schedulers
            # Run scheduler
            schedule_result = self.scheduler.schedule(bsp_hardware, task_graph)

            # BSP schedulers return BSPSchedule objects with makespan property
            makespan = schedule_result.makespan

            return {
                'makespan': makespan,
                'schedule': schedule_result
            }

    def _calculate_makespan_from_async_schedule(self, schedule_dict):
        """Calculate makespan from async scheduler result dictionary."""
        max_end_time = 0.0
        for processor_schedule in schedule_dict.values():
            for task in processor_schedule:
                if hasattr(task, 'end'):
                    max_end_time = max(max_end_time, task.end)
        return max_end_time

    def _calculate_makespan_from_schedule(self, schedule_dict):
        """Calculate makespan from BSP scheduler result dictionary."""
        max_end_time = 0.0
        for node_tasks in schedule_dict.values():
            for task in node_tasks:
                if hasattr(task, 'end'):
                    max_end_time = max(max_end_time, task.end)
        return max_end_time


# Scheduler ordering for visualizations
# HeftBusyCommScheduler (async/delay model) should be first, then BSP schedulers
SCHEDULER_ORDER = [
    "HeftBusyCommScheduler",           # First - async/delay model (special treatment)
    "HEFT-BSP-EarliestNext",          # BSP conversions
    "HEFT-BSP-Eager",
    "FillInSplitBSPScheduler-HEFT",   # Native BSP schedulers
    "FillInSplitBSPScheduler-CPoP",
    "FillInSplitBSPScheduler-DS",
    "FillInSplitBSPScheduler-HEFT-Merge",
    "BCSHScheduler-NoEFT",
    "BCSHScheduler-EFT"
]

# Scheduler renames for display
SCHEDULER_RENAMES = {
    "HeftBusyCommScheduler": "HEFT",
    "HEFT-BSP-EarliestNext": "HEFT + EFN",
    "HEFT-BSP-Eager": "HEFT + Eager",
    "FillInSplitBSPScheduler-HEFT": "BALS Upward",
    "FillInSplitBSPScheduler-CPoP": "BALS Combined",
    # "FillInSplitBSPScheduler-DS": "BALS Dyn. Comb.",
    "FillInSplitBSPScheduler-HEFT-Merge": "BALS Upw. + Elim.",
    "BCSHScheduler-NoEFT": "BCSH (LDSH)",
    "BCSHScheduler-EFT": "BCSH (EFT)"
}

def create_bsp_schedulers() -> Dict[str, object]:
    """Create BSP schedulers matching current benchmarking setup.

    Returns:
        Dictionary mapping scheduler names to UnifiedSchedulerWrapper instances
    """
    schedulers = {}

    # 1. Async scheduler (delay model) - auto-detected as async
    schedulers["HeftBusyCommScheduler"] = UnifiedSchedulerWrapper(
        HeftBusyCommScheduler()
    )
    
    # schedulers["Heft"] = UnifiedSchedulerWrapper(
    #     HeftScheduler()
    # )

    # 2. Async-to-BSP conversion schedulers - auto-detected as bsp
    heft_async = HeftScheduler()

    schedulers["HEFT-BSP-EarliestNext"] = UnifiedSchedulerWrapper(
        AsyncToBSPScheduler(
            async_scheduler=heft_async,
            strategy="earliest-finishing-next"
        )
    )

    schedulers["HEFT-BSP-Eager"] = UnifiedSchedulerWrapper(
        AsyncToBSPScheduler(
            async_scheduler=heft_async,
            strategy="eager"
        )
    )

    # 3. Native BSP schedulers - auto-detected as bsp

    schedulers["FillInSplitBSPScheduler-HEFT"] = UnifiedSchedulerWrapper(
        FillInSplitBSPScheduler(priority_mode='heft')
    )

    schedulers["FillInSplitBSPScheduler-CPoP"] = UnifiedSchedulerWrapper(
        FillInSplitBSPScheduler(priority_mode='cpop')
    )
    
    # schedulers["FillInSplitBSPScheduler-DS"] = UnifiedSchedulerWrapper(
    #     FillInSplitBSPScheduler(priority_mode='ds')
    # )
    
    schedulers["FillInSplitBSPScheduler-HEFT-Merge"] = UnifiedSchedulerWrapper(
        FillInSplitBSPScheduler(priority_mode='heft', optimize_merging=True)
    )
    
    # schedulers["FillInSplitBSPScheduler-CPOP-Merge"] = UnifiedSchedulerWrapper(
    #     FillInSplitBSPScheduler(priority_mode='cpop', optimize_merging=True)
    # )

    schedulers["BCSHScheduler-NoEFT"] = UnifiedSchedulerWrapper(
        BCSHScheduler(use_eft=False)
    )

    schedulers["BCSHScheduler-EFT"] = UnifiedSchedulerWrapper(
        BCSHScheduler(use_eft=True)
    )

    return schedulers

def get_bsp_schedulers(scheduler_names: Optional[List[str]] = None) -> Dict[str, object]:
    """Get BSP schedulers for benchmarking.

    Args:
        scheduler_names: Optional list of specific scheduler names to include

    Returns:
        Dictionary mapping scheduler names to UnifiedSchedulerWrapper instances
    """
    schedulers = create_bsp_schedulers()

    # Filter by requested scheduler names if provided
    if scheduler_names:
        filtered_schedulers = {}
        for name in scheduler_names:
            if name in schedulers:
                filtered_schedulers[name] = schedulers[name]
            else:
                logger.warning(f"Requested scheduler '{name}' not found. Available: {list(schedulers.keys())}")
        schedulers = filtered_schedulers

    logger.info(f"Created {len(schedulers)} schedulers: {list(schedulers.keys())}")
    return schedulers

def get_scheduler_display_name(scheduler_name: str) -> str:
    """Get display name for a scheduler."""
    return SCHEDULER_RENAMES.get(scheduler_name, scheduler_name)

def get_ordered_scheduler_names(scheduler_names: List[str]) -> List[str]:
    """Order scheduler names according to visualization requirements.

    Args:
        scheduler_names: List of scheduler names to order

    Returns:
        Ordered list of scheduler names
    """
    # Create a mapping for ordering
    order_map = {name: i for i, name in enumerate(SCHEDULER_ORDER)}

    # Sort scheduler names according to the predefined order
    def get_order(name):
        return order_map.get(name, len(SCHEDULER_ORDER))

    ordered_names = sorted(scheduler_names, key=get_order)

    return ordered_names

def is_delay_model_scheduler(scheduler_name: str) -> bool:
    """Check if scheduler uses delay/async model (not BSP model).

    Args:
        scheduler_name: Name of the scheduler

    Returns:
        True if scheduler uses delay/async model
    """
    return scheduler_name in ["HeftBusyCommScheduler"]