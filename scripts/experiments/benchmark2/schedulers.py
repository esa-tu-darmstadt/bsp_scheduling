"""
Scheduler configuration for BSP benchmarking.

This module defines the schedulers currently enabled in the BSP benchmarking,
including the async HeftBusyCommScheduler and BSP schedulers.
"""

from typing import List, Dict, Optional
import logging

# Import optimized delay model schedulers (replaces slow SAGA schedulers)
from saga_bsp import schedulers
from saga_bsp.misc.saga_scheduler_wrapper import preprocess_task_graph
from saga_bsp.schedulers.delaymodel import HeftScheduler

# Import BSP schedulers and utilities
from saga_bsp.misc.heft_busy_communication import HeftBusyCommScheduler
from saga_bsp.schedulers import FillInSplitBSPScheduler, BCSHScheduler, HDaggScheduler
from saga_bsp.schedulers import BSPgScheduler, SourceScheduler, MultilevelScheduler
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


# Scheduler groups for visualization labels and separators
# Groups are ordered: baseline -> proposed (ours) -> existing (comparison)
SCHEDULER_GROUPS = {
    "baseline": [
        "HeftBusyCommScheduler",
    ],
    "proposed": [
        "HEFT-BSP-EarliestNext",
        "HEFT-BSP-Eager",
        "FillInSplitBSPScheduler-CPoP",
        "FillInSplitBSPScheduler-CPoP-Merge",
        "FillInSplitBSPScheduler-HEFT",
        "FillInSplitBSPScheduler-HEFT-Merge",
        
        "FillInSplitBSPScheduler-HEFT-BoundaryDeps",
        "FillInSplitBSPScheduler-HEFT-BoundaryDeps-Merge",
    ],
    "existing": [
        "HDaggScheduler-0.01",
        "HDaggScheduler-0.1",
        "HDaggScheduler-0.5",
        "BCSHScheduler-NoEFT",
        "BCSHScheduler-EFT",
        "BSPgScheduler",
        "SourceScheduler",
        "MultilevelScheduler-15",
        "MultilevelScheduler-30",
    ],
}

# Display labels for scheduler groups
SCHEDULER_GROUP_LABELS = {
    "baseline": "Async Baseline",
    "proposed": "Our Work",
    "existing": "Existing BSP Schedulers",
}

# Scheduler renames for display
SCHEDULER_RENAMES = {
    "HeftBusyCommScheduler": "HEFT, delay-model",
    "HEFT-BSP-EarliestNext": "HEFT + EFN",
    "HEFT-BSP-Eager": "HEFT + Eager",
    "FillInSplitBSPScheduler-HEFT": "BALS Upward",
    "FillInSplitBSPScheduler-CPoP": "BALS Combined",
    "FillInSplitBSPScheduler-HEFT-Merge": "BALS Upw. + Elim.",
    "FillInSplitBSPScheduler-CPoP-Merge": "BALS Comb. + Elim.",
    
    "FillInSplitBSPScheduler-HEFT-BoundaryDeps": "BALS Upw. Snap",
    "FillInSplitBSPScheduler-HEFT-BoundaryDeps-Merge": "BALS Upw. Snap + Elim.",
    
    # HDagg schedulers
    "HDaggScheduler-0.01": "HDagg (ε=0.01)",
    "HDaggScheduler-0.1": "HDagg (ε=0.1)",
    "HDaggScheduler-0.5": "HDagg (ε=0.5)",
    "BCSHScheduler-NoEFT": "BCSH (LDSH)",
    "BCSHScheduler-EFT": "BCSH (EFT)",
    # Papp et al. 2024 schedulers
    "BSPgScheduler": "BSPg",
    "SourceScheduler": "Source",
    "MultilevelScheduler-15": "Multilevel (15%)",
    "MultilevelScheduler-30": "Multilevel (30%)",
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
    
    schedulers["FillInSplitBSPScheduler-CPoP-Merge"] = UnifiedSchedulerWrapper(
        FillInSplitBSPScheduler(priority_mode='cpop', optimize_merging=True)
    )
    
    schedulers["FillInSplitBSPScheduler-HEFT-BoundaryDeps"] = UnifiedSchedulerWrapper(
    FillInSplitBSPScheduler(priority_mode='heft', reduce_fragmentation=True, boundary_slack_factor=10)
    )
    
    schedulers["FillInSplitBSPScheduler-HEFT-BoundaryDeps-Merge"] = UnifiedSchedulerWrapper(
    FillInSplitBSPScheduler(priority_mode='heft', optimize_merging=True, reduce_fragmentation=True, boundary_slack_factor=10)
    )

    schedulers["HDaggScheduler-0.01"] = UnifiedSchedulerWrapper(
        HDaggScheduler(epsilon=0.01)
    )
    
    schedulers["HDaggScheduler-0.1"] = UnifiedSchedulerWrapper(
        HDaggScheduler(epsilon=0.1)
    )
        
    # schedulers["HDaggScheduler-0.5"] = UnifiedSchedulerWrapper(
    #     HDaggScheduler(epsilon=0.5)
    # )

    schedulers["BCSHScheduler-NoEFT"] = UnifiedSchedulerWrapper(
        BCSHScheduler(use_eft=False)
    )

    schedulers["BCSHScheduler-EFT"] = UnifiedSchedulerWrapper(
        BCSHScheduler(use_eft=True)
    )

    # 4. Papp et al. 2024 schedulers
    schedulers["BSPgScheduler"] = UnifiedSchedulerWrapper(
        BSPgScheduler()
    )

    schedulers["SourceScheduler"] = UnifiedSchedulerWrapper(
        SourceScheduler()
    )

    # TODO: Uncomment for full comparison
    schedulers["MultilevelScheduler-15"] = UnifiedSchedulerWrapper(
        MultilevelScheduler(coarsening_ratios=[0.15], hc_interval=100, hc_max_steps=20)
    )

    schedulers["MultilevelScheduler-30"] = UnifiedSchedulerWrapper(
        MultilevelScheduler(coarsening_ratios=[0.30], hc_interval=100, hc_max_steps=20)
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
    # Derive order from SCHEDULER_GROUPS (preserves dict insertion order)
    order_list = [name for group in SCHEDULER_GROUPS.values() for name in group]
    order_map = {name: i for i, name in enumerate(order_list)}

    # Sort scheduler names according to the predefined order
    def get_order(name):
        return order_map.get(name, len(order_list))

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


def get_scheduler_group(scheduler_name: str) -> str:
    """Get the group that a scheduler belongs to.

    Args:
        scheduler_name: Name of the scheduler

    Returns:
        Group name ('baseline', 'proposed', or 'existing')
    """
    for group_name, schedulers in SCHEDULER_GROUPS.items():
        if scheduler_name in schedulers:
            return group_name
    return "existing"  # Default to existing for unknown schedulers


def organize_schedulers_by_group(scheduler_names: List[str]) -> tuple:
    """Organize schedulers by their group for visualization.

    Args:
        scheduler_names: List of scheduler names to organize

    Returns:
        Tuple of (ordered_schedulers, group_boundaries)
        where group_boundaries maps group_label to (start_idx, end_idx)
    """
    # First, order the schedulers
    ordered_schedulers = get_ordered_scheduler_names(scheduler_names)

    # Group order for visualization
    group_order = ["baseline", "proposed", "existing"]

    # Calculate boundaries based on actual scheduler positions
    group_boundaries = {}
    current_group = None
    group_start = 0

    for idx, scheduler in enumerate(ordered_schedulers):
        scheduler_group = get_scheduler_group(scheduler)

        if scheduler_group != current_group:
            # Close previous group if exists
            if current_group is not None and current_group in group_order:
                label = SCHEDULER_GROUP_LABELS.get(current_group, current_group)
                group_boundaries[label] = (group_start, idx)

            # Start new group
            current_group = scheduler_group
            group_start = idx

    # Close the last group
    if current_group is not None and current_group in group_order:
        label = SCHEDULER_GROUP_LABELS.get(current_group, current_group)
        group_boundaries[label] = (group_start, len(ordered_schedulers))

    return ordered_schedulers, group_boundaries