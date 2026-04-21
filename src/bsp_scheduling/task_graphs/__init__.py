"""
Task graph generation utilities for bsp_scheduling.

This module provides utilities for generating task graphs from various sources:
- WfCommons workflow recipes
- Sum-Product Networks (SPNs)

Usage examples:

# WfCommons task graphs
from bsp_scheduling.task_graphs import WfCommonsTaskGraphGenerator
wf_gen = WfCommonsTaskGraphGenerator(cache_dir=Path("./cache"))
task_graph, metadata = wf_gen.generate_task_graph("montage", task_count=100)

# SPN task graphs
from bsp_scheduling.task_graphs import SPNTaskGraphGenerator
spn_gen = SPNTaskGraphGenerator(cache_dir=Path("./cache"))
task_graph, metadata = spn_gen.generate_task_graph("path/to/spn.bin")
"""

from .wfcommons_generator import WfCommonsTaskGraphGenerator
from .spn_generator import SPNTaskGraphGenerator
from .base import TaskGraphGenerator, TaskGraphMetadata
from .ccr_adjustment import calculate_ccr, adjust_task_graph_to_ccr, generate_ccr_variants, get_ccr_statistics

__all__ = [
    'WfCommonsTaskGraphGenerator',
    'SPNTaskGraphGenerator',
    'TaskGraphGenerator',
    'TaskGraphMetadata',
    'calculate_ccr',
    'adjust_task_graph_to_ccr',
    'generate_ccr_variants',
    'get_ccr_statistics'
]