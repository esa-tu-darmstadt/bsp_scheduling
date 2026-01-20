"""Papp et al. 2024 BSP Scheduling Algorithms

Implementation of scheduling algorithms from:
"Efficient Multi-Processor Scheduling in Increasingly Realistic Models"
by Papp, Anegg, Karanasiou, and Yzelman (SPAA 2024)

Components:
- BSPgScheduler: Greedy BSP scheduler (Algorithm 1)
- SourceScheduler: Source-layer based scheduler (Algorithm 2)
- MultilevelScheduler: Coarsen-solve-refine framework
- DAGCoarsener: DAG coarsening utilities
"""

from .bspg_scheduler import BSPgScheduler
from .source_scheduler import SourceScheduler
from .coarsening import DAGCoarsener, ContractionRecord
from .multilevel_scheduler import MultilevelScheduler

__all__ = [
    'BSPgScheduler',
    'SourceScheduler',
    'DAGCoarsener',
    'ContractionRecord',
    'MultilevelScheduler',
]
