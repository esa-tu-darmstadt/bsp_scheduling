"""Optimized delay model schedulers for bsp_scheduling.

This module contains reimplemented versions of SAGA schedulers with optimizations
for better performance, avoiding the overhead of precalculating large arrays.

Usage Examples:

Replace SAGA schedulers in existing code:

    # OLD (SAGA - slow for large problems):
    from saga.schedulers import HeftScheduler

    # NEW (Optimized - fast for large problems):
    from bsp_scheduling.schedulers.delaymodel import HeftScheduler

Replace SAGA priority functions:

    # OLD (SAGA - slow for large problems):
    from saga.schedulers.cpop import upward_rank, cpop_ranks

    # NEW (Optimized - fast for large problems):
    from bsp_scheduling.schedulers.delaymodel.priorities import upward_rank, cpop_ranks

The optimized versions provide identical results but with significantly better
performance by:
- Avoiding O(N*M*E) precalculation of all runtime/communication combinations
- Computing values on-the-fly only when needed
- Pre-calculating average speeds once instead of repeatedly
"""

from .heft import HeftScheduler
from .priorities import upward_rank, downward_rank, cpop_ranks

__all__ = ['HeftScheduler', 'upward_rank', 'downward_rank', 'cpop_ranks']