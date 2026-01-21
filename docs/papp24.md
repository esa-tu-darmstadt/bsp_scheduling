# Deviations from Papp et al. 2024

This document describes the differences between our implementation and the algorithms described in:

> Papp, Pál András, et al. "Scheduling DAGs on Heterogeneous BSP Systems with Synchronization Costs." (2024)

## 1. Incremental Coarsening Edge Maintenance

**Location:** `saga_bsp/schedulers/papp/coarsening.py` - `DAGCoarsener` class

**Paper's approach:** The coarsening algorithm (Section 4.5, Appendix A.5) contracts edges one at a time until the graph reaches a target size. For each contraction step, the paper implies recomputing which edges are contractable (i.e., have no alternate path).

**Our optimization:** We added an `incremental` mode that maintains the set of contractable edges incrementally rather than recomputing from scratch after each contraction.

After contracting edge (u, v) where u is merged into v:
1. Remove edges involving the absorbed node u from the contractable set
2. Only recheck edges that could have been affected by the contraction:
   - Rule 1: Edges involving the absorbing node v
   - Rule 2: Edges from/to neighbors of new predecessors of v
   - Rule 3: Edges from/to neighbors of new successors of v
   - Rule 4: Edges between predecessors and successors of v
   - Rule 5: Edges (pred_of_v, b) where b is reachable from v's new successors
   - Rule 6: Edges (a, b) where a is an ancestor of v (not just direct predecessor) and b is reachable from v's new successors

Rule 6 is critical for correctness: when contracting creates a new edge v → new_succ, any edge (a, b) can gain an alternate path a → ... → v → new_succ → ... → b. Without this rule, edges from indirect predecessors of v would not be rechecked, potentially leaving edges with alternate paths in the contractable set and causing cycles when contracted.

**Performance impact:** ~50x speedup on large graphs. On a 3641-node, 15968-edge graph:
- Standard mode: Timed out after 120s with only 107/3095 contractions complete (~58 minutes estimated)
- Incremental mode: Completed all 3095 contractions in ~72 seconds

**Result equivalence:** Results may differ from the standard (non-incremental) mode due to different edge iteration order. The paper does not specify tie-breaking behavior when multiple edges have the same selection score, so both orderings produce valid results per the algorithm specification.

**Testing:** The `deterministic` flag can be enabled alongside `incremental` to sort edges before selection, ensuring reproducible results for testing that both modes select the same edges.

**Flags:**
- `DAGCoarsener(incremental=True)` - Enable incremental mode
- `DAGCoarsener(deterministic=True)` - Enable deterministic edge ordering
- `MultilevelScheduler(incremental_coarsening=True)` - Enable incremental mode via scheduler

---

## 2. Alternate Path Detection Without Graph Copy

**Location:** `saga_bsp/schedulers/papp/coarsening.py` - `_has_alternate_path()` method

**Paper's approach:** To check if edge (u, v) can be contracted, verify there is no other directed path from u to v besides the direct edge.

**Original implementation issue:** Our naive implementation copied the entire graph, removed the edge (u, v), then checked for a path. This is O(V + E) per edge check just for the copy operation.

**Our optimization:** We implemented `_has_alternate_path()` which performs a single BFS from u, skipping the direct edge to v. If v is reached through any other path, an alternate path exists. This is O(reachable nodes) without any graph copying.

**Performance impact:** Eliminates the O(V + E) graph copy overhead per edge check.

**Result equivalence:** Produces identical results to the graph-copy approach.

---

## 3. Edge Selection Optimizations

**Location:** `saga_bsp/schedulers/papp/coarsening.py` - `_select_edge_to_contract()` method

**Paper's approach:** To select the best edge to contract:
1. Sort all contractable edges by w(u) + w(v) ascending
2. Take the first 1/3 of the sorted list
3. From that subset, pick the edge with largest c(u)

**Our optimizations:**

### 3a. Partial Sort with heapq.nsmallest

Instead of fully sorting all edges O(n log n), we use `heapq.nsmallest()` to find only the smallest 1/3 of edges. This is O(n log k) where k = n/3.

### 3b. Communication Weight Caching

The `comm_weight(edge)` function computes the average outgoing edge weight for a node. Since this only depends on the source node `u` (not the edge), we cache the result per node to avoid redundant computation when multiple edges share the same source.

**Performance impact:** On a 3641-node graph with 3095 contractions:
- Before: `_select_edge_to_contract` took 64.5s cumulative (6.2M `comm_weight` calls, 20.7s in sorting)
- After: Reduced to ~48.5s cumulative with caching and partial sort

**Result equivalence:** Produces identical results to the original approach.

---

## 4. BSPg Scheduler Optimizations

**Location:** `saga_bsp/schedulers/papp/bspg_scheduler.py` - `BSPgScheduler` class

**Paper's approach:** The BSPg greedy scheduler (Algorithm 1, Section 4.2) assigns tasks to processors as they become free, tracking finish times and idle processors.

**Our optimizations:**

### 4a. Min-Heap for Finish Times

The original implementation used `min(finish_times.values())` to find the earliest time a processor becomes free, which is O(P) per iteration.

We replaced this with a min-heap that provides O(log P) updates and O(1) min lookup. The heap uses lazy deletion - stale entries are filtered out when encountered.

### 4b. Incremental Idle Count Tracking

The original implementation counted idle processors each iteration with `sum(1 for p in processors if finish_times[p] == float('inf'))` and checked if all are idle with `all(...)`.

We now maintain `idle_count` incrementally, incrementing it when a processor becomes idle and resetting it when a new superstep starts.

### 4c. Cached Processor Speeds

Processor speeds are cached at the start of scheduling to avoid repeated dictionary lookups during task assignment.

**Performance impact:** Reduces per-iteration cost from O(P) to O(log P) for processor management.

**Result equivalence:** Produces identical results to the standard implementation (verified by tests).

**Flags:**
- `BSPgScheduler(optimized=True)` - Enable all BSPg optimizations
