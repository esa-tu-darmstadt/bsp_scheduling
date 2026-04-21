# Papp et al. 2024 - Implementation Notes

This document describes implementation details, optimizations, and findings related to:

> Papp, Pál András, et al. "Scheduling DAGs on Heterogeneous BSP Systems with Synchronization Costs." (2024)

---

# Part 1: Implementation Deviations

The following sections describe optimizations and differences from the algorithms as described in the paper.

## 1.1 Incremental Coarsening Edge Maintenance

**Location:** `bsp_scheduling/schedulers/papp/coarsening.py` - `DAGCoarsener` class

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

## 1.2 Alternate Path Detection Without Graph Copy

**Location:** `bsp_scheduling/schedulers/papp/coarsening.py` - `_has_alternate_path()` method

**Paper's approach:** To check if edge (u, v) can be contracted, verify there is no other directed path from u to v besides the direct edge.

**Original implementation issue:** Our naive implementation copied the entire graph, removed the edge (u, v), then checked for a path. This is O(V + E) per edge check just for the copy operation.

**Our optimization:** We implemented `_has_alternate_path()` which performs a single BFS from u, skipping the direct edge to v. If v is reached through any other path, an alternate path exists. This is O(reachable nodes) without any graph copying.

**Performance impact:** Eliminates the O(V + E) graph copy overhead per edge check.

**Result equivalence:** Produces identical results to the graph-copy approach.

---

## 1.3 Edge Selection Optimizations

**Location:** `bsp_scheduling/schedulers/papp/coarsening.py` - `_select_edge_to_contract()` method

**Paper's approach:** To select the best edge to contract:
1. Sort all contractable edges by w(u) + w(v) ascending
2. Take the first 1/3 of the sorted list
3. From that subset, pick the edge with largest c(u)

**Our optimizations:**

### 1.3a Partial Sort with heapq.nsmallest

Instead of fully sorting all edges O(n log n), we use `heapq.nsmallest()` to find only the smallest 1/3 of edges. This is O(n log k) where k = n/3.

### 1.3b Communication Weight Caching

The `comm_weight(edge)` function computes the average outgoing edge weight for a node. Since this only depends on the source node `u` (not the edge), we cache the result per node to avoid redundant computation when multiple edges share the same source.

**Performance impact:** On a 3641-node graph with 3095 contractions:
- Before: `_select_edge_to_contract` took 64.5s cumulative (6.2M `comm_weight` calls, 20.7s in sorting)
- After: Reduced to ~48.5s cumulative with caching and partial sort

**Result equivalence:** Produces identical results to the original approach.

---

## 1.4 BSPg Scheduler Optimizations

**Location:** `bsp_scheduling/schedulers/papp/bspg_scheduler.py` - `BSPgScheduler` class

**Paper's approach:** The BSPg greedy scheduler (Algorithm 1, Section 4.2) assigns tasks to processors as they become free, tracking finish times and idle processors.

**Our optimizations:**

### 1.4a Min-Heap for Finish Times

The original implementation used `min(finish_times.values())` to find the earliest time a processor becomes free, which is O(P) per iteration.

We replaced this with a min-heap that provides O(log P) updates and O(1) min lookup. The heap uses lazy deletion - stale entries are filtered out when encountered.

### 1.4b Incremental Idle Count Tracking

The original implementation counted idle processors each iteration with `sum(1 for p in processors if finish_times[p] == float('inf'))` and checked if all are idle with `all(...)`.

We now maintain `idle_count` incrementally, incrementing it when a processor becomes idle and resetting it when a new superstep starts.

### 1.4c Cached Processor Speeds

Processor speeds are cached at the start of scheduling to avoid repeated dictionary lookups during task assignment.

**Performance impact:** Reduces per-iteration cost from O(P) to O(log P) for processor management.

**Result equivalence:** Produces identical results to the standard implementation (verified by tests).

**Flags:**
- `BSPgScheduler(optimized=True)` - Enable all BSPg optimizations

---

# Part 2: Findings and Observations

The following sections document observed behaviors and limitations of the algorithms that are not explicitly discussed in the paper.

## 2.1 Source Scheduler Pathological Behavior on Out-Trees

**Location:** `bsp_scheduling/schedulers/papp/source_scheduler.py` - `SourceScheduler` class

**Algorithm:** The Source scheduler (Algorithm 2, Section 4.2 / Appendix A.2) assigns tasks in supersteps based on source layers:
1. First superstep: cluster sources by common successors, round-robin assign clusters
2. Subsequent supersteps: sort sources by work weight descending, round-robin assign
3. After each superstep: add successors if **all predecessors are on the same processor**

**Problem discovered:** The Source scheduler exhibits catastrophic performance degradation on out-tree graph structures, scheduling all tasks onto a single processor.

### Root Cause

The successor-addition rule (step 3) causes a cascade effect on out-trees:

1. **Out-tree structure:** Single root → branches down → many leaves → optional sink
2. **Superstep 0:** Root (only source) assigned to processor 0
3. **Successor addition:** All children have only 1 predecessor (root) on processor 0 → **all children added to processor 0**
4. **Cascade:** Each subsequent level has all predecessors on processor 0 → entire tree ends up on processor 0

**Example (16-node out-tree, 8 processors):**
```
OUT-TREE: 16 nodes
  Supersteps: 1
  Tasks per processor: {'0': 16}  # All on one processor!

IN-TREE: 15 nodes
  Supersteps: 3
  Tasks per processor: {'0': 5, '1': 4, '2': 3, '3': 3}  # Distributed
```

### Benchmark Results

From `benchmark2` experiments comparing out-tree vs in-tree performance:

| Metric | Out-tree | In-tree |
|--------|----------|---------|
| Mean makespan ratio | **24.32x** | 1.21x |
| Median makespan ratio | 14.04x | 1.15x |
| Max makespan ratio | 113.61x | 1.82x |

**Performance gap:** Source scheduler is **20x worse on out-trees compared to in-trees**.

On out-trees, Source scheduler ranks last among all 16 tested schedulers (tied with BSPgScheduler and MultilevelScheduler variants at 24.32x).

### Paper's Acknowledgment

The paper does not explicitly acknowledge this limitation. However, the experimental results (Tables 4-5, page 20) implicitly show it:

> "Source is rather effective for the shallow spmv DAGs, but not very useful otherwise."

- Source wins on `spmv` graphs (wide, shallow, longest path = 3)
- Source loses on `exp`, `cg`, `kNN` graphs (deeper DAGs)
- The authors did not test on tree-structured graphs

### Why In-Trees Work Better

In-trees avoid the cascade because:
1. Many sources exist initially (the leaves)
2. Sources get distributed across processors via clustering/round-robin
3. Successors have predecessors on **different processors** → cannot be added to same superstep
4. Forces multiple supersteps → maintains processor distribution

### Conclusion

The Source scheduler is designed for wide, shallow DAGs where parallelism exists at the source level. It fundamentally fails on "narrow top" structures like out-trees where the single root causes all descendants to cascade onto one processor. This is **intended behavior** per the algorithm design, but the **pathological case is not documented** in the paper.

**Recommendation:** Do not use SourceScheduler on out-tree or similar narrow-top graph structures. Use HeftBSPScheduler, HDaggScheduler, or BALSScheduler instead.

---

## 2.2 BSPg Scheduler Pathological Behavior on Out-Trees

**Location:** `bsp_scheduling/schedulers/papp/bspg_scheduler.py` - `BSPgScheduler` class

**Algorithm:** The BSPg greedy scheduler (Algorithm 1, Section 4.2 / Appendix A.2) assigns tasks to processors as they become free:
1. Maintain `ready_p[p]` = nodes assignable to processor p without communication (all predecessors on p or in earlier supersteps)
2. Maintain `ready_all` = nodes requiring cross-processor communication (predecessors on multiple processors in current superstep)
3. When processor p becomes free, **prefer `ready_p[p]` over `ready_all`**
4. Close superstep when ≥P/2 processors are idle without assignable nodes

**Problem discovered:** BSPg exhibits the same catastrophic performance degradation on out-trees as SourceScheduler, scheduling all tasks onto a single processor.

### Root Cause

The `ready_p` priority mechanism causes processor hogging on out-trees:

1. **Out-tree structure:** Single root → branches down → many leaves
2. **Initial state:** Root (only source) in `ready_all`
3. **Processor 0 picks root:** First free processor (0) takes root from `ready_all`
4. **Children go to `ready_p[0]`:** All children have only 1 predecessor (root) on processor 0 → they go into `ready_p[0]`, not `ready_all`
5. **Processor 0 keeps working:** When processor 0 finishes root, it picks from `ready_p[0]` first → gets children
6. **Cascade:** Children's children also have all predecessors on processor 0 → go into `ready_p[0]`
7. **Result:** Processor 0 always has work in `ready_p[0]`, never goes idle, never gives other processors a chance

**Example (16-node out-tree, 8 processors):**
```
OUT-TREE: 16 nodes
  Supersteps: 1
  Tasks per processor: {'0': 16}  # All on one processor!

IN-TREE: 15 nodes
  Supersteps: 4
  Tasks per processor: {'0': 4, '1': 3, '2': 2, '3': 2, '4': 1, '5': 1, '6': 1, '7': 1}  # Distributed
```

### Benchmark Results

From `benchmark2` experiments comparing out-tree vs in-tree performance:

| Metric | Out-tree | In-tree |
|--------|----------|---------|
| Mean makespan ratio | **24.32x** | 1.09x |
| Median makespan ratio | 14.04x | 1.07x |
| Max makespan ratio | 113.61x | 1.25x |

**Performance gap:** BSPg scheduler is **22x worse on out-trees compared to in-trees**.

### Comparison with Source Scheduler

Both schedulers fail on out-trees but through different mechanisms:

| Scheduler | Mechanism | Effect |
|-----------|-----------|--------|
| **Source** | Greedy successor addition: adds successors if all predecessors on same processor | Entire tree added to processor 0 in one superstep |
| **BSPg** | `ready_p` priority: processor prefers local-data nodes over `ready_all` | Processor 0 always has local work, never yields |

Both result in **zero parallelism** - all work on one processor regardless of available processor count.

### Paper's Acknowledgment

The paper does not explicitly acknowledge this limitation. Section 4.5 (page 6) notes a related issue:

> "While our algorithms above perform well in general, we have found that they are often unable to find good solutions in problems dominated by communication costs... both our initialization heuristics and our local search algorithms attempt to (re)schedule single nodes separately, so they do not perform well in this case."

This confirms BSPg is a greedy heuristic that can get trapped in poor local decisions.

### Why In-Trees Work Better

In-trees avoid the processor hogging because:
1. Many sources exist initially (the leaves) → distributed across processors via `ready_all`
2. Successors have predecessors on **different processors** → go into `ready_all`, not `ready_p`
3. Work distribution is maintained throughout execution

### Conclusion

The BSPg scheduler's locality optimization (preferring `ready_p` over `ready_all`) backfires on out-trees where the single root creates a cascade of local-only dependencies. This is **intended behavior** per the algorithm design (locality is generally good), but the **pathological case is not documented** in the paper.

**Recommendation:** Do not use BSPgScheduler on out-tree or similar narrow-top graph structures. Use HeftBSPScheduler, HDaggScheduler, or BALSScheduler instead.
