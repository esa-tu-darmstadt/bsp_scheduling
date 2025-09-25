"""BCSH: Bulk Communication Scheduling Heuristic

Implements the algorithm described in the paper
"A Task Scheduling Algorithm to Package Messages on Distributed Memory Parallel Machines"
by Fujimoto et al. (BCSH).

Notes on interpretation for this implementation:
- The user can choose between the original implementation for placing node groups on processors (LDSH)
or our custom implementation based on a earliest-finishing time heuristic (EFT).
LDSH ignores processor heterogeneity, while EFT takes it into account.
- Levels: computed by number of nodes (topological distance to exit), not weighted path length.

The algorithm proceeds in two main phases:
1) Grouping (Algorithm 4.4): build independent group layers bottom-up using duplication and
   controlled merging to reduce excessive duplication while maintaining balance.
2) Layer scheduling (Algorithm 4.3): choose an effective number of processors per layer (processor
   saving) and schedule each node group entirely on a single processor in topological order.

The resulting schedule is bulk synchronous: one superstep per group layer, no inter-processor
dependencies within a superstep, and all communication happens between supersteps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Sequence, Set, Tuple, Optional

import networkx as nx
import logging

from .base import BSPScheduler
from ..schedule import BSPSchedule, BSPHardware

logger = logging.getLogger(__name__)


# ------------------------------
# Utility functions (pure, local)
# ------------------------------


def _node_weight(G: nx.DiGraph, u: Hashable) -> float:
    return float(G.nodes[u].get("weight", 1.0))


def _edge_weight(G: nx.DiGraph, u: Hashable, v: Hashable) -> float:
    return float(G.edges[(u, v)].get("weight", 0.0))


def _tau_max(G: nx.DiGraph) -> float:
    if G.number_of_edges() == 0:
        return 0.0
    return max(_edge_weight(G, u, v) for u, v in G.edges())


def _omega_of_nodes(G: nx.DiGraph, nodes: Iterable[Hashable]) -> float:
    return sum(_node_weight(G, u) for u in nodes)


def _levels_by_node_count(G: nx.DiGraph) -> Dict[Hashable, int]:
    """Compute levels measured by number of nodes to an exit (exit nodes at level 1).

    level(u) = 1 + max(level(v) for v in Succ(u)) if u has successors, else 1.
    """
    # Post-order over reverse graph to ensure successors processed before predecessors
    R = G.reverse(copy=False)
    order = list(
        nx.topological_sort(R)
    )  # This is a reverse-topo order of G (exits first)
    level: Dict[Hashable, int] = {}
    for u in order:
        succs = list(G.successors(u))
        if not succs:
            level[u] = 1
        else:
            level[u] = 1 + max(level[v] for v in succs)
    return level


def _V_levels(
    G: nx.DiGraph, level_map: Dict[Hashable, int]
) -> Dict[int, List[Hashable]]:
    buckets: Dict[int, List[Hashable]] = {}
    for u, l in level_map.items():
        buckets.setdefault(l, []).append(u)
    return buckets


def _LM_LDSH(weights: Sequence[float], p: int) -> float:
    """Graham's LDSH (LPT) makespan for a multiset of job weights on p identical processors.

    Args:
        weights: job (group) weights
        p: number of processors

    Returns:
        Makespan (max processor load)
    """
    if p <= 0:
        raise ValueError("Number of processors must be positive")
    if not weights:
        return 0.0
    # Sort descending as LPT
    sorted_w = sorted(weights, reverse=True)
    loads = [0.0 for _ in range(p)]
    for w in sorted_w:
        # Assign to the least-loaded processor
        idx = min(range(p), key=lambda i: loads[i])
        loads[idx] += w
    return max(loads)


def _LM_LDSH_with_assignment(
    weights: Sequence[float], p: int
) -> Tuple[float, List[int]]:
    """LDSH makespan and assignment indices (processor indices 0..p-1) for each weight in the
    original order of `weights`.
    """
    if p <= 0:
        raise ValueError("Number of processors must be positive")
    n = len(weights)
    if n == 0:
        return 0.0, []
    # Use stable indices to map back to original order
    items = list(enumerate(weights))
    items.sort(key=lambda x: x[1], reverse=True)  # sort by weight desc
    loads = [0.0 for _ in range(p)]
    assignment = [0 for _ in range(n)]
    for idx, w in items:
        proc = min(range(p), key=lambda i: loads[i])
        assignment[idx] = proc
        loads[proc] += w
    return max(loads), assignment


def _choose_P_processor_saving(
    weights: Sequence[float], p_max: int
) -> Tuple[int, float, List[int]]:
    """Choose P in [ceil(W/wmax), p_max] minimizing LM, and return (P, LM, assignment) for that P.
    Assignment is per weight to processor index in range [0..P-1].
    """
    if not weights:
        return 1, 0.0, []
    W = sum(weights)
    wmax = max(weights)
    lower = max(1, int((W + wmax - 1) // wmax))  # ceil(W/wmax)
    upper = max(lower, p_max)
    best_P = lower
    best_LM = float("inf")
    best_assign: List[int] = []
    for P in range(lower, p_max + 1):
        LM, assign = _LM_LDSH_with_assignment(weights, P)
        if LM < best_LM - 1e-12:  # stability against FP
            best_LM = LM
            best_P = P
            best_assign = assign
    return best_P, best_LM, best_assign


def _EFT_scheduling(
    groups: List[Set[Hashable]],
    group_weights: List[float],
    processors: List[Hashable],
    processor_speeds: Dict[Hashable, float],
) -> Tuple[float, List[Hashable]]:
    """Schedule groups using earliest finish time (EFT) approach considering processor speeds.

    Returns:
        (makespan, assignment) where assignment maps each group to a processor
    """
    n_groups = len(groups)
    if n_groups == 0:
        return 0.0, []

    # Track when each processor will be available (finish time)
    processor_finish_times = {p: 0.0 for p in processors}
    assignment = [None] * n_groups

    # Sort groups by weight descending (schedule larger groups first)
    sorted_indices = sorted(
        range(n_groups), key=lambda i: group_weights[i], reverse=True
    )

    for idx in sorted_indices:
        weight = group_weights[idx]

        # Find processor that will finish this group earliest
        best_proc = None
        best_finish_time = float("inf")

        for proc in processors:
            # Calculate when this group would finish on this processor
            speed = processor_speeds[proc]
            duration = weight / speed
            finish_time = processor_finish_times[proc] + duration

            if finish_time < best_finish_time:
                best_finish_time = finish_time
                best_proc = proc

        # Assign group to best processor
        assignment[idx] = best_proc
        processor_finish_times[best_proc] = best_finish_time

    # Makespan is the maximum finish time
    makespan = max(processor_finish_times.values())
    return makespan, assignment


def _choose_P_processor_saving_EFT(
    groups: List[Set[Hashable]],
    weights: Sequence[float],
    processors: List[Hashable],
    processor_speeds: Dict[Hashable, float],
) -> Tuple[int, float, List[Hashable]]:
    """Choose P processors minimizing makespan using EFT, with processor saving.

    Returns:
        (P, makespan, assignment) where assignment maps each group to a processor
    """
    if not weights:
        return 1, 0.0, []

    p_max = len(processors)
    W = sum(weights)
    wmax = max(weights)

    # Consider average speed for lower bound calculation
    avg_speed = (
        sum(processor_speeds.values()) / len(processor_speeds)
        if processor_speeds
        else 1.0
    )
    lower = max(1, int((W / avg_speed) / (wmax / avg_speed)))  # Adjusted for speed
    lower = min(lower, p_max)

    best_P = lower
    best_makespan = float("inf")
    best_assign: List[Hashable] = []

    # Try different numbers of processors
    for P in range(lower, p_max + 1):
        active_procs = processors[:P]
        active_speeds = {p: processor_speeds[p] for p in active_procs}

        makespan, assign = _EFT_scheduling(
            groups, list(weights), active_procs, active_speeds
        )

        if makespan < best_makespan - 1e-12:
            best_makespan = makespan
            best_P = P
            best_assign = assign

    return best_P, best_makespan, best_assign


# ------------------------------
# Grouping (Algorithm 4.4)
# ------------------------------


def _group_layers(
    G: nx.DiGraph, p: int, verbose: bool = False
) -> List[List[Set[Hashable]]]:
    """Implement Algorithm 4.4 to construct group layers.

    Returns a list of layers, each a list of node-groups (sets of task ids).
    The first element in the returned list corresponds to L1 in the paper (closest to exits).
    """
    # Precompute helpers
    levels = _levels_by_node_count(G)
    V_levels = _V_levels(G, levels)
    lmax = max(levels.values(), default=0)
    tau_max = _tau_max(G)
    succ_cache: Dict[Hashable, Set[Hashable]] = {
        u: set(G.successors(u)) for u in G.nodes
    }

    def layer_weight(groups: List[Set[Hashable]]) -> float:
        return sum(_omega_of_nodes(G, C) for C in groups)

    def LM_of_groups(groups: List[Set[Hashable]], p_: int) -> float:
        weights = [_omega_of_nodes(G, C) for C in groups]
        return _LM_LDSH(weights, p_)

    def extend_by_level(L: List[Set[Hashable]], level_idx: int) -> List[Set[Hashable]]:
        # Duplicate new level's nodes into groups that contain any of their successors
        CC = [set(C) for C in L]  # deep-copy
        for u in V_levels.get(level_idx, []):
            S = succ_cache[u]
            if not S:
                continue
            for C in CC:
                if C & S:
                    C.add(u)
        if verbose:
            logger.debug(f"Extended to level {level_idx}: |groups|={len(CC)}")
        # Reduce excessive duplication by merging pairs
        LM_threshold = LM_of_groups(CC, p)
        if verbose:
            logger.debug(
                f"Initial LM threshold at level {level_idx}: {LM_threshold:.3f}"
            )
        # Repeat merging until no merge occurs in a full pass
        merged_in_pass = True
        while merged_in_pass:
            merged_in_pass = False
            # Sort groups by weight increasing
            CC_sorted = sorted(CC, key=lambda C: _omega_of_nodes(G, C))
            # We'll operate on the working list CC; map sorted order to actual instances
            for Ci in list(CC_sorted):
                if Ci not in CC:
                    continue  # may have been merged/removed already
                # Build list of other groups with overlap weight to Ci
                candidates = [C for C in CC if C is not Ci]
                # Sort by overlap weight decreasing
                candidates.sort(key=lambda D: _omega_of_nodes(G, D & Ci), reverse=True)
                # Try to merge Ci into the best-overlap candidate that keeps union under LM_threshold
                for Dj in candidates:
                    union = Dj | Ci
                    if _omega_of_nodes(G, union) < LM_threshold - 1e-12:
                        # Replace Dj with union, remove Ci
                        CC.remove(Dj)
                        CC.remove(Ci)
                        CC.append(union)
                        merged_in_pass = True
                        if verbose:
                            logger.debug(
                                f"Merged groups (|Ci|={len(Ci)}, |Dj|={len(Dj)}) -> |union|={len(union)}"
                            )
                        break
                if merged_in_pass:
                    break
        return CC

    # Build layers from exits upward
    layers: List[List[Set[Hashable]]] = []
    l = 1
    while l <= lmax:
        # Start a tentative layer with nodes exactly at level l
        L: List[Set[Hashable]] = [set([v]) for v in V_levels.get(l, [])]
        if not L:
            # No nodes at this level (shouldn't happen), skip
            l += 1
            continue
        current_top = l
        # Try to bring up L by including higher levels while balanced (or few nodes remain)
        while current_top < lmax:
            next_level = current_top + 1
            CC = extend_by_level(L, next_level)
            # Remaining work beyond next_level
            # t := ω(V(next_level+1 .. lmax))
            remain_nodes: List[Hashable] = []
            for lev in range(next_level + 1, lmax + 1):
                remain_nodes.extend(V_levels.get(lev, []))
            t_remain = _omega_of_nodes(G, remain_nodes)
            if t_remain >= (t_remain / max(1, p)) + tau_max - 1e-12:
                # Enforce balancedness
                LM_val = LM_of_groups(CC, p)
                W_val = layer_weight(CC)
                if LM_val <= (W_val / max(1, p)) + tau_max + 1e-12:
                    # Accept extension
                    L = CC
                    current_top = next_level
                    if verbose:
                        logger.debug(
                            f"Accepted balanced extension to level {current_top}: LM={LM_val:.3f}, W/p+tau={W_val/p+tau_max:.3f}"
                        )
                    continue
                else:
                    # Not balanced; stop extending this layer
                    if verbose:
                        logger.debug(
                            f"Rejected extension at level {next_level}: LM={LM_val:.3f} > W/p+tau={W_val/p+tau_max:.3f}"
                        )
                    break
            else:
                # Few nodes remain; don't force a comm phase yet—accept extension unconditionally
                L = CC
                current_top = next_level
                if verbose:
                    logger.debug(
                        f"Accepted extension (few remain) to level {current_top}: t_remain={t_remain:.3f} < t_remain/p+tau={t_remain/max(1,p)+tau_max:.3f}"
                    )
                continue
        # Finalize this layer
        layers.append(L)
        if verbose:
            logger.debug(
                f"Finalized layer with top level {current_top}: |groups|={len(L)}, layer_weight={layer_weight(L):.3f}"
            )
        l = current_top + 1
    return layers


# ------------------------------
# Layer scheduling (Algorithm 4.3) and BCSH scheduler
# ------------------------------


def _topo_order_within_group(
    G: nx.DiGraph, group_nodes: Set[Hashable]
) -> List[Hashable]:
    """Return a topological order of nodes restricted to `group_nodes`.

    Uses the global topological order filtered to group nodes to respect inter-group dependencies.
    """
    global_order = list(nx.topological_sort(G))
    group_set = set(group_nodes)
    return [u for u in global_order if u in group_set]


@dataclass
class BCSHScheduler(BSPScheduler):
    """BCSH scheduler: produces BSP schedules with bulk synchronous supersteps.

    This class implements the paper's Algorithm 4.4 (grouping) and Algorithm 4.3 (layer scheduling).

    Args:
        verbose: Enable debug logging
        use_eft: Use Earliest Finish Time scheduling instead of LDSH (considers processor speeds)
    """

    verbose: bool = False
    use_eft: bool = False

    @property
    def name(self) -> str:
        return "BCSH+" + ("EFT" if self.use_eft else "LDSH")

    def schedule(self, hardware: BSPHardware, task_graph: nx.DiGraph) -> BSPSchedule:
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        # p = number of available processors
        p = len(hardware.network.nodes)
        if p <= 0:
            raise ValueError("Hardware must have at least one processor")

        schedule = BSPSchedule(hardware, task_graph)
        if self.verbose:
            logger.debug("Starting BCSH scheduling")

        # 1) Build group layers (L1 closest to exits)
        layers = _group_layers(task_graph, p, verbose=self.verbose)
        if self.verbose:
            logger.debug(f"Constructed {len(layers)} group layers")

        # 2) Schedule each layer as one superstep, from sources to exits
        #    The paper defines L1 < L2 < ... < Ln (Ji < Jj means Ji depends on Jj),
        #    so we execute in order Ln, Ln-1, ..., L1.
        processors_all = list(hardware.network.nodes)

        # Get processor speeds if using EFT
        processor_speeds = None
        if self.use_eft:
            processor_speeds = {
                p: hardware.network.nodes[p].get("weight", 1.0) for p in processors_all
            }

        for layer_idx, layer_groups in enumerate(reversed(layers), start=1):
            superstep = schedule.add_superstep()

            # Algorithm 4.3: processor saving and scheduling of node groups
            group_weights = [_omega_of_nodes(task_graph, C) for C in layer_groups]

            if self.use_eft:
                # Use EFT-based scheduling with processor speeds
                P, _, assignment = _choose_P_processor_saving_EFT(
                    layer_groups, group_weights, processors_all, processor_speeds
                )
                active_procs = processors_all[:P]
            else:
                # Use original LDSH-based scheduling
                P, _, assignment = _choose_P_processor_saving(group_weights, p)
                active_procs = processors_all[:P]

            if self.verbose:
                method = "EFT" if self.use_eft else "LDSH"
                logger.debug(
                    f"Layer {layer_idx}: |groups|={len(layer_groups)}, W={sum(group_weights):.3f}, P={P}, method={method}"
                )

            # Schedule groups according to assignment
            for idx, C in enumerate(layer_groups):
                if self.use_eft:
                    # EFT returns processor names directly
                    proc = assignment[idx] if assignment else processors_all[0]
                else:
                    # LDSH returns processor indices
                    proc_local_idx = assignment[idx] if assignment else 0
                    proc = active_procs[proc_local_idx]

                # Topologically order tasks within the group
                ordered_nodes = _topo_order_within_group(task_graph, C)
                for node in ordered_nodes:
                    schedule.schedule(node, proc, superstep)
                if self.verbose:
                    logger.debug(
                        f"  Scheduled group of size {len(C)} on proc {proc}: tasks={len(ordered_nodes)}"
                    )

        # Validate BSP constraints
        schedule.assert_valid()
        if self.verbose:
            logger.debug(
                f"Finished BCSH scheduling with {schedule.num_supersteps} supersteps, makespan={schedule.makespan:.3f}"
            )
        return schedule
