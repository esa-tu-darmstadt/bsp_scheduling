"""BSPg Scheduler - Algorithm 1 from Papp et al. 2024

This implements the greedy BSP scheduling algorithm (BSPg) that:
- Consecutively assigns nodes to processors when processors become idle
- Maintains ready_p (nodes only assignable to processor p) and ready_all (nodes for any processor)
- Closes superstep when ≥P/2 processors are idle without assignable nodes
- Uses tie-breaking score based on communication costs

Reference: Section 4.2, Algorithm 1 (Appendix A.2, page 14)
"""

from collections import defaultdict
from typing import Dict, Set, List, Optional, Tuple
import heapq
import networkx as nx

from ..base import BSPScheduler
from ...schedule import BSPSchedule, BSPHardware


class BSPgScheduler(BSPScheduler):
    """BSPg greedy scheduler from Papp et al. 2024.

    Algorithm outline:
    1. Maintain ready_p[p] = nodes whose predecessors are all on processor p or earlier supersteps
    2. Maintain ready_all = nodes that can be assigned to any processor
    3. When processor p becomes free, assign from ready_p[p] first, then ready_all
    4. Use tie-breaking score: Σ c(u)/outdeg(u) for relevant predecessors
    5. Close superstep when ≥P/2 processors are idle without assignable nodes
    """

    def __init__(self, verbose: bool = False, optimized: bool = True):
        """Initialize the BSPg scheduler.

        Args:
            verbose: Print progress information
            optimized: Use optimized implementation with min-heap for finish_times
                       and incremental idle count tracking. This is our optimization
                       (not from the paper) that provides better performance on large
                       graphs. Results should be identical to the standard implementation.
        """
        super().__init__()
        self.name = "BSPg"
        self.verbose = verbose
        self.optimized = optimized

    def schedule(self, hardware: BSPHardware, task_graph: nx.DiGraph) -> BSPSchedule:
        """Schedule tasks using BSPg algorithm."""
        if self.optimized:
            return self._schedule_optimized(hardware, task_graph)
        else:
            return self._schedule_standard(hardware, task_graph)

    def _schedule_standard(self, hardware: BSPHardware, task_graph: nx.DiGraph) -> BSPSchedule:
        """Schedule tasks using BSPg algorithm."""
        schedule = BSPSchedule(hardware, task_graph)

        num_processors = len(hardware.network.nodes())
        processors = list(hardware.network.nodes())

        # Track processor states
        # finish_times[p] = time when processor p becomes free
        finish_times: Dict[str, float] = {p: 0.0 for p in processors}

        # Track which nodes are assigned
        assigned: Set[str] = set()

        # Ready sets: ready_p[p] = nodes only assignable to p in current superstep
        #             ready_all = nodes assignable to any processor (no cross-proc deps in current superstep)
        ready_p: Dict[str, Set[str]] = {p: set() for p in processors}
        ready_all: Set[str] = set()

        # Track processor assignments (processor -> superstep -> list of tasks)
        proc_assignments: Dict[str, int] = {}  # node -> processor
        superstep_assignments: Dict[str, int] = {}  # node -> superstep index

        # Initialize: find source nodes (no predecessors)
        ready = set()
        for node in task_graph.nodes():
            if task_graph.in_degree(node) == 0:
                ready.add(node)
        ready_all = ready.copy()

        current_superstep_idx = 0
        current_superstep = schedule.add_superstep()
        end_step = False

        while len(assigned) < len(task_graph.nodes()):
            if end_step and all(finish_times[p] == float('inf') for p in processors):
                # Start new superstep
                ready_p = {p: set() for p in processors}
                ready_all = ready.copy()
                current_superstep_idx += 1
                current_superstep = schedule.add_superstep()
                end_step = False
                finish_times = {p: 0.0 for p in processors}

            # Find the earliest time when a processor becomes free
            min_finish_time = min(finish_times.values())

            # Find all processors that finish at this time
            free_processors = [p for p in processors if finish_times[p] == min_finish_time]

            # Try to assign nodes to free processors
            nodes_assigned_this_step = False

            for proc in free_processors:
                if end_step:
                    # In end_step mode, only assign from ready_p (not ready_all)
                    node = self._choose_node(proc, ready_p[proc], set(), task_graph, proc_assignments)
                else:
                    node = self._choose_node(proc, ready_p[proc], ready_all, task_graph, proc_assignments)

                if node is not None:
                    # Assign node to processor in current superstep
                    schedule.schedule(node, proc, current_superstep)
                    assigned.add(node)
                    proc_assignments[node] = proc
                    superstep_assignments[node] = current_superstep_idx

                    # Remove from ready sets
                    ready_p[proc].discard(node)
                    ready_all.discard(node)
                    ready.discard(node)

                    # Update finish time
                    work_weight = task_graph.nodes[node]['weight']
                    proc_speed = hardware.network.nodes[proc]['weight']
                    duration = work_weight / proc_speed
                    finish_times[proc] = min_finish_time + duration

                    nodes_assigned_this_step = True

                    # Update ready sets for successors
                    for succ in task_graph.successors(node):
                        if succ in assigned:
                            continue

                        # Check if all predecessors are assigned
                        all_preds_assigned = all(
                            pred in assigned for pred in task_graph.predecessors(succ)
                        )

                        if all_preds_assigned:
                            ready.add(succ)

                            # Check if succ can be scheduled in current superstep on proc
                            # (all preds must be on proc in current superstep or earlier supersteps)
                            can_schedule_on_proc = True
                            all_preds_on_proc_or_earlier = True

                            for pred in task_graph.predecessors(succ):
                                pred_ss = superstep_assignments[pred]
                                pred_proc = proc_assignments[pred]

                                if pred_ss == current_superstep_idx:
                                    # Pred in current superstep - must be on same proc
                                    if pred_proc != proc:
                                        all_preds_on_proc_or_earlier = False
                                        break

                            if all_preds_on_proc_or_earlier:
                                ready_p[proc].add(succ)
                            else:
                                # Has predecessors on multiple processors in current superstep
                                # Will be available in ready_all after superstep ends
                                pass
                else:
                    # Processor cannot be assigned a node
                    finish_times[proc] = float('inf')  # Mark as idle

            # Check if we should close the superstep
            idle_count = sum(1 for p in processors if finish_times[p] == float('inf'))
            if not end_step and ready_all == set() and idle_count >= num_processors // 2:
                end_step = True

        # Merge supersteps where possible
        schedule.merge_supersteps()

        schedule.assert_valid()
        return schedule

    def _schedule_optimized(self, hardware: BSPHardware, task_graph: nx.DiGraph) -> BSPSchedule:
        """Optimized BSPg scheduling with heap-based processor management.

        Optimizations over standard implementation (not from the paper):
        - Uses min-heap for finish_times: O(log P) updates instead of O(P) min search
        - Tracks idle count incrementally instead of O(P) count each iteration
        - Caches processor speeds to avoid repeated dict lookups
        """
        schedule = BSPSchedule(hardware, task_graph)

        num_processors = len(hardware.network.nodes())
        processors = list(hardware.network.nodes())

        # Cache processor speeds for faster lookup
        proc_speeds = {p: hardware.network.nodes[p]['weight'] for p in processors}

        # Track processor states using a min-heap for efficient min-time lookup
        # Heap entries: (finish_time, processor_id)
        # Also maintain a dict for O(1) lookup of current finish time per processor
        finish_times: Dict[str, float] = {p: 0.0 for p in processors}
        # Initial heap: all processors available at time 0
        finish_heap: List[Tuple[float, str]] = [(0.0, p) for p in processors]
        heapq.heapify(finish_heap)

        # Track idle processors incrementally
        idle_count = 0

        # Track which nodes are assigned
        assigned: Set[str] = set()

        # Ready sets
        ready_p: Dict[str, Set[str]] = {p: set() for p in processors}
        ready_all: Set[str] = set()

        # Track processor assignments
        proc_assignments: Dict[str, str] = {}  # node -> processor
        superstep_assignments: Dict[str, int] = {}  # node -> superstep index

        # Initialize: find source nodes (no predecessors)
        ready = set()
        for node in task_graph.nodes():
            if task_graph.in_degree(node) == 0:
                ready.add(node)
        ready_all = ready.copy()

        current_superstep_idx = 0
        current_superstep = schedule.add_superstep()
        end_step = False

        total_nodes = len(task_graph.nodes())

        while len(assigned) < total_nodes:
            if end_step and idle_count == num_processors:
                # Start new superstep - all processors are idle
                ready_p = {p: set() for p in processors}
                ready_all = ready.copy()
                current_superstep_idx += 1
                current_superstep = schedule.add_superstep()
                end_step = False
                # Reset finish times and heap
                finish_times = {p: 0.0 for p in processors}
                finish_heap = [(0.0, p) for p in processors]
                heapq.heapify(finish_heap)
                idle_count = 0

            # Find the earliest time when a processor becomes free
            # Pop stale entries from heap (where heap time doesn't match current finish_times)
            while finish_heap:
                min_time, min_proc = finish_heap[0]
                if finish_times[min_proc] == min_time:
                    break
                heapq.heappop(finish_heap)  # Stale entry, remove it

            if not finish_heap:
                # All processors are idle (shouldn't happen if idle_count is correct)
                break

            min_finish_time = finish_heap[0][0]

            # Find all processors that finish at this time
            # Pop all entries with min_finish_time
            free_processors = []
            while finish_heap and finish_heap[0][0] == min_finish_time:
                t, proc = heapq.heappop(finish_heap)
                if finish_times[proc] == t:  # Not stale
                    free_processors.append(proc)

            # Try to assign nodes to free processors
            for proc in free_processors:
                if end_step:
                    node = self._choose_node(proc, ready_p[proc], set(), task_graph, proc_assignments)
                else:
                    node = self._choose_node(proc, ready_p[proc], ready_all, task_graph, proc_assignments)

                if node is not None:
                    # Assign node to processor in current superstep
                    schedule.schedule(node, proc, current_superstep)
                    assigned.add(node)
                    proc_assignments[node] = proc
                    superstep_assignments[node] = current_superstep_idx

                    # Remove from ready sets
                    ready_p[proc].discard(node)
                    ready_all.discard(node)
                    ready.discard(node)

                    # Update finish time using cached proc speed
                    work_weight = task_graph.nodes[node]['weight']
                    duration = work_weight / proc_speeds[proc]
                    new_finish_time = min_finish_time + duration
                    finish_times[proc] = new_finish_time
                    heapq.heappush(finish_heap, (new_finish_time, proc))

                    # Update ready sets for successors
                    for succ in task_graph.successors(node):
                        if succ in assigned:
                            continue

                        # Check if all predecessors are assigned
                        all_preds_assigned = all(
                            pred in assigned for pred in task_graph.predecessors(succ)
                        )

                        if all_preds_assigned:
                            ready.add(succ)

                            # Check if succ can be scheduled in current superstep on proc
                            all_preds_on_proc_or_earlier = True

                            for pred in task_graph.predecessors(succ):
                                pred_ss = superstep_assignments[pred]
                                pred_proc = proc_assignments[pred]

                                if pred_ss == current_superstep_idx:
                                    if pred_proc != proc:
                                        all_preds_on_proc_or_earlier = False
                                        break

                            if all_preds_on_proc_or_earlier:
                                ready_p[proc].add(succ)
                else:
                    # Processor cannot be assigned a node - mark as idle
                    finish_times[proc] = float('inf')
                    idle_count += 1
                    # Don't push inf to heap - we track idle count separately

            # Check if we should close the superstep
            if not end_step and not ready_all and idle_count >= num_processors // 2:
                end_step = True

        # Merge supersteps where possible
        schedule.merge_supersteps()

        schedule.assert_valid()
        return schedule

    def _choose_node(
        self,
        processor: str,
        ready_p: Set[str],
        ready_all: Set[str],
        task_graph: nx.DiGraph,
        proc_assignments: Dict[str, str]
    ) -> Optional[str]:
        """Choose the best node to assign to a processor.

        First tries ready_p (nodes only available for this processor),
        then ready_all (nodes available for any processor).
        Uses tie-breaking score based on communication costs.

        Args:
            processor: The processor to assign to
            ready_p: Nodes only assignable to this processor
            ready_all: Nodes assignable to any processor
            task_graph: The task graph
            proc_assignments: Current processor assignments

        Returns:
            Best node to assign, or None if no node available
        """
        # First try ready_p
        if ready_p:
            return self._select_best_node(ready_p, processor, task_graph, proc_assignments)

        # Then try ready_all
        if ready_all:
            return self._select_best_node(ready_all, processor, task_graph, proc_assignments)

        return None

    def _select_best_node(
        self,
        candidates: Set[str],
        processor: str,
        task_graph: nx.DiGraph,
        proc_assignments: Dict[str, str]
    ) -> str:
        """Select the best node from candidates using tie-breaking score.

        Score for node v: Σ c(u)/outdeg(u) for predecessors u where
        u or one of its successors is on processor p.

        Higher score = more likely to save communication costs.
        """
        best_node = None
        best_score = -1.0

        for node in candidates:
            score = self._compute_tie_breaking_score(node, processor, task_graph, proc_assignments)
            if score > best_score:
                best_score = score
                best_node = node

        return best_node

    def _compute_tie_breaking_score(
        self,
        node: str,
        processor: str,
        task_graph: nx.DiGraph,
        proc_assignments: Dict[str, str]
    ) -> float:
        """Compute tie-breaking score for assigning node to processor.

        Score = Σ c(u)/outdeg(u) for predecessors u where:
        - u is assigned to processor, OR
        - one of u's successors is assigned to processor

        This estimates the potential to save communication costs.

        Per Papp et al. 2024: c(u) is the communication weight (edge weight).
        We use the average outgoing edge weight (edges are uniform per source).
        """
        score = 0.0

        for pred in task_graph.predecessors(node):
            outdeg = task_graph.out_degree(pred)
            if outdeg == 0:
                continue

            # c(u) = average communication weight of outgoing edges from pred
            comm_weight = sum(
                task_graph.edges[pred, succ]['weight']
                for succ in task_graph.successors(pred)
            ) / outdeg

            # Check if pred or any of its successors is on processor
            pred_on_proc = proc_assignments.get(pred) == processor
            succ_on_proc = any(
                proc_assignments.get(s) == processor
                for s in task_graph.successors(pred)
            )

            if pred_on_proc or succ_on_proc:
                score += comm_weight / outdeg

        return score
