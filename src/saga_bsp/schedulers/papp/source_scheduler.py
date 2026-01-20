"""Source Scheduler - Algorithm 2 from Papp et al. 2024

This implements the Source scheduling algorithm that:
- Forms supersteps from source node layers
- First superstep: clusters sources sharing common successors, round-robin assignment
- Subsequent supersteps: sorts by work weight descending, round-robin assignment
- After each superstep: adds successors if ALL predecessors are on same processor

Reference: Section 4.2, Algorithm 2 (Appendix A.2, page 15)
"""

from collections import defaultdict
from typing import Dict, Set, List, Optional, Tuple
import networkx as nx

from ..base import BSPScheduler
from ...schedule import BSPSchedule, BSPHardware


class SourceScheduler(BSPScheduler):
    """Source-layer based scheduler from Papp et al. 2024.

    Algorithm outline:
    1. Process DAG layer by layer (source nodes at each step)
    2. First superstep: cluster sources by common successors, round-robin assign
    3. Later supersteps: sort sources by work weight descending, round-robin assign
    4. After round-robin: add successors whose ALL predecessors are on same processor
    """

    def __init__(self, verbose: bool = False):
        super().__init__()
        self.name = "Source"
        self.verbose = verbose

    def schedule(self, hardware: BSPHardware, task_graph: nx.DiGraph) -> BSPSchedule:
        """Schedule tasks using Source algorithm."""
        schedule = BSPSchedule(hardware, task_graph)

        num_processors = len(hardware.network.nodes())
        processors = list(hardware.network.nodes())

        # Track assigned nodes
        assigned: Set[str] = set()

        # Track processor assignments: node -> processor
        proc_assignments: Dict[str, str] = {}

        # Track superstep assignments: node -> superstep index
        superstep_assignments: Dict[str, int] = {}

        # Create working copy of graph to track remaining nodes
        remaining_nodes = set(task_graph.nodes())

        superstep_idx = 0

        while remaining_nodes:
            # Find current source nodes (no unassigned predecessors)
            sources = self._compute_sources(remaining_nodes, assigned, task_graph)

            if not sources:
                # No sources found - shouldn't happen in valid DAG
                raise RuntimeError("No source nodes found but remaining nodes exist")

            # Create new superstep
            current_superstep = schedule.add_superstep()
            superstep_idx = current_superstep.index

            if superstep_idx == 0:
                # First superstep: cluster sources by common successors
                proc_assignment = self._assign_with_clustering(
                    sources, task_graph, processors
                )
            else:
                # Later supersteps: sort by work weight descending, round-robin
                proc_assignment = self._assign_round_robin_by_weight(
                    sources, task_graph, processors
                )

            # Schedule the assigned nodes
            for node, proc in proc_assignment.items():
                schedule.schedule(node, proc, current_superstep)
                assigned.add(node)
                remaining_nodes.discard(node)
                proc_assignments[node] = proc
                superstep_assignments[node] = superstep_idx

            # Try to add successors if all their predecessors are on same processor
            additional_nodes = self._find_addable_successors(
                list(proc_assignment.keys()),
                remaining_nodes,
                assigned,
                proc_assignments,
                task_graph
            )

            for node, proc in additional_nodes.items():
                schedule.schedule(node, proc, current_superstep)
                assigned.add(node)
                remaining_nodes.discard(node)
                proc_assignments[node] = proc
                superstep_assignments[node] = superstep_idx

        # Merge supersteps where possible
        schedule.merge_supersteps()

        schedule.assert_valid()
        return schedule

    def _compute_sources(
        self,
        remaining_nodes: Set[str],
        assigned: Set[str],
        task_graph: nx.DiGraph
    ) -> Set[str]:
        """Find current source nodes (nodes with all predecessors assigned).

        Args:
            remaining_nodes: Nodes not yet assigned
            assigned: Nodes already assigned
            task_graph: The task graph

        Returns:
            Set of source nodes (nodes whose predecessors are all assigned)
        """
        sources = set()
        for node in remaining_nodes:
            all_preds_assigned = all(
                pred in assigned for pred in task_graph.predecessors(node)
            )
            if all_preds_assigned:
                sources.add(node)
        return sources

    def _assign_with_clustering(
        self,
        sources: Set[str],
        task_graph: nx.DiGraph,
        processors: List[str]
    ) -> Dict[str, str]:
        """Assign sources with clustering for first superstep.

        Cluster sources that share common successors, then round-robin assign clusters.

        Args:
            sources: Source nodes to assign
            task_graph: The task graph
            processors: List of processor IDs

        Returns:
            Dict mapping node -> processor
        """
        # Build clusters: nodes sharing successors go together
        # Use union-find approach
        parent = {node: node for node in sources}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Build successor -> sources mapping
        succ_to_sources: Dict[str, List[str]] = defaultdict(list)
        for node in sources:
            for succ in task_graph.successors(node):
                succ_to_sources[succ].append(node)

        # Union sources that share a successor
        for succ, src_list in succ_to_sources.items():
            if len(src_list) > 1:
                for i in range(1, len(src_list)):
                    union(src_list[0], src_list[i])

        # Group nodes by cluster
        clusters: Dict[str, List[str]] = defaultdict(list)
        for node in sources:
            clusters[find(node)].append(node)

        # Assign clusters to processors round-robin
        assignment: Dict[str, str] = {}
        proc_idx = 0

        for cluster_id, cluster_nodes in clusters.items():
            for node in cluster_nodes:
                assignment[node] = processors[proc_idx]
            proc_idx = (proc_idx + 1) % len(processors)

        return assignment

    def _assign_round_robin_by_weight(
        self,
        sources: Set[str],
        task_graph: nx.DiGraph,
        processors: List[str]
    ) -> Dict[str, str]:
        """Assign sources round-robin, sorted by work weight descending.

        Args:
            sources: Source nodes to assign
            task_graph: The task graph
            processors: List of processor IDs

        Returns:
            Dict mapping node -> processor
        """
        # Sort by work weight descending
        sorted_sources = sorted(
            sources,
            key=lambda n: task_graph.nodes[n].get('weight', 1.0),
            reverse=True
        )

        # Round-robin assignment
        assignment: Dict[str, str] = {}
        for idx, node in enumerate(sorted_sources):
            proc = processors[idx % len(processors)]
            assignment[node] = proc

        return assignment

    def _find_addable_successors(
        self,
        just_assigned: List[str],
        remaining_nodes: Set[str],
        assigned: Set[str],
        proc_assignments: Dict[str, str],
        task_graph: nx.DiGraph
    ) -> Dict[str, str]:
        """Find successors that can be added to current superstep.

        A successor can be added if ALL its predecessors are on the same processor.

        Args:
            just_assigned: Nodes just assigned in this superstep
            remaining_nodes: Nodes not yet assigned
            assigned: Nodes already assigned
            proc_assignments: Current processor assignments
            task_graph: The task graph

        Returns:
            Dict mapping addable node -> processor
        """
        addable: Dict[str, str] = {}

        # Check successors of just-assigned nodes
        for node in just_assigned:
            for succ in task_graph.successors(node):
                if succ in remaining_nodes and succ not in addable:
                    # Check if all predecessors are assigned and on same processor
                    preds = list(task_graph.predecessors(succ))
                    all_assigned = all(pred in assigned or pred in addable for pred in preds)

                    if all_assigned and preds:
                        # Get processors of all predecessors
                        pred_procs = set()
                        for pred in preds:
                            if pred in proc_assignments:
                                pred_procs.add(proc_assignments[pred])
                            elif pred in addable:
                                pred_procs.add(addable[pred])

                        # If all on same processor, add successor to that processor
                        if len(pred_procs) == 1:
                            proc = pred_procs.pop()
                            addable[succ] = proc

        # Recursively check if we can add more
        if addable:
            # Update assigned temporarily
            temp_assigned = assigned | set(addable.keys())
            temp_proc = proc_assignments.copy()
            temp_proc.update(addable)

            more_addable = self._find_addable_successors(
                list(addable.keys()),
                remaining_nodes - set(addable.keys()),
                temp_assigned,
                temp_proc,
                task_graph
            )
            addable.update(more_addable)

        return addable
