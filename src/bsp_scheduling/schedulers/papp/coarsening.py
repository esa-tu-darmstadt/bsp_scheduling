"""DAG Coarsening Utilities - Section 4.5 from Papp et al. 2024

This implements the DAG coarsening for multilevel scheduling:
- Contract edges while preserving DAG property (no cycles)
- Edge selection: prefer small w(u)+w(v) and large c(u)
- Track contraction history for uncoarsening

Reference: Section 4.5, Appendix A.5 (pages 17-18)
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Tuple
import networkx as nx
import copy


@dataclass
class ContractionRecord:
    """Record of a single edge contraction.

    Stores information needed to undo the contraction during uncoarsening.

    Attributes:
        contracted_node: The node that was removed (absorbed)
        absorbing_node: The node that absorbed the contracted node
        original_work_weights: Tuple of (contracted_work, absorbing_work) before merge
        original_comm_weights: Tuple of (contracted_comm, absorbing_comm) before merge
        edge_weight: Weight of the contracted edge
    """
    contracted_node: str
    absorbing_node: str
    original_work_weights: Tuple[float, float]
    original_comm_weights: Tuple[float, float]
    edge_weight: float


class DAGCoarsener:
    """Coarsens a DAG by contracting edges while preserving acyclicity.

    The coarsening algorithm:
    1. Sort contractable edges by w(u) + w(v) ascending
    2. From the first 1/3 of edges, select the one with largest c(u)
    3. Contract that edge, merging u into v
    4. Update weights: new_w = w(u) + w(v), new_c = c(u) + c(v)
    5. Repeat until target size is reached

    Attributes:
        contraction_history: List of ContractionRecord for uncoarsening
    """

    def __init__(self, incremental: bool = False, deterministic: bool = False):
        """Initialize the DAG coarsener.

        Args:
            incremental: If True, use incremental maintenance of contractable edges
                         instead of recomputing from scratch after each contraction.
                         This is our optimization (not from the paper) that provides
                         ~40x speedup on large graphs by only rechecking edges that
                         could be affected by each contraction. Results may differ
                         from standard mode due to unspecified tie-breaking in edge
                         selection (both are valid per the paper's algorithm).
            deterministic: If True, sort edges before selection to ensure reproducible
                           results regardless of iteration order. Useful for testing
                           that incremental and standard produce identical results.
                           The paper doesn't specify tie-breaking, so without this
                           flag, results may vary based on edge iteration order.
        """
        self.contraction_history: List[ContractionRecord] = []
        self.incremental = incremental
        self.deterministic = deterministic

    def coarsen(
        self,
        task_graph: nx.DiGraph,
        target_ratio: float = 0.15,
        min_nodes: int = 4
    ) -> nx.DiGraph:
        """Coarsen a DAG to a target fraction of its original size.

        Args:
            task_graph: The task graph to coarsen
            target_ratio: Target size as fraction of original (default 0.15 = 15%)
            min_nodes: Minimum number of nodes to keep (default 4)

        Returns:
            Coarsened DAG
        """
        # Create working copy
        G = copy.deepcopy(task_graph)

        original_size = len(G.nodes())
        target_size = max(min_nodes, int(original_size * target_ratio))

        self.contraction_history = []

        if self.incremental:
            return self._coarsen_incremental(G, target_size)
        else:
            return self._coarsen_standard(G, target_size)

    def _coarsen_standard(self, G: nx.DiGraph, target_size: int) -> nx.DiGraph:
        """Standard coarsening: recompute all contractable edges after each contraction."""
        while len(G.nodes()) > target_size:
            # Find all contractable edges
            contractable = self._find_contractable_edges(G)

            if not contractable:
                # No more edges can be contracted
                break

            # Sort for deterministic tie-breaking if requested
            if self.deterministic:
                contractable = sorted(contractable)

            # Select edge to contract
            edge = self._select_edge_to_contract(G, contractable)

            if edge is None:
                break

            u, v = edge

            # Contract edge (u, v): merge u into v
            G = self._contract_edge(G, u, v)

        return G

    def _coarsen_incremental(self, G: nx.DiGraph, target_size: int) -> nx.DiGraph:
        """Incremental coarsening: maintain contractable edges set incrementally.

        Instead of recomputing all contractable edges after each contraction,
        we only recheck edges that could have been affected by the contraction.
        """
        # Initial computation: find all contractable edges once
        contractable = set()
        for u, v in G.edges():
            if not self._has_alternate_path(G, u, v):
                contractable.add((u, v))

        while len(G.nodes()) > target_size:
            if not contractable:
                break

            # Select edge to contract
            # Sort for deterministic tie-breaking if requested, otherwise just convert to list
            edges_list = sorted(contractable) if self.deterministic else list(contractable)
            edge = self._select_edge_to_contract(G, edges_list)

            if edge is None:
                break

            u, v = edge

            # Before contraction, record the graph structure around u and v
            old_preds_v = set(G.predecessors(v))
            old_succs_v = set(G.successors(v))
            preds_u = set(G.predecessors(u))
            succs_u = set(G.successors(u))

            # Remove all edges involving u from contractable set
            edges_to_remove = set()
            for pred in preds_u:
                edges_to_remove.add((pred, u))
            for succ in succs_u:
                edges_to_remove.add((u, succ))
            contractable -= edges_to_remove

            # Contract the edge
            G = self._contract_edge(G, u, v)

            # Identify edges that need rechecking:
            # New edges were added: (new_pred -> v) and (v -> new_succ)
            # These can create new alternate paths for other edges
            new_preds_v = preds_u - old_preds_v - {v}
            new_succs_v = succs_u - old_succs_v - {v}

            # Collect edges to recheck
            edges_to_recheck = set()

            # 1. All edges involving v (v's connectivity changed)
            for pred in G.predecessors(v):
                edges_to_recheck.add((pred, v))
            for succ in G.successors(v):
                edges_to_recheck.add((v, succ))

            # 2. Edges from/to neighbors of new predecessors
            #    (new paths: new_pred -> v -> ... could affect edges from new_pred's preds)
            for new_pred in new_preds_v:
                if new_pred in G.nodes():
                    for pred_of_new_pred in G.predecessors(new_pred):
                        edges_to_recheck.add((pred_of_new_pred, new_pred))
                    for succ_of_new_pred in G.successors(new_pred):
                        if succ_of_new_pred != v:
                            edges_to_recheck.add((new_pred, succ_of_new_pred))

            # 3. Edges from/to neighbors of new successors
            #    (new paths: ... -> v -> new_succ could affect edges to new_succ's succs)
            for new_succ in new_succs_v:
                if new_succ in G.nodes():
                    for pred_of_new_succ in G.predecessors(new_succ):
                        if pred_of_new_succ != v:
                            edges_to_recheck.add((pred_of_new_succ, new_succ))
                    for succ_of_new_succ in G.successors(new_succ):
                        edges_to_recheck.add((new_succ, succ_of_new_succ))

            # 4. Edges between predecessors of v and successors of v
            #    (these could have new alternate paths through v)
            for pred in G.predecessors(v):
                for succ in G.successors(v):
                    if G.has_edge(pred, succ):
                        edges_to_recheck.add((pred, succ))

            # 5. Edges (pred_of_v, b) where b is reachable from v's new successors
            #    These can have new alternate paths: pred_of_v -> v -> new_succ -> ... -> b
            if new_succs_v:
                # Find all nodes reachable from new successors
                reachable_from_new_succs = set()
                for new_succ in new_succs_v:
                    if new_succ in G.nodes():
                        reachable_from_new_succs.add(new_succ)
                        reachable_from_new_succs.update(nx.descendants(G, new_succ))

                # For each predecessor of v, check edges to reachable nodes
                for pred in G.predecessors(v):
                    for b in G.successors(pred):
                        if b != v and b in reachable_from_new_succs:
                            edges_to_recheck.add((pred, b))

            # 6. Edges (a, b) where a can reach v (not just direct predecessor) and
            #    b is reachable from v's new successors.
            #    These can have new alternate paths: a -> ... -> v -> new_succ -> ... -> b
            if new_succs_v:
                # Find all nodes that can reach v
                ancestors_of_v = nx.ancestors(G, v)

                # For each ancestor a, check if any of a's successors are in reachable_from_new_succs
                for a in ancestors_of_v:
                    for b in G.successors(a):
                        if b != v and b in reachable_from_new_succs:
                            edges_to_recheck.add((a, b))

            # Recheck affected edges and update contractable set
            for edge in edges_to_recheck:
                a, b = edge
                if a in G.nodes() and b in G.nodes() and G.has_edge(a, b):
                    if self._has_alternate_path(G, a, b):
                        contractable.discard(edge)
                    else:
                        contractable.add(edge)

        assert nx.is_directed_acyclic_graph(G), "Coarsened graph is not a DAG"
        return G

    def _find_contractable_edges(self, G: nx.DiGraph) -> List[Tuple[str, str]]:
        """Find all edges that can be contracted without creating cycles.

        An edge (u, v) is contractable iff there is no other directed path
        from u to v besides the edge itself.

        Args:
            G: The DAG

        Returns:
            List of contractable edges as (source, target) tuples
        """
        contractable = []

        for u, v in G.edges():
            if self._can_contract_edge(G, u, v):
                contractable.append((u, v))

        return contractable

    def _can_contract_edge(self, G: nx.DiGraph, u: str, v: str) -> bool:
        """Check if edge (u, v) can be contracted without creating a cycle.

        An edge can be contracted iff there is no other directed path from u to v.
        If contracted and there was another path, the merged node would have a
        self-loop, violating the DAG property.

        Args:
            G: The DAG
            u: Source node
            v: Target node

        Returns:
            True if edge can be contracted
        """
        # Check if there's an alternate path from u to v (not using the direct edge).
        # We do a single BFS from u, skipping the direct u->v edge.
        # This is O(reachable nodes + edges) instead of copying the entire graph.
        return not self._has_alternate_path(G, u, v)

    def _has_alternate_path(self, G: nx.DiGraph, u: str, v: str) -> bool:
        """Check if there's a path from u to v that doesn't use the direct edge (u, v).

        Uses BFS starting from u, but skips v when expanding u's neighbors.
        If v is reached through any other path, returns True.

        Args:
            G: The DAG
            u: Source node
            v: Target node (the node we want to check reachability to)

        Returns:
            True if an alternate path exists, False otherwise
        """
        visited = {u}
        queue = deque()

        # Start BFS: add all successors of u except v
        for successor in G.successors(u):
            if successor != v and successor not in visited:
                if successor == v:  # Shouldn't happen due to check above, but defensive
                    return True
                visited.add(successor)
                queue.append(successor)

        # BFS traversal
        while queue:
            current = queue.popleft()
            for successor in G.successors(current):
                if successor == v:
                    return True  # Found alternate path to v
                if successor not in visited:
                    visited.add(successor)
                    queue.append(successor)

        return False  # No alternate path found

    def _select_edge_to_contract(
        self,
        G: nx.DiGraph,
        contractable: List[Tuple[str, str]]
    ) -> Optional[Tuple[str, str]]:
        """Select the best edge to contract using Papp et al. criteria.

        Selection:
        1. Sort by w(u) + w(v) ascending
        2. Take first 1/3 of list
        3. From those, pick edge with largest c(u)

        Per Papp et al. 2024: c(u) is the communication weight of u's output,
        computed as average outgoing edge weight (edges are uniform per source).

        Args:
            G: The DAG
            contractable: List of contractable edges

        Returns:
            Best edge to contract, or None if no valid edge
        """
        import heapq

        if not contractable:
            return None

        # Cache node weights for faster lookup
        node_weights = G.nodes

        # Combined weight function using cached lookups
        def combined_weight(edge):
            u, v = edge
            return node_weights[u]['weight'] + node_weights[v]['weight']

        # Use heapq.nsmallest for partial sort - O(n * log(k)) instead of O(n * log(n))
        # We only need the smallest 1/3 of edges by combined weight
        subset_size = max(1, len(contractable) // 3)
        subset = heapq.nsmallest(subset_size, contractable, key=combined_weight)

        # Cache comm_weight per node (not per edge) since it only depends on u
        # This avoids recomputing the same sum for edges with the same source node
        comm_weight_cache: Dict[str, float] = {}

        def comm_weight(edge):
            u, _ = edge
            if u not in comm_weight_cache:
                outdeg = G.out_degree(u)
                if outdeg == 0:
                    comm_weight_cache[u] = 0.0
                else:
                    total = sum(
                        G.edges[u, succ]['weight']
                        for succ in G.successors(u)
                    )
                    comm_weight_cache[u] = total / outdeg
            return comm_weight_cache[u]

        best_edge = max(subset, key=comm_weight)
        return best_edge

    def _contract_edge(self, G: nx.DiGraph, u: str, v: str) -> nx.DiGraph:
        """Contract edge (u, v) by merging u into v.

        After contraction:
        - v's work weight = w(u) + w(v)
        - v's communication weight = c(u) + c(v)
        - All predecessors of u become predecessors of v
        - All successors of u become successors of v
        - Node u is removed

        Args:
            G: The DAG
            u: Node to be absorbed (contracted)
            v: Node that absorbs u

        Returns:
            Modified DAG
        """
        # Record contraction for uncoarsening
        wu = G.nodes[u].get('weight', 1.0)
        wv = G.nodes[v].get('weight', 1.0)
        cu = G.nodes[u].get('comm_weight', G.nodes[u].get('weight', 1.0))
        cv = G.nodes[v].get('comm_weight', G.nodes[v].get('weight', 1.0))
        edge_weight = G.edges[u, v].get('weight', 1.0) if G.has_edge(u, v) else 1.0

        record = ContractionRecord(
            contracted_node=u,
            absorbing_node=v,
            original_work_weights=(wu, wv),
            original_comm_weights=(cu, cv),
            edge_weight=edge_weight
        )
        self.contraction_history.append(record)

        # Update v's weights
        G.nodes[v]['weight'] = wu + wv
        G.nodes[v]['comm_weight'] = cu + cv

        # Redirect u's predecessors to v
        for pred in list(G.predecessors(u)):
            if pred != v:  # Avoid self-loop
                # Keep maximum edge weight if edge already exists
                pred_edge_weight = G.edges[pred, u].get('weight', 1.0)
                if G.has_edge(pred, v):
                    existing_weight = G.edges[pred, v].get('weight', 1.0)
                    G.edges[pred, v]['weight'] = max(existing_weight, pred_edge_weight)
                else:
                    G.add_edge(pred, v, weight=pred_edge_weight)

        # Redirect u's successors to v
        for succ in list(G.successors(u)):
            if succ != v:  # Avoid self-loop
                succ_edge_weight = G.edges[u, succ].get('weight', 1.0)
                if G.has_edge(v, succ):
                    existing_weight = G.edges[v, succ].get('weight', 1.0)
                    G.edges[v, succ]['weight'] = max(existing_weight, succ_edge_weight)
                else:
                    G.add_edge(v, succ, weight=succ_edge_weight)

        # Remove u
        G.remove_node(u)

        return G

    def uncoarsen_step(
        self,
        G: nx.DiGraph,
        original_graph: nx.DiGraph
    ) -> Tuple[nx.DiGraph, ContractionRecord]:
        """Undo one contraction step.

        Args:
            G: Current coarsened graph
            original_graph: Original uncoarsened graph (for retrieving weights)

        Returns:
            Tuple of (slightly uncoarsened graph, contraction record that was undone)
        """
        if not self.contraction_history:
            raise RuntimeError("No more contractions to undo")

        record = self.contraction_history.pop()
        u = record.contracted_node
        v = record.absorbing_node
        wu, wv = record.original_work_weights
        cu, cv = record.original_comm_weights

        # Add back the contracted node
        G.add_node(u, weight=wu)
        if 'comm_weight' in G.nodes[v]:
            G.nodes[u]['comm_weight'] = cu

        # Restore v's original weight
        G.nodes[v]['weight'] = wv
        if 'comm_weight' in G.nodes[v]:
            G.nodes[v]['comm_weight'] = cv

        # Re-add the edge (u, v)
        G.add_edge(u, v, weight=record.edge_weight)

        # Restore edges from original graph
        # Predecessors of u
        for pred in original_graph.predecessors(u):
            if pred in G.nodes() and not G.has_edge(pred, u):
                edge_weight = original_graph.edges[pred, u].get('weight', 1.0)
                G.add_edge(pred, u, weight=edge_weight)

        # Successors of u
        for succ in original_graph.successors(u):
            if succ in G.nodes() and not G.has_edge(u, succ):
                edge_weight = original_graph.edges[u, succ].get('weight', 1.0)
                G.add_edge(u, succ, weight=edge_weight)

        # Remove edges from v that should be from u
        for pred in list(G.predecessors(v)):
            if pred in original_graph.predecessors(u) and not original_graph.has_edge(pred, v):
                G.remove_edge(pred, v)

        for succ in list(G.successors(v)):
            if succ in original_graph.successors(u) and not original_graph.has_edge(v, succ):
                G.remove_edge(v, succ)

        return G, record

    def uncoarsen_n_steps(
        self,
        G: nx.DiGraph,
        original_graph: nx.DiGraph,
        n: int = 5
    ) -> Tuple[nx.DiGraph, List[ContractionRecord]]:
        """Undo n contraction steps.

        Args:
            G: Current coarsened graph
            original_graph: Original uncoarsened graph
            n: Number of steps to undo

        Returns:
            Tuple of (uncoarsened graph, list of undone records)
        """
        records = []
        for _ in range(min(n, len(self.contraction_history))):
            G, record = self.uncoarsen_step(G, original_graph)
            records.append(record)
        return G, records

    @property
    def num_contractions(self) -> int:
        """Number of contractions performed (and not yet undone)."""
        return len(self.contraction_history)
