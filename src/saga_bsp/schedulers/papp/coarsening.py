"""DAG Coarsening Utilities - Section 4.5 from Papp et al. 2024

This implements the DAG coarsening for multilevel scheduling:
- Contract edges while preserving DAG property (no cycles)
- Edge selection: prefer small w(u)+w(v) and large c(u)
- Track contraction history for uncoarsening

Reference: Section 4.5, Appendix A.5 (pages 17-18)
"""

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

    def __init__(self):
        self.contraction_history: List[ContractionRecord] = []

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

        while len(G.nodes()) > target_size:
            # Find all contractable edges
            contractable = self._find_contractable_edges(G)

            if not contractable:
                # No more edges can be contracted
                break

            # Select edge to contract
            edge = self._select_edge_to_contract(G, contractable)

            if edge is None:
                break

            u, v = edge

            # Contract edge (u, v): merge u into v
            G = self._contract_edge(G, u, v)

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
        # Temporarily remove the edge
        G_temp = G.copy()
        G_temp.remove_edge(u, v)

        # Check if there's still a path from u to v
        try:
            # If a path exists without the edge, we cannot contract
            path = nx.has_path(G_temp, u, v)
            return not path
        except nx.NetworkXError:
            return True

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
        if not contractable:
            return None

        # Sort by combined work weight (ascending)
        def combined_weight(edge):
            u, v = edge
            wu = G.nodes[u]['weight']
            wv = G.nodes[v]['weight']
            return wu + wv

        sorted_edges = sorted(contractable, key=combined_weight)

        # Take first 1/3 of edges (at least 1)
        subset_size = max(1, len(sorted_edges) // 3)
        subset = sorted_edges[:subset_size]

        # From subset, pick edge with largest c(u)
        # c(u) = average outgoing edge weight from u
        def comm_weight(edge):
            u, v = edge
            outdeg = G.out_degree(u)
            if outdeg == 0:
                return 0.0
            return sum(
                G.edges[u, succ]['weight']
                for succ in G.successors(u)
            ) / outdeg

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
