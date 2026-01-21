"""Tests for DAG Coarsening.

Tests that the incremental coarsening optimization produces identical results
to the standard (recompute-all) approach.
"""

import pytest
import networkx as nx
import random
import copy
from saga_bsp.schedulers.papp.coarsening import DAGCoarsener


def create_random_dag(num_nodes: int, edge_probability: float = 0.3, seed: int = None) -> nx.DiGraph:
    """Create a random DAG with weights on nodes and edges.

    Args:
        num_nodes: Number of nodes in the DAG
        edge_probability: Probability of an edge between any two nodes (where valid)
        seed: Random seed for reproducibility

    Returns:
        A random DAG with 'weight' attributes on nodes and edges
    """
    if seed is not None:
        random.seed(seed)

    G = nx.DiGraph()

    # Add nodes with random weights
    for i in range(num_nodes):
        G.add_node(str(i), weight=random.uniform(1.0, 10.0))

    # Add edges (only from lower to higher index to ensure DAG property)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_probability:
                G.add_edge(str(i), str(j), weight=random.uniform(1.0, 5.0))

    # Ensure the graph is connected (add edges to isolated nodes)
    for node in G.nodes():
        if G.in_degree(node) == 0 and G.out_degree(node) == 0:
            # Connect to a random other node
            other_nodes = [n for n in G.nodes() if n != node]
            if other_nodes:
                target = random.choice(other_nodes)
                if int(node) < int(target):
                    G.add_edge(node, target, weight=random.uniform(1.0, 5.0))
                else:
                    G.add_edge(target, node, weight=random.uniform(1.0, 5.0))

    return G


def graphs_equal(G1: nx.DiGraph, G2: nx.DiGraph) -> bool:
    """Check if two graphs have identical structure and weights."""
    if set(G1.nodes()) != set(G2.nodes()):
        return False
    if set(G1.edges()) != set(G2.edges()):
        return False

    # Check node weights
    for node in G1.nodes():
        if abs(G1.nodes[node].get('weight', 0) - G2.nodes[node].get('weight', 0)) > 1e-9:
            return False

    # Check edge weights
    for edge in G1.edges():
        if abs(G1.edges[edge].get('weight', 0) - G2.edges[edge].get('weight', 0)) > 1e-9:
            return False

    return True


class TestCoarseningEquivalence:
    """Tests that incremental and standard coarsening produce identical results."""

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1000])
    def test_small_dag_equivalence(self, seed):
        """Test on small random DAGs."""
        G = create_random_dag(num_nodes=20, edge_probability=0.25, seed=seed)

        # Make two independent copies
        G_standard = copy.deepcopy(G)
        G_incremental = copy.deepcopy(G)

        # Run standard coarsening
        coarsener_standard = DAGCoarsener(incremental=False, deterministic=True)
        result_standard = coarsener_standard.coarsen(G_standard, target_ratio=0.3)

        # Run incremental coarsening
        coarsener_incremental = DAGCoarsener(incremental=True, deterministic=True)
        result_incremental = coarsener_incremental.coarsen(G_incremental, target_ratio=0.3)

        # Results should be identical
        assert graphs_equal(result_standard, result_incremental), \
            f"Graphs differ for seed {seed}"

        # Contraction histories should match
        assert len(coarsener_standard.contraction_history) == len(coarsener_incremental.contraction_history), \
            f"Contraction history lengths differ for seed {seed}"

        for i, (rec_std, rec_inc) in enumerate(zip(
            coarsener_standard.contraction_history,
            coarsener_incremental.contraction_history
        )):
            assert rec_std.contracted_node == rec_inc.contracted_node, \
                f"Contraction {i}: contracted nodes differ ({rec_std.contracted_node} vs {rec_inc.contracted_node})"
            assert rec_std.absorbing_node == rec_inc.absorbing_node, \
                f"Contraction {i}: absorbing nodes differ ({rec_std.absorbing_node} vs {rec_inc.absorbing_node})"

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_medium_dag_equivalence(self, seed):
        """Test on medium-sized random DAGs."""
        G = create_random_dag(num_nodes=50, edge_probability=0.15, seed=seed)

        G_standard = copy.deepcopy(G)
        G_incremental = copy.deepcopy(G)

        coarsener_standard = DAGCoarsener(incremental=False, deterministic=True)
        result_standard = coarsener_standard.coarsen(G_standard, target_ratio=0.2)

        coarsener_incremental = DAGCoarsener(incremental=True, deterministic=True)
        result_incremental = coarsener_incremental.coarsen(G_incremental, target_ratio=0.2)

        assert graphs_equal(result_standard, result_incremental), \
            f"Graphs differ for seed {seed}"
        assert len(coarsener_standard.contraction_history) == len(coarsener_incremental.contraction_history)

    def test_chain_dag(self):
        """Test on a simple chain DAG."""
        G = nx.DiGraph()
        for i in range(10):
            G.add_node(str(i), weight=1.0)
        for i in range(9):
            G.add_edge(str(i), str(i+1), weight=1.0)

        G_standard = copy.deepcopy(G)
        G_incremental = copy.deepcopy(G)

        coarsener_standard = DAGCoarsener(incremental=False, deterministic=True)
        result_standard = coarsener_standard.coarsen(G_standard, target_ratio=0.3)

        coarsener_incremental = DAGCoarsener(incremental=True, deterministic=True)
        result_incremental = coarsener_incremental.coarsen(G_incremental, target_ratio=0.3)

        assert graphs_equal(result_standard, result_incremental)

    def test_diamond_dag(self):
        """Test on a diamond-shaped DAG (has alternate paths)."""
        G = nx.DiGraph()
        # Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        for i in range(4):
            G.add_node(str(i), weight=1.0)
        G.add_edge('0', '1', weight=1.0)
        G.add_edge('0', '2', weight=1.0)
        G.add_edge('1', '3', weight=1.0)
        G.add_edge('2', '3', weight=1.0)

        G_standard = copy.deepcopy(G)
        G_incremental = copy.deepcopy(G)

        coarsener_standard = DAGCoarsener(incremental=False, deterministic=True)
        result_standard = coarsener_standard.coarsen(G_standard, target_ratio=0.5)

        coarsener_incremental = DAGCoarsener(incremental=True, deterministic=True)
        result_incremental = coarsener_incremental.coarsen(G_incremental, target_ratio=0.5)

        assert graphs_equal(result_standard, result_incremental)

    def test_wide_dag(self):
        """Test on a wide DAG with many parallel paths."""
        G = nx.DiGraph()
        # Single source, many parallel middle nodes, single sink
        G.add_node('source', weight=1.0)
        G.add_node('sink', weight=1.0)
        for i in range(8):
            G.add_node(f'mid_{i}', weight=1.0)
            G.add_edge('source', f'mid_{i}', weight=1.0)
            G.add_edge(f'mid_{i}', 'sink', weight=1.0)

        G_standard = copy.deepcopy(G)
        G_incremental = copy.deepcopy(G)

        coarsener_standard = DAGCoarsener(incremental=False, deterministic=True)
        result_standard = coarsener_standard.coarsen(G_standard, target_ratio=0.3)

        coarsener_incremental = DAGCoarsener(incremental=True, deterministic=True)
        result_incremental = coarsener_incremental.coarsen(G_incremental, target_ratio=0.3)

        assert graphs_equal(result_standard, result_incremental)


class TestHasAlternatePath:
    """Tests for the _has_alternate_path helper function."""

    def test_no_alternate_path_in_chain(self):
        """In a chain, no edge has an alternate path."""
        G = nx.DiGraph()
        for i in range(5):
            G.add_node(str(i))
        for i in range(4):
            G.add_edge(str(i), str(i+1))

        coarsener = DAGCoarsener()

        # No edge in a chain has an alternate path
        for i in range(4):
            assert not coarsener._has_alternate_path(G, str(i), str(i+1))

    def test_alternate_path_in_diamond(self):
        """In a diamond, the diagonal edges have alternate paths."""
        G = nx.DiGraph()
        for i in range(4):
            G.add_node(str(i))
        # Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        G.add_edge('0', '1')
        G.add_edge('0', '2')
        G.add_edge('1', '3')
        G.add_edge('2', '3')

        coarsener = DAGCoarsener()

        # Edges 0->1 and 0->2 have no alternate paths
        assert not coarsener._has_alternate_path(G, '0', '1')
        assert not coarsener._has_alternate_path(G, '0', '2')

        # Edges 1->3 and 2->3 have no alternate paths
        assert not coarsener._has_alternate_path(G, '1', '3')
        assert not coarsener._has_alternate_path(G, '2', '3')

    def test_alternate_path_with_transitive_edge(self):
        """An edge that skips a node has an alternate path through that node."""
        G = nx.DiGraph()
        for i in range(3):
            G.add_node(str(i))
        G.add_edge('0', '1')
        G.add_edge('1', '2')
        G.add_edge('0', '2')  # Transitive edge

        coarsener = DAGCoarsener()

        # 0->1 and 1->2 have no alternate paths
        assert not coarsener._has_alternate_path(G, '0', '1')
        assert not coarsener._has_alternate_path(G, '1', '2')

        # 0->2 has an alternate path (0 -> 1 -> 2)
        assert coarsener._has_alternate_path(G, '0', '2')
