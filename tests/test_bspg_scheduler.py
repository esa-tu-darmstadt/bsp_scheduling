"""Tests for BSPg Scheduler.

Tests that the optimized BSPg implementation produces identical results
to the standard implementation.
"""

import pytest
import networkx as nx
import random
from saga_bsp.schedulers.papp.bspg_scheduler import BSPgScheduler
from saga_bsp.schedule import BSPSchedule, BSPHardware


def create_hardware(num_processors: int, sync_time: float = 1.0) -> BSPHardware:
    """Create a simple fully-connected hardware configuration."""
    network = nx.complete_graph(num_processors)
    for node in network.nodes():
        network.nodes[node]['weight'] = 1.0  # Uniform processor speed
    for edge in network.edges():
        network.edges[edge]['weight'] = 1.0  # Uniform network speed
    return BSPHardware(network, sync_time)


def create_random_dag(num_nodes: int, edge_probability: float = 0.3, seed: int = None) -> nx.DiGraph:
    """Create a random DAG with weights on nodes and edges."""
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

    # Ensure connectivity
    for node in G.nodes():
        if G.in_degree(node) == 0 and G.out_degree(node) == 0:
            other_nodes = [n for n in G.nodes() if n != node]
            if other_nodes:
                target = random.choice(other_nodes)
                if int(node) < int(target):
                    G.add_edge(node, target, weight=random.uniform(1.0, 5.0))
                else:
                    G.add_edge(target, node, weight=random.uniform(1.0, 5.0))

    return G


def schedules_equal(s1: BSPSchedule, s2: BSPSchedule) -> bool:
    """Check if two schedules have identical structure."""
    # Check number of supersteps
    if len(s1.supersteps) != len(s2.supersteps):
        return False

    # Check makespan (with tolerance for floating point)
    if abs(s1.makespan - s2.makespan) > 1e-9:
        return False

    # Check task assignments in each superstep
    for ss1, ss2 in zip(s1.supersteps, s2.supersteps):
        # Get task->processor mapping for each superstep
        tasks1 = {}
        tasks2 = {}
        for proc, tasks in ss1.tasks.items():
            for task in tasks:
                tasks1[task.node] = proc
        for proc, tasks in ss2.tasks.items():
            for task in tasks:
                tasks2[task.node] = proc

        if tasks1 != tasks2:
            return False

    return True


class TestBSPgEquivalence:
    """Tests that optimized and standard BSPg produce identical results."""

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1000])
    def test_small_dag_equivalence(self, seed):
        """Test on small random DAGs."""
        G = create_random_dag(num_nodes=30, edge_probability=0.2, seed=seed)
        hardware = create_hardware(num_processors=4)

        scheduler_standard = BSPgScheduler(optimized=False)
        scheduler_optimized = BSPgScheduler(optimized=True)

        schedule_standard = scheduler_standard.schedule(hardware, G)
        schedule_optimized = scheduler_optimized.schedule(hardware, G)

        assert schedules_equal(schedule_standard, schedule_optimized), \
            f"Schedules differ for seed {seed}"

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_medium_dag_equivalence(self, seed):
        """Test on medium-sized random DAGs."""
        G = create_random_dag(num_nodes=100, edge_probability=0.1, seed=seed)
        hardware = create_hardware(num_processors=8)

        scheduler_standard = BSPgScheduler(optimized=False)
        scheduler_optimized = BSPgScheduler(optimized=True)

        schedule_standard = scheduler_standard.schedule(hardware, G)
        schedule_optimized = scheduler_optimized.schedule(hardware, G)

        assert schedules_equal(schedule_standard, schedule_optimized), \
            f"Schedules differ for seed {seed}"

    def test_chain_dag(self):
        """Test on a simple chain DAG."""
        G = nx.DiGraph()
        for i in range(20):
            G.add_node(str(i), weight=1.0)
        for i in range(19):
            G.add_edge(str(i), str(i+1), weight=1.0)

        hardware = create_hardware(num_processors=4)

        scheduler_standard = BSPgScheduler(optimized=False)
        scheduler_optimized = BSPgScheduler(optimized=True)

        schedule_standard = scheduler_standard.schedule(hardware, G)
        schedule_optimized = scheduler_optimized.schedule(hardware, G)

        assert schedules_equal(schedule_standard, schedule_optimized)

    def test_wide_dag(self):
        """Test on a wide DAG with many parallel paths."""
        G = nx.DiGraph()
        G.add_node('source', weight=1.0)
        G.add_node('sink', weight=1.0)
        for i in range(16):
            G.add_node(f'mid_{i}', weight=1.0)
            G.add_edge('source', f'mid_{i}', weight=1.0)
            G.add_edge(f'mid_{i}', 'sink', weight=1.0)

        hardware = create_hardware(num_processors=4)

        scheduler_standard = BSPgScheduler(optimized=False)
        scheduler_optimized = BSPgScheduler(optimized=True)

        schedule_standard = scheduler_standard.schedule(hardware, G)
        schedule_optimized = scheduler_optimized.schedule(hardware, G)

        assert schedules_equal(schedule_standard, schedule_optimized)

    def test_diamond_dag(self):
        """Test on a diamond-shaped DAG."""
        G = nx.DiGraph()
        for i in range(4):
            G.add_node(str(i), weight=1.0)
        G.add_edge('0', '1', weight=1.0)
        G.add_edge('0', '2', weight=1.0)
        G.add_edge('1', '3', weight=1.0)
        G.add_edge('2', '3', weight=1.0)

        hardware = create_hardware(num_processors=2)

        scheduler_standard = BSPgScheduler(optimized=False)
        scheduler_optimized = BSPgScheduler(optimized=True)

        schedule_standard = scheduler_standard.schedule(hardware, G)
        schedule_optimized = scheduler_optimized.schedule(hardware, G)

        assert schedules_equal(schedule_standard, schedule_optimized)
