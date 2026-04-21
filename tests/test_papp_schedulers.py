"""Tests for Papp et al. 2024 BSP Schedulers

Tests for:
- BSPgScheduler
- SourceScheduler
- MultilevelScheduler
- DAGCoarsener
- HillClimbing / HCcs
- ILPcs / ILPpart
"""

import pytest
import networkx as nx
from bsp_scheduling import BSPHardware, BSPSchedule
from bsp_scheduling.schedulers.papp import (
    BSPgScheduler,
    SourceScheduler,
    MultilevelScheduler,
    DAGCoarsener,
    ContractionRecord,
)
from bsp_scheduling.optimization import HillClimbing, HCcs, ILPcs, ILPpart


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_hardware():
    """Create simple 2-processor hardware."""
    network = nx.complete_graph(2)
    for node in network.nodes():
        network.nodes[node]['weight'] = 1.0
    for edge in network.edges():
        network.edges[edge]['weight'] = 1.0
    return BSPHardware(network, sync_time=5.0)


@pytest.fixture
def four_proc_hardware():
    """Create 4-processor hardware."""
    network = nx.complete_graph(4)
    for node in network.nodes():
        network.nodes[node]['weight'] = 1.0
    for edge in network.edges():
        network.edges[edge]['weight'] = 1.0
    return BSPHardware(network, sync_time=10.0)


@pytest.fixture
def chain_task_graph():
    """Create a simple chain: A -> B -> C -> D."""
    G = nx.DiGraph()
    for i, name in enumerate(['A', 'B', 'C', 'D']):
        G.add_node(name, weight=10.0)
    G.add_edge('A', 'B', weight=5.0)
    G.add_edge('B', 'C', weight=5.0)
    G.add_edge('C', 'D', weight=5.0)
    return G


@pytest.fixture
def diamond_task_graph():
    """Create diamond pattern: A -> B,C -> D."""
    G = nx.DiGraph()
    G.add_node('A', weight=10.0)
    G.add_node('B', weight=20.0)
    G.add_node('C', weight=15.0)
    G.add_node('D', weight=10.0)
    G.add_edge('A', 'B', weight=5.0)
    G.add_edge('A', 'C', weight=5.0)
    G.add_edge('B', 'D', weight=5.0)
    G.add_edge('C', 'D', weight=5.0)
    return G


@pytest.fixture
def fork_join_task_graph():
    """Create fork-join pattern: A -> B,C,D -> E."""
    G = nx.DiGraph()
    G.add_node('A', weight=10.0)
    G.add_node('B', weight=10.0)
    G.add_node('C', weight=10.0)
    G.add_node('D', weight=10.0)
    G.add_node('E', weight=10.0)
    G.add_edge('A', 'B', weight=5.0)
    G.add_edge('A', 'C', weight=5.0)
    G.add_edge('A', 'D', weight=5.0)
    G.add_edge('B', 'E', weight=5.0)
    G.add_edge('C', 'E', weight=5.0)
    G.add_edge('D', 'E', weight=5.0)
    return G


@pytest.fixture
def larger_task_graph():
    """Create a larger task graph for coarsening tests."""
    G = nx.DiGraph()
    # Create 20 nodes
    for i in range(20):
        G.add_node(f'T{i}', weight=10.0 + i)

    # Create edges forming a layered structure
    # Layer 0: T0-T3 (sources)
    # Layer 1: T4-T7
    # Layer 2: T8-T11
    # Layer 3: T12-T15
    # Layer 4: T16-T19 (sinks)

    # Connect layer 0 to layer 1
    for i in range(4):
        for j in range(4, 8):
            if (i + j) % 3 == 0:
                G.add_edge(f'T{i}', f'T{j}', weight=5.0)

    # Connect layer 1 to layer 2
    for i in range(4, 8):
        for j in range(8, 12):
            if (i + j) % 3 == 1:
                G.add_edge(f'T{i}', f'T{j}', weight=5.0)

    # Connect layer 2 to layer 3
    for i in range(8, 12):
        for j in range(12, 16):
            if (i + j) % 3 == 2:
                G.add_edge(f'T{i}', f'T{j}', weight=5.0)

    # Connect layer 3 to layer 4
    for i in range(12, 16):
        for j in range(16, 20):
            if (i + j) % 3 == 0:
                G.add_edge(f'T{i}', f'T{j}', weight=5.0)

    return G


# ============================================================================
# BSPgScheduler Tests
# ============================================================================

class TestBSPgScheduler:
    """Tests for BSPgScheduler."""

    def test_simple_chain(self, simple_hardware, chain_task_graph):
        """Test BSPg on a simple chain graph."""
        scheduler = BSPgScheduler()
        schedule = scheduler.schedule(simple_hardware, chain_task_graph)

        # Verify schedule is valid
        is_valid, errors = schedule.is_valid()
        assert is_valid, f"Invalid schedule: {errors}"

        # All tasks should be scheduled
        for node in chain_task_graph.nodes():
            assert schedule.task_scheduled(node)

    def test_diamond(self, simple_hardware, diamond_task_graph):
        """Test BSPg on a diamond graph."""
        scheduler = BSPgScheduler()
        schedule = scheduler.schedule(simple_hardware, diamond_task_graph)

        is_valid, errors = schedule.is_valid()
        assert is_valid, f"Invalid schedule: {errors}"

    def test_fork_join(self, four_proc_hardware, fork_join_task_graph):
        """Test BSPg on a fork-join graph."""
        scheduler = BSPgScheduler()
        schedule = scheduler.schedule(four_proc_hardware, fork_join_task_graph)

        is_valid, errors = schedule.is_valid()
        assert is_valid, f"Invalid schedule: {errors}"

    def test_larger_graph(self, four_proc_hardware, larger_task_graph):
        """Test BSPg on a larger graph."""
        scheduler = BSPgScheduler()
        schedule = scheduler.schedule(four_proc_hardware, larger_task_graph)

        is_valid, errors = schedule.is_valid()
        assert is_valid, f"Invalid schedule: {errors}"

        # All tasks should be scheduled
        for node in larger_task_graph.nodes():
            assert schedule.task_scheduled(node)


# ============================================================================
# SourceScheduler Tests
# ============================================================================

class TestSourceScheduler:
    """Tests for SourceScheduler."""

    def test_simple_chain(self, simple_hardware, chain_task_graph):
        """Test Source on a simple chain graph."""
        scheduler = SourceScheduler()
        schedule = scheduler.schedule(simple_hardware, chain_task_graph)

        is_valid, errors = schedule.is_valid()
        assert is_valid, f"Invalid schedule: {errors}"

    def test_diamond(self, simple_hardware, diamond_task_graph):
        """Test Source on a diamond graph."""
        scheduler = SourceScheduler()
        schedule = scheduler.schedule(simple_hardware, diamond_task_graph)

        is_valid, errors = schedule.is_valid()
        assert is_valid, f"Invalid schedule: {errors}"

    def test_fork_join(self, four_proc_hardware, fork_join_task_graph):
        """Test Source on a fork-join graph."""
        scheduler = SourceScheduler()
        schedule = scheduler.schedule(four_proc_hardware, fork_join_task_graph)

        is_valid, errors = schedule.is_valid()
        assert is_valid, f"Invalid schedule: {errors}"

    def test_larger_graph(self, four_proc_hardware, larger_task_graph):
        """Test Source on a larger graph."""
        scheduler = SourceScheduler()
        schedule = scheduler.schedule(four_proc_hardware, larger_task_graph)

        is_valid, errors = schedule.is_valid()
        assert is_valid, f"Invalid schedule: {errors}"


# ============================================================================
# DAGCoarsener Tests
# ============================================================================

class TestDAGCoarsener:
    """Tests for DAG coarsening utilities."""

    def test_simple_coarsening(self, chain_task_graph):
        """Test basic coarsening on a chain."""
        coarsener = DAGCoarsener()

        # Coarsen to 50% - use min_nodes=2 since chain only has 4 nodes
        coarsened = coarsener.coarsen(chain_task_graph, target_ratio=0.5, min_nodes=2)

        # Should have fewer nodes
        assert len(coarsened.nodes()) < len(chain_task_graph.nodes())

        # Should still be a DAG
        assert nx.is_directed_acyclic_graph(coarsened)

    def test_coarsening_preserves_dag(self, larger_task_graph):
        """Test that coarsening always produces a valid DAG."""
        coarsener = DAGCoarsener()

        coarsened = coarsener.coarsen(larger_task_graph, target_ratio=0.3)

        assert nx.is_directed_acyclic_graph(coarsened)

    def test_uncoarsening(self, larger_task_graph):
        """Test uncoarsening reverses contraction."""
        coarsener = DAGCoarsener()

        # Coarsen
        coarsened = coarsener.coarsen(larger_task_graph, target_ratio=0.5)
        coarsened_size = len(coarsened.nodes())

        # Uncoarsen
        uncoarsened, record = coarsener.uncoarsen_step(coarsened, larger_task_graph)

        # Should have one more node
        assert len(uncoarsened.nodes()) == coarsened_size + 1

    def test_contraction_record(self, chain_task_graph):
        """Test contraction records are created correctly."""
        coarsener = DAGCoarsener()

        # Do one contraction
        coarsened = coarsener.coarsen(chain_task_graph, target_ratio=0.8, min_nodes=3)

        if coarsener.num_contractions > 0:
            record = coarsener.contraction_history[0]
            assert isinstance(record, ContractionRecord)
            assert record.contracted_node is not None
            assert record.absorbing_node is not None


# ============================================================================
# HillClimbing Tests
# ============================================================================

class TestHillClimbing:
    """Tests for hill climbing optimization."""

    def test_basic_optimization(self, simple_hardware, diamond_task_graph):
        """Test basic HC optimization."""
        # First create a schedule
        scheduler = BSPgScheduler()
        schedule = scheduler.schedule(simple_hardware, diamond_task_graph)
        initial_cost = schedule.makespan

        # Apply HC optimization
        hc = HillClimbing(max_iterations=100)
        optimized = hc.optimize(schedule, time_limit=10.0)

        # Schedule should still be valid
        is_valid, errors = optimized.is_valid()
        assert is_valid, f"Invalid after optimization: {errors}"

        # Cost should not increase
        assert optimized.makespan <= initial_cost

    @pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="see HCcs.optimize")
    def test_hccs_optimization(self, simple_hardware, diamond_task_graph):
        """Test HCcs optimization."""
        scheduler = BSPgScheduler()
        schedule = scheduler.schedule(simple_hardware, diamond_task_graph)

        hccs = HCcs(max_iterations=100)
        optimized = hccs.optimize(schedule)

        is_valid, errors = optimized.is_valid()
        assert is_valid, f"Invalid after HCcs: {errors}"


# ============================================================================
# ILP Solvers Tests
# ============================================================================

class TestILPSolvers:
    """Tests for ILP-based optimization."""

    @pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="see ILPcs.optimize")
    def test_ilpcs_basic(self, simple_hardware, diamond_task_graph):
        """Test ILPcs basic functionality."""
        scheduler = BSPgScheduler()
        schedule = scheduler.schedule(simple_hardware, diamond_task_graph)

        ilpcs = ILPcs(time_limit_seconds=10.0)
        optimized = ilpcs.optimize(schedule)

        is_valid, errors = optimized.is_valid()
        assert is_valid, f"Invalid after ILPcs: {errors}"

    @pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="see ILPpart.optimize")
    def test_ilppart_basic(self, simple_hardware, diamond_task_graph):
        """Test ILPpart basic functionality."""
        scheduler = BSPgScheduler()
        schedule = scheduler.schedule(simple_hardware, diamond_task_graph)

        ilppart = ILPpart(time_limit_seconds=10.0)
        optimized = ilppart.optimize(schedule)

        is_valid, errors = optimized.is_valid()
        assert is_valid, f"Invalid after ILPpart: {errors}"


# ============================================================================
# MultilevelScheduler Tests
# ============================================================================

class TestMultilevelScheduler:
    """Tests for multilevel scheduling."""

    def test_basic_multilevel(self, four_proc_hardware, larger_task_graph):
        """Test basic multilevel scheduling."""
        scheduler = MultilevelScheduler(
            coarsening_ratios=[0.5],  # Less aggressive for testing
            hc_interval=3,
            hc_max_steps=50
        )
        schedule = scheduler.schedule(four_proc_hardware, larger_task_graph)

        is_valid, errors = schedule.is_valid()
        assert is_valid, f"Invalid schedule: {errors}"

        # All tasks should be scheduled
        for node in larger_task_graph.nodes():
            assert schedule.task_scheduled(node), f"Task {node} not scheduled"

    def test_multilevel_with_different_ratios(self, four_proc_hardware, larger_task_graph):
        """Test multilevel with multiple coarsening ratios."""
        scheduler = MultilevelScheduler(
            coarsening_ratios=[0.3, 0.5],
            hc_interval=5,
            hc_max_steps=30
        )
        schedule = scheduler.schedule(four_proc_hardware, larger_task_graph)

        is_valid, errors = schedule.is_valid()
        assert is_valid, f"Invalid schedule: {errors}"


# ============================================================================
# Integration Tests
# ============================================================================

class TestPappIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="HC+HCcs pipeline; see HCcs.optimize")
    def test_full_pipeline(self, four_proc_hardware, larger_task_graph):
        """Test the full Papp et al. pipeline."""
        from bsp_scheduling.optimization import optimize_with_hill_climbing

        # Step 1: Initial schedule with BSPg
        bspg = BSPgScheduler()
        schedule = bspg.schedule(four_proc_hardware, larger_task_graph)
        initial_cost = schedule.makespan

        # Step 2: Apply HC+HCcs
        schedule = optimize_with_hill_climbing(
            schedule,
            hc_time_limit=30.0,
            hccs_time_limit=10.0,
            hc_max_iterations=100
        )

        # Should still be valid
        is_valid, errors = schedule.is_valid()
        assert is_valid, f"Invalid after optimization: {errors}"

        # Cost should not increase
        assert schedule.makespan <= initial_cost * 1.01  # Allow small tolerance

    def test_scheduler_comparison(self, four_proc_hardware, larger_task_graph):
        """Compare different schedulers on the same graph."""
        bspg = BSPgScheduler()
        source = SourceScheduler()
        multilevel = MultilevelScheduler(coarsening_ratios=[0.5], hc_max_steps=20)

        bspg_schedule = bspg.schedule(four_proc_hardware, larger_task_graph)
        source_schedule = source.schedule(four_proc_hardware, larger_task_graph)
        multilevel_schedule = multilevel.schedule(four_proc_hardware, larger_task_graph)

        # All should produce valid schedules
        assert bspg_schedule.is_valid()[0]
        assert source_schedule.is_valid()[0]
        assert multilevel_schedule.is_valid()[0]

        # All should have finite makespan
        assert bspg_schedule.makespan > 0
        assert source_schedule.makespan > 0
        assert multilevel_schedule.makespan > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
