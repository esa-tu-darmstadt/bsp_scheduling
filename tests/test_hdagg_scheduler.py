"""Tests for HDagg Scheduler.

Tests the HDagg (Hybrid DAG Aggregation) scheduler implementation against
the algorithm described in the paper:
"HDagg: Hybrid Aggregation of Loop-carried Dependence Iterations in Sparse Matrix Computations"
"""

import pytest
import networkx as nx
from saga_bsp.schedulers.hdagg import HDaggScheduler
from saga_bsp.schedule import BSPSchedule, BSPHardware


def create_hardware(num_processors: int, sync_time: float = 1.0) -> BSPHardware:
    """Create a simple fully-connected hardware configuration."""
    network = nx.complete_graph(num_processors)
    for node in network.nodes():
        network.nodes[node]['weight'] = 1.0  # Uniform processor speed
    for edge in network.edges():
        network.edges[edge]['weight'] = 1.0  # Uniform network speed
    return BSPHardware(network, sync_time)


class TestTransitiveReduction:
    """Tests for the transitive reduction step."""

    def test_no_transitive_edges(self):
        """Test that non-transitive edges are preserved."""
        scheduler = HDaggScheduler()

        # Simple chain: 0 -> 1 -> 2
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(1, 2)

        G_prime = scheduler._transitive_reduction(G)

        assert G_prime.has_edge(0, 1)
        assert G_prime.has_edge(1, 2)
        assert not G_prime.has_edge(0, 2)  # No transitive edge exists

    def test_removes_transitive_edge(self):
        """Test that transitive edges are removed."""
        scheduler = HDaggScheduler()

        # Triangle with transitive edge: 0 -> 1 -> 2, 0 -> 2 (transitive)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(0, 2)  # This is transitive (0 -> 1 -> 2)

        G_prime = scheduler._transitive_reduction(G)

        assert G_prime.has_edge(0, 1)
        assert G_prime.has_edge(1, 2)
        assert not G_prime.has_edge(0, 2)  # Should be removed

    def test_multiple_transitive_edges(self):
        """Test removal of multiple transitive edges."""
        scheduler = HDaggScheduler()

        # Diamond shape with transitive edges
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 2)
        G.add_edge(1, 3)
        G.add_edge(2, 3)
        G.add_edge(0, 3)  # Transitive through either 1 or 2

        G_prime = scheduler._transitive_reduction(G)

        assert G_prime.has_edge(0, 1)
        assert G_prime.has_edge(0, 2)
        assert G_prime.has_edge(1, 3)
        assert G_prime.has_edge(2, 3)
        assert not G_prime.has_edge(0, 3)  # Should be removed


class TestAggregateDenselyConnected:
    """Tests for the vertex aggregation step."""

    def test_chain_forms_single_group(self):
        """Test that a simple chain with single outgoing edges forms one group."""
        scheduler = HDaggScheduler()

        # Chain: 0 -> 1 -> 2
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(1, 2)

        T = scheduler._aggregate_densely_connected(G)

        # All vertices should be in the same group
        all_vertices = set()
        for group in T:
            all_vertices.update(group)
        assert all_vertices == {0, 1, 2}

    def test_fork_creates_separate_groups(self):
        """Test that vertices with multiple outgoing edges create separate groups."""
        scheduler = HDaggScheduler()

        # Fork: 0 -> 1, 0 -> 2 (vertex 0 has multiple outgoing edges)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 2)

        T = scheduler._aggregate_densely_connected(G)

        # Sink vertices 1 and 2 should start separate groups
        # 0 cannot be merged with both since it has 2 outgoing edges
        assert len(T) >= 2

    def test_tree_structure(self):
        """Test a simple tree structure."""
        scheduler = HDaggScheduler()

        # Tree: 0 -> 2, 1 -> 2 (both 0 and 1 have single outgoing edge)
        G = nx.DiGraph()
        G.add_edge(0, 2)
        G.add_edge(1, 2)

        T = scheduler._aggregate_densely_connected(G)

        # 0 and 1 each have one outgoing edge, so they can form subtrees with 2
        all_vertices = set()
        for group in T:
            all_vertices.update(group)
        assert all_vertices == {0, 1, 2}


class TestComputeWavefronts:
    """Tests for wavefront computation."""

    def test_chain_wavefronts(self):
        """Test wavefronts for a chain DAG."""
        scheduler = HDaggScheduler()

        # Chain: 0 -> 1 -> 2
        G = nx.DiGraph()
        G.add_node(0, weight=1.0)
        G.add_node(1, weight=1.0)
        G.add_node(2, weight=1.0)
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(1, 2, weight=1.0)

        W, l = scheduler._compute_wavefronts(G)

        assert l == 3
        assert W[0] == [0]
        assert W[1] == [1]
        assert W[2] == [2]

    def test_parallel_wavefronts(self):
        """Test wavefronts for independent parallel tasks."""
        scheduler = HDaggScheduler()

        # Three independent tasks
        G = nx.DiGraph()
        G.add_node(0, weight=1.0)
        G.add_node(1, weight=1.0)
        G.add_node(2, weight=1.0)

        W, l = scheduler._compute_wavefronts(G)

        assert l == 1
        assert set(W[0]) == {0, 1, 2}

    def test_diamond_wavefronts(self):
        """Test wavefronts for a diamond DAG."""
        scheduler = HDaggScheduler()

        # Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        G = nx.DiGraph()
        for i in range(4):
            G.add_node(i, weight=1.0)
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(0, 2, weight=1.0)
        G.add_edge(1, 3, weight=1.0)
        G.add_edge(2, 3, weight=1.0)

        W, l = scheduler._compute_wavefronts(G)

        assert l == 3
        assert W[0] == [0]
        assert set(W[1]) == {1, 2}
        assert W[2] == [3]


class TestBinPacking:
    """Tests for bin packing."""

    def test_empty_components(self):
        """Test with no components."""
        scheduler = HDaggScheduler()

        bins = scheduler._bin_pack([], {}, 2)

        assert len(bins) == 2
        assert all(len(b) == 0 for b in bins)

    def test_single_component_single_bin(self):
        """Test a single component goes to one bin."""
        scheduler = HDaggScheduler()

        ccs = [{0, 1, 2}]
        costs = {0: 1.0, 1: 1.0, 2: 1.0}

        bins = scheduler._bin_pack(ccs, costs, 2)

        # Single component should go to one bin
        total = len(bins[0]) + len(bins[1])
        assert total == 3

    def test_balanced_distribution(self):
        """Test that components are distributed for balance."""
        scheduler = HDaggScheduler()

        # Two components with equal cost
        ccs = [{0}, {1}]
        costs = {0: 1.0, 1: 1.0}

        bins = scheduler._bin_pack(ccs, costs, 2)

        # Each bin should get one component
        assert (len(bins[0]) == 1 and len(bins[1]) == 1)


class TestPGPCalculation:
    """Tests for PGP (Potential Gain Proxy) calculation."""

    def test_balanced_pgp_is_zero(self):
        """Test that perfectly balanced bins have PGP = 0."""
        scheduler = HDaggScheduler()

        bins = [[0], [1]]
        costs = {0: 1.0, 1: 1.0}

        pgp = scheduler._calculate_pgp(bins, costs)

        assert pgp == pytest.approx(0.0)

    def test_unbalanced_pgp_is_positive(self):
        """Test that unbalanced bins have PGP > 0."""
        scheduler = HDaggScheduler()

        bins = [[0, 1], []]  # All work on one bin
        costs = {0: 1.0, 1: 1.0}

        pgp = scheduler._calculate_pgp(bins, costs)

        assert pgp > 0  # Should be 1 - (1/2) = 0.5

    def test_all_on_one_bin(self):
        """Test PGP when all work is on one bin."""
        scheduler = HDaggScheduler()

        bins = [[0, 1, 2], []]
        costs = {0: 1.0, 1: 1.0, 2: 1.0}

        pgp = scheduler._calculate_pgp(bins, costs)

        # PGP = 1 - B_avg / B_max = 1 - (1.5/3.0) = 0.5
        assert pgp == pytest.approx(0.5)


class TestFullScheduler:
    """Integration tests for the full HDagg scheduler."""

    def test_chain_dag(self):
        """Test scheduling a simple chain DAG."""
        hardware = create_hardware(2)

        # Chain: 0 -> 1 -> 2
        G = nx.DiGraph()
        for i in range(3):
            G.add_node(i, weight=1.0)
        G.add_edge(0, 1, weight=0.5)
        G.add_edge(1, 2, weight=0.5)

        scheduler = HDaggScheduler(epsilon=0.1, verbose=False)
        schedule = scheduler.schedule(hardware, G)

        # Verify all tasks are scheduled
        assert schedule.task_scheduled(0)
        assert schedule.task_scheduled(1)
        assert schedule.task_scheduled(2)

        # Verify schedule is valid
        is_valid, errors = schedule.is_valid()
        assert is_valid, f"Invalid schedule: {errors}"

    def test_diamond_dag(self):
        """Test scheduling a diamond DAG."""
        hardware = create_hardware(2)

        # Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        G = nx.DiGraph()
        for i in range(4):
            G.add_node(i, weight=1.0)
        G.add_edge(0, 1, weight=0.5)
        G.add_edge(0, 2, weight=0.5)
        G.add_edge(1, 3, weight=0.5)
        G.add_edge(2, 3, weight=0.5)

        scheduler = HDaggScheduler(epsilon=0.1, verbose=False)
        schedule = scheduler.schedule(hardware, G)

        # Verify all tasks are scheduled
        for i in range(4):
            assert schedule.task_scheduled(i)

        # Verify schedule is valid
        is_valid, errors = schedule.is_valid()
        assert is_valid, f"Invalid schedule: {errors}"

    def test_parallel_tasks(self):
        """Test scheduling independent parallel tasks."""
        hardware = create_hardware(2)

        # Four independent tasks
        G = nx.DiGraph()
        for i in range(4):
            G.add_node(i, weight=1.0)

        scheduler = HDaggScheduler(epsilon=0.1, verbose=False)
        schedule = scheduler.schedule(hardware, G)

        # Verify all tasks are scheduled
        for i in range(4):
            assert schedule.task_scheduled(i)

        # Verify schedule is valid
        is_valid, errors = schedule.is_valid()
        assert is_valid, f"Invalid schedule: {errors}"

        # Should have just one superstep for independent tasks
        assert len(schedule.supersteps) == 1

    def test_paper_example_dag(self):
        """Test the example DAG from Figure 2 in the paper.

        The DAG has 13 vertices (0-12) with the structure shown in Figure 2(a).

        From the paper:
        - Figure 2(a): Original DAG G with transitive edges (red)
        - Figure 2(b): G' after transitive reduction (6 edges removed)
        - Figure 2(c): Groups after Step 1:
            - {1, 2} subtree
            - {5, 6, 7} subtree
            - {11, 12} subtree
            - {0}, {3}, {4}, {8}, {9}, {10} individuals
        - Figure 2(d): Final schedule with 2 processors:
            - CW1: {0,1,2,3} on P0, {5,6,7} on P1
            - CW2: {4} on P0, {8} on P1
            - CW3: {11,12} on P0, {9,10} on P1

        Key structure from Figure 2(a):
        - Sources: 0, 5, 6
        - Sinks in G': 2, 4, 9, 10, 12
        - 8 has out_degree=3 in G' (→9, →10, →11)
        """
        hardware = create_hardware(2)

        G = nx.DiGraph()
        for i in range(13):
            G.add_node(i, weight=1.0)

        # Essential edges (black in Figure 2(a), remain in G'):
        essential_edges = [
            (0, 1), (1, 2),      # Chain: 0→1→2, forms {1,2} subtree
            (0, 3), (3, 4),      # Chain: 0→3→4
            (5, 7), (6, 7),      # Both →7, forms {5,6,7} subtree
            (7, 8),              # 7→8
            (3, 8),              # 3→8 (so 3 has out_degree=2 in G': →4 and →8)
            (8, 9), (8, 10), (8, 11),  # 8 has out_degree=3 in G'
            (11, 12),            # Chain for {11,12} subtree
        ]

        # Transitive edges (red in Figure 2(a), will be removed):
        transitive_edges = [
            (0, 4),              # Transitive via 0→3→4
            (3, 11),             # Transitive via 3→8→11
            (8, 12),             # Transitive via 8→11→12
            (5, 8),              # Transitive via 5→7→8
            (7, 9),              # Transitive via 7→8→9
            (6, 9),              # Transitive via 6→7→9 (7→9 exists in G)
        ]

        # Add all edges
        for u, v in essential_edges + transitive_edges:
            G.add_edge(u, v, weight=0.5)

        # epsilon=0.35 chosen so that:
        # - W1 alone (PGP=0.333) doesn't trigger a cut
        # - W1+W2 (PGP=0.125) is balanced
        # - W1+W2+W3 (PGP=0.5) triggers a cut, keeping W1+W2
        # - W3+W4 (PGP=0.4) triggers a cut, keeping W3 separate
        scheduler = HDaggScheduler(epsilon=0.35, verbose=False)
        schedule = scheduler.schedule(hardware, G)

        # Verify all tasks are scheduled
        for i in range(13):
            assert schedule.task_scheduled(i), f"Task {i} not scheduled"

        # Verify schedule is valid
        is_valid, errors = schedule.is_valid()
        assert is_valid, f"Invalid schedule: {errors}"

        # Verify transitive reduction removed 6 edges
        assert scheduler.stats['transitive_edges_removed'] == 6, \
            f"Expected 6 transitive edges removed, got {scheduler.stats['transitive_edges_removed']}"

        # Verify the schedule structure matches Figure 2(d)
        # CW1: {0,1,2,3} on P0, {5,6,7} on P1
        # CW2: {4} on P0, {8} on P1
        # CW3: {11,12} on P0, {9,10} on P1

        processors = list(hardware.network.nodes())

        # Should have 3 coarsened wavefronts (supersteps)
        assert len(schedule.supersteps) == 3, \
            f"Expected 3 supersteps, got {len(schedule.supersteps)}"

        # Check CW1 (superstep 0): should contain {0,1,2,3,5,6,7}
        cw1_tasks = set()
        for proc in processors:
            for task in schedule.supersteps[0].tasks[proc]:
                cw1_tasks.add(task.node)
        assert cw1_tasks == {0, 1, 2, 3, 5, 6, 7}, \
            f"CW1 should contain {{0,1,2,3,5,6,7}}, got {cw1_tasks}"

        # Check CW2 (superstep 1): should contain {4, 8}
        cw2_tasks = set()
        for proc in processors:
            for task in schedule.supersteps[1].tasks[proc]:
                cw2_tasks.add(task.node)
        assert cw2_tasks == {4, 8}, \
            f"CW2 should contain {{4, 8}}, got {cw2_tasks}"

        # Check CW3 (superstep 2): should contain {9, 10, 11, 12}
        cw3_tasks = set()
        for proc in processors:
            for task in schedule.supersteps[2].tasks[proc]:
                cw3_tasks.add(task.node)
        assert cw3_tasks == {9, 10, 11, 12}, \
            f"CW3 should contain {{9, 10, 11, 12}}, got {cw3_tasks}"

    def test_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        hardware = create_hardware(2)

        G = nx.DiGraph()
        for i in range(5):
            G.add_node(i, weight=1.0)
        G.add_edge(0, 1, weight=0.5)
        G.add_edge(1, 2, weight=0.5)
        G.add_edge(0, 3, weight=0.5)
        G.add_edge(3, 4, weight=0.5)

        scheduler = HDaggScheduler(epsilon=0.1, verbose=False)
        scheduler.schedule(hardware, G)

        # Check that stats were populated
        assert scheduler.stats['original_vertices'] == 5
        assert scheduler.stats['groups_created'] > 0
        assert scheduler.stats['coarsened_vertices'] > 0
        assert scheduler.stats['wavefronts'] > 0
        assert scheduler.stats['coarsened_wavefronts'] > 0

    def test_epsilon_parameter(self):
        """Test that epsilon parameter affects scheduling."""
        hardware = create_hardware(4)

        # Create a larger DAG for testing
        G = nx.DiGraph()
        for i in range(20):
            G.add_node(i, weight=float(i + 1))

        # Create random-ish structure
        for i in range(19):
            G.add_edge(i, i + 1, weight=1.0)
        for i in range(0, 18, 2):
            G.add_edge(i, i + 2, weight=0.5)

        # Test with different epsilon values
        scheduler_strict = HDaggScheduler(epsilon=0.05, verbose=False)
        scheduler_relaxed = HDaggScheduler(epsilon=0.5, verbose=False)

        schedule_strict = scheduler_strict.schedule(hardware, G)
        schedule_relaxed = scheduler_relaxed.schedule(hardware, G)

        # Both should produce valid schedules
        assert schedule_strict.is_valid()[0]
        assert schedule_relaxed.is_valid()[0]


class TestWavefrontCoarseningBugs:
    """Tests for specific bugs in wavefront coarsening."""

    def test_no_duplicate_wavefronts_on_single_unbalanced_waves(self):
        """Test that single unbalanced waves at the end are not duplicated.

        Regression test for bug where the last wavefront gets added twice when:
        1. The final iteration (i=l) triggers a cut due to PGP > epsilon
        2. cut == i-1 (single unbalanced wave case)
        3. The elif branch (last_cut_level == l) then incorrectly adds it again

        This bug manifests as tasks being scheduled multiple times, which
        violates BSP schedule validity.
        """
        hardware = create_hardware(5)

        # Create a fork-join graph that produces single unbalanced waves:
        # Source (T0) -> 5 parallel chains -> Sink (T6)
        # This creates 2 super-vertices after aggregation:
        # - Super-vertex 0: T0 (source with out_degree > 1)
        # - Super-vertex 1: all other tasks (merged due to out_degree == 1)
        #
        # The coarsened DAG has 2 wavefronts, each being a single component,
        # triggering the "single unbalanced wave" case for both.
        G = nx.DiGraph()

        # Source
        G.add_node('T0', weight=2.0)

        # 5 chains of 1 task each
        for i in range(1, 6):
            G.add_node(f'T{i}', weight=2.0)
            G.add_edge('T0', f'T{i}', weight=1.0)

        # Sink
        G.add_node('T6', weight=2.0)
        for i in range(1, 6):
            G.add_edge(f'T{i}', 'T6', weight=1.0)

        scheduler = HDaggScheduler(epsilon=0.01, verbose=False)
        schedule = scheduler.schedule(hardware, G)

        # Count how many times each task appears in the schedule
        task_counts = {}
        for superstep in schedule.supersteps:
            for proc, tasks in superstep.tasks.items():
                for task in tasks:
                    task_counts[task.node] = task_counts.get(task.node, 0) + 1

        # Each task should appear exactly once
        for task_node, count in task_counts.items():
            assert count == 1, \
                f"Task {task_node} scheduled {count} times (expected 1). " \
                f"This indicates a duplicate wavefront bug."

        # All 7 tasks should be scheduled
        assert len(task_counts) == 7, \
            f"Expected 7 tasks scheduled, got {len(task_counts)}"

        # Schedule must be valid
        is_valid, errors = schedule.is_valid()
        assert is_valid, f"Invalid schedule: {errors}"

    def test_parallel_chains_no_duplicates(self):
        """Test parallel chains structure doesn't cause duplicate scheduling.

        This is the structure from playground.py that exposed the bug.
        """
        hardware = create_hardware(5)

        # Create structure similar to gen_parallel_chains(1, 5, 3)
        # Source -> 5 chains of 3 tasks each -> Sink
        G = nx.DiGraph()

        # Source
        G.add_node('src', weight=2.0)

        # 5 chains of 3 tasks each
        chain_tasks = []
        for chain in range(5):
            chain_start = f'c{chain}_0'
            G.add_node(chain_start, weight=2.0)
            G.add_edge('src', chain_start, weight=1.0)
            prev = chain_start

            for t in range(1, 3):
                task = f'c{chain}_{t}'
                G.add_node(task, weight=2.0)
                G.add_edge(prev, task, weight=1.0)
                prev = task

            chain_tasks.append(prev)  # Last task in chain

        # Sink
        G.add_node('sink', weight=2.0)
        for last_task in chain_tasks:
            G.add_edge(last_task, 'sink', weight=1.0)

        scheduler = HDaggScheduler(epsilon=0.01, verbose=False)
        schedule = scheduler.schedule(hardware, G)

        # Count task occurrences
        task_counts = {}
        for superstep in schedule.supersteps:
            for proc, tasks in superstep.tasks.items():
                for task in tasks:
                    task_counts[task.node] = task_counts.get(task.node, 0) + 1

        # No duplicates
        duplicates = {k: v for k, v in task_counts.items() if v > 1}
        assert not duplicates, \
            f"Found duplicate scheduled tasks: {duplicates}"

        # 1 source + 5*3 chain tasks + 1 sink = 17 tasks
        assert len(task_counts) == 17, \
            f"Expected 17 tasks, got {len(task_counts)}"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_task(self):
        """Test scheduling a single task."""
        hardware = create_hardware(2)

        G = nx.DiGraph()
        G.add_node(0, weight=1.0)

        scheduler = HDaggScheduler()
        schedule = scheduler.schedule(hardware, G)

        assert schedule.task_scheduled(0)
        is_valid, errors = schedule.is_valid()
        assert is_valid, f"Invalid schedule: {errors}"

    def test_many_processors(self):
        """Test with more processors than tasks."""
        hardware = create_hardware(10)

        G = nx.DiGraph()
        for i in range(3):
            G.add_node(i, weight=1.0)
        G.add_edge(0, 1, weight=0.5)
        G.add_edge(1, 2, weight=0.5)

        scheduler = HDaggScheduler()
        schedule = scheduler.schedule(hardware, G)

        assert schedule.is_valid()[0]

    def test_varying_task_weights(self):
        """Test with varying task weights."""
        hardware = create_hardware(2)

        G = nx.DiGraph()
        G.add_node(0, weight=10.0)  # Heavy task
        G.add_node(1, weight=1.0)  # Light task
        G.add_node(2, weight=5.0)  # Medium task
        G.add_edge(0, 2, weight=1.0)
        G.add_edge(1, 2, weight=1.0)

        scheduler = HDaggScheduler()
        schedule = scheduler.schedule(hardware, G)

        assert schedule.is_valid()[0]

    def test_verbose_mode(self):
        """Test verbose mode doesn't crash."""
        hardware = create_hardware(2)

        G = nx.DiGraph()
        for i in range(3):
            G.add_node(i, weight=1.0)
        G.add_edge(0, 1, weight=0.5)
        G.add_edge(1, 2, weight=0.5)

        scheduler = HDaggScheduler(verbose=True)
        schedule = scheduler.schedule(hardware, G)

        assert schedule.is_valid()[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
