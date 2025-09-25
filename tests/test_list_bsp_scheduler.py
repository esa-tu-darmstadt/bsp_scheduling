"""Comprehensive tests for the ListBSPScheduler."""

import pytest
import networkx as nx
from saga_bsp.schedulers import ListBSPScheduler
from saga_bsp.schedule import BSPSchedule, BSPHardware, Superstep


def create_simple_hardware(num_processors=3, sync_time=10.0):
    """Create a simple fully-connected hardware configuration."""
    network = nx.complete_graph(num_processors)
    
    # Set processor speeds
    for node in network.nodes():
        network.nodes[node]['weight'] = 1.0  # All processors same speed
    
    # Set network speeds
    for edge in network.edges():
        network.edges[edge]['weight'] = 1.0  # All connections same speed
    
    return BSPHardware(network, sync_time)


def create_chain_task_graph(num_tasks=4):
    """Create a simple chain task graph: A -> B -> C -> D."""
    task_graph = nx.DiGraph()
    
    # Add tasks with computation weights
    for i in range(num_tasks):
        task_name = chr(65 + i)  # A, B, C, D
        task_graph.add_node(task_name, weight=100.0)
    
    # Add dependencies
    for i in range(num_tasks - 1):
        src = chr(65 + i)
        dst = chr(65 + i + 1)
        task_graph.add_edge(src, dst, weight=50.0)  # Communication weight
    
    return task_graph


def create_diamond_task_graph():
    """Create a diamond-shaped task graph."""
    task_graph = nx.DiGraph()
    
    # Add tasks
    task_graph.add_node('A', weight=100.0)
    task_graph.add_node('B', weight=150.0)
    task_graph.add_node('C', weight=200.0)
    task_graph.add_node('D', weight=100.0)
    
    # Add dependencies (diamond shape)
    task_graph.add_edge('A', 'B', weight=50.0)
    task_graph.add_edge('A', 'C', weight=75.0)
    task_graph.add_edge('B', 'D', weight=50.0)
    task_graph.add_edge('C', 'D', weight=50.0)
    
    return task_graph


def create_fork_join_task_graph():
    """Create a fork-join task graph."""
    task_graph = nx.DiGraph()
    
    # Add tasks
    task_graph.add_node('A', weight=100.0)
    task_graph.add_node('B', weight=150.0)
    task_graph.add_node('C', weight=150.0)
    task_graph.add_node('D', weight=150.0)
    task_graph.add_node('E', weight=100.0)
    
    # Fork from A to B, C, D
    task_graph.add_edge('A', 'B', weight=50.0)
    task_graph.add_edge('A', 'C', weight=50.0)
    task_graph.add_edge('A', 'D', weight=50.0)
    
    # Join from B, C, D to E
    task_graph.add_edge('B', 'E', weight=50.0)
    task_graph.add_edge('C', 'E', weight=50.0)
    task_graph.add_edge('D', 'E', weight=50.0)
    
    return task_graph


class TestListBSPScheduler:
    """Test suite for ListBSPScheduler."""
    
    def test_initialization(self):
        """Test scheduler initialization."""
        scheduler = ListBSPScheduler()
        assert scheduler.name == "ListBSP"
        
        scheduler_verbose = ListBSPScheduler(verbose=True)
        assert scheduler_verbose.verbose is True
    
    def test_chain_schedule(self):
        """Test scheduling a chain task graph."""
        hardware = create_simple_hardware()
        task_graph = create_chain_task_graph()
        scheduler = ListBSPScheduler()
        
        schedule = scheduler.schedule(hardware, task_graph)
        
        # Verify all tasks are scheduled
        assert len(schedule.task_mapping) == 4
        for task in ['A', 'B', 'C', 'D']:
            assert schedule.task_scheduled(task)
        
        # Verify schedule is valid
        is_valid, errors = schedule.is_valid()
        assert is_valid, f"Schedule validation failed: {errors}"
        
        # Check that dependencies are respected
        assert schedule['A'][0].end <= schedule['B'][0].start
        assert schedule['B'][0].end <= schedule['C'][0].start
        assert schedule['C'][0].end <= schedule['D'][0].start
    
    def test_diamond_schedule(self):
        """Test scheduling a diamond task graph."""
        hardware = create_simple_hardware()
        task_graph = create_diamond_task_graph()
        scheduler = ListBSPScheduler()
        
        schedule = scheduler.schedule(hardware, task_graph)
        
        # Verify all tasks are scheduled
        assert len(schedule.task_mapping) == 4
        for task in ['A', 'B', 'C', 'D']:
            assert schedule.task_scheduled(task)
        
        # Verify schedule is valid
        is_valid, errors = schedule.is_valid()
        assert is_valid, f"Schedule validation failed: {errors}"
        
        # Check that A finishes before B and C start
        assert schedule['A'][0].end <= schedule['B'][0].start
        assert schedule['A'][0].end <= schedule['C'][0].start
        
        # Check that B and C finish before D starts
        assert schedule['B'][0].end <= schedule['D'][0].start
        assert schedule['C'][0].end <= schedule['D'][0].start
    
    def test_fork_join_schedule(self):
        """Test scheduling a fork-join task graph."""
        hardware = create_simple_hardware()
        task_graph = create_fork_join_task_graph()
        scheduler = ListBSPScheduler(verbose=True)
        
        schedule = scheduler.schedule(hardware, task_graph)
        
        # Verify all tasks are scheduled
        assert len(schedule.task_mapping) == 5
        for task in ['A', 'B', 'C', 'D', 'E']:
            assert schedule.task_scheduled(task)
        
        # Verify schedule is valid
        is_valid, errors = schedule.is_valid()
        assert is_valid, f"Schedule validation failed: {errors}"
        
        # Check fork dependencies
        assert schedule['A'][0].end <= schedule['B'][0].start
        assert schedule['A'][0].end <= schedule['C'][0].start
        assert schedule['A'][0].end <= schedule['D'][0].start
        
        # Check join dependencies
        assert schedule['B'][0].end <= schedule['E'][0].start
        assert schedule['C'][0].end <= schedule['E'][0].start
        assert schedule['D'][0].end <= schedule['E'][0].start
    
    def test_sequential_execution_high_sync_cost(self):
        """Test that with high sync cost, tasks execute sequentially on same processor."""
        # High sync cost makes it better to avoid creating new supersteps
        hardware = create_simple_hardware(num_processors=3, sync_time=1000.0)
        
        # Create diamond with small communication costs
        task_graph = nx.DiGraph()
        task_graph.add_node('A', weight=100.0)
        task_graph.add_node('B', weight=150.0)
        task_graph.add_node('C', weight=200.0)
        task_graph.add_node('D', weight=100.0)
        task_graph.add_edge('A', 'B', weight=10.0)  # Small communication
        task_graph.add_edge('A', 'C', weight=10.0)  # Small communication
        task_graph.add_edge('B', 'D', weight=10.0)
        task_graph.add_edge('C', 'D', weight=10.0)
        
        scheduler = ListBSPScheduler()
        schedule = scheduler.schedule(hardware, task_graph)
        
        # Validate the schedule
        schedule.assert_valid()
        
        # With high sync cost, optimal is to put everything in one superstep
        assert len(schedule.supersteps) == 1, f"Expected 1 superstep, got {len(schedule.supersteps)}"
        
        # All tasks should be on the same processor to avoid communication
        a_proc = schedule['A'][0].proc
        b_proc = schedule['B'][0].proc
        c_proc = schedule['C'][0].proc
        d_proc = schedule['D'][0].proc
        
        assert a_proc == b_proc == c_proc == d_proc, \
            f"All tasks should be on same processor: A={a_proc}, B={b_proc}, C={c_proc}, D={d_proc}"
    
    def test_parallel_execution_low_sync_cost(self):
        """Test that with low sync cost, independent tasks execute in parallel."""
        # Low sync cost makes parallel execution worthwhile
        hardware = create_simple_hardware(num_processors=3, sync_time=1.0)
        
        # Create diamond with larger tasks to make parallel execution beneficial
        task_graph = nx.DiGraph()
        task_graph.add_node('A', weight=100.0)
        task_graph.add_node('B', weight=500.0)  # Large task
        task_graph.add_node('C', weight=500.0)  # Large task
        task_graph.add_node('D', weight=100.0)
        task_graph.add_edge('A', 'B', weight=10.0)  # Small communication
        task_graph.add_edge('A', 'C', weight=10.0)  # Small communication
        task_graph.add_edge('B', 'D', weight=10.0)
        task_graph.add_edge('C', 'D', weight=10.0)
        
        scheduler = ListBSPScheduler(verbose=True)
        schedule = scheduler.schedule(hardware, task_graph)
        
        # Validate the schedule
        schedule.assert_valid()
        
        # B and C should execute in parallel (same superstep, different processors)
        b_task = schedule['B'][0]
        c_task = schedule['C'][0]
        
        # They should be in the same superstep
        assert b_task.superstep.index == c_task.superstep.index, \
            f"B and C should be in same superstep: B={b_task.superstep.index}, C={c_task.superstep.index}"
        
        # They should be on different processors for true parallelism
        assert b_task.proc != c_task.proc, \
            f"B and C should be on different processors: B={b_task.proc}, C={c_task.proc}"
    
    def test_superstep_splitting(self):
        """Test that the scheduler can split supersteps when beneficial."""
        hardware = create_simple_hardware(num_processors=2, sync_time=5.0)
        
        # Create a task graph that would benefit from splitting
        task_graph = nx.DiGraph()
        task_graph.add_node('A', weight=100.0)
        task_graph.add_node('B', weight=200.0)
        task_graph.add_node('C', weight=100.0)
        task_graph.add_node('D', weight=100.0)
        
        task_graph.add_edge('A', 'C', weight=50.0)
        task_graph.add_edge('B', 'D', weight=50.0)
        
        scheduler = ListBSPScheduler(verbose=True)
        schedule = scheduler.schedule(hardware, task_graph)
        
        # Verify schedule is valid
        is_valid, errors = schedule.is_valid()
        assert is_valid, f"Schedule validation failed: {errors}"
        
        # Check that all tasks are scheduled
        assert len(schedule.task_mapping) == 4
    
    def test_makespan_optimization(self):
        """Test that the scheduler optimizes for makespan."""
        hardware = create_simple_hardware()
        task_graph = create_diamond_task_graph()
        
        scheduler = ListBSPScheduler()
        schedule = scheduler.schedule(hardware, task_graph)
        
        # Compare with a naive approach (all tasks in sequence)
        # With optimization, the makespan should be better than sequential
        
        # Calculate sequential makespan (all tasks on one processor)
        sequential_makespan = 0.0
        for node in ['A', 'B', 'C', 'D']:
            sequential_makespan += task_graph.nodes[node]['weight'] / 1.0
        
        # Add synchronization overhead for each transition
        sequential_makespan += hardware.sync_time * 3  # 3 transitions
        
        # Optimized schedule should be better
        assert schedule.makespan <= sequential_makespan
    
    def test_empty_graph(self):
        """Test scheduling an empty task graph."""
        hardware = create_simple_hardware()
        task_graph = nx.DiGraph()
        scheduler = ListBSPScheduler()
        
        schedule = scheduler.schedule(hardware, task_graph)
        
        # Should handle empty graph gracefully
        assert len(schedule.task_mapping) == 0
        assert schedule.makespan == 0.0
    
    def test_single_task(self):
        """Test scheduling a single task."""
        hardware = create_simple_hardware()
        task_graph = nx.DiGraph()
        task_graph.add_node('A', weight=100.0)
        
        scheduler = ListBSPScheduler()
        schedule = scheduler.schedule(hardware, task_graph)
        
        # Task should be scheduled
        assert schedule.task_scheduled('A')
        assert len(schedule.supersteps) >= 1
        
        # Makespan should be task execution time plus sync time (BSP includes sync)
        # For BSP, even a single superstep has sync overhead
        expected_makespan = 100.0 + hardware.sync_time
        assert schedule.makespan == expected_makespan
    
    def test_heterogeneous_processors(self):
        """Test scheduling with processors of different speeds."""
        # Create hardware with different processor speeds
        network = nx.complete_graph(3)
        network.nodes[0]['weight'] = 1.0   # Slow processor
        network.nodes[1]['weight'] = 2.0   # Fast processor
        network.nodes[2]['weight'] = 1.5   # Medium processor
        
        for edge in network.edges():
            network.edges[edge]['weight'] = 1.0
        
        hardware = BSPHardware(network, 10.0)
        task_graph = create_chain_task_graph()
        
        scheduler = ListBSPScheduler()
        schedule = scheduler.schedule(hardware, task_graph)
        
        # Verify schedule is valid
        is_valid, errors = schedule.is_valid()
        assert is_valid, f"Schedule validation failed: {errors}"
        
        # Fast processor should be preferred for critical tasks
        # (though exact placement depends on the algorithm's decisions)
        assert all(schedule.task_scheduled(task) for task in ['A', 'B', 'C', 'D'])


class TestSuperstepSplitting:
    """Test suite specifically for superstep splitting functionality."""
    
    def test_split_superstep_basic(self):
        """Test basic superstep splitting."""
        hardware = create_simple_hardware()
        task_graph = create_chain_task_graph()
        
        # Create a schedule manually to test splitting
        schedule = BSPSchedule(hardware, task_graph)
        superstep = schedule.add_superstep()
        
        # Schedule A and B in the same superstep on same processor
        schedule.schedule('A', 0, superstep)
        schedule.schedule('B', 0, superstep)
        
        # Get the time when A finishes
        a_task = schedule['A'][0]
        split_time = a_task.end
        
        # Split the superstep
        new_superstep = schedule.split_superstep(superstep, split_time)
        
        if new_superstep:
            # B should be in the new superstep
            b_task = schedule['B'][0]
            assert b_task.superstep == new_superstep
            
            # A should remain in original superstep
            assert a_task.superstep == superstep
    
    def test_split_empty_result(self):
        """Test splitting when no tasks would be moved."""
        hardware = create_simple_hardware()
        task_graph = nx.DiGraph()
        task_graph.add_node('A', weight=100.0)
        task_graph.add_node('B', weight=100.0)
        
        schedule = BSPSchedule(hardware, task_graph)
        superstep = schedule.add_superstep()
        schedule.schedule('A', 0, superstep)
        schedule.schedule('B', 0, superstep)
        
        # Try to split right after A but before B starts
        a_task = schedule['A'][0]
        b_task = schedule['B'][0]
        
        # Split at a time when A has finished but B hasn't started yet
        split_time = (a_task.end + b_task.start) / 2
        
        # This should move B to a new superstep
        new_superstep = schedule.split_superstep(superstep, split_time)
        assert new_superstep is not None
    
    def test_split_invalid_time(self):
        """Test splitting with invalid split times."""
        hardware = create_simple_hardware()
        task_graph = nx.DiGraph()
        task_graph.add_node('A', weight=100.0)
        
        schedule = BSPSchedule(hardware, task_graph)
        superstep = schedule.add_superstep()
        schedule.schedule('A', 0, superstep)
        
        # Try to split before superstep starts
        with pytest.raises(ValueError):
            schedule.split_superstep(superstep, -1.0)
        
        # Try to split after superstep ends
        with pytest.raises(ValueError):
            schedule.split_superstep(superstep, superstep.end_time + 1)
    
    def test_scheduler_uses_splitting(self):
        """Test that ListBSPScheduler can use splitting when beneficial."""
        # Very low sync cost to make splitting attractive
        hardware = create_simple_hardware(num_processors=2, sync_time=1.0)
        
        # Create a scenario where splitting would be beneficial
        task_graph = nx.DiGraph()
        
        # First task
        task_graph.add_node('A', weight=100.0)
        
        # Long task that A doesn't depend on
        task_graph.add_node('B', weight=1000.0)
        
        # Task dependent on A that should run in parallel with B
        task_graph.add_node('C', weight=100.0)
        task_graph.add_edge('A', 'C', weight=1.0)
        
        scheduler = ListBSPScheduler(verbose=False)
        schedule = scheduler.schedule(hardware, task_graph)
        
        # Verify schedule is valid
        is_valid, errors = schedule.is_valid()
        assert is_valid, f"Schedule validation failed: {errors}"
        
        # The scheduler should recognize that running C in parallel with B is beneficial
        # This could be achieved through splitting or smart initial placement
        c_task = schedule['C'][0]
        b_task = schedule['B'][0]
        
        # Check that C doesn't wait for B to finish (they should overlap in execution)
        # If C starts after B ends, that's suboptimal
        assert c_task.start < b_task.end, \
            f"C should start before B ends for parallelism: C.start={c_task.start}, B.end={b_task.end}"