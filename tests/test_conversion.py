import pytest
import networkx as nx
from saga.scheduler import Task
from saga_bsp.conversion import convert_async_to_bsp
from saga_bsp.schedule import BSPSchedule, BSPHardware


def create_simple_hardware():
    """Create a simple 2-processor BSP hardware."""
    network = nx.Graph()
    network.add_node("proc1", weight=1.0)
    network.add_node("proc2", weight=1.0)
    network.add_edge("proc1", "proc2", weight=1.0)
    return BSPHardware(network=network, sync_time=0.0)


def create_diamond_task_graph():
    """Create a diamond-shaped task dependency graph"""
    task_graph = nx.DiGraph()
    task_graph.add_node("A", weight=10.0)
    task_graph.add_node("B", weight=10.0)
    task_graph.add_node("C", weight=10.0)
    task_graph.add_node("D", weight=10.0)
    
    task_graph.add_edge("A", "B", weight=5.0)
    task_graph.add_edge("A", "C", weight=5.0)
    task_graph.add_edge("B", "D", weight=5.0)
    task_graph.add_edge("C", "D", weight=5.0)
    
    return task_graph


def test_simple_conversion():
    """Test basic conversion of a simple async schedule"""
    hardware = create_simple_hardware()
    task_graph = create_diamond_task_graph()
    
    # Create async schedule: A on proc1, B and C on proc2, D on proc1
    async_schedule = {
        "proc1": [
            Task("proc1", "A", 0.0, 10.0),
            Task("proc1", "D", 25.0, 35.0)
        ],
        "proc2": [
            Task("proc2", "B", 15.0, 25.0),
            Task("proc2", "C", 15.0, 25.0)
        ]
    }
    
    # Convert to BSP
    bsp_schedule = convert_async_to_bsp(hardware, task_graph, async_schedule)
    
    # Basic validation
    assert isinstance(bsp_schedule, BSPSchedule)
    assert bsp_schedule.num_supersteps > 0
    
    # Check that all tasks are scheduled
    scheduled_tasks = set()
    for superstep in bsp_schedule.supersteps:
        for proc, tasks in superstep.tasks.items():
            for task in tasks:
                scheduled_tasks.add(task.node)
    
    assert scheduled_tasks == {"A", "B", "C", "D"}


def test_superstep_dependencies():
    """Test that BSP supersteps respect communication dependencies"""
    hardware = create_simple_hardware()
    task_graph = create_diamond_task_graph()
    
    # Create async schedule with inter-processor dependencies
    async_schedule = {
        "proc1": [
            Task("proc1", "A", 0.0, 10.0),  # First task
        ],
        "proc2": [
            Task("proc2", "B", 15.0, 25.0),  # Depends on A (different proc)
            Task("proc2", "C", 15.0, 25.0),  # Depends on A (different proc)
            Task("proc2", "D", 30.0, 40.0)   # Depends on B and C (same proc)
        ]
    }
    
    bsp_schedule = convert_async_to_bsp(hardware, task_graph, async_schedule)
    
    # Verify task placement in supersteps
    task_to_superstep = {}
    for i, superstep in enumerate(bsp_schedule.supersteps):
        for proc, tasks in superstep.tasks.items():
            for task in tasks:
                task_to_superstep[task.node] = i
    
    # A should be in earlier superstep than B and C (inter-processor dependency)
    assert task_to_superstep["A"] < task_to_superstep["B"]
    assert task_to_superstep["A"] < task_to_superstep["C"]
    
    # B and C can be in same superstep (no dependency between them)
    # D should be in later superstep than B and C
    assert task_to_superstep["D"] >= task_to_superstep["B"]
    assert task_to_superstep["D"] >= task_to_superstep["C"]


def test_linear_chain():
    """Test conversion of a linear task chain"""
    hardware = create_simple_hardware()
    
    # Linear chain task graph
    task_graph = nx.DiGraph()
    task_graph.add_node("T1", weight=10.0)
    task_graph.add_node("T2", weight=10.0)
    task_graph.add_node("T3", weight=10.0)
    task_graph.add_edge("T1", "T2", weight=5.0)
    task_graph.add_edge("T2", "T3", weight=5.0)
    
    # Schedule tasks on alternating processors (forces communication)
    async_schedule = {
        "proc1": [
            Task("proc1", "T1", 0.0, 10.0),
            Task("proc1", "T3", 25.0, 35.0)
        ],
        "proc2": [
            Task("proc2", "T2", 15.0, 25.0)
        ]
    }
    
    bsp_schedule = convert_async_to_bsp(hardware, task_graph, async_schedule)
    
    # Should have 3 supersteps due to alternating processor assignments
    assert bsp_schedule.num_supersteps >= 2
    
    # Verify ordering
    task_to_superstep = {}
    for i, superstep in enumerate(bsp_schedule.supersteps):
        for proc, tasks in superstep.tasks.items():
            for task in tasks:
                task_to_superstep[task.node] = i
    
    assert task_to_superstep["T1"] < task_to_superstep["T2"]
    assert task_to_superstep["T2"] < task_to_superstep["T3"]


def test_empty_schedule():
    """Test handling of empty async schedule"""
    hardware = create_simple_hardware()
    task_graph = nx.DiGraph()
    
    async_schedule = {"proc1": [], "proc2": []}
    
    bsp_schedule = convert_async_to_bsp(hardware, task_graph, async_schedule)
    
    assert bsp_schedule.num_supersteps == 0
    assert bsp_schedule.makespan == 0.0


def test_single_task():
    """Test conversion with single task"""
    hardware = create_simple_hardware()
    
    task_graph = nx.DiGraph()
    task_graph.add_node("T1", weight=10.0)
    
    async_schedule = {
        "proc1": [Task("proc1", "T1", 0.0, 10.0)],
        "proc2": []
    }
    
    bsp_schedule = convert_async_to_bsp(hardware, task_graph, async_schedule)
    
    assert bsp_schedule.num_supersteps == 1
    assert len(bsp_schedule.supersteps[0].tasks["proc1"]) == 1
    assert bsp_schedule.supersteps[0].tasks["proc1"][0].node == "T1"


if __name__ == "__main__":
    pytest.main([__file__])