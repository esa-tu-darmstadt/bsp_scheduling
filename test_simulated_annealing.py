#!/usr/bin/env python3
"""Test script for BSP simulated annealing optimization"""

import networkx as nx
import logging
from typing import List
from src.saga_bsp.schedule import BSPSchedule, BSPHardware
from src.saga_bsp.optimization import BSPSimulatedAnnealing, MoveTaskToSuperstep, MoveTaskToProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_hardware():
    """Create a simple test hardware configuration"""
    # Create a simple network with 3 processors
    network = nx.Graph()
    network.add_node("P0", weight=1.0)  # Processing speed
    network.add_node("P1", weight=2.0)
    network.add_node("P2", weight=1.5)
    
    # Add communication links with bandwidth
    network.add_edge("P0", "P1", weight=10.0)  # Communication speed
    network.add_edge("P1", "P2", weight=8.0)
    network.add_edge("P0", "P2", weight=6.0)
    
    hardware = BSPHardware(network=network, sync_time=1.0)
    return hardware


def create_test_task_graph():
    """Create a simple test task graph"""
    # Create a diamond-shaped task dependency graph
    task_graph = nx.DiGraph()
    
    # Add tasks with computation costs
    task_graph.add_node("T1", weight=10.0)  # Entry task
    task_graph.add_node("T2", weight=15.0)  # Left branch
    task_graph.add_node("T3", weight=12.0)  # Right branch  
    task_graph.add_node("T4", weight=8.0)   # Join task
    task_graph.add_node("T5", weight=20.0)  # Exit task
    
    # Add dependencies with communication costs
    task_graph.add_edge("T1", "T2", weight=5.0)
    task_graph.add_edge("T1", "T3", weight=3.0)
    task_graph.add_edge("T2", "T4", weight=4.0)
    task_graph.add_edge("T3", "T4", weight=6.0)
    task_graph.add_edge("T4", "T5", weight=2.0)
    
    return task_graph


def create_initial_schedule(hardware, task_graph):
    """Create an initial (possibly suboptimal) BSP schedule"""
    schedule = BSPSchedule(hardware, task_graph)
    
    # Create supersteps and assign tasks
    # Superstep 0: T1
    ss0 = schedule.add_superstep()
    schedule.schedule("T1", "P0", ss0)
    
    # Superstep 1: T2, T3
    ss1 = schedule.add_superstep()
    schedule.schedule("T2", "P1", ss1)
    schedule.schedule("T3", "P2", ss1)
    
    # Superstep 2: T4
    ss2 = schedule.add_superstep()
    schedule.schedule("T4", "P0", ss2)
    
    # Superstep 3: T5
    ss3 = schedule.add_superstep()
    schedule.schedule("T5", "P1", ss3)
    
    return schedule


def print_schedule(schedule, title):
    """Print schedule details"""
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Makespan: {schedule.makespan:.2f}")
    print(f"Number of supersteps: {schedule.num_supersteps}")
    
    for i, superstep in enumerate(schedule.supersteps):
        print(f"\nSuperstep {i} (start: {superstep.start_time:.2f}, end: {superstep.end_time:.2f}):")
        print(f"  Sync time: {superstep.sync_time:.2f}")
        print(f"  Exchange time: {superstep.exchange_time:.2f}")
        print(f"  Compute time: {superstep.compute_time:.2f}")
        print(f"  Total time: {superstep.total_time:.2f}")
        
        for proc, tasks in superstep.tasks.items():
            if tasks:
                task_info = ", ".join([f"{t.node}({t.duration:.1f})" for t in tasks])
                print(f"    {proc}: {task_info}")


def test_individual_actions():
    """Test individual actions"""
    print("\n" + "="*50)
    print("TESTING INDIVIDUAL ACTIONS")
    print("="*50)
    
    hardware = create_test_hardware()
    task_graph = create_test_task_graph()
    schedule = create_initial_schedule(hardware, task_graph)
    
    print_schedule(schedule, "Initial Schedule")
    
    # Test MoveTaskToSuperstep
    print("\n--- Testing MoveTaskToSuperstep ---")
    action = MoveTaskToSuperstep(2)
    
    # Try to move T2 from superstep 1 to superstep 2
    feasible = action.is_feasible(schedule, "T2")
    print(f"Moving T2 to superstep 2 feasible: {feasible}")
    
    if feasible:
        test_schedule = schedule.copy()
        success = action.apply(test_schedule, "T2")
        print(f"Action applied successfully: {success}")
        print_schedule(test_schedule, "After Moving T2 to Superstep 2")
    
    # Test MoveTaskToProcessor 
    print("\n--- Testing MoveTaskToProcessor ---")
    action2 = MoveTaskToProcessor("P2")
    
    # Try to move T2 from P1 to P2 (in same superstep)
    feasible2 = action2.is_feasible(schedule, "T2")
    print(f"Moving T2 to processor P2 feasible: {feasible2}")
    
    if feasible2:
        test_schedule2 = schedule.copy()
        success2 = action2.apply(test_schedule2, "T2")
        print(f"Action applied successfully: {success2}")
        print_schedule(test_schedule2, "After Moving T2 to Processor P2")


def test_simulated_annealing():
    """Test the full simulated annealing optimization"""
    print("\n" + "="*50)
    print("TESTING SIMULATED ANNEALING OPTIMIZATION")
    print("="*50)
    
    hardware = create_test_hardware()
    task_graph = create_test_task_graph()
    initial_schedule = create_initial_schedule(hardware, task_graph)
    
    print_schedule(initial_schedule, "Initial Schedule")
    
    # Create optimizer
    optimizer = BSPSimulatedAnnealing(
        max_iterations=100,
        max_temp=10.0,
        min_temp=0.1,
        cooling_rate=0.95
    )
    
    # Run optimization
    print("\nRunning simulated annealing optimization...")
    optimized_schedule = optimizer.optimize(initial_schedule)
    
    print_schedule(optimized_schedule, "Optimized Schedule")
    
    # Print optimization statistics
    stats = optimizer.get_optimization_stats()
    print("\nOptimization Statistics:")
    print(f"  Total iterations: {stats['total_iterations']}")
    print(f"  Accepted iterations: {stats['accepted_iterations']}")
    print(f"  Applied iterations: {stats['applied_iterations']}")
    print(f"  Acceptance rate: {stats['acceptance_rate']:.2%}")
    print(f"  Initial energy: {stats['initial_energy']:.2f}")
    print(f"  Final energy: {stats['final_energy']:.2f}")
    print(f"  Improvement: {stats['improvement']:.2f} ({stats['improvement_percent']:.1f}%)")


def test_extensibility():
    """Test that the framework is extensible by creating a custom action"""
    print("\n" + "="*50)
    print("TESTING EXTENSIBILITY")
    print("="*50)
    
    from src.saga_bsp.optimization.simulated_annealing import ScheduleAction
    
    class SwapTwoTasks(ScheduleAction):
        """Custom action to swap two tasks' positions"""
        
        def __init__(self, other_task: str):
            self.other_task = other_task
        
        def is_feasible(self, schedule, task_node: str) -> bool:
            # For simplicity, just check if both tasks exist
            return (task_node in schedule.task_mapping and 
                    self.other_task in schedule.task_mapping and
                    task_node != self.other_task)
        
        def apply(self, schedule, task_node: str) -> bool:
            if not self.is_feasible(schedule, task_node):
                return False
            
            task1 = schedule.task_mapping[task_node]
            task2 = schedule.task_mapping[self.other_task]
            
            # Swap processors and supersteps
            old_proc1, old_ss1 = task1.proc, task1.superstep
            old_proc2, old_ss2 = task2.proc, task2.superstep
            
            # Remove both tasks
            schedule.unschedule(task1)
            schedule.unschedule(task2)
            
            # Add them in swapped positions
            schedule.schedule(task_node, old_proc2, old_ss2)
            schedule.schedule(self.other_task, old_proc1, old_ss1)
            
            return True
        
        def get_possible_targets(self, schedule, task_node: str) -> List:
            return [t for t in schedule.task_mapping.keys() if t != task_node]
        
        def __str__(self) -> str:
            return f"SwapTwoTasks(other={self.other_task})"
    
    # Test the custom action
    hardware = create_test_hardware()
    task_graph = create_test_task_graph()
    schedule = create_initial_schedule(hardware, task_graph)
    
    print_schedule(schedule, "Before Swap")
    
    # Try to swap T2 and T3
    swap_action = SwapTwoTasks("T3")
    feasible = swap_action.is_feasible(schedule, "T2")
    print(f"\nSwapping T2 and T3 feasible: {feasible}")
    
    if feasible:
        test_schedule = schedule.copy()
        success = swap_action.apply(test_schedule, "T2")
        print(f"Swap applied successfully: {success}")
        print_schedule(test_schedule, "After Swapping T2 and T3")
    
    print("\nExtensibility test shows that custom actions can be easily added!")


def main():
    """Main test function"""
    print("BSP Simulated Annealing Optimization Test")
    print("=" * 50)
    
    try:
        # Test individual actions
        test_individual_actions()
        
        # Test full simulated annealing
        test_simulated_annealing()
        
        # Test extensibility
        test_extensibility()
        
        print("\n" + "="*50)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()