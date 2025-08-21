#!/usr/bin/env python3
"""Debug version of BSP simulated annealing test with more detailed analysis"""

import networkx as nx
import logging
from typing import List
from src.saga_bsp.schedule import BSPSchedule, BSPHardware
from src.saga_bsp.optimization import BSPSimulatedAnnealing, MoveTaskToSuperstep, MoveTaskToProcessor

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_hardware():
    """Create a simple test hardware configuration"""
    # Create a simple network with 3 processors of different speeds
    network = nx.Graph()
    network.add_node("P0", weight=1.0)  # Slow processor
    network.add_node("P1", weight=3.0)  # Fast processor  
    network.add_node("P2", weight=2.0)  # Medium processor
    
    # Add communication links with different bandwidths
    network.add_edge("P0", "P1", weight=10.0)  # Fast communication
    network.add_edge("P1", "P2", weight=5.0)   # Medium communication
    network.add_edge("P0", "P2", weight=2.0)   # Slow communication
    
    hardware = BSPHardware(network=network, sync_time=0.5)
    return hardware


def create_test_task_graph():
    """Create a more complex test task graph"""
    task_graph = nx.DiGraph()
    
    # Create a more complex graph with multiple paths
    tasks = {
        "T1": 20.0,   # Entry task - heavy computation
        "T2": 5.0,    # Light task
        "T3": 15.0,   # Medium task
        "T4": 30.0,   # Heavy task
        "T5": 8.0,    # Light task
        "T6": 12.0,   # Medium task
        "T7": 25.0    # Exit task - heavy computation
    }
    
    for task, weight in tasks.items():
        task_graph.add_node(task, weight=weight)
    
    # Create dependencies with communication costs
    dependencies = [
        ("T1", "T2", 2.0),
        ("T1", "T3", 4.0),
        ("T2", "T4", 1.0),
        ("T3", "T4", 3.0),
        ("T2", "T5", 1.5),
        ("T4", "T6", 2.5),
        ("T5", "T6", 1.0),
        ("T4", "T7", 5.0),
        ("T6", "T7", 3.0)
    ]
    
    for src, dst, weight in dependencies:
        task_graph.add_edge(src, dst, weight=weight)
    
    return task_graph


def create_deliberately_bad_schedule(hardware, task_graph):
    """Create a deliberately suboptimal BSP schedule for testing"""
    schedule = BSPSchedule(hardware, task_graph)
    
    # Strategy: Put all tasks on the slowest processor (P0) in separate supersteps
    # This should be highly suboptimal
    
    # Superstep 0: T1 on slow processor
    ss0 = schedule.add_superstep()
    schedule.schedule("T1", "P0", ss0)
    
    # Superstep 1: T2 on slow processor (could be parallel with T3)
    ss1 = schedule.add_superstep()
    schedule.schedule("T2", "P0", ss1)
    
    # Superstep 2: T3 on slow processor (could be parallel with T2)
    ss2 = schedule.add_superstep()
    schedule.schedule("T3", "P0", ss2)
    
    # Superstep 3: T4 on slow processor
    ss3 = schedule.add_superstep()
    schedule.schedule("T4", "P0", ss3)
    
    # Superstep 4: T5 on slow processor
    ss4 = schedule.add_superstep()
    schedule.schedule("T5", "P0", ss4)
    
    # Superstep 5: T6 on slow processor
    ss5 = schedule.add_superstep()
    schedule.schedule("T6", "P0", ss5)
    
    # Superstep 6: T7 on slow processor
    ss6 = schedule.add_superstep()
    schedule.schedule("T7", "P0", ss6)
    
    return schedule


def create_better_initial_schedule(hardware, task_graph):
    """Create a reasonable initial schedule for comparison"""
    schedule = BSPSchedule(hardware, task_graph)
    
    # Better strategy: Use parallelism and faster processors
    
    # Superstep 0: T1 on fastest processor
    ss0 = schedule.add_superstep()
    schedule.schedule("T1", "P1", ss0)
    
    # Superstep 1: T2 and T3 in parallel on different processors
    ss1 = schedule.add_superstep()
    schedule.schedule("T2", "P1", ss1)  # Fast processor for T2
    schedule.schedule("T3", "P2", ss1)  # Medium processor for T3
    
    # Superstep 2: T4, T5 in parallel
    ss2 = schedule.add_superstep()
    schedule.schedule("T4", "P1", ss2)  # Heavy task on fast processor
    schedule.schedule("T5", "P2", ss2)  # Light task on medium processor
    
    # Superstep 3: T6
    ss3 = schedule.add_superstep()
    schedule.schedule("T6", "P1", ss3)
    
    # Superstep 4: T7
    ss4 = schedule.add_superstep()
    schedule.schedule("T7", "P1", ss4)
    
    return schedule


def analyze_feasible_actions(schedule, task_node):
    """Analyze what actions are feasible for a given task"""
    print(f"\n--- Analyzing feasible actions for {task_node} ---")
    
    if task_node not in schedule.task_mapping:
        print(f"Task {task_node} not found in schedule")
        return
    
    task = schedule.task_mapping[task_node]
    current_superstep = task.superstep.index
    current_processor = task.proc
    
    print(f"Current position: Superstep {current_superstep}, Processor {current_processor}")
    
    # Test MoveTaskToSuperstep actions
    print("\nSuperstep move feasibility:")
    for i in range(len(schedule.supersteps)):
        if i != current_superstep:
            action = MoveTaskToSuperstep(i)
            feasible = action.is_feasible(schedule, task_node)
            print(f"  Move to superstep {i}: {'✓' if feasible else '✗'}")
            
            if not feasible:
                # Analyze why it's not feasible
                print(f"    Checking constraints:")
                
                # Check predecessors
                for pred_node in schedule.task_graph.predecessors(task_node):
                    if pred_node in schedule.task_mapping:
                        pred_task = schedule.task_mapping[pred_node]
                        pred_superstep = pred_task.superstep.index
                        print(f"    Predecessor {pred_node} in superstep {pred_superstep} (need < {i}): {'✓' if pred_superstep < i else '✗'}")
                
                # Check successors
                for succ_node in schedule.task_graph.successors(task_node):
                    if succ_node in schedule.task_mapping:
                        succ_task = schedule.task_mapping[succ_node]
                        succ_superstep = succ_task.superstep.index
                        print(f"    Successor {succ_node} in superstep {succ_superstep} (need > {i}): {'✓' if succ_superstep > i else '✗'}")
    
    # Test MoveTaskToProcessor actions
    print("\nProcessor move feasibility:")
    for proc in schedule.hardware.network.nodes:
        if proc != current_processor:
            action = MoveTaskToProcessor(proc)
            feasible = action.is_feasible(schedule, task_node)
            speed = schedule.hardware.network.nodes[proc]["weight"]
            print(f"  Move to processor {proc} (speed {speed}): {'✓' if feasible else '✗'}")


def print_detailed_schedule(schedule, title):
    """Print detailed schedule information"""
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Makespan: {schedule.makespan:.2f}")
    print(f"Number of supersteps: {schedule.num_supersteps}")
    
    for i, superstep in enumerate(schedule.supersteps):
        print(f"\nSuperstep {i} (start: {superstep.start_time:.2f}, end: {superstep.end_time:.2f}):")
        print(f"  Times: sync={superstep.sync_time:.2f}, exchange={superstep.exchange_time:.2f}, compute={superstep.compute_time:.2f}, total={superstep.total_time:.2f}")
        
        for proc, tasks in superstep.tasks.items():
            if tasks:
                proc_speed = schedule.hardware.network.nodes[proc]["weight"]
                task_info = []
                for t in tasks:
                    task_cost = schedule.task_graph.nodes[t.node]["weight"]
                    task_info.append(f"{t.node}(cost={task_cost:.0f}, dur={t.duration:.1f})")
                print(f"    {proc} (speed {proc_speed}): {', '.join(task_info)}")


def debug_simulated_annealing():
    """Debug simulated annealing with detailed analysis"""
    print("DEBUG: BSP Simulated Annealing Optimization")
    print("=" * 60)
    
    hardware = create_test_hardware()
    task_graph = create_test_task_graph()
    
    # Test with deliberately bad schedule
    bad_schedule = create_deliberately_bad_schedule(hardware, task_graph)
    print_detailed_schedule(bad_schedule, "Deliberately Bad Initial Schedule")
    
    # Analyze feasible actions for a few tasks
    analyze_feasible_actions(bad_schedule, "T2")
    analyze_feasible_actions(bad_schedule, "T3")
    
    # Create a better schedule for comparison
    better_schedule = create_better_initial_schedule(hardware, task_graph)
    print_detailed_schedule(better_schedule, "Better Manual Schedule (for comparison)")
    
    # Run optimization on bad schedule
    print(f"\n{'='*60}")
    print("RUNNING OPTIMIZATION ON BAD SCHEDULE")
    print("=" * 60)
    
    optimizer = BSPSimulatedAnnealing(
        max_iterations=200,
        max_temp=50.0,
        min_temp=0.01,
        cooling_rate=0.95
    )
    
    print(f"Starting optimization...")
    optimized_schedule = optimizer.optimize(bad_schedule)
    
    print_detailed_schedule(optimized_schedule, "Optimized Schedule")
    
    # Print detailed optimization statistics
    stats = optimizer.get_optimization_stats()
    print(f"\nDetailed Optimization Statistics:")
    print(f"  Total iterations: {stats['total_iterations']}")
    print(f"  Accepted iterations: {stats['accepted_iterations']}")
    print(f"  Applied iterations: {stats['applied_iterations']}")
    print(f"  Acceptance rate: {stats['acceptance_rate']:.2%}")
    print(f"  Initial energy: {stats['initial_energy']:.2f}")
    print(f"  Final energy: {stats['final_energy']:.2f}")
    print(f"  Improvement: {stats['improvement']:.2f} ({stats['improvement_percent']:.1f}%)")
    
    # Analyze the last few iterations
    print(f"\nLast 10 iterations analysis:")
    for i, iteration in enumerate(optimizer.iterations[-10:]):
        accept_str = "ACCEPT" if iteration.accepted else "reject"
        action_str = str(iteration.action) if iteration.action else "None"
        print(f"  {len(optimizer.iterations)-10+i:3d}: temp={iteration.temperature:.2f}, "
              f"current={iteration.current_energy:.2f}, neighbor={iteration.neighbor_energy:.2f}, "
              f"{accept_str}, {action_str}")


if __name__ == "__main__":
    debug_simulated_annealing()