#!/usr/bin/env python3
"""Debug script for testing individual actions"""

import networkx as nx
from src.saga_bsp.schedule import BSPSchedule, BSPHardware
from src.saga_bsp.optimization import MoveTaskToSuperstep, MoveTaskToProcessor


def create_simple_hardware():
    network = nx.Graph()
    network.add_node("P0", weight=1.0)
    network.add_node("P1", weight=2.0)
    network.add_edge("P0", "P1", weight=10.0)
    return BSPHardware(network=network, sync_time=1.0)


def create_simple_task_graph():
    task_graph = nx.DiGraph()
    task_graph.add_node("A", weight=10.0)
    task_graph.add_node("B", weight=15.0)  
    task_graph.add_node("C", weight=20.0)
    task_graph.add_edge("A", "B", weight=5.0)  # A -> B
    task_graph.add_edge("B", "C", weight=3.0)  # B -> C
    return task_graph


def test_move_task_to_superstep_targets():
    """Debug MoveTaskToSuperstep target generation"""
    hardware = create_simple_hardware()
    task_graph = create_simple_task_graph()
    schedule = BSPSchedule(hardware, task_graph)
    
    # Create schedule: A in ss0, B in ss1, C in ss2
    ss0 = schedule.add_superstep()
    schedule.schedule("A", "P0", ss0)
    
    ss1 = schedule.add_superstep()
    schedule.schedule("B", "P0", ss1)
    
    ss2 = schedule.add_superstep()
    schedule.schedule("C", "P0", ss2)
    
    print("Initial schedule:")
    for i, ss in enumerate(schedule.supersteps):
        tasks = []
        for proc, task_list in ss.tasks.items():
            for task in task_list:
                tasks.append(f"{task.node}@{proc}")
        print(f"  Superstep {i}: {tasks}")
    
    print("\nTesting B movement possibilities:")
    
    # Test each possible superstep for B
    for target_ss in range(len(schedule.supersteps)):
        if target_ss != 1:  # B is currently in superstep 1
            action = MoveTaskToSuperstep(target_ss)
            feasible = action.is_feasible(schedule, "B")
            print(f"  Move B to superstep {target_ss}: {'✓' if feasible else '✗'}")
            
            if not feasible:
                # Debug why not feasible
                print(f"    Checking constraints for B -> superstep {target_ss}:")
                
                # Check predecessors  
                for pred in task_graph.predecessors("B"):
                    if pred in schedule.task_mapping:
                        pred_ss = schedule.task_mapping[pred].superstep.index
                        valid_pred = pred_ss < target_ss
                        print(f"      Predecessor {pred} in superstep {pred_ss} (need < {target_ss}): {'✓' if valid_pred else '✗'}")
                
                # Check successors
                for succ in task_graph.successors("B"):
                    if succ in schedule.task_mapping:
                        succ_ss = schedule.task_mapping[succ].superstep.index  
                        valid_succ = succ_ss > target_ss
                        print(f"      Successor {succ} in superstep {succ_ss} (need > {target_ss}): {'✓' if valid_succ else '✗'}")
    
    # Test get_possible_targets
    action = MoveTaskToSuperstep(0)  # Dummy
    targets = action.get_possible_targets(schedule, "B")
    print(f"\nget_possible_targets for B: {targets}")


def test_processor_feasibility():
    """Debug processor movement feasibility"""
    hardware = create_simple_hardware()
    task_graph = create_simple_task_graph()  
    schedule = BSPSchedule(hardware, task_graph)
    
    # Create scenario with intra-superstep dependency  
    ss0 = schedule.add_superstep()
    schedule.schedule("A", "P0", ss0)
    schedule.schedule("B", "P0", ss0)  # B depends on A, same superstep
    
    print("Schedule with intra-superstep dependency:")
    print("  Superstep 0: A@P0, B@P0")
    
    # Try to move B to P1
    action = MoveTaskToProcessor("P1")
    feasible = action.is_feasible(schedule, "B")
    print(f"\nMove B from P0 to P1: {'✓' if feasible else '✗'}")
    
    if not feasible:
        print("  Reason: B depends on A in same superstep")


if __name__ == "__main__":
    print("=== Testing MoveTaskToSuperstep ===")
    test_move_task_to_superstep_targets()
    
    print("\n=== Testing MoveTaskToProcessor ===")
    test_processor_feasibility()