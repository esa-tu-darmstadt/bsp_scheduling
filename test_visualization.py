#!/usr/bin/env python3
"""
Test script to demonstrate BSP schedule visualization functionality.
"""

import sys
import os
sys.path.insert(0, 'src')

import networkx as nx
import matplotlib.pyplot as plt
from saga.scheduler import Task
from saga.utils.draw import  draw_task_graph
from saga_bsp.conversion import convert_async_to_bsp
from saga_bsp.utils.visualization import draw_bsp_gantt, draw_superstep_breakdown, draw_tile_activity


def create_test_network():
    """Create a simple 3-processor network"""
    network = nx.Graph()
    network.add_node("P0", weight=2.0)  # Fast processor
    network.add_node("P1", weight=1.0)  # Medium processor  
    network.add_node("P2", weight=1.5)  # Fast processor
    
    # Connect all processors
    network.add_edge("P0", "P1", weight=1.0)
    network.add_edge("P1", "P2", weight=1.0)
    network.add_edge("P0", "P2", weight=1.0)
    
    return network


def create_test_task_graph():
    """Create a more complex task dependency graph"""
    task_graph = nx.DiGraph()
    
    # Add tasks with weights (computation costs)
    tasks = {
        "A": 20.0, "B": 15.0, "C": 10.0, "D": 25.0, 
        "E": 12.0, "F": 18.0, "G": 8.0, "H": 22.0
    }
    
    for task, weight in tasks.items():
        task_graph.add_node(task, weight=weight)
    
    # Add dependencies (communication costs)
    dependencies = [
        ("A", "B", 5.0), ("A", "C", 3.0), ("B", "D", 4.0), 
        ("C", "D", 2.0), ("B", "E", 6.0), ("D", "F", 8.0),
        ("E", "G", 3.0), ("F", "G", 5.0), ("G", "H", 4.0)
    ]
    
    for src, dst, weight in dependencies:
        task_graph.add_edge(src, dst, weight=weight)
    
    return task_graph


def create_complex_async_schedule():
    """Create a complex async schedule for demonstration"""
    # Simulate a more realistic async schedule with inter-processor dependencies
    async_schedule = {
        "P0": [
            Task("P0", "A", 0.0, 10.0),   # First task
            Task("P0", "F", 45.0, 57.0),  # Later task after communication
        ],
        "P1": [
            Task("P1", "B", 15.0, 30.0),  # Depends on A
            Task("P1", "E", 35.0, 47.0),  # Depends on B
            Task("P1", "H", 70.0, 92.0),  # Final task
        ],
        "P2": [
            Task("P2", "C", 15.0, 21.7),  # Depends on A (faster processor)
            Task("P2", "D", 32.0, 48.7),  # Depends on B and C
            Task("P2", "G", 60.0, 65.3),  # Depends on E and F
        ]
    }
    
    return async_schedule


def main():
    """Main demonstration function"""
    print("Creating test network and task graph...")
    network = create_test_network()
    task_graph = create_test_task_graph()
    
    draw_task_graph(task_graph)
    
    print("Creating async schedule...")
    async_schedule = create_complex_async_schedule()
    
    print("Converting to BSP schedule...")
    bsp_schedule = convert_async_to_bsp(network, task_graph, async_schedule)
    
    print(f"BSP Schedule created with {bsp_schedule.num_supersteps} supersteps")
    print(f"Total makespan: {bsp_schedule.makespan:.2f}")
    
    # Create visualizations
    print("Creating visualizations...")
    
    # 1. BSP Gantt Chart
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    draw_bsp_gantt(bsp_schedule, show_phases=True, axis=ax1)
    ax1.set_title('BSP Schedule Gantt Chart', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('bsp_gantt_chart.png', dpi=150, bbox_inches='tight')
    print("Saved: bsp_gantt_chart.png")
    
    # 2. Superstep Breakdown
    fig2 = draw_superstep_breakdown(bsp_schedule, figsize=(12, 8))
    plt.savefig('superstep_breakdown.png', dpi=150, bbox_inches='tight')
    print("Saved: superstep_breakdown.png")
    
    # 3. Tile Activity Heatmap  
    ax3 = draw_tile_activity(bsp_schedule, figsize=(12, 6))
    plt.savefig('tile_activity_heatmap.png', dpi=150, bbox_inches='tight')
    print("Saved: tile_activity_heatmap.png")
    
    # Print schedule details
    print("\nSchedule Details:")
    print("================")
    for i, superstep in enumerate(bsp_schedule.supersteps):
        print(f"Superstep {i}:")
        print(f"  Start: {superstep.start_time:.2f}, End: {superstep.end_time:.2f}")
        print(f"  Sync: {superstep.sync_time:.2f}, Exchange: {superstep.exchange_time:.2f}, Compute: {superstep.compute_time:.2f}")
        
        for proc, tasks in superstep.tasks.items():
            if tasks:
                task_names = [t.node for t in tasks]
                print(f"  {proc}: {task_names}")
        print()
    
    # Show plots if running interactively
    try:
        plt.show()
    except:
        print("Plots saved to files (cannot display in this environment)")


if __name__ == "__main__":
    main()