import logging
import sys
import pathlib

# Add paths for imports
sys.path.append(str(pathlib.Path(__file__).parent / "src"))
sys.path.append(str(pathlib.Path(__file__).parent.parent / "saga" / "src"))

from saga.schedulers.data.random import gen_out_trees
from saga_bsp.misc import HeftBusyCommScheduler
from saga.utils.draw import draw_gantt
import saga_bsp as bsp

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

# Create the same test data as playground
from saga.schedulers.data.random import gen_random_networks
network = gen_random_networks(1, 5)[0]  # A simple network with 5 nodes
task_graph = gen_out_trees(1, 3, 2)[0]  # out tree with 3 levels, branching 2

# Preprocess
[task_graph, _] = bsp.preprocess_task_graph(task_graph)

print("Task graph nodes:", list(task_graph.nodes()))
print("Task graph edges:", list(task_graph.edges(data=True)))
print("\nNetwork nodes:", list(network.nodes()))
print("Network edges:", list(network.edges(data=True)))
print("\n" + "="*50)

# Test the busy comm scheduler
scheduler = HeftBusyCommScheduler()
schedule = scheduler.schedule(network, task_graph)

print("\n" + "="*50)
print("FINAL SCHEDULE:")
for node, tasks in schedule.items():
    if tasks:
        print(f"\n{node}:")
        for task in tasks:
            print(f"  {task.name}: [{task.start:.2f}, {task.end:.2f}]")

# Check specifically for T8 and T3
print("\n" + "="*50)
print("CHECKING T8 and T3:")
for node, tasks in schedule.items():
    for task in tasks:
        if task.name == "T3":
            print(f"T3 on {node}: ends at {task.end:.2f}")
        if task.name == "T8":
            print(f"T8 on {node}: starts at {task.start:.2f}, ends at {task.end:.2f}")
            # Check its predecessors
            print(f"  T8 predecessors: {list(task_graph.predecessors('T8'))}")
            
# Check the specific calculation for T8
print("\n" + "="*50)
print("DETAILED T8 SCHEDULING ANALYSIS:")
# The schedule should show that T8 needs communication time from T3
# T3 ends at 4.0 on N0
# T8 is on N1, so needs 1.0 comm time
# T8 should have been blocked from 4.0-5.0 (receiving data) then compute 5.0-6.0