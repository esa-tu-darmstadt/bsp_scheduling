import logging
import random
import saga
from saga.schedulers.data.random import (gen_in_trees, gen_out_trees,
                                         gen_parallel_chains,
                                         gen_random_networks)
from saga.schedulers import HeftScheduler, BruteForceScheduler
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph

import matplotlib.pyplot as plt
import networkx as nx

import saga_bsp as bsp
from saga_bsp.schedulers import MSAScheduler, FillInSplitBSPScheduler, BCSHScheduler
from saga_bsp.optimization.simulated_annealing_v2 import BSPSimulatedAnnealing
from saga_bsp.misc import HeftBusyCommScheduler

import tikzplotlib as tikz

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sa_logger = logging.getLogger('saga_bsp.optimization.simulated_annealing_v2')
sa_logger.setLevel(logging.DEBUG)
  
# Create test data
network = gen_random_networks(1, 5)[0]
task_graph = gen_parallel_chains(1, 5, 5, get_task_weight=lambda x:2)[0]
# task_graph = gen_in_trees(1, 3, 3)[0]
# task_graph = gen_out_trees(1, 3, 3)[0]

# min_levels, max_levels = 2, 4
# min_branching, max_branching = 2, 3
# min_nodes, max_nodes = 3, 5
# pairs = []
# network = gen_random_networks(
#     num=1,
#     num_nodes=random.randint(min_nodes, max_nodes)
# )[0]
# task_graph = gen_out_trees(
#     num=1,
#     num_levels=random.randint(min_levels, max_levels),
#     branching_factor=random.randint(min_branching, max_branching)
# )[0]

[task_graph, _] = bsp.preprocess_task_graph(task_graph)

print("Generated test data:")
print(f"Network nodes: {list(network.nodes())}")
print(f"Task graph nodes: {list(task_graph.nodes())}")
print()

print("Network edges:", list(network.edges(data=True)))
print("Task graph edges:", list(task_graph.edges(data=True)))

draw_network(network)
draw_task_graph(task_graph)

# Test async scheduler first
async_scheduler = HeftBusyCommScheduler()
async_schedule = async_scheduler.schedule(network, task_graph)

# draw_gantt(async_schedule)
bsp.draw_busy_comm_gantt(async_schedule, title="HEFT with Busy Communication")

bsp_hardware = bsp.BSPHardware(network=network, sync_time=0)

# bsp_schedule = bsp.convert_async_to_bsp(
#     bsp_hardware, task_graph, async_schedule, 
#     strategy="eager"
# )

# # Draw the unoptimized schedule
# bsp.draw_bsp_gantt(bsp_schedule)

# simulated_annealing = BSPSimulatedAnnealing(
#     max_iterations=100,
#     max_temp=100.0,
#     min_temp=0.1,
#     cooling_rate=0.99
# )

# # Run simulated annealing optimization
# optimized_schedule = simulated_annealing.optimize(bsp_schedule)
# simulated_annealing.print_optimization_stats()
# simulated_annealing.draw_energy_history()
# bsp.print_bsp_schedule(optimized_schedule)



# # Draw the optimized schedule
# bsp.draw_bsp_gantt(optimized_schedule)

# Test the HEFT BSP scheduler
# heft_bsp_scheduler = FillInSplitBSPScheduler(verbose=True, draw_after_each_step=False)
bsp_schedule = BCSHScheduler(verbose=True, use_eft=True).schedule(bsp_hardware, task_graph)
bsp.draw_bsp_gantt(bsp_schedule, title="BCSH")

bsp_schedule = FillInSplitBSPScheduler(priority_mode="heft").schedule(bsp_hardware, task_graph)
bsp.draw_bsp_gantt(bsp_schedule, title="FillInSplitBSP+heft")


bsp_schedule = FillInSplitBSPScheduler(priority_mode="cpop").schedule(bsp_hardware, task_graph)
bsp.draw_bsp_gantt(bsp_schedule, title="FillInSplitBSP+cpop")

bsp_schedule.assert_valid()

plt.show()


# # Test just earliest-finishing-next with debugging
# strategies = [
#     ("eager", None),  # Eager strategy
#     ("level-based", None),
#     ("earliest-finishing-next", None),
#     ("earliest-finishing-next", 0.20),  # 20% backfill threshold
# ]

# for strategy, backfill in strategies:
#     suffix = f" (backfill {backfill*100:.0f}%)" if backfill else ""
#     print(f"\n=== Testing {strategy}{suffix} strategy ===")
    
#     bsp_schedule = bsp.convert_async_to_bsp(
#         bsp_hardware, task_graph, async_schedule, 
#         strategy=strategy, backfill_threshold_percent=backfill, verbose=True
#     )
    
#     bsp.draw_bsp_gantt(bsp_schedule)
#     print(f"Number of supersteps: {bsp_schedule.num_supersteps}")
#     print(f"Makespan: {bsp_schedule.makespan:.2f}")
    
#     # Show task placement
#     for i, superstep in enumerate(bsp_schedule.supersteps):
#         print(f"  Superstep {i}:")
#         for proc, tasks in superstep.tasks.items():
#             task_names = [task.node for task in tasks]
#             if task_names:
#                 print(f"    {proc}: {task_names}")

# # Draw gantt for eager strategy
# print(f"\nDrawing Gantt chart for eager strategy...")
# eager_schedule = bsp.convert_async_to_bsp(bsp_hardware, task_graph, async_schedule, strategy="earliest-finishing-next")
# bsp.draw_bsp_gantt(eager_schedule)


