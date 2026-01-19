"""Debug script to visualize schedules and understand timing scales."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from saga.schedulers.data.random import gen_in_trees

import saga_bsp as bsp
from saga_bsp.hardware.graphcore import IPUHardware
from saga_bsp.task_graphs.ccr_adjustment import adjust_task_graph_to_ccr, calculate_ccr
from saga_bsp.misc.saga_scheduler_wrapper import preprocess_task_graph
from saga_bsp.misc.heft_busy_communication import HeftBusyCommScheduler
from saga_bsp.schedulers import FillInSplitBSPScheduler, BCSHScheduler

# Generate one graph
np.random.seed(42)
task_graph = gen_in_trees(
    1, 4, 3,
    lambda x: np.random.uniform(1.0, 10.0),
    lambda src, dst: np.random.uniform(0.5, 5.0)
)[0]

# Preprocess
task_graph, metadata = preprocess_task_graph(task_graph)
print(f"Task graph: {task_graph.number_of_nodes()} nodes, {task_graph.number_of_edges()} edges")

# Print task weights
print("\nTask weights:")
for node in task_graph.nodes():
    weight = task_graph.nodes[node].get('weight', 0)
    print(f"  {node}: {weight:.2f}")

# Print edge weights
print("\nEdge weights:")
for u, v in task_graph.edges():
    weight = task_graph.edges[u, v].get('weight', 0)
    print(f"  {u} -> {v}: {weight:.2f}")

# Create hardware
num_tiles = 4
sync_time_ns = 100.0  # 100 nanoseconds
hardware = IPUHardware(num_tiles=num_tiles, sync_time=sync_time_ns)

print(f"\nHardware: {num_tiles} tiles, sync_time={sync_time_ns}ns")
print(f"Network node weights (speed): {[hardware.network.nodes[n]['weight'] for n in list(hardware.network.nodes())[:2]]}...")

# Check CCR before adjustment
ccr_before = calculate_ccr(task_graph, hardware.network)
print(f"\nCCR before adjustment: {ccr_before}")

# Adjust CCR
target_ccr = 1.0
adjust_task_graph_to_ccr(task_graph, hardware.network, target_ccr)
ccr_after = calculate_ccr(task_graph, hardware.network)
print(f"CCR after adjustment to {target_ccr}: {ccr_after}")

# Print adjusted edge weights
print("\nAdjusted edge weights:")
for u, v in list(task_graph.edges())[:5]:
    weight = task_graph.edges[u, v].get('weight', 0)
    print(f"  {u} -> {v}: {weight:.2e}")

# Run schedulers
output_dir = Path("./results/debug_schedules")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. HeftBusyCommScheduler
print("\n--- HeftBusyCommScheduler ---")
heft_scheduler = HeftBusyCommScheduler()
heft_schedule = heft_scheduler.schedule(hardware.network, task_graph)

# Calculate makespan
heft_makespan = 0.0
for proc_schedule in heft_schedule.values():
    for task in proc_schedule:
        if hasattr(task, 'end'):
            heft_makespan = max(heft_makespan, task.end)
print(f"HEFT makespan: {heft_makespan:.2e}")

# Plot HEFT schedule
bsp.draw_busy_comm_gantt(heft_schedule, title=f"HEFT (makespan={heft_makespan:.2e})")
plt.savefig(output_dir / "heft_schedule.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved HEFT schedule to {output_dir / 'heft_schedule.png'}")

# 2. FillInSplitBSPScheduler
print("\n--- FillInSplitBSPScheduler ---")
bsp_scheduler = FillInSplitBSPScheduler(priority_mode='heft')
bsp_schedule = bsp_scheduler.schedule(hardware, task_graph)
print(f"BSP makespan: {bsp_schedule.makespan:.2e}")
print(f"Number of supersteps: {bsp_schedule.num_supersteps}")

# Plot BSP schedule
bsp.draw_bsp_gantt(bsp_schedule, title=f"BALS (makespan={bsp_schedule.makespan:.2e})")
plt.savefig(output_dir / "bsp_schedule.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved BSP schedule to {output_dir / 'bsp_schedule.png'}")

# 3. BCSHScheduler
print("\n--- BCSHScheduler ---")
bcsh_scheduler = BCSHScheduler(use_eft=True)
bcsh_schedule = bcsh_scheduler.schedule(hardware, task_graph)
print(f"BCSH makespan: {bcsh_schedule.makespan:.2e}")
print(f"Number of supersteps: {bcsh_schedule.num_supersteps}")

# Plot BCSH schedule
bsp.draw_bsp_gantt(bcsh_schedule, title=f"BCSH (makespan={bcsh_schedule.makespan:.2e})")
plt.savefig(output_dir / "bcsh_schedule.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved BCSH schedule to {output_dir / 'bcsh_schedule.png'}")

print(f"\n--- Ratios ---")
print(f"BALS / HEFT: {bsp_schedule.makespan / heft_makespan:.2f}")
print(f"BCSH / HEFT: {bcsh_schedule.makespan / heft_makespan:.2f}")
