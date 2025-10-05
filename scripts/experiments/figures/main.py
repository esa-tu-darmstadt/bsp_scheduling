#!/usr/bin/env python

import pathlib

import numpy
import saga_bsp as bsp
from saga_bsp.schedulers import (
    MSAScheduler,
    FillInSplitBSPScheduler,
    BCSHScheduler
)
from saga_bsp.optimization.simulated_annealing_v2 import BSPSimulatedAnnealing
from saga_bsp.misc import HeftBusyCommScheduler

from saga.schedulers.data.random import (
    gen_in_trees,
    gen_out_trees,
    gen_parallel_chains,
    gen_random_networks,
)
from saga.schedulers import HeftScheduler, BruteForceScheduler
from saga.utils.draw import draw_gantt, draw_network, draw_task_graph


# This script creates the explanatory Figures in the BSP scheduling paper.
# For the benchmarking experiments, please see the benchmarking folder.

base_output_dir = pathlib.Path(__file__).parent.resolve().joinpath("output")
base_output_dir.mkdir(exist_ok=True, parents=True)

def default_network():
    return gen_random_networks(1, 4, get_node_weight=lambda i: 1.0, get_edge_weight=lambda i, j: 1.0)[0]

def default_task_graph():
    numpy.random.seed(45)
    task_graph = gen_parallel_chains(1, 4, 3, get_task_weight=lambda i: numpy.random.uniform(2.0, 5.0), get_dependency_weight=lambda i,j: numpy.random.uniform(1.0, 2.0))[0]
    #task_graph = gen_parallel_chains(1, 4, 3, get_task_weight=lambda i: 3.0, get_dependency_weight=lambda i,j: 1.0)[0]
    [task_graph, _] = bsp.preprocess_task_graph(task_graph)
    return task_graph

# -----------------------------------------------------------------
# Figures from background section showing task graph, async schedule
# and BSP schedule.
# -----------------------------------------------------------------
output_dir = base_output_dir.joinpath("figures_background")
output_dir.mkdir(exist_ok=True, parents=True)

network = default_network()
# network=gen_random_networks(1, 5, get_node_weight=lambda i: 1.0, get_edge_weight=lambda i, j: 1.0)[0]
# set seed for reproducibility
# numpy.random.seed(42)
# task_graph = gen_parallel_chains(1, 4, 3, get_task_weight=lambda i: numpy.random.uniform(2.0, 5.0), get_dependency_weight=lambda i,j: numpy.random.uniform(1.0, 2.0))[0]
task_graph = default_task_graph()
hardware = bsp.BSPHardware(network=network, sync_time=1.0)

task_graph_scale = 2
draw_task_graph(
    task_graph,
    use_latex=False,
    font_size=10 * task_graph_scale,
    arrowsize=10 * task_graph_scale,
    draw_edge_weights=False,
    draw_node_weights=False,
    figsize=tuple(task_graph_scale * i for i in (2.2, 3)),
).get_figure().savefig(
    output_dir.joinpath("task_graph.pdf"), dpi=300, bbox_inches="tight"
)

bsp.draw_busy_comm_gantt(
    HeftScheduler().schedule(network, task_graph),
    figsize=(2.5, 2.5),
    font_size=10,
    tick_font_size=10,
    use_latex=True,
    legend_loc=None,
).get_figure().savefig(
    output_dir.joinpath("async_schedule.pdf"), dpi=300, bbox_inches="tight"
)

bsp.draw_busy_comm_gantt(
    HeftBusyCommScheduler().schedule(network, task_graph),
    figsize=(2.5, 2.5),
    font_size=10,
    tick_font_size=10,
    use_latex=True,
    legend_loc=None,
).get_figure().savefig(
    output_dir.joinpath("async_schedule_busy.pdf"), dpi=300, bbox_inches="tight"
)

bsp.draw_bsp_gantt(
    bsp.convert_async_to_bsp(
        hardware,
        task_graph,
        HeftScheduler().schedule(network, task_graph),
        strategy='earliest-finishing-next',
    ),
    # FillInSplitBSPScheduler().schedule(hardware, task_graph),
    figsize=(4, 2.5),
    font_size=10,
    tick_font_size=10,
    y_label=None,
    use_latex=True,
    legend_loc="upper right",
).get_figure().savefig(
    output_dir.joinpath("bsp_schedule.pdf"), dpi=300, bbox_inches="tight"
)

# -----------------------------------------------------------------
# Figures for BSP schedules
# -----------------------------------------------------------------
output_dir = base_output_dir.joinpath("figures_async_to_bsp")
output_dir.mkdir(exist_ok=True, parents=True)

x_max = 40

network = default_network()
task_graph = default_task_graph()
# task_graph = gen_in_trees(1, 2, 3)[0]
# task_graph = gen_out_trees(1, 3, 2)[0]
hardware = bsp.BSPHardware(network=network, sync_time=1.0)

task_graph_scale = 2
# draw_task_graph(
#     task_graph,
#     use_latex=False,
#     font_size=10 * task_graph_scale,
#     arrowsize=10 * task_graph_scale,
#     draw_edge_weights=False,
#     draw_node_weights=False,
#     figsize=tuple(task_graph_scale * i for i in (2.2, 3)),
# ).get_figure().savefig(
#     output_dir.joinpath("task_graph.pdf"), dpi=300, bbox_inches="tight"
# )

# bsp.draw_busy_comm_gantt(
#     HeftBusyCommScheduler().schedule(network, task_graph),
#     figsize=(2.5, 2.5),
#     font_size=10,
#     tick_font_size=10,
#     use_latex=True,
#     legend_loc=None,
# ).get_figure().savefig(
#     output_dir.joinpath("async_schedule_busy.pdf"), dpi=300, bbox_inches="tight"
# )

ax = bsp.draw_bsp_gantt(
    bsp.convert_async_to_bsp(
        hardware,
        task_graph,
        HeftScheduler().schedule(network, task_graph),
        strategy='earliest-finishing-next',
    ),
    figsize=(4, 2.5),
    font_size=10,
    tick_font_size=10,
    y_label=None,
    use_latex=True,
    legend_loc=None,
    # title="Conversion Strategy: Earliest Finishing Next",
)
ax.set_xlim(0, x_max)
ax.get_figure().savefig(
    output_dir.joinpath("earliest_finishing_next.pdf"), dpi=300, bbox_inches="tight"
)

ax = bsp.draw_bsp_gantt(
    bsp.convert_async_to_bsp(
        hardware,
        task_graph,
        HeftScheduler().schedule(network, task_graph),
        strategy='eager',
    ),
    figsize=(4, 2.5),
    font_size=10,
    tick_font_size=10,
    # title="Conversion Strategy: Eager",
    y_label=None,
    use_latex=True,
    legend_loc=None,
)
ax.set_xlim(0, x_max)
ax.get_figure().savefig(
    output_dir.joinpath("eager.pdf"), dpi=300, bbox_inches="tight"
)

output_dir = base_output_dir.joinpath("figures_native")
output_dir.mkdir(exist_ok=True, parents=True)

ax = bsp.draw_busy_comm_gantt(
    HeftBusyCommScheduler().schedule(network, task_graph),
    figsize=(2.5, 2.5),
    font_size=10,
    tick_font_size=10,
    use_latex=True,
    legend_loc=None,
)
ax.set_xlim(0, x_max)
ax.get_figure().savefig(
    output_dir.joinpath("async_schedule_busy.pdf"), dpi=300, bbox_inches="tight"
)

for priority_mode in ["heft", "cpop"]:
    for optimize_merging in [True, False]:
        ax = bsp.draw_bsp_gantt(
            FillInSplitBSPScheduler(priority_mode=priority_mode, optimize_merging=optimize_merging).schedule(hardware, task_graph),
            figsize=(4, 2.5),
            font_size=10,
            tick_font_size=10,
        y_label=None,
        use_latex=True,
        legend_loc=None,
        # title=f"BALS with {priority_mode.capitalize()} priority (ours)",
        )
        ax.set_xlim(0, x_max)
        ax.get_figure().savefig(
            output_dir.joinpath(f"bals_{priority_mode}_{'elim' if optimize_merging else 'no_elim'}.pdf"), dpi=300, bbox_inches="tight"
        )

ax = bsp.draw_bsp_gantt(
    BCSHScheduler().schedule(hardware, task_graph),
    figsize=(4, 2.5),
    font_size=10,
    tick_font_size=10,
    y_label=None,
    use_latex=True,
    legend_loc=None,
    # title="BCSH Scheduler with EFT placement (ours)",
)
ax.set_xlim(0, x_max)
ax.get_figure().savefig(
    output_dir.joinpath("bcsh_schedule.pdf"), dpi=300, bbox_inches="tight"
)