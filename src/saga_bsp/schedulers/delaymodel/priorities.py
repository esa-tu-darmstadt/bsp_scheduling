"""Optimized priority calculation functions.

This module provides optimized implementations of HEFT and CPOP priority functions
that avoid the performance bottlenecks present in the original SAGA implementations.
"""

from typing import Dict, Hashable
import networkx as nx
import numpy as np


def upward_rank(network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, float]:
    """Optimized upward rank calculation that pre-computes averages.

    This is identical to the implementation in heft_busy_communication.py
    but kept separate for modularity.
    """
    ranks = {}

    # Pre-calculate average computation time for each task (across all processors)
    avg_comp_times = {}
    for task in task_graph.nodes:
        task_weight = task_graph.nodes[task]['weight']
        avg_comp_times[task] = np.mean([
            task_weight / network.nodes[node]['weight']
            for node in network.nodes
        ])

    # Pre-calculate average communication time for each edge (across all network links)
    avg_comm_times = {}
    network_speeds = [network.edges[src, dst]['weight'] for src, dst in network.edges]
    avg_network_speed = np.mean(network_speeds)

    for src_task, dst_task in task_graph.edges:
        edge_weight = task_graph.edges[src_task, dst_task]['weight']
        avg_comm_times[(src_task, dst_task)] = edge_weight / avg_network_speed

    # Calculate ranks in reverse topological order
    topological_order = list(nx.topological_sort(task_graph))
    for task in topological_order[::-1]:
        avg_comp_time = avg_comp_times[task]

        # Find maximum successor rank + communication time
        max_comm_time = 0.0
        if task_graph.out_degree(task) > 0:
            max_comm_time = max(
                ranks[successor] + avg_comm_times.get((task, successor), 0.0)
                for successor in task_graph.successors(task)
            )

        ranks[task] = avg_comp_time + max_comm_time

    return ranks


def downward_rank(network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, float]:
    """Optimized downward rank calculation that pre-computes averages."""
    ranks = {}

    # Pre-calculate average computation time for each task (across all processors)
    avg_comp_times = {}
    for task in task_graph.nodes:
        task_weight = task_graph.nodes[task]['weight']
        avg_comp_times[task] = np.mean([
            task_weight / network.nodes[node]['weight']
            for node in network.nodes
        ])

    # Pre-calculate average communication time for each edge (across all network links)
    avg_comm_times = {}
    network_speeds = [network.edges[src, dst]['weight'] for src, dst in network.edges]
    avg_network_speed = np.mean(network_speeds)

    for src_task, dst_task in task_graph.edges:
        edge_weight = task_graph.edges[src_task, dst_task]['weight']
        avg_comm_times[(src_task, dst_task)] = edge_weight / avg_network_speed

    # Calculate ranks in topological order
    topological_order = list(nx.topological_sort(task_graph))
    for task in topological_order:
        if task_graph.in_degree(task) <= 0:
            ranks[task] = 0.0
        else:
            max_pred_time = max(
                ranks[pred] + avg_comm_times.get((pred, task), 0.0) + avg_comp_times[pred]
                for pred in task_graph.predecessors(task)
            )
            ranks[task] = max_pred_time

    return ranks


def cpop_ranks(network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, float]:
    """Computes the ranks of the tasks in the task graph using for the CPoP algorithm.

    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph.

    Returns:
        Dict[Hashable, float]: The ranks of the tasks in the task graph.
            Keys are task names and values are the ranks.
    """
    upward_ranks = upward_rank(network, task_graph)
    downward_ranks = downward_rank(network, task_graph)
    return {
        task_name: (upward_ranks[task_name] + downward_ranks[task_name])
        for task_name in task_graph.nodes
    }