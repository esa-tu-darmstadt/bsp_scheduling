"""Optimized priority calculation functions.

This module provides optimized implementations of HEFT and CPOP priority functions
that avoid the performance bottlenecks present in the original SAGA implementations.
"""

from typing import Dict, Hashable
import networkx as nx
import numpy as np


def upward_rank(network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, float]:
    """Optimized upward rank calculation that pre-computes average speeds.

    Precalculates average processor speed and average network speed once,
    then computes execution times on the fly for better efficiency.
    """
    ranks = {}

    # Precalculate average processor speed (scalar)
    processor_speeds = [network.nodes[node]['weight'] for node in network.nodes]
    avg_processor_speed = np.mean(processor_speeds)

    # Precalculate average network speed (scalar)
    network_speeds = [network.edges[src, dst]['weight'] for src, dst in network.edges]
    avg_network_speed = np.mean(network_speeds)

    # Calculate ranks in reverse topological order
    topological_order = list(nx.topological_sort(task_graph))
    for task in topological_order[::-1]:
        # Compute average computation time
        task_weight = task_graph.nodes[task]['weight']
        avg_comp_time = task_weight / avg_processor_speed

        # Find maximum successor rank + communication time
        max_comm_time = 0.0
        if task_graph.out_degree(task) > 0:
            max_comm_time = max(
                ranks[successor] + (task_graph.edges[task, successor]['weight'] / avg_network_speed)
                for successor in task_graph.successors(task)
            )

        ranks[task] = avg_comp_time + max_comm_time

    return ranks


def downward_rank(network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, float]:
    """Optimized downward rank calculation that pre-computes average speeds.

    Precalculates average processor speed and average network speed once,
    then computes execution times on the fly for better efficiency.
    """
    ranks = {}

    # Precalculate average processor speed (scalar)
    processor_speeds = [network.nodes[node]['weight'] for node in network.nodes]
    avg_processor_speed = np.mean(processor_speeds)

    # Precalculate average network speed (scalar)
    network_speeds = [network.edges[src, dst]['weight'] for src, dst in network.edges]
    avg_network_speed = np.mean(network_speeds)

    # Calculate ranks in topological order
    topological_order = list(nx.topological_sort(task_graph))
    for task in topological_order:
        if task_graph.in_degree(task) <= 0:
            ranks[task] = 0.0
        else:
            max_pred_time = max(
                ranks[pred] + (task_graph.edges[pred, task]['weight'] / avg_network_speed) +
                (task_graph.nodes[pred]['weight'] / avg_processor_speed)
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


def calculate_dynamic_downward_rank(task_name: str, task_graph: nx.DiGraph, schedule) -> float:
    """Calculate dynamic downward rank (earliest start time) for a task.

    The dynamic downward rank is the actual earliest starting time of the task,
    which is the latest finishing time of any of its predecessors.

    Args:
        task_name: Task to calculate rank for
        task_graph: Task dependency graph
        schedule: Current partial schedule

    Returns:
        Dynamic downward rank (earliest start time)
    """
    earliest_start = 0.0

    for pred_name in task_graph.predecessors(task_name):
        if schedule.task_scheduled(pred_name):
            pred_instance = schedule.get_single_instance(pred_name)
            earliest_start = max(earliest_start, pred_instance.end)

    return earliest_start