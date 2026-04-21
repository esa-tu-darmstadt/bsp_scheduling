"""
CCR (Communication-to-Computation Ratio) adjustment utilities.

This module provides functions to calculate and adjust the CCR of task graphs
based on both the task graph and the network hardware.
"""

import logging
from typing import Tuple

import networkx as nx
import numpy as np
from ..schedule import BSPHardware

logger = logging.getLogger(__name__)


def calculate_ccr(task_graph: nx.DiGraph, network: nx.Graph) -> float:
    """Calculate the Communication-to-Computation Ratio (CCR) for a task graph on a network.

    CCR = (average communication time) / (average computation time)

    Args:
        task_graph: Task graph with node weights (computation) and edge weights (communication data)
        network: Network graph with node weights (processing speed) and edge weights (communication speed)

    Returns:
        CCR value
    """
    if len(task_graph.nodes()) == 0 or len(task_graph.edges()) == 0:
        return 0.0

    task_weights = [task_graph.nodes[node].get('weight', 1.0) for node in task_graph.nodes()]
    network_node_weights = [network.nodes[node].get('weight', 1.0) for node in network.nodes()]

    edge_data_weights = [task_graph.edges[edge].get('weight', 1.0) for edge in task_graph.edges()]
    network_edge_weights = [0 if edge[0] == edge[1] else network.edges[edge].get('weight', 1.0) for edge in network.edges()]

    avg_task_cost = np.mean(task_weights)
    avg_processor_speed = np.mean(network_node_weights)
    avg_data_size = np.mean(edge_data_weights)
    avg_link_speed = np.mean(network_edge_weights)
    
    mean_computation_time = avg_task_cost / avg_processor_speed
    mean_communication_time = avg_data_size / avg_link_speed
    
    ccr = mean_communication_time / mean_computation_time if mean_computation_time > 0 else 0.0
    
    return ccr


def adjust_task_graph_to_ccr(task_graph: nx.DiGraph, network: nx.Graph, target_ccr: float):
    """Adjust task graph edge weights to achieve target CCR.

    Args:
        task_graph: Task graph to adjust (will be modified in place)
        network: Network hardware specification
        target_ccr: Target CCR value

    Returns:
        modified task graph with adjusted edge weights
    """
    if len(task_graph.edges()) == 0:
        logger.warning("Task graph has no edges, cannot adjust CCR")
        return task_graph

    current_ccr = calculate_ccr(task_graph, network)
    scaling_factor = target_ccr / current_ccr

    # Apply scaling to all edge weights
    for u, v in task_graph.edges():
        original_weight = task_graph.edges[u, v].get('weight', 1.0)
        new_weight = original_weight * scaling_factor
        task_graph.edges[u, v]['weight'] = new_weight
        
    return  task_graph


def generate_ccr_variants(task_graph: nx.DiGraph, network: nx.Graph,
                         target_ccrs: list) -> list:
    """Generate multiple variants of a task graph with different CCRs.

    Args:
        task_graph: Base task graph (will not be modified)
        network: Network hardware specification
        target_ccrs: List of target CCR values

    Returns:
        List of (task_graph_variant, actual_ccr) tuples
    """
    variants = []

    for target_ccr in target_ccrs:
        # Create a copy of the task graph
        graph_copy = task_graph.copy()

        # Adjust to target CCR
        adjust_task_graph_to_ccr(graph_copy, network, target_ccr)

        variants.append((graph_copy, target_ccr))

    return variants


def calculate_avg_computation_time(task_graph: nx.DiGraph, network: nx.Graph) -> float:
    """Calculate the average computation time for tasks in a graph on given hardware.

    Args:
        task_graph: Task graph with node weights (computation work)
        network: Network graph with node weights (processing speed)

    Returns:
        Average computation time per task (in same time units as weight/speed ratio)
    """
    if len(task_graph.nodes()) == 0:
        return 0.0

    task_weights = [task_graph.nodes[node].get('weight', 1.0) for node in task_graph.nodes()]
    network_node_weights = [network.nodes[node].get('weight', 1.0) for node in network.nodes()]

    avg_task_weight = np.mean(task_weights)
    avg_processor_speed = np.mean(network_node_weights)

    return avg_task_weight / avg_processor_speed


def calculate_sync_time(task_graph: nx.DiGraph, network: nx.Graph, sync_ratio: float) -> float:
    """Calculate sync time as a ratio of average computation time.

    This provides a hardware-aware way to set sync_time for BSPHardware,
    where sync_ratio=1.0 means sync takes as long as one average task.

    Args:
        task_graph: Task graph with node weights (computation work)
        network: Network graph with node weights (processing speed)
        sync_ratio: Sync time as multiple of avg computation time
                   (e.g., 0.1 = 10% of avg task runtime, 1.0 = same as avg task)

    Returns:
        Sync time in the same units as computation time
    """
    avg_computation_time = calculate_avg_computation_time(task_graph, network)
    return avg_computation_time * sync_ratio


def get_ccr_statistics(task_graph: nx.DiGraph, network: nx.Graph) -> dict:
    """Get detailed CCR statistics for a task graph on a network.

    Args:
        task_graph: Task graph
        network: Network graph

    Returns:
        Dictionary with CCR statistics
    """
    if len(task_graph.nodes()) == 0 or len(task_graph.edges()) == 0:
        return {
            'ccr': 0.0,
            'avg_computation_time': 0.0,
            'avg_communication_time': 0.0,
            'task_count': len(task_graph.nodes()),
            'edge_count': len(task_graph.edges())
        }

    # Calculate components
    task_weights = [task_graph.nodes[node].get('weight', 1.0) for node in task_graph.nodes()]
    edge_weights = [task_graph.edges[edge].get('weight', 1.0) for edge in task_graph.edges()]
    network_node_weights = [network.nodes[node].get('weight', 1.0) for node in network.nodes()]
    network_edge_weights = [network.edges[edge].get('weight', 1.0) for edge in network.edges()]

    avg_task_weight = np.mean(task_weights)
    avg_edge_weight = np.mean(edge_weights)
    avg_processor_speed = np.mean(network_node_weights)
    avg_link_speed = np.mean(network_edge_weights)

    total_task_weight = np.sum(task_weights)
    total_edge_weight = np.sum(edge_weights)

    total_computation_time = total_task_weight / avg_processor_speed
    total_communication_time = total_edge_weight / avg_link_speed

    ccr = total_communication_time / total_computation_time if total_computation_time > 0 else 0.0

    return {
        'ccr': ccr,
        'avg_computation_time': total_computation_time,
        'avg_communication_time': total_communication_time,
        'avg_task_weight': avg_task_weight,
        'avg_edge_weight': avg_edge_weight,
        'avg_processor_speed': avg_processor_speed,
        'avg_link_speed': avg_link_speed,
        'task_count': len(task_graph.nodes()),
        'edge_count': len(task_graph.edges())
    }