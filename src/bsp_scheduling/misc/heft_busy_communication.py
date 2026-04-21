"""HEFT Scheduler with Busy Communication.

This is a modified version of the HEFT scheduler that accounts for "busy communication",
where processors are blocked while receiving data and cannot perform computation during
communication periods. Communication and computation form a single contiguous block.
"""

import logging
import pathlib
from typing import Dict, Hashable, List

import networkx as nx
import numpy as np

from saga.scheduler import Scheduler, Task
from saga.utils.tools import get_insert_loc

thisdir = pathlib.Path(__file__).resolve().parent


def upward_rank(network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, float]:
    """Optimized upward rank calculation that pre-computes averages."""
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


def heft_rank_sort(network: nx.Graph, task_graph: nx.DiGraph) -> List[Hashable]:
    """Sort tasks based on their rank (as defined in the HEFT paper).

    Args:
        network (nx.Graph): The network graph.
        task_graph (nx.DiGraph): The task graph.

    Returns:
        List[Hashable]: The sorted list of tasks.
    """
    rank = upward_rank(network, task_graph)
    topological_sort = {node: i for i, node in enumerate(reversed(list(nx.topological_sort(task_graph))))}
    rank = {node: (rank[node] + topological_sort[node]) for node in rank}
    return sorted(list(rank.keys()), key=rank.get, reverse=True)


class HeftBusyCommScheduler(Scheduler):
    """HEFT scheduler with busy communication.
    
    In this variant, processors are blocked during communication and cannot
    perform computation while receiving data. Communication and computation
    form a single contiguous block: first all data is received (sum of all
    communication times), then the task is computed.
    """

    name = "HEFT (Busy Comm)"

    @staticmethod
    def calculate_runtime(network: nx.Graph, task_graph: nx.DiGraph, node: Hashable, task: Hashable) -> float:
        """Calculate runtime of a task on a specific node.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            node: The processor node.
            task: The task to calculate runtime for.

        Returns:
            float: The runtime of the task on the node.
        """
        speed: float = network.nodes[node]["weight"]
        cost: float = task_graph.nodes[task]["weight"]
        return cost / speed

    @staticmethod
    def calculate_commtime(network: nx.Graph, task_graph: nx.DiGraph,
                          src_node: Hashable, dst_node: Hashable,
                          src_task: Hashable, dst_task: Hashable) -> float:
        """Calculate communication time between two tasks on different nodes.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            src_node: Source processor node.
            dst_node: Destination processor node.
            src_task: Source task.
            dst_task: Destination task.

        Returns:
            float: The communication time between the tasks.
        """
        if src_node == dst_node:
            return 0.0

        # Get network edge speed (bidirectional)
        if network.has_edge(src_node, dst_node):
            speed: float = network.edges[src_node, dst_node]["weight"]
        elif network.has_edge(dst_node, src_node):
            speed: float = network.edges[dst_node, src_node]["weight"]
        else:
            raise ValueError(f"No network connection between {src_node} and {dst_node}")

        cost = task_graph.edges[src_task, dst_task]["weight"]
        return cost / speed

    def _schedule(
        self,
        network: nx.Graph,
        task_graph: nx.DiGraph,
        schedule_order: List[Hashable],
    ) -> Dict[Hashable, List[Task]]:
        """Schedule the tasks on the network with busy communication.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            schedule_order: The order in which to schedule the tasks.

        Returns:
            Dict[Hashable, List[Task]]: The schedule.
        """
        comp_schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        task_schedule: Dict[Hashable, Task] = {}

        logging.debug("Schedule order: %s", schedule_order)
        
        for task_name in schedule_order:
            min_finish_time = np.inf
            best_node = None
            best_idx = None
            best_start = None
            
            for node in network.nodes:  # Find the best node to run the task
                # Calculate total communication time (sum of all incoming communications)
                total_comm_time = 0.0
                max_parent_end = 0.0
                comm_predecessors = []  # Track which predecessors require communication
                
                for parent in task_graph.predecessors(task_name):
                    parent_task = task_schedule[parent]
                    max_parent_end = max(max_parent_end, parent_task.end)
                    
                    if parent_task.node != node:
                        # Need communication from different processor
                        comm_time = self.calculate_commtime(network, task_graph, parent_task.node, node, parent, task_name)
                        total_comm_time += comm_time  # Sum all communication times
                        comm_predecessors.append(parent)  # Track this predecessor
                
                # Total duration of this task (communication + computation as one block)
                runtime = self.calculate_runtime(network, task_graph, node, task_name)
                total_duration = total_comm_time + runtime
                
                # Find earliest time this task can start (when all parents are done)
                earliest_start = max_parent_end
                
                # Find a slot that can fit the entire block (comm + compute)
                idx, start_time = get_insert_loc(
                    comp_schedule[node], earliest_start, total_duration
                )
                
                # The actual computation starts after communication
                task_start = start_time + total_comm_time
                task_finish = start_time + total_duration
                
                # logging.debug(
                #     "Testing task %s on node %s: comm_time=%s, runtime=%s, start=%s, finish=%s",
                #     task_name,
                #     node,
                #     total_comm_time,
                #     runtime,
                #     task_start,
                #     task_finish
                # )
                
                if task_finish < min_finish_time:
                    min_finish_time = task_finish
                    best_node = node
                    best_idx = idx
                    best_start = start_time  # Start of the entire block (comm + compute)
                    best_comm_time = total_comm_time
                    best_comm_predecessors = comm_predecessors  # Save the list of communicating predecessors

            # Schedule the task on the best processor
            # The Task object represents the entire block (communication + computation)
            task = Task(
                best_node, task_name, best_start, min_finish_time
            )
            # Store communication time and predecessors separately to enable visualization
            task.comm_time = best_comm_time
            task.comm_predecessors = best_comm_predecessors  # List of predecessors that require communication
            comp_schedule[best_node].insert(best_idx, task)
            task_schedule[task_name] = task
            
            logging.debug(
                "Scheduled task %s on node %s: [%s, %s]",
                task_name,
                best_node,
                task.start,
                task.end
            )

        return comp_schedule

    def schedule(
        self, network: nx.Graph, task_graph: nx.DiGraph
    ) -> Dict[str, List[Task]]:
        """Schedule the tasks on the network with busy communication.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[str, List[Task]]: The schedule.
        """
        schedule_order = heft_rank_sort(network, task_graph)
        return self._schedule(network, task_graph, schedule_order)