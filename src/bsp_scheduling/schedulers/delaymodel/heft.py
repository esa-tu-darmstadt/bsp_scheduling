"""Optimized HEFT Scheduler.

This is a reimplemented version of SAGA's HEFT scheduler that eliminates
the performance bottlenecks from precalculating large arrays.
"""

import logging
from typing import Dict, Hashable, List

import networkx as nx
import numpy as np

from saga.scheduler import Scheduler, Task
from saga.utils.tools import get_insert_loc

from .priorities import upward_rank


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


class HeftScheduler(Scheduler):
    """Schedules tasks using the HEFT algorithm.

    This is an optimized version that calculates runtime and communication
    times on-the-fly instead of precalculating large arrays.

    Source: https://dx.doi.org/10.1109/71.993206
    """

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
        """Schedule the tasks on the network.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.
            schedule_order (List[Hashable]): The order in which to schedule the tasks.

        Returns:
            Dict[Hashable, List[Task]]: The schedule.

        Raises:
            ValueError: If the instance is invalid.
        """
        comp_schedule: Dict[Hashable, List[Task]] = {node: [] for node in network.nodes}
        task_schedule: Dict[Hashable, Task] = {}

        task_name: Hashable
        logging.debug("Schedule order: %s", schedule_order)
        for task_name in schedule_order:
            min_finish_time = np.inf
            best_node = None
            for node in network.nodes:  # Find the best node to run the task
                max_arrival_time: float = max(
                    [
                        0.0,
                        *[
                            task_schedule[parent].end
                            + self.calculate_commtime(network, task_graph,
                                                     task_schedule[parent].node, node,
                                                     parent, task_name)
                            for parent in task_graph.predecessors(task_name)
                        ],
                    ]
                )

                runtime = self.calculate_runtime(network, task_graph, node, task_name)
                idx, start_time = get_insert_loc(
                    comp_schedule[node], max_arrival_time, runtime
                )

                logging.debug(
                    "Testing task %s on node %s: start time %s, finish time %s",
                    task_name,
                    node,
                    start_time,
                    start_time + runtime,
                )

                finish_time = start_time + runtime
                if finish_time < min_finish_time:
                    min_finish_time = finish_time
                    best_node = node, idx

            new_runtime = self.calculate_runtime(network, task_graph, best_node[0], task_name)
            task = Task(
                best_node[0], task_name, min_finish_time - new_runtime, min_finish_time
            )
            comp_schedule[best_node[0]].insert(best_node[1], task)
            task_schedule[task_name] = task

        return comp_schedule

    def schedule(
        self, network: nx.Graph, task_graph: nx.DiGraph
    ) -> Dict[str, List[Task]]:
        """Schedule the tasks on the network.

        Args:
            network (nx.Graph): The network graph.
            task_graph (nx.DiGraph): The task graph.

        Returns:
            Dict[str, List[Task]]: The schedule.

        Raises:
            ValueError: If the instance is invalid.
        """
        schedule_order = heft_rank_sort(network, task_graph)
        return self._schedule(network, task_graph, schedule_order)