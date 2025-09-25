"""Module for collecting and analyzing dataset statistics."""

import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import statistics

import networkx as nx
import numpy as np

from saga.data import Dataset


@dataclass
class StatisticsSummary:
    """Summary statistics for a value (mean, min, max, std)."""
    mean: float
    min: float
    max: float
    std: float

    def to_dict(self) -> Dict:
        return {
            'mean': self.mean,
            'min': self.min,
            'max': self.max,
            'std': self.std
        }


@dataclass
class TaskGraphStats:
    """Statistics for task graphs."""
    num_tasks: StatisticsSummary
    task_weights: StatisticsSummary
    edge_weights: StatisticsSummary
    ccr: StatisticsSummary  # Communication to Computation Ratio
    critical_path_length: StatisticsSummary
    parallelism: StatisticsSummary  # Number of levels in DAG
    avg_degree: StatisticsSummary  # Average node degree

    def to_dict(self) -> Dict:
        return {
            'num_tasks': self.num_tasks.to_dict(),
            'task_weights': self.task_weights.to_dict(),
            'edge_weights': self.edge_weights.to_dict(),
            'ccr': self.ccr.to_dict(),
            'critical_path_length': self.critical_path_length.to_dict(),
            'parallelism': self.parallelism.to_dict(),
            'avg_degree': self.avg_degree.to_dict()
        }


@dataclass
class NetworkGraphStats:
    """Statistics for network graphs."""
    num_nodes: StatisticsSummary
    node_weights: StatisticsSummary
    edge_weights: StatisticsSummary
    avg_degree: StatisticsSummary
    diameter: StatisticsSummary  # Maximum shortest path between any two nodes
    clustering_coefficient: StatisticsSummary

    def to_dict(self) -> Dict:
        return {
            'num_nodes': self.num_nodes.to_dict(),
            'node_weights': self.node_weights.to_dict(),
            'edge_weights': self.edge_weights.to_dict(),
            'avg_degree': self.avg_degree.to_dict(),
            'diameter': self.diameter.to_dict(),
            'clustering_coefficient': self.clustering_coefficient.to_dict()
        }


@dataclass
class DatasetStatistics:
    """Complete statistics for a dataset."""
    name: str
    num_instances: int
    task_graph_stats: TaskGraphStats
    network_graph_stats: NetworkGraphStats

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'num_instances': self.num_instances,
            'task_graph_stats': self.task_graph_stats.to_dict(),
            'network_graph_stats': self.network_graph_stats.to_dict()
        }


def calculate_summary_stats(values: List[float]) -> StatisticsSummary:
    """Calculate summary statistics for a list of values."""
    if not values:
        return StatisticsSummary(0.0, 0.0, 0.0, 0.0)

    return StatisticsSummary(
        mean=statistics.mean(values),
        min=min(values),
        max=max(values),
        std=statistics.stdev(values) if len(values) > 1 else 0.0
    )


def analyze_task_graph(task_graph: nx.DiGraph) -> Dict[str, float]:
    """Analyze a single task graph and return metrics."""
    metrics = {}

    # Basic metrics
    metrics['num_tasks'] = len(task_graph.nodes)

    # Task weights
    task_weights = [task_graph.nodes[node].get('weight', 1.0) for node in task_graph.nodes]
    metrics['avg_task_weight'] = np.mean(task_weights) if task_weights else 0.0

    # Edge weights
    edge_weights = [task_graph.edges[edge].get('weight', 1.0) for edge in task_graph.edges]
    metrics['avg_edge_weight'] = np.mean(edge_weights) if edge_weights else 0.0

    # Communication to Computation Ratio (CCR)
    if metrics['avg_task_weight'] > 0:
        metrics['ccr'] = metrics['avg_edge_weight'] / metrics['avg_task_weight']
    else:
        metrics['ccr'] = 0.0

    # Critical path length
    try:
        if nx.is_directed_acyclic_graph(task_graph):
            metrics['critical_path_length'] = nx.dag_longest_path_length(task_graph, weight='weight')
        else:
            metrics['critical_path_length'] = 0.0
    except:
        metrics['critical_path_length'] = 0.0

    # Parallelism (number of levels)
    try:
        if nx.is_directed_acyclic_graph(task_graph):
            generations = list(nx.topological_generations(task_graph))
            metrics['parallelism'] = len(generations)
        else:
            metrics['parallelism'] = 1.0
    except:
        metrics['parallelism'] = 1.0

    # Average degree
    degrees = [task_graph.degree(node) for node in task_graph.nodes]
    metrics['avg_degree'] = np.mean(degrees) if degrees else 0.0

    return metrics


def analyze_network_graph(network: nx.Graph) -> Dict[str, float]:
    """Analyze a single network graph and return metrics."""
    metrics = {}

    # Basic metrics
    metrics['num_nodes'] = len(network.nodes)

    # Node weights
    node_weights = [network.nodes[node].get('weight', 1.0) for node in network.nodes]
    metrics['avg_node_weight'] = np.mean(node_weights) if node_weights else 0.0

    # Edge weights
    edge_weights = [network.edges[edge].get('weight', 1.0) for edge in network.edges]
    metrics['avg_edge_weight'] = np.mean(edge_weights) if edge_weights else 0.0

    # Average degree
    degrees = [network.degree(node) for node in network.nodes]
    metrics['avg_degree'] = np.mean(degrees) if degrees else 0.0

    # Diameter (max shortest path length)
    try:
        if nx.is_connected(network) and len(network.nodes) > 1:
            metrics['diameter'] = nx.diameter(network)
        else:
            metrics['diameter'] = 0.0
    except:
        metrics['diameter'] = 0.0

    # Clustering coefficient
    try:
        metrics['clustering_coefficient'] = nx.average_clustering(network)
    except:
        metrics['clustering_coefficient'] = 0.0

    return metrics


def analyze_dataset(dataset: Dataset, max_instances: int = 100) -> DatasetStatistics:
    """Analyze a complete dataset and return statistics."""
    print(f"Analyzing dataset: {dataset.name}")

    # Collect metrics for all instances
    task_metrics = {
        'num_tasks': [],
        'avg_task_weight': [],
        'avg_edge_weight': [],
        'ccr': [],
        'critical_path_length': [],
        'parallelism': [],
        'avg_degree': []
    }

    network_metrics = {
        'num_nodes': [],
        'avg_node_weight': [],
        'avg_edge_weight': [],
        'avg_degree': [],
        'diameter': [],
        'clustering_coefficient': []
    }

    num_instances = min(len(dataset), max_instances)

    for i in range(num_instances):
        if i % 10 == 0:
            print(f"  Processing instance {i+1}/{num_instances}")

        try:
            network, task_graph = dataset[i]

            # Analyze task graph
            task_stats = analyze_task_graph(task_graph)
            for key, value in task_stats.items():
                if key in task_metrics:
                    task_metrics[key].append(value)

            # Analyze network
            network_stats = analyze_network_graph(network)
            for key, value in network_stats.items():
                if key in network_metrics:
                    network_metrics[key].append(value)
        except Exception as e:
            print(f"  Error processing instance {i}: {e}")
            continue

    # Calculate summary statistics
    task_graph_stats = TaskGraphStats(
        num_tasks=calculate_summary_stats(task_metrics['num_tasks']),
        task_weights=calculate_summary_stats(task_metrics['avg_task_weight']),
        edge_weights=calculate_summary_stats(task_metrics['avg_edge_weight']),
        ccr=calculate_summary_stats(task_metrics['ccr']),
        critical_path_length=calculate_summary_stats(task_metrics['critical_path_length']),
        parallelism=calculate_summary_stats(task_metrics['parallelism']),
        avg_degree=calculate_summary_stats(task_metrics['avg_degree'])
    )

    network_graph_stats = NetworkGraphStats(
        num_nodes=calculate_summary_stats(network_metrics['num_nodes']),
        node_weights=calculate_summary_stats(network_metrics['avg_node_weight']),
        edge_weights=calculate_summary_stats(network_metrics['avg_edge_weight']),
        avg_degree=calculate_summary_stats(network_metrics['avg_degree']),
        diameter=calculate_summary_stats(network_metrics['diameter']),
        clustering_coefficient=calculate_summary_stats(network_metrics['clustering_coefficient'])
    )

    return DatasetStatistics(
        name=dataset.name,
        num_instances=len(task_metrics['num_tasks']),  # Actual instances processed
        task_graph_stats=task_graph_stats,
        network_graph_stats=network_graph_stats
    )


def save_dataset_statistics(stats: DatasetStatistics, output_dir: pathlib.Path) -> pathlib.Path:
    """Save dataset statistics to a text file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed JSON
    json_file = output_dir / f"{stats.name}_statistics.json"
    with open(json_file, 'w') as f:
        json.dump(stats.to_dict(), f, indent=2)

    # Save human-readable summary
    txt_file = output_dir / f"{stats.name}_summary.txt"
    with open(txt_file, 'w') as f:
        f.write(f"Dataset Statistics Summary: {stats.name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of instances analyzed: {stats.num_instances}\n\n")

        f.write("Task Graph Statistics:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Number of tasks:        avg={stats.task_graph_stats.num_tasks.mean:.2f}, "
               f"min={stats.task_graph_stats.num_tasks.min:.0f}, "
               f"max={stats.task_graph_stats.num_tasks.max:.0f}, "
               f"std={stats.task_graph_stats.num_tasks.std:.2f}\n")
        f.write(f"Task weights:           avg={stats.task_graph_stats.task_weights.mean:.2f}, "
               f"min={stats.task_graph_stats.task_weights.min:.2f}, "
               f"max={stats.task_graph_stats.task_weights.max:.2f}, "
               f"std={stats.task_graph_stats.task_weights.std:.2f}\n")
        f.write(f"Edge weights:           avg={stats.task_graph_stats.edge_weights.mean:.2f}, "
               f"min={stats.task_graph_stats.edge_weights.min:.2f}, "
               f"max={stats.task_graph_stats.edge_weights.max:.2f}, "
               f"std={stats.task_graph_stats.edge_weights.std:.2f}\n")
        f.write(f"CCR (Comm/Comp ratio):  avg={stats.task_graph_stats.ccr.mean:.2f}, "
               f"min={stats.task_graph_stats.ccr.min:.2f}, "
               f"max={stats.task_graph_stats.ccr.max:.2f}, "
               f"std={stats.task_graph_stats.ccr.std:.2f}\n")
        f.write(f"Critical path length:   avg={stats.task_graph_stats.critical_path_length.mean:.2f}, "
               f"min={stats.task_graph_stats.critical_path_length.min:.2f}, "
               f"max={stats.task_graph_stats.critical_path_length.max:.2f}, "
               f"std={stats.task_graph_stats.critical_path_length.std:.2f}\n")
        f.write(f"Parallelism (levels):   avg={stats.task_graph_stats.parallelism.mean:.2f}, "
               f"min={stats.task_graph_stats.parallelism.min:.0f}, "
               f"max={stats.task_graph_stats.parallelism.max:.0f}, "
               f"std={stats.task_graph_stats.parallelism.std:.2f}\n")
        f.write(f"Average node degree:    avg={stats.task_graph_stats.avg_degree.mean:.2f}, "
               f"min={stats.task_graph_stats.avg_degree.min:.2f}, "
               f"max={stats.task_graph_stats.avg_degree.max:.2f}, "
               f"std={stats.task_graph_stats.avg_degree.std:.2f}\n\n")

        f.write("Network Graph Statistics:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Number of nodes:        avg={stats.network_graph_stats.num_nodes.mean:.2f}, "
               f"min={stats.network_graph_stats.num_nodes.min:.0f}, "
               f"max={stats.network_graph_stats.num_nodes.max:.0f}, "
               f"std={stats.network_graph_stats.num_nodes.std:.2f}\n")
        f.write(f"Node weights:           avg={stats.network_graph_stats.node_weights.mean:.2f}, "
               f"min={stats.network_graph_stats.node_weights.min:.2f}, "
               f"max={stats.network_graph_stats.node_weights.max:.2f}, "
               f"std={stats.network_graph_stats.node_weights.std:.2f}\n")
        f.write(f"Edge weights:           avg={stats.network_graph_stats.edge_weights.mean:.2f}, "
               f"min={stats.network_graph_stats.edge_weights.min:.2f}, "
               f"max={stats.network_graph_stats.edge_weights.max:.2f}, "
               f"std={stats.network_graph_stats.edge_weights.std:.2f}\n")
        f.write(f"Average node degree:    avg={stats.network_graph_stats.avg_degree.mean:.2f}, "
               f"min={stats.network_graph_stats.avg_degree.min:.2f}, "
               f"max={stats.network_graph_stats.avg_degree.max:.2f}, "
               f"std={stats.network_graph_stats.avg_degree.std:.2f}\n")
        f.write(f"Network diameter:       avg={stats.network_graph_stats.diameter.mean:.2f}, "
               f"min={stats.network_graph_stats.diameter.min:.0f}, "
               f"max={stats.network_graph_stats.diameter.max:.0f}, "
               f"std={stats.network_graph_stats.diameter.std:.2f}\n")
        f.write(f"Clustering coefficient: avg={stats.network_graph_stats.clustering_coefficient.mean:.3f}, "
               f"min={stats.network_graph_stats.clustering_coefficient.min:.3f}, "
               f"max={stats.network_graph_stats.clustering_coefficient.max:.3f}, "
               f"std={stats.network_graph_stats.clustering_coefficient.std:.3f}\n")

    print(f"Saved statistics to {txt_file} and {json_file}")
    return txt_file


def generate_dot_files(dataset: Dataset, output_dir: pathlib.Path, num_samples: int = 1) -> List[pathlib.Path]:
    """Generate DOT files for sample task graphs and networks from the dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dot_files = []

    # Generate samples
    for i in range(min(num_samples, len(dataset))):
        network, task_graph = dataset[i]

        # Generate task graph DOT file
        task_dot_file = output_dir / f"{dataset.name}_task_graph_sample_{i}.dot"
        with open(task_dot_file, 'w') as f:
            f.write("digraph TaskGraph {\n")
            f.write("  rankdir=TB;\n")
            f.write("  node [shape=circle];\n")

            # Add nodes with weights
            for node in task_graph.nodes:
                weight = task_graph.nodes[node].get('weight', 1.0)
                f.write(f'  "{node}" [label="{node}\\n{weight:.2f}"];\n')

            # Add edges with weights
            for src, dst in task_graph.edges:
                weight = task_graph.edges[(src, dst)].get('weight', 1.0)
                f.write(f'  "{src}" -> "{dst}" [label="{weight:.2f}"];\n')

            f.write("}\n")

        dot_files.append(task_dot_file)

        # Generate network graph DOT file
        network_dot_file = output_dir / f"{dataset.name}_network_graph_sample_{i}.dot"
        with open(network_dot_file, 'w') as f:
            f.write("graph NetworkGraph {\n")
            f.write("  layout=fdp;\n")
            f.write("  node [shape=box];\n")

            # Add nodes with weights
            for node in network.nodes:
                weight = network.nodes[node].get('weight', 1.0)
                f.write(f'  "{node}" [label="{node}\\n{weight:.2f}"];\n')

            # Add edges with weights
            for src, dst in network.edges:
                weight = network.edges[(src, dst)].get('weight', 1.0)
                f.write(f'  "{src}" -- "{dst}" [label="{weight:.2f}"];\n')

            f.write("}\n")

        dot_files.append(network_dot_file)

    print(f"Generated {len(dot_files)} DOT files in {output_dir}")
    return dot_files