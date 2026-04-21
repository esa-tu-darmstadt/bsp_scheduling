from typing import Dict, Hashable, List, Tuple
import networkx as nx
from saga.scheduler import Scheduler, Task

from ..schedulers.base import BSPScheduler
from ..schedule import BSPHardware
from ..conversion import convert_bsp_to_async


def preprocess_task_graph(task_graph: nx.DiGraph, weight_threshold: float = 1e-9) -> Tuple[nx.DiGraph, Dict]:
    """Preprocess task graph by removing nodes and edges with negligible weights.
    
    SAGA often uses nodes with very small weights (e.g., <= 1e-9) as entry/exit nodes
    to structure the task graph. While these have minimal computation/communication cost,
    they force synchronization in BSP which can falsify the schedule. This function
    removes such nodes and edges while preserving the essential dependencies.
    
    Args:
        task_graph: The original task dependency graph
        weight_threshold: Threshold below which weights are considered negligible (default: 1e-9)
        
    Returns:
        Tuple of:
            - Preprocessed task graph with low-weight nodes/edges removed
            - Dictionary with preprocessing metadata (removed nodes, edges, etc.)
    """
    # Create a copy to avoid modifying the original
    processed_graph = task_graph.copy()
    
    # Track what we remove for debugging/analysis
    metadata = {
        'removed_nodes': [],
        'removed_edges': [],
        'rerouted_edges': []
    }
    
    # First, identify nodes to remove (those with weight <= threshold)
    nodes_to_remove = []
    for node in processed_graph.nodes():
        node_weight = processed_graph.nodes[node].get('weight', 0.0)
        if node_weight <= weight_threshold:
            nodes_to_remove.append(node)
            metadata['removed_nodes'].append((node, node_weight))
    
    # For each node to remove, reroute dependencies
    for node in nodes_to_remove:
        # Get predecessors and successors
        predecessors = list(processed_graph.predecessors(node))
        successors = list(processed_graph.successors(node))
        
        # Connect each predecessor directly to each successor
        for pred in predecessors:
            for succ in successors:
                if pred != succ and not processed_graph.has_edge(pred, succ):
                    # Combine edge weights if multiple paths existed
                    pred_to_node_weight = processed_graph.edges[pred, node].get('weight', 0.0)
                    node_to_succ_weight = processed_graph.edges[node, succ].get('weight', 0.0)
                    combined_weight = max(pred_to_node_weight, node_to_succ_weight)
                    
                    # Only add edge if combined weight is above threshold
                    if combined_weight > weight_threshold:
                        processed_graph.add_edge(pred, succ, weight=combined_weight)
                        metadata['rerouted_edges'].append((pred, succ, combined_weight))
        
        # Remove the node (this also removes its incident edges)
        processed_graph.remove_node(node)
    
    # Second pass: remove edges with weight <= threshold
    edges_to_remove = []
    for u, v, data in processed_graph.edges(data=True):
        edge_weight = data.get('weight', 0.0)
        if edge_weight <= weight_threshold:
            edges_to_remove.append((u, v))
            metadata['removed_edges'].append((u, v, edge_weight))
    
    # Remove low-weight edges
    for u, v in edges_to_remove:
        processed_graph.remove_edge(u, v)
    
    # Clean up: remove any isolated nodes (nodes with no edges)
    isolated_nodes = list(nx.isolates(processed_graph))
    for node in isolated_nodes:
        node_weight = processed_graph.nodes[node].get('weight', 0.0)
        # Only remove if it's truly isolated and not a significant computation node
        if node_weight <= weight_threshold:
            processed_graph.remove_node(node)
            metadata['removed_nodes'].append((node, node_weight))
    
    # Add summary statistics to metadata
    metadata['summary'] = {
        'original_nodes': task_graph.number_of_nodes(),
        'original_edges': task_graph.number_of_edges(),
        'processed_nodes': processed_graph.number_of_nodes(),
        'processed_edges': processed_graph.number_of_edges(),
        'nodes_removed': len(metadata['removed_nodes']),
        'edges_removed': len(metadata['removed_edges']),
        'edges_rerouted': len(metadata['rerouted_edges'])
    }
    
    return processed_graph, metadata


class SagaSchedulerWrapper(Scheduler):
    """SAGA-compatible wrapper for BSP schedulers.
    
    This wrapper allows any BSP scheduler to be used within SAGA's
    infrastructure (benchmarking, simulated annealing, etc.) by converting
    BSP schedules back to the async format with proper BSP timing information.
    
    The wrapper bridges the gap between BSP scheduling algorithms and SAGA's
    async-focused analysis tools.
    
    Args:
        bsp_scheduler: Any BSP scheduler to wrap
        sync_time: BSP synchronization overhead time
    """
    
    def __init__(self, bsp_scheduler: BSPScheduler, sync_time: float = 1.0, 
                 preprocess: bool = False, weight_threshold: float = 1e-9):
        """Initialize the wrapper.
        
        Args:
            bsp_scheduler: Any BSP scheduler to wrap
            sync_time: BSP synchronization overhead time
            preprocess: Whether to preprocess task graphs to remove low-weight nodes/edges
            weight_threshold: Weight threshold for preprocessing (nodes/edges <= this are removed)
        """
        self.bsp_scheduler = bsp_scheduler
        self.sync_time = sync_time
        self.preprocess = preprocess
        self.weight_threshold = weight_threshold
        
        # Use the BSP scheduler's name directly (already clean)
        self.name = bsp_scheduler.__name__
    
    def schedule(self, network: nx.Graph, task_graph: nx.DiGraph) -> Dict[Hashable, List[Task]]:
        """Schedule tasks using BSP model and convert to SAGA async format.
        
        Args:
            network: The network topology from SAGA
            task_graph: The task dependency graph
            
        Returns:
            Dict mapping processors to lists of Tasks with BSP timing
        """
        # Optionally preprocess the task graph
        if self.preprocess:
            processed_graph, metadata = preprocess_task_graph(task_graph, self.weight_threshold)
            # Store metadata for debugging if needed
            self._last_preprocessing_metadata = metadata
            # Use the processed graph for scheduling
            task_graph = processed_graph
        
        # Create BSP hardware from SAGA's network and our sync time
        bsp_hardware = BSPHardware(network=network, sync_time=self.sync_time)
        
        # Get BSP schedule
        bsp_schedule = self.bsp_scheduler.schedule(bsp_hardware, task_graph)
        
        # Convert to SAGA-compatible async format with BSP timing
        async_schedule = convert_bsp_to_async(bsp_schedule)
        
        return async_schedule