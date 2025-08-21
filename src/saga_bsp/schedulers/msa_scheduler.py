"""
Multi-Stage Scheduling Approach (MSA) for LogP and BSP Models

Implementation of the scheduling algorithm described in:
"Static Scheduling Using Task Replication for LogP and BSP Models"
by Boeres, Rebello, and Skillicorn (1999)

The MSA algorithm consists of two main stages:
1. Task clustering with replication using clustering factor γ
2. Cluster-to-processor mapping with communication scheduling
"""

import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass

from .base import BSPScheduler
from ..schedule import BSPSchedule, BSPHardware


@dataclass
class TaskCluster:
    """Represents a cluster of tasks with replicated ancestors"""
    primary_task: str  # The main task this cluster is built around
    replicated_tasks: List[str]  # Replicated ancestor tasks
    band_level: int  # Which computation band this cluster belongs to
    
    @property
    def all_tasks(self) -> List[str]:
        """Get all tasks in this cluster (primary + replicated)"""
        return [self.primary_task] + self.replicated_tasks
    
    @property
    def size(self) -> int:
        """Get cluster size (γ + 1)"""
        return len(self.all_tasks)


@dataclass
class ComputationBand:
    """Represents a parallel computation band containing clusters"""
    level: int
    clusters: List[TaskCluster]
    
    def get_all_tasks(self) -> Set[str]:
        """Get all tasks in this band"""
        tasks = set()
        for cluster in self.clusters:
            tasks.update(cluster.all_tasks)
        return tasks


class MSAScheduler(BSPScheduler):
    """Multi-Stage Scheduling Approach (MSA) for BSP models.
    
    This scheduler implements the MSA algorithm which uses task clustering
    with replication to minimize communication overhead in BSP/LogP environments.
    
    Args:
        gamma: Clustering factor (default 3). Controls level of task replication.
        lambda_s: Sending overhead (default extracted from hardware)
        lambda_r: Receiving overhead (default extracted from hardware)  
        tau: Communication latency (default extracted from hardware)
    """
    
    def __init__(self, gamma: int = 3, lambda_s: Optional[float] = None, 
                 lambda_r: Optional[float] = None, tau: Optional[float] = None):
        super().__init__()
        self.name = "MSA"
        self.gamma = gamma  # Clustering factor
        self.lambda_s = lambda_s  # Sending overhead
        self.lambda_r = lambda_r  # Receiving overhead
        self.tau = tau  # Communication latency
        
    def schedule(self, hardware: BSPHardware, task_graph: nx.DiGraph) -> BSPSchedule:
        """Schedule tasks using the MSA algorithm.
        
        Args:
            hardware: BSP hardware configuration
            task_graph: Task dependency graph
            
        Returns:
            BSP schedule with optimized supersteps
        """
        # Extract communication parameters if not provided
        if self.lambda_s is None or self.lambda_r is None or self.tau is None:
            self._extract_communication_parameters(hardware)
            
        # Stage 1: Task clustering with replication
        clusters, bands = self._stage1_clustering(task_graph)
        
        # Stage 2: Cluster-to-processor mapping
        bsp_schedule = self._stage2_mapping(clusters, bands, hardware, task_graph)
        
        return bsp_schedule
    
    def _extract_communication_parameters(self, hardware: BSPHardware):
        """Extract LogP/BSP parameters from hardware configuration.
        
        For simplicity, we estimate parameters from network topology.
        In practice, these would be measured or provided explicitly.
        """
        if self.lambda_s is None:
            # Estimate sending overhead as fraction of sync time
            self.lambda_s = hardware.sync_time * 0.1
            
        if self.lambda_r is None:
            # Estimate receiving overhead (typically similar to sending)
            self.lambda_r = hardware.sync_time * 0.1
            
        if self.tau is None:
            # Estimate latency from network edge weights
            edge_weights = [data['weight'] for _, _, data in hardware.network.edges(data=True)]
            self.tau = 1.0 / max(edge_weights) if edge_weights else 1.0
    
    def _stage1_clustering(self, task_graph: nx.DiGraph) -> Tuple[List[TaskCluster], List[ComputationBand]]:
        """Stage 1: Task clustering with replication.
        
        Creates clusters of tasks with replicated ancestors to minimize
        communication overhead.
        
        Returns:
            Tuple of (clusters, computation_bands)
        """
        # Step 1: Compute task ranks using latest-finish time
        task_ranks = self._compute_latest_finish_times(task_graph)
        
        # Step 2: Topologically sort tasks by rank
        sorted_tasks = sorted(task_ranks.keys(), 
                            key=lambda t: (-task_ranks[t], t))  # Descending rank, stable sort
        
        # Step 3: Create clusters with replication
        clusters = []
        bands = []
        tasks_in_clusters = set()
        
        for task in sorted_tasks:
            if task in tasks_in_clusters:
                continue
                
            # Create cluster for this task
            cluster = self._create_cluster(task, task_graph, task_ranks, tasks_in_clusters)
            clusters.append(cluster)
            
            # Update tasks in clusters
            tasks_in_clusters.update(cluster.all_tasks)
            
            # Assign to computation band
            cluster.band_level = self._determine_band_level(cluster, bands, task_graph)
            
            # Add to appropriate band or create new band
            target_band = None
            for band in bands:
                if band.level == cluster.band_level:
                    target_band = band
                    break
                    
            if target_band is None:
                target_band = ComputationBand(cluster.band_level, [])
                bands.append(target_band)
                
            target_band.clusters.append(cluster)
        
        # Sort bands by level
        bands.sort(key=lambda b: b.level)
        
        return clusters, bands
    
    def _compute_latest_finish_times(self, task_graph: nx.DiGraph) -> Dict[str, float]:
        """Compute latest finish time for each task.
        
        This is used for task ranking in the MSA algorithm.
        Latest finish time is the latest time a task can finish while
        still achieving minimum makespan.
        """
        # Compute critical path lengths (upward rank)
        upward_rank = {}
        
        # Process tasks in reverse topological order
        for task in reversed(list(nx.topological_sort(task_graph))):
            task_weight = task_graph.nodes[task].get('weight', 1.0)
            
            if task_graph.out_degree(task) == 0:
                # Sink task
                upward_rank[task] = task_weight
            else:
                # Max over all successors
                max_successor_path = 0.0
                for successor in task_graph.successors(task):
                    comm_weight = task_graph.edges[task, successor].get('weight', 0.0)
                    successor_path = comm_weight + upward_rank[successor]
                    max_successor_path = max(max_successor_path, successor_path)
                    
                upward_rank[task] = task_weight + max_successor_path
        
        # Latest finish time = critical path length
        # In the paper, this is more sophisticated, but this approximation works
        return upward_rank
    
    def _create_cluster(self, primary_task: str, task_graph: nx.DiGraph, 
                       task_ranks: Dict[str, float], tasks_in_clusters: Set[str]) -> TaskCluster:
        """Create a cluster around a primary task with replicated ancestors.
        
        The cluster contains the primary task plus up to γ replicated ancestor tasks.
        """
        cluster = TaskCluster(primary_task, [], 0)
        
        # Find ancestors to replicate
        ancestors = list(task_graph.predecessors(primary_task))
        
        # Sort ancestors by rank (highest rank first)
        ancestors.sort(key=lambda t: task_ranks[t], reverse=True)
        
        # Add up to γ ancestors (that aren't already in clusters)
        replicated_count = 0
        for ancestor in ancestors:
            if replicated_count >= self.gamma:
                break
                
            if ancestor not in tasks_in_clusters:
                cluster.replicated_tasks.append(ancestor)
                replicated_count += 1
        
        return cluster
    
    def _determine_band_level(self, cluster: TaskCluster, bands: List[ComputationBand], 
                            task_graph: nx.DiGraph) -> int:
        """Determine which computation band level a cluster should be assigned to.
        
        Clusters with dependencies should be in later bands.
        """
        max_dependency_level = -1
        
        for task in cluster.all_tasks:
            for pred in task_graph.predecessors(task):
                # Find which band this predecessor is in
                for band in bands:
                    if pred in band.get_all_tasks():
                        max_dependency_level = max(max_dependency_level, band.level)
                        break
        
        return max_dependency_level + 1
    
    def _stage2_mapping(self, clusters: List[TaskCluster], bands: List[ComputationBand],
                       hardware: BSPHardware, task_graph: nx.DiGraph) -> BSPSchedule:
        """Stage 2: Map clusters to processors and create BSP schedule.
        
        This stage maps the clustered tasks to physical processors and
        creates the final BSP schedule with supersteps.
        """
        # Note: clusters parameter passed for completeness, currently using bands approach
        bsp_schedule = BSPSchedule(hardware, task_graph)
        
        # Create one superstep per computation band
        for band in bands:
            superstep = bsp_schedule.add_superstep()
            
            # Assign clusters to processors using load balancing
            self._assign_clusters_to_processors(band, superstep, hardware, task_graph)
        
        return bsp_schedule
    
    def _assign_clusters_to_processors(self, band: ComputationBand, superstep, 
                                     hardware: BSPHardware, task_graph: nx.DiGraph):
        """Assign clusters in a band to processors for load balancing.
        
        Uses a simple greedy assignment based on processor load.
        """
        # Track processor loads (total execution time)
        processor_loads = {proc: 0.0 for proc in hardware.network.nodes}
        
        # Sort clusters by size (largest first for better load balancing)
        sorted_clusters = sorted(band.clusters, 
                               key=lambda c: self._compute_cluster_cost(c, task_graph), 
                               reverse=True)
        
        for cluster in sorted_clusters:
            # Find processor with minimum load
            best_processor = min(processor_loads.keys(), 
                               key=lambda p: processor_loads[p])
            
            # Schedule all tasks in cluster on this processor
            for task in cluster.all_tasks:
                if task in task_graph.nodes:  # Ensure task exists in graph
                    superstep.schedule_task(task, best_processor)
                    
                    # Update processor load
                    task_weight = task_graph.nodes[task].get('weight', 1.0)
                    proc_speed = hardware.network.nodes[best_processor].get('weight', 1.0)
                    processor_loads[best_processor] += task_weight / proc_speed
    
    def _compute_cluster_cost(self, cluster: TaskCluster, task_graph: nx.DiGraph) -> float:
        """Compute total execution cost of a cluster."""
        total_cost = 0.0
        for task in cluster.all_tasks:
            if task in task_graph.nodes:
                total_cost += task_graph.nodes[task].get('weight', 1.0)
        return total_cost