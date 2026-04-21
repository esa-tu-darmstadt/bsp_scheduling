"""
Sum-Product Network (SPN) based task graph generation.

This module converts SPNs serialized with CapnProto into task graphs
where each SPN node becomes a task in the computation graph.
"""

import logging
import pathlib
from typing import Tuple, Dict, List, Optional

import capnp
import networkx as nx
import numpy as np

from .base import TaskGraphGenerator, TaskGraphMetadata

logger = logging.getLogger(__name__)


class SPNTaskGraphGenerator(TaskGraphGenerator):
    """Generator for task graphs from Sum-Product Networks."""

    def __init__(self, cache_dir: Optional[pathlib.Path] = None, spn_schema_path: Optional[pathlib.Path] = None):
        """Initialize the SPN task graph generator.

        Args:
            cache_dir: Directory to store cached task graphs (optional)
            spn_schema_path: Path to the spflow.capnp schema file
        """
        super().__init__(cache_dir)

        if spn_schema_path is None:
            # Try to find the schema in the standard location
            spn_schema_path = pathlib.Path(__file__).parent.parent.parent.parent / "data" / "spn" / "spflow.capnp"

        if not spn_schema_path.exists():
            raise FileNotFoundError(f"SPN schema file not found: {spn_schema_path}")

        # Load the CapnProto schema
        self.spflow_schema = capnp.load(str(spn_schema_path))

    def _calculate_node_weight(self, node, node_type: str) -> float:
        """Calculate computational weight for an SPN node in clock cycles.

        Args:
            node: The SPN node (from capnp)
            node_type: Type of the node ('product', 'sum', 'histogram', 'gaussian', 'categorical')

        Returns:
            Computational weight in clock cycles
        """
        if node_type == 'product':
            # Product nodes: multiplication operations, cycles proportional to children
            # Base cost + cost per child multiplication
            return 10.0 + 5.0 * len(node.children)

        elif node_type == 'sum':
            # Sum nodes: weighted sum operations, more complex than product
            # Base cost + cost per child (includes weight multiplication and addition)
            return 15.0 + 8.0 * len(node.children)

        elif node_type == 'histogram':
            # Histogram leaf: lookup and interpolation operations
            # Base cost + cost proportional to number of bins for lookup
            return 20.0 + 2.0 * len(node.breaks)

        elif node_type == 'gaussian':
            # Gaussian leaf: exponential and square operations (expensive)
            return 50.0

        elif node_type == 'categorical':
            # Categorical leaf: simple lookup operation
            # Base cost + small cost per category
            return 10.0 + 1.0 * len(node.probabilities)

        else:
            raise ValueError(f"Unknown SPN node type: {node_type}")

    def _calculate_edge_weight(self, parent_type: str, child_type: str) -> float:
        """Calculate communication weight between SPN nodes in bits.

        Args:
            parent_type: Type of the parent node
            child_type: Type of the child node

        Returns:
            Communication weight in bits (32-bit floats)
        """
        # Base data transfer: one 32-bit float
        base_bits = 32.0

        return base_bits

    def load_spn_from_file(self, spn_file_path: pathlib.Path) -> Tuple[any, Dict]:
        """Load SPN from binary file.

        Args:
            spn_file_path: Path to the serialized SPN file

        Returns:
            Tuple of (spn_model, node_info_dict)
        """
        if not spn_file_path.exists():
            raise FileNotFoundError(f"SPN file not found: {spn_file_path}")

        try:
            with open(spn_file_path, 'rb') as f:
                # Read the header
                header = self.spflow_schema.Header.read(f)

                if header.which() == 'model':
                    model = header.model
                elif header.which() == 'query':
                    # Extract the model from the query's joint probability
                    query = header.query
                    if hasattr(query, 'joint') and hasattr(query.joint, 'model'):
                        model = query.joint.model
                        logger.info(f"Extracted SPN model from query in {spn_file_path.name}")
                    else:
                        raise ValueError("Query structure does not contain a model")
                else:
                    raise ValueError(f"Unknown header type: {header.which()}")

                # Create a mapping of node ID to node info for easy lookup
                node_info = {}
                for node in model.nodes:
                    node_type = node.which()
                    if node_type == 'product':
                        node_data = node.product
                    elif node_type == 'sum':
                        node_data = node.sum
                    elif node_type == 'hist':
                        node_data = node.hist
                        node_type = 'histogram'
                    elif node_type == 'gaussian':
                        node_data = node.gaussian
                    elif node_type == 'categorical':
                        node_data = node.categorical
                    else:
                        continue

                    node_info[node.id] = {
                        'type': node_type,
                        'data': node_data,
                        'is_root': node.rootNode
                    }

                return model, node_info

        except Exception as e:
            logger.error(f"Failed to load SPN from {spn_file_path}: {e}")
            raise

    def generate_task_graph(self, spn_file_path: str) -> Tuple[nx.DiGraph, TaskGraphMetadata]:
        """Generate a task graph from an SPN file.

        Args:
            spn_file_path: Path to the serialized SPN file

        Returns:
            Tuple of (task_graph, metadata)
        """
        spn_path = pathlib.Path(spn_file_path)

        # Load the SPN
        model, node_info = self.load_spn_from_file(spn_path)

        # Create directed graph
        task_graph = nx.DiGraph()

        # Add nodes with weights
        for node_id, info in node_info.items():
            weight = self._calculate_node_weight(info['data'], info['type'])
            task_graph.add_node(node_id, weight=weight, spn_type=info['type'])

        # Add edges based on SPN structure
        for node_id, info in node_info.items():
            if info['type'] in ['product', 'sum']:
                # These nodes have children
                children = info['data'].children
                for child_id in children:
                    if child_id in node_info:
                        # Add edge from child to parent (data flow direction)
                        edge_weight = self._calculate_edge_weight(
                            info['type'], node_info[child_id]['type']
                        )
                        task_graph.add_edge(child_id, node_id, weight=edge_weight)

        # Ensure the graph is connected and has proper structure
        if not nx.is_weakly_connected(task_graph):
            logger.warning(f"SPN graph from {spn_file_path} is not connected")

        # Calculate statistics
        task_weights = [task_graph.nodes[node].get('weight', 1.0) for node in task_graph.nodes()]
        edge_weights = [task_graph.edges[edge].get('weight', 1.0) for edge in task_graph.edges()]

        avg_task_weight = np.mean(task_weights) if task_weights else 1.0
        avg_edge_weight = np.mean(edge_weights) if edge_weights else 1.0

        # Collect additional info about the SPN
        node_type_counts = {}
        for info in node_info.values():
            node_type = info['type']
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1

        # Create metadata
        metadata = TaskGraphMetadata(
            source_type='spn',
            source_name=spn_path.name,
            task_count=len(task_graph.nodes()),
            edge_count=len(task_graph.edges()),
            avg_task_weight=avg_task_weight,
            avg_edge_weight=avg_edge_weight,
            additional_info={
                'spn_name': model.name,
                'num_features': model.numFeatures,
                'node_type_counts': node_type_counts,
                'root_node_id': model.rootNode
            }
        )

        return task_graph, metadata

    def generate_dataset_from_directory(self, spn_dir: pathlib.Path,
                                      target_ccrs: Optional[List[float]] = None,
                                      overwrite_cache: bool = False) -> Tuple[List[nx.DiGraph], List[TaskGraphMetadata]]:
        """Generate task graphs from all SPN files in a directory.

        Args:
            spn_dir: Directory containing SPN files
            target_ccrs: List of target CCRs to apply (None for no scaling)
            overwrite_cache: Whether to overwrite cached data

        Returns:
            Tuple of (task_graphs, metadata_list)
        """
        if not spn_dir.exists():
            raise FileNotFoundError(f"SPN directory not found: {spn_dir}")

        # Find all .bin files (SPN files)
        spn_files = list(spn_dir.glob("*.bin"))
        if not spn_files:
            raise ValueError(f"No SPN files (.bin) found in {spn_dir}")

        cache_key = f"spn_dir_{spn_dir.name}"
        if target_ccrs:
            cache_key += f"_ccrs_{'-'.join(f'{ccr:.2f}' for ccr in target_ccrs)}"

        # Check cache first
        if not overwrite_cache:
            cached_data = self.load_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"Loaded cached SPN dataset: {len(cached_data[0])} graphs")
                return cached_data

        logger.info(f"Generating SPN dataset from {len(spn_files)} files in {spn_dir}")

        task_graphs = []
        metadata_list = []

        # Default CCRs if none specified
        if target_ccrs is None:
            target_ccrs = [None]  # No scaling

        total_combinations = len(spn_files) * len(target_ccrs)
        processed = 0

        for spn_file in spn_files:
            for target_ccr in target_ccrs:
                try:
                    task_graph, metadata = self.generate_task_graph(str(spn_file), target_ccr)
                    task_graphs.append(task_graph)
                    metadata_list.append(metadata)

                    processed += 1
                    if processed % 5 == 0:
                        logger.info(f"Processed {processed}/{total_combinations} combinations")

                except Exception as e:
                    logger.warning(f"Failed to process {spn_file.name} with CCR {target_ccr}: {e}")

        # Save to cache
        if task_graphs and self.cache_dir:
            self.save_to_cache(cache_key, task_graphs, metadata_list)
            logger.info(f"Cached {len(task_graphs)} SPN-based task graphs")

        return task_graphs, metadata_list

    def list_available_spns(self, spn_dir: pathlib.Path) -> List[str]:
        """Get list of available SPN files in directory.

        Args:
            spn_dir: Directory to search for SPN files

        Returns:
            List of SPN filenames
        """
        if not spn_dir.exists():
            return []

        return [f.name for f in spn_dir.glob("*.bin")]