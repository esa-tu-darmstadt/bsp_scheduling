"""
Base classes and common utilities for task graph generation.
"""

import json
import pathlib
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import networkx as nx


@dataclass
class TaskGraphMetadata:
    """Metadata for generated task graphs (task graph properties only)."""
    source_type: str  # 'wfcommons' or 'spn'
    source_name: str  # recipe name or spn filename
    task_count: int
    edge_count: int
    avg_task_weight: float
    avg_edge_weight: float
    additional_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        """Convert metadata to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'TaskGraphMetadata':
        """Create metadata from dictionary."""
        return cls(**data)


class TaskGraphGenerator(ABC):
    """Abstract base class for task graph generators."""

    def __init__(self, cache_dir: Optional[pathlib.Path] = None):
        """Initialize the task graph generator.

        Args:
            cache_dir: Directory to store cached task graphs (optional)
        """
        self.cache_dir = cache_dir
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def generate_task_graph(self, source_name: str, **kwargs) -> Tuple[nx.DiGraph, TaskGraphMetadata]:
        """Generate a single task graph.

        Args:
            source_name: Name/identifier of the source
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (task_graph, metadata)
        """
        pass

    def get_cached_path(self, cache_key: str, suffix: str) -> pathlib.Path:
        """Get path for cached file."""
        if not self.cache_dir:
            raise ValueError("Cache directory not configured")
        return self.cache_dir / f"{cache_key}_{suffix}"

    def save_to_cache(self, cache_key: str, task_graphs: List[nx.DiGraph],
                     metadata_list: List[TaskGraphMetadata]) -> None:
        """Save generated data to cache."""
        if not self.cache_dir:
            return

        # Save task graphs
        with open(self.get_cached_path(cache_key, "graphs.pkl"), 'wb') as f:
            pickle.dump(task_graphs, f)

        # Save metadata
        metadata_dicts = [md.to_dict() for md in metadata_list]
        with open(self.get_cached_path(cache_key, "metadata.json"), 'w') as f:
            json.dump(metadata_dicts, f, indent=2)

    def load_from_cache(self, cache_key: str) -> Optional[Tuple[List[nx.DiGraph], List[TaskGraphMetadata]]]:
        """Load data from cache."""
        if not self.cache_dir:
            return None

        graphs_path = self.get_cached_path(cache_key, "graphs.pkl")
        metadata_path = self.get_cached_path(cache_key, "metadata.json")

        if not all(p.exists() for p in [graphs_path, metadata_path]):
            return None

        try:
            with open(graphs_path, 'rb') as f:
                task_graphs = pickle.load(f)

            with open(metadata_path, 'r') as f:
                metadata_dicts = json.load(f)
                metadata_list = [TaskGraphMetadata.from_dict(md) for md in metadata_dicts]

            return task_graphs, metadata_list

        except Exception:
            return None

