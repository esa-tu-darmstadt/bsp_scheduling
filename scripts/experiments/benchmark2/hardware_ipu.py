"""
IPU Hardware implementation for BSP scheduling.

This module implements the IPU topology as specified:
- 4 tiles per island, connected at 128 bit/cycle
- 16 islands per column, connected at 64 bit/cycle
- 23 columns per IPU, inter-column tiles connected at 32 bit/cycle
- Multiple IPUs connected at 8 bit/cycle

ALL tiles are connected to ALL other tiles, with speeds determined by their hierarchical relationship.
"""

import networkx as nx
import math
from typing import Union, Optional
import logging

from saga_bsp import BSPHardware

logger = logging.getLogger(__name__)

class IPUHardware(BSPHardware):
    """IPU-specific BSP hardware implementation."""

    # IPU topology constants
    TILES_PER_ISLAND = 4
    ISLANDS_PER_COLUMN = 16
    COLUMNS_PER_IPU = 23

    TILES_PER_COLUMN = TILES_PER_ISLAND * ISLANDS_PER_COLUMN  # 64
    TILES_PER_IPU = TILES_PER_COLUMN * COLUMNS_PER_IPU        # 1472

    # Connection speeds (bit/cycle)
    INTRA_ISLAND_SPEED = 128    # tiles within same island
    INTRA_COLUMN_SPEED = 64     # islands within same column
    INTRA_IPU_SPEED = 32        # columns within same IPU
    INTER_IPU_SPEED = 8         # between IPUs

    def __init__(self,
                 num_tiles: Optional[int] = None,
                 num_islands: Optional[int] = None,
                 num_columns: Optional[int] = None,
                 num_ipus: Optional[int] = None,
                 sync_time: float = 100.0):
        """Initialize IPU hardware configuration.

        User can specify either num_tiles, num_islands, num_columns, or num_ipus.
        Only one parameter should be provided.

        Args:
            num_tiles: Total number of tiles
            num_islands: Number of islands (each has 4 tiles)
            num_columns: Number of columns (each has 64 tiles)
            num_ipus: Number of IPUs (each has 1472 tiles)
            sync_time: Synchronization time in nanoseconds
        """
        # Validate input - exactly one parameter should be provided
        params = [num_tiles, num_islands, num_columns, num_ipus]
        non_none_params = [p for p in params if p is not None]

        if len(non_none_params) != 1:
            raise ValueError("Exactly one of num_tiles, num_islands, num_columns, or num_ipus must be specified")

        # Calculate total tiles based on input
        if num_tiles is not None:
            self.num_tiles = num_tiles
        elif num_islands is not None:
            self.num_tiles = num_islands * self.TILES_PER_ISLAND
        elif num_columns is not None:
            self.num_tiles = num_columns * self.TILES_PER_COLUMN
        elif num_ipus is not None:
            self.num_tiles = num_ipus * self.TILES_PER_IPU

        # Calculate hierarchy counts
        self.num_ipus = math.ceil(self.num_tiles / self.TILES_PER_IPU)
        self.num_columns = math.ceil(self.num_tiles / self.TILES_PER_COLUMN)
        self.num_islands = math.ceil(self.num_tiles / self.TILES_PER_ISLAND)

        logger.info(f"IPU Hardware: {self.num_tiles} tiles, {self.num_islands} islands, "
                   f"{self.num_columns} columns, {self.num_ipus} IPUs")

        # Generate network topology
        network = self._generate_network_topology()

        # Convert sync time from nanoseconds to seconds
        sync_time_seconds = sync_time / 1e9

        super().__init__(network=network, sync_time=sync_time_seconds)

    def _generate_network_topology(self) -> nx.Graph:
        """Generate the network topology for the IPU configuration.

        Creates a fully connected graph where ALL tiles are connected to ALL other tiles,
        with connection speeds determined by their hierarchical relationship.
        """
        network = nx.Graph()

        # Add all tiles as nodes with uniform processing speed
        for tile_id in range(self.num_tiles):
            network.add_node(f"tile_{tile_id}", weight=1.0)  # Uniform processing capability

        # Connect ALL tiles to ALL other tiles (including self-loops for SAGA compatibility)
        for tile1_id in range(self.num_tiles):
            for tile2_id in range(self.num_tiles):
                tile1_name = f"tile_{tile1_id}"
                tile2_name = f"tile_{tile2_id}"

                if tile1_id == tile2_id:
                    # Self-loops have zero communication cost (same node)
                    network.add_edge(tile1_name, tile2_name, weight=1e9)
                else:
                    # Determine connection speed based on hierarchical relationship
                    speed = self._get_connection_speed(tile1_id, tile2_id)
                    # Only add edge if it doesn't exist (since we want undirected graph)
                    if not network.has_edge(tile1_name, tile2_name):
                        network.add_edge(tile1_name, tile2_name, weight=speed)

        logger.info(f"Generated fully connected network with {network.number_of_nodes()} nodes and {network.number_of_edges()} edges")
        return network

    def _get_tile_location(self, tile_id: int) -> dict:
        """Get hierarchical location of a tile."""
        island_id = tile_id // self.TILES_PER_ISLAND
        column_id = tile_id // self.TILES_PER_COLUMN
        ipu_id = tile_id // self.TILES_PER_IPU

        return {
            'tile_id': tile_id,
            'island_id': island_id,
            'column_id': column_id,
            'ipu_id': ipu_id,
            'tile_in_island': tile_id % self.TILES_PER_ISLAND,
            'island_in_column': island_id % self.ISLANDS_PER_COLUMN,
            'column_in_ipu': column_id % self.COLUMNS_PER_IPU
        }

    def _get_connection_speed(self, tile1_id: int, tile2_id: int) -> float:
        """Get connection speed between two tiles based on their hierarchical relationship."""
        if tile1_id == tile2_id:
            return float('inf')  # Same tile

        loc1 = self._get_tile_location(tile1_id)
        loc2 = self._get_tile_location(tile2_id)

        # Same island - highest speed (128 bit/cycle)
        if loc1['island_id'] == loc2['island_id']:
            return self.INTRA_ISLAND_SPEED

        # Same column - medium-high speed (64 bit/cycle)
        elif loc1['column_id'] == loc2['column_id']:
            return self.INTRA_COLUMN_SPEED

        # Same IPU - medium speed (32 bit/cycle)
        elif loc1['ipu_id'] == loc2['ipu_id']:
            return self.INTRA_IPU_SPEED

        # Different IPUs - lowest speed (8 bit/cycle)
        else:
            return self.INTER_IPU_SPEED

    def get_connection_speed(self, tile1_id: int, tile2_id: int) -> float:
        """Public method to get connection speed between two tiles."""
        return self._get_connection_speed(tile1_id, tile2_id)

    def get_tile_info(self, tile_name: str) -> dict:
        """Get hierarchical information about a tile."""
        if not tile_name.startswith("tile_"):
            raise ValueError(f"Invalid tile name: {tile_name}")

        tile_id = int(tile_name.split('_')[1])
        return self._get_tile_location(tile_id)

    def __str__(self) -> str:
        return (f"IPUHardware(tiles={self.num_tiles}, islands={self.num_islands}, "
                f"columns={self.num_columns}, ipus={self.num_ipus})")

    def __repr__(self) -> str:
        return self.__str__()


# Convenience functions for creating hardware configurations
def create_ipu_from_tiles(num_tiles: int, sync_time: float = 100.0) -> IPUHardware:
    """Create IPU hardware from tile count."""
    return IPUHardware(num_tiles=num_tiles, sync_time=sync_time)

def create_ipu_from_islands(num_islands: int, sync_time: float = 100.0) -> IPUHardware:
    """Create IPU hardware from island count."""
    return IPUHardware(num_islands=num_islands, sync_time=sync_time)

def create_ipu_from_columns(num_columns: int, sync_time: float = 100.0) -> IPUHardware:
    """Create IPU hardware from column count."""
    return IPUHardware(num_columns=num_columns, sync_time=sync_time)

def create_ipu_from_ipus(num_ipus: int, sync_time: float = 100.0) -> IPUHardware:
    """Create IPU hardware from IPU count."""
    return IPUHardware(num_ipus=num_ipus, sync_time=sync_time)