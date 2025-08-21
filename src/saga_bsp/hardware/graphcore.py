"""
Graphcore IPU hardware specialization for BSP scheduling.

This module provides hardware configurations for Graphcore Intelligence Processing Units (IPUs),
including single IPU configurations and multi-IPU M2000 systems with torus topology.
"""

from typing import List
import networkx as nx
import math
from ..schedule import BSPHardware


class GraphcoreIPUHardware(BSPHardware):
    """Specialized BSP hardware configuration for Graphcore IPUs.
    
    Architecture:
    - Each IPU contains 1472 tiles (processors)
    - All tiles within an IPU are fully connected
    - Communication bus: 4 bytes width
    - Default IPU system clock: 1.8 GHz
    - Multiple IPUs connected via IPU-Links in M2000 devices (4 IPUs per M2000)
    - M2000 systems connected in torus topology
    """
    
    TILES_PER_IPU = 1472
    IPUS_PER_M2000 = 4
    COMMUNICATION_BUS_WIDTH = 4  # bytes
    DEFAULT_IPU_CLOCK_GHZ = 1.8
    
    def __init__(self, 
                 num_tiles: int,
                 ipu_clock_ghz: float = DEFAULT_IPU_CLOCK_GHZ,
                 sync_time_ns: float = 100.0):
        """Initialize Graphcore IPU hardware configuration.
        
        Args:
            num_tiles: Total number of tiles to configure
            ipu_clock_ghz: IPU system clock frequency in GHz (default: 1.8)
            sync_time_ns: Synchronization time in nanoseconds (default: 100.0)
        """
        self.num_tiles = num_tiles
        self.ipu_clock_ghz = ipu_clock_ghz
        self.sync_time_ns = sync_time_ns
        
        # Calculate number of IPUs and M2000 systems needed
        self.num_ipus = math.ceil(num_tiles / self.TILES_PER_IPU)
        self.num_m2000s = math.ceil(self.num_ipus / self.IPUS_PER_M2000)
        
        # Generate network topology and calculate sync time
        network = self._generate_network_topology()
        sync_time_seconds = sync_time_ns / 1e9  # Convert ns to seconds
        
        super().__init__(network=network, sync_time=sync_time_seconds)
    
    def _generate_network_topology(self) -> nx.Graph:
        """Generate the network topology for the IPU configuration."""
        network = nx.Graph()
        
        # Add all tiles as nodes
        for tile_id in range(self.num_tiles):
            # Calculate which IPU and M2000 this tile belongs to
            ipu_id = tile_id // self.TILES_PER_IPU
            m2000_id = ipu_id // self.IPUS_PER_M2000
            
            # Tile compute capability (all tiles equal for simplicity)
            tile_speed = self.ipu_clock_ghz  # Operations per second (simplified)
            
            network.add_node(
                f"tile_{tile_id}",
                weight=tile_speed,
                ipu_id=ipu_id,
                m2000_id=m2000_id
            )
        
        # Add edges based on topology
        self._add_intra_ipu_connections(network)
        self._add_inter_ipu_connections(network)
        self._add_inter_m2000_connections(network)
        
        return network
    
    def _add_intra_ipu_connections(self, network: nx.Graph):
        """Add full connectivity within each IPU."""
        for ipu_id in range(self.num_ipus):
            # Get all tiles in this IPU
            ipu_tiles = [
                f"tile_{tile_id}" 
                for tile_id in range(self.num_tiles)
                if tile_id // self.TILES_PER_IPU == ipu_id
            ]
            
            # Full connectivity within IPU (very high bandwidth)
            intra_ipu_bandwidth = self.COMMUNICATION_BUS_WIDTH * self.ipu_clock_ghz * 1e9  # bytes/second
            
            for i, tile1 in enumerate(ipu_tiles):
                for tile2 in ipu_tiles[i+1:]:
                    network.add_edge(tile1, tile2, weight=intra_ipu_bandwidth)
    
    def _add_inter_ipu_connections(self, network: nx.Graph):
        """Add IPU-Link connections between IPUs within M2000 systems."""
        if self.num_ipus <= 1:
            return
            
        # IPU-Link bandwidth (lower than intra-IPU)
        ipu_link_bandwidth = 64e9  # 64 GB/s per IPU-Link (typical value)
        
        for m2000_id in range(self.num_m2000s):
            # Get IPUs in this M2000
            m2000_ipus = []
            for ipu_id in range(self.num_ipus):
                if ipu_id // self.IPUS_PER_M2000 == m2000_id:
                    m2000_ipus.append(ipu_id)
            
            # Connect IPUs within M2000 according to the topology shown in images
            # Based on the image, it appears IPUs are connected in a specific pattern
            # For simplicity, we'll use a ring topology with cross-connections
            if len(m2000_ipus) > 1:
                self._connect_ipus_in_m2000(network, m2000_ipus, ipu_link_bandwidth)
    
    def _connect_ipus_in_m2000(self, network: nx.Graph, ipu_ids: List[int], bandwidth: float):
        """Connect IPUs within an M2000 system based on the hardware topology."""
        # Create representative nodes for each IPU (use first tile of each IPU)
        ipu_representatives = []
        for ipu_id in ipu_ids:
            first_tile_id = ipu_id * self.TILES_PER_IPU
            if first_tile_id < self.num_tiles:
                ipu_representatives.append(f"tile_{first_tile_id}")
        
        # Based on the M2000 topology from images, create ring + cross connections
        for i, ipu_rep1 in enumerate(ipu_representatives):
            for ipu_rep2 in ipu_representatives[i+1:]:
                # Connect representative tiles between IPUs
                network.add_edge(ipu_rep1, ipu_rep2, weight=bandwidth)
                
                # Also connect a few more tiles between IPUs for better connectivity
                ipu1_id = int(ipu_rep1.split('_')[1]) // self.TILES_PER_IPU
                ipu2_id = int(ipu_rep2.split('_')[1]) // self.TILES_PER_IPU
                
                # Add a few more inter-IPU connections
                for k in range(min(4, self.TILES_PER_IPU // 100)):  # Sparse connections
                    tile1_id = ipu1_id * self.TILES_PER_IPU + k * 100
                    tile2_id = ipu2_id * self.TILES_PER_IPU + k * 100
                    
                    if tile1_id < self.num_tiles and tile2_id < self.num_tiles:
                        network.add_edge(f"tile_{tile1_id}", f"tile_{tile2_id}", weight=bandwidth)
    
    def _add_inter_m2000_connections(self, network: nx.Graph):
        """Add connections between M2000 systems in torus topology."""
        if self.num_m2000s <= 1:
            return
            
        # Inter-M2000 bandwidth (typically lower than IPU-Links)
        inter_m2000_bandwidth = 32e9  # 32 GB/s (example value)
        
        # Create torus topology between M2000 systems
        # For simplicity, we'll connect representative tiles from each M2000
        m2000_representatives = []
        for m2000_id in range(self.num_m2000s):
            # Use the first tile of the first IPU in each M2000 as representative
            first_ipu_in_m2000 = m2000_id * self.IPUS_PER_M2000
            if first_ipu_in_m2000 < self.num_ipus:
                first_tile_id = first_ipu_in_m2000 * self.TILES_PER_IPU
                if first_tile_id < self.num_tiles:
                    m2000_representatives.append(f"tile_{first_tile_id}")
        
        # Torus topology: each M2000 connects to its neighbors
        for i, rep1 in enumerate(m2000_representatives):
            # Connect to next M2000 (with wraparound for torus)
            next_idx = (i + 1) % len(m2000_representatives)
            rep2 = m2000_representatives[next_idx]
            
            if rep1 != rep2:  # Avoid self-loops
                network.add_edge(rep1, rep2, weight=inter_m2000_bandwidth)
    
    def get_tile_info(self, tile_name: str) -> dict:
        """Get information about a specific tile."""
        if not tile_name.startswith("tile_"):
            raise ValueError(f"Invalid tile name: {tile_name}")
        
        tile_id = int(tile_name.split('_')[1])
        ipu_id = tile_id // self.TILES_PER_IPU
        m2000_id = ipu_id // self.IPUS_PER_M2000
        
        return {
            "tile_id": tile_id,
            "ipu_id": ipu_id,
            "m2000_id": m2000_id,
            "tile_in_ipu": tile_id % self.TILES_PER_IPU,
            "ipu_in_m2000": ipu_id % self.IPUS_PER_M2000
        }
    
    def __str__(self) -> str:
        return (f"GraphcoreIPUHardware(tiles={self.num_tiles}, "
                f"ipus={self.num_ipus}, m2000s={self.num_m2000s}, "
                f"clock={self.ipu_clock_ghz}GHz)")
    
    def __repr__(self) -> str:
        return self.__str__()


def create_ipu_hardware(num_tiles: int, 
                       ipu_clock_ghz: float = GraphcoreIPUHardware.DEFAULT_IPU_CLOCK_GHZ,
                       sync_time_ns: float = 100.0) -> GraphcoreIPUHardware:
    """Convenience function to create Graphcore IPU hardware configuration.
    
    Args:
        num_tiles: Number of tiles to configure
        ipu_clock_ghz: IPU clock frequency in GHz
        sync_time_ns: Synchronization time in nanoseconds
        
    Returns:
        GraphcoreIPUHardware: Configured IPU hardware
    """
    return GraphcoreIPUHardware(
        num_tiles=num_tiles,
        ipu_clock_ghz=ipu_clock_ghz,
        sync_time_ns=sync_time_ns
    )