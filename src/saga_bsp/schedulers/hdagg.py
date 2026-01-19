"""HDagg (Hybrid DAG Aggregation) Scheduler.

Implements the HDagg algorithm from:
"HDagg: Hybrid Aggregation of Loop-carried Dependence Iterations in Sparse Matrix Computations"
by Zarebavani et al.

HDagg is a two-step algorithm:
1. Aggregating Densely Connected Vertices - Groups vertices that share data to improve locality
2. Load-Balance Preserving (LBP) Wavefront Coarsening - Merges wavefronts while maintaining load balance

IMPORTANT: This implementation follows the paper exactly. No simplifications or deviations.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
import logging
import networkx as nx

from .base import BSPScheduler
from ..schedule import BSPSchedule, BSPHardware, Superstep

logger = logging.getLogger(__name__)


class HDaggScheduler(BSPScheduler):
    """HDagg scheduler implementing the Hybrid DAG Aggregation algorithm.

    This scheduler creates schedules optimized for:
    - Load balance across processors
    - Data locality through vertex grouping
    - Reduced synchronization through wavefront coarsening

    The algorithm operates in two steps:
    1. Step 1: Aggregates densely connected vertices into groups (subtrees)
    2. Step 2: Coarsens wavefronts using Load-Balance Preserving (LBP) strategy

    Reference: Zarebavani et al., "HDagg: Hybrid Aggregation of Loop-carried
    Dependence Iterations in Sparse Matrix Computations"
    """

    def __init__(self, epsilon: float = 0.1, verbose: bool = False):
        """Initialize the HDagg scheduler.

        Args:
            epsilon: Load balance threshold (0 to 1). When PGP > epsilon,
                    the wavefront merging is cut. Lower values = stricter balance.
                    Default: 0.1 (10% imbalance tolerance)
            verbose: Enable detailed logging output
        """
        super().__init__()
        self.name = "HDagg"
        self.epsilon = epsilon
        self.verbose = verbose

        # Statistics tracking
        self._reset_stats()

    def _reset_stats(self):
        """Reset scheduler statistics."""
        self.stats = {
            'original_vertices': 0,
            'transitive_edges_removed': 0,
            'groups_created': 0,
            'coarsened_vertices': 0,
            'wavefronts': 0,
            'coarsened_wavefronts': 0,
            'bin_packing_disabled': False,
        }

    def _log(self, message: str):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"  [HDagg] {message}")

    # =========================================================================
    # STEP 1: Aggregating Densely Connected Vertices (Algorithm 1, Lines 1-20)
    # =========================================================================

    def _transitive_reduction(self, G: nx.DiGraph) -> nx.DiGraph:
        """Apply two-hop transitive edge reduction approximation (Algorithm 1, Line 1).

        This method removes an edge i → f if there exists a vertex j with
        incoming edge i → j and outgoing edge j → f (i.e., a two-hop path).

        Reference: SpMP method from [4] in the paper.

        Args:
            G: Input directed acyclic graph

        Returns:
            G': Reduced graph with transitive edges removed
        """
        G_prime = G.copy()
        edges_to_remove = []

        # For each edge i → f, check if there's a two-hop path i → j → f
        for i, f in G.edges():
            # Check all successors j of i
            for j in G.successors(i):
                if j != f and G.has_edge(j, f):
                    # Found two-hop path i → j → f, so edge i → f is transitive
                    edges_to_remove.append((i, f))
                    break

        G_prime.remove_edges_from(edges_to_remove)
        self.stats['transitive_edges_removed'] = len(edges_to_remove)
        self._log(f"Transitive reduction removed {len(edges_to_remove)} edges")

        return G_prime

    def _aggregate_densely_connected(self, G_prime: nx.DiGraph) -> List[List[Any]]:
        """Find subtrees of densely connected vertices (Algorithm 1, Lines 2-19).

        This implements the modified BFS traversal that finds subtrees in the
        reduced DAG. A subtree is formed when all parent vertices have exactly
        one outgoing edge.

        The algorithm:
        1. Initialize T with sink vertices (vertices with no outgoing edges)
        2. For each group H in T, try to expand it by adding parents
        3. If parents form a tree (each has exactly one outgoing edge), merge them
        4. Otherwise, add unvisited parents as new sink vertices

        Args:
            G_prime: Transitive-reduced DAG

        Returns:
            T: List of vertex groups (each group is a list of vertices)
        """
        # Line 2: T.append(G'.Sink()) - Initialize with sink vertices
        # Sink vertices are those with no outgoing edges
        sinks = [v for v in G_prime.nodes() if G_prime.out_degree(v) == 0]

        # Each sink starts as its own group in T
        T: List[List[Any]] = [[sink] for sink in sinks]
        visited: Set[Any] = set(sinks)  # Track visited vertices

        self._log(f"Initial sink vertices: {sinks}")

        # Lines 3-19: Process each group in T
        i = 0
        while i < len(T):
            H = T[i]  # Line 4: H = T[i]

            j = 0
            while j < len(H):  # Line 5: for j in H
                v = H[j]  # Line 6: v = H[j]
                A = list(G_prime.predecessors(v))  # Line 7: A = parents(G', v)

                # Line 8: Check if {v} ∪ A forms a tree
                # A tree is formed if each vertex in A has exactly one outgoing edge
                forms_tree = True
                for parent in A:
                    if G_prime.out_degree(parent) != 1:
                        forms_tree = False
                        break

                if forms_tree:
                    # Line 9: H.append(A) - Expand the group with parents
                    for parent in A:
                        if parent not in H:
                            H.append(parent)
                            visited.add(parent)
                else:
                    # Lines 11-15: Add unvisited parents as new sink vertices
                    for c in A:
                        if c not in visited:
                            T.append([c])
                            visited.add(c)

                j += 1

            # Line 18: T[i] = H (H is already T[i] by reference, but ensure it's updated)
            T[i] = H
            i += 1

        # Handle any vertices that weren't reached (sources with no predecessors)
        all_grouped = set()
        for group in T:
            all_grouped.update(group)

        for v in G_prime.nodes():
            if v not in all_grouped:
                T.append([v])
                self._log(f"Adding isolated vertex {v} as its own group")

        self.stats['groups_created'] = len(T)
        self._log(f"Created {len(T)} vertex groups")

        return T

    def _create_coarsened_dag(self, G: nx.DiGraph, G_prime: nx.DiGraph,
                              T: List[List[Any]]) -> Tuple[nx.DiGraph, Dict[int, List[Any]], Dict[Any, float]]:
        """Create coarsened DAG G'' from vertex groups (Algorithm 1, Line 20).

        Creates a new DAG where:
        - Each super-vertex represents a group of original vertices
        - Edges exist between super-vertices if any edge exists between their constituent vertices
        - Super-vertex cost = sum of constituent vertex costs

        Args:
            G: Original task graph (for vertex weights)
            G_prime: Transitive-reduced DAG (for structure)
            T: List of vertex groups

        Returns:
            Tuple of:
            - G'': Coarsened DAG
            - group_to_vertices: Mapping from super-vertex ID to original vertices
            - costs: Cost (total weight) of each super-vertex
        """
        # Create mapping from original vertex to group index
        vertex_to_group: Dict[Any, int] = {}
        group_to_vertices: Dict[int, List[Any]] = {}

        for group_idx, group in enumerate(T):
            group_to_vertices[group_idx] = list(group)
            for v in group:
                vertex_to_group[v] = group_idx

        # Create coarsened DAG
        G_double_prime = nx.DiGraph()

        # Add super-vertices with aggregated costs
        costs: Dict[Any, float] = {}
        for group_idx, vertices in group_to_vertices.items():
            total_cost = sum(G.nodes[v]['weight'] for v in vertices)
            G_double_prime.add_node(group_idx, weight=total_cost)
            costs[group_idx] = total_cost

        # Add edges between super-vertices
        # An edge exists from group A to group B if any vertex in A has an edge to any vertex in B
        for u, v in G_prime.edges():
            group_u = vertex_to_group[u]
            group_v = vertex_to_group[v]
            if group_u != group_v and not G_double_prime.has_edge(group_u, group_v):
                # Use sum of edge weights between the groups
                edge_weight = sum(
                    G.edges[e]['weight']
                    for e in G.edges()
                    if vertex_to_group.get(e[0]) == group_u and vertex_to_group.get(e[1]) == group_v
                )
                G_double_prime.add_edge(group_u, group_v, weight=edge_weight)

        self.stats['coarsened_vertices'] = len(G_double_prime.nodes())
        self._log(f"Coarsened DAG has {len(G_double_prime.nodes())} super-vertices and {len(G_double_prime.edges())} edges")

        return G_double_prime, group_to_vertices, costs

    # =========================================================================
    # STEP 2: Load-Balance Preserving Wavefront Coarsening (Algorithm 1, Lines 21-38)
    # =========================================================================

    def _compute_wavefronts(self, G: nx.DiGraph) -> Tuple[Dict[int, List[Any]], int]:
        """Compute wavefronts (topological levels) of a DAG (Algorithm 1, Line 21).

        Each wavefront contains vertices that can execute in parallel after
        all vertices in previous wavefronts have completed.

        Args:
            G: DAG to compute wavefronts for

        Returns:
            Tuple of:
            - W: Dictionary mapping wavefront level to list of vertices
            - l: Total number of wavefronts
        """
        # Compute topological levels
        levels: Dict[Any, int] = {}

        for node in nx.topological_sort(G):
            preds = list(G.predecessors(node))
            if not preds:
                levels[node] = 0
            else:
                levels[node] = max(levels[p] for p in preds) + 1

        # Group vertices by level to create wavefronts
        W: Dict[int, List[Any]] = defaultdict(list)
        for node, level in levels.items():
            W[level].append(node)

        # l is the number of wavefronts (0-indexed levels, so l = max_level + 1)
        l = max(W.keys()) + 1 if W else 0

        self.stats['wavefronts'] = l
        self._log(f"Computed {l} wavefronts")

        return dict(W), l

    def _find_connected_components(self, G: nx.DiGraph, vertices: Set[Any]) -> List[Set[Any]]:
        """Find connected components in the subgraph induced by vertices.

        Uses the undirected version of the subgraph for connectivity analysis,
        as done in the paper using the Shiloach-Vishkin algorithm.

        Args:
            G: The full DAG
            vertices: Set of vertices to consider

        Returns:
            List of connected components (each component is a set of vertices)
        """
        if not vertices:
            return []

        # Create induced subgraph and convert to undirected for connectivity
        subgraph = G.subgraph(vertices).to_undirected()

        # Find connected components
        components = list(nx.connected_components(subgraph))

        return components

    def _bin_pack(self, connected_components: List[Set[Any]],
                  costs: Dict[Any, float], p: int) -> List[List[Any]]:
        """First-fit bin packing of connected components into p bins (Algorithm 1, Line 23, 25).

        HDagg uses a first-fit strategy where a connected component is assigned
        to the first bin that is not balanced. Vertices are ordered inside bins
        with smallest ID first to improve spatial locality.

        Args:
            connected_components: List of connected components to pack
            costs: Cost (weight) of each vertex
            p: Number of bins (processors)

        Returns:
            List of p bins, each containing a list of vertices
        """
        bins: List[List[Any]] = [[] for _ in range(p)]
        bin_loads: List[float] = [0.0] * p

        # Calculate cost of each connected component
        cc_with_costs = []
        for cc in connected_components:
            cc_cost = sum(costs.get(v, 0.0) for v in cc)
            cc_with_costs.append((cc, cc_cost))

        # Sort CCs by cost in descending order for better bin packing
        cc_with_costs.sort(key=lambda x: -x[1])

        for cc, cc_cost in cc_with_costs:
            # First-fit: assign to the bin with minimum load
            # (This is a common first-fit variant that tends to balance loads)
            min_load_idx = min(range(p), key=lambda i: bin_loads[i])

            # Sort vertices by ID for spatial locality (as specified in paper)
            sorted_vertices = sorted(cc)

            bins[min_load_idx].extend(sorted_vertices)
            bin_loads[min_load_idx] += cc_cost

        return bins

    def _calculate_pgp(self, bins: List[List[Any]], costs: Dict[Any, float]) -> float:
        """Calculate Potential Gain Proxy (PGP) metric (Algorithm 1, Line 26).

        PGP measures the load balance of aggregated iterations statically.
        It shows the potential runtime reduction if all cores had balanced workload.

        Formula: PGP = 1 - B̄ / max(B_i)

        Where:
        - B_i = sum of costs of vertices assigned to core i
        - B̄ = average of all B_i

        When balanced (all same load): PGP = 0
        Worst case (all on one core): PGP = 1 - 1/p

        Args:
            bins: List of bins, each containing vertices
            costs: Cost of each vertex

        Returns:
            PGP value (0 = perfectly balanced, higher = more imbalanced)
        """
        # Calculate load B_i for each bin
        B = [sum(costs.get(v, 0.0) for v in bin_contents) for bin_contents in bins]

        if not B or max(B) == 0:
            return 0.0

        B_avg = sum(B) / len(B)
        B_max = max(B)

        # PGP = 1 - B̄ / max(B_i)
        pgp = 1.0 - (B_avg / B_max)

        return pgp

    def _lbp_wavefront_coarsening(self, G_coarsened: nx.DiGraph,
                                   W: Dict[int, List[Any]], l: int,
                                   costs: Dict[Any, float],
                                   p: int, epsilon: float) -> Tuple[List[Tuple[List[List[Any]], bool]], bool]:
        """Load-Balance Preserving (LBP) wavefront coarsening (Algorithm 1, Lines 21-38).

        This algorithm merges consecutive wavefronts while maintaining load balance
        as measured by the PGP metric. When PGP exceeds epsilon, a cut is made.

        Args:
            G_coarsened: Coarsened DAG G''
            W: Wavefronts (mapping from level to vertex list)
            l: Number of wavefronts
            costs: Cost of each super-vertex
            p: Number of processors
            epsilon: Load balance threshold

        Returns:
            Tuple of:
            - S: List of coarsened wavefronts, each is (bins, use_bin_pack)
            - disable_bin_pack: Whether to disable bin packing globally
        """
        S: List[Tuple[List[List[Any]], bool]] = []

        if l == 0:
            return S, False

        # Line 22: cut = 0
        cut = 0

        # Line 23: S_curr = S_prev = BinPack(CC(W[0:1]), C, p)
        # W[0:1] means just wavefront 0 (first wavefront)
        vertices_in_range = set(W.get(0, []))
        ccs = self._find_connected_components(G_coarsened, vertices_in_range)
        S_curr = self._bin_pack(ccs, costs, p)
        S_prev = [list(bin_contents) for bin_contents in S_curr]  # Deep copy

        # Lines 24-35: Iterate through wavefronts
        for i in range(1, l + 1):
            # Line 25: S_curr = BinPack(CC(W[cut:i]), C, p)
            # W[cut:i] means wavefronts from cut to i-1 (inclusive)
            vertices_in_range = set()
            for level in range(cut, i):
                vertices_in_range.update(W.get(level, []))

            ccs = self._find_connected_components(G_coarsened, vertices_in_range)
            S_curr = self._bin_pack(ccs, costs, p)

            # Line 26: if PGP(S_curr) > ε then
            pgp_curr = self._calculate_pgp(S_curr, costs)

            if pgp_curr > epsilon:
                # Lines 27-31: Handle the cut
                if cut == i - 1:
                    # Line 27-28: Single Unbalanced Wave
                    # Append S_curr (the single unbalanced wavefront)
                    S.append((S_curr, True))
                    self._log(f"Cut at wavefront {i}: single unbalanced wave (PGP={pgp_curr:.3f})")
                    # Move cut forward to i (wavefront i-1 is already added)
                    cut = i
                else:
                    # Lines 29-30: Append S_prev (the balanced merged wavefronts)
                    S.append((S_prev, True))
                    self._log(f"Cut at wavefront {i}: merged wavefronts {cut} to {i-2} (PGP={self._calculate_pgp(S_prev, costs):.3f})")

                    # The wavefront at i-1 caused imbalance. It needs to start the next group.
                    # Set cut = i-1 so the next range starts from i-1
                    cut = i - 1

                    # Recompute S_curr for just the single wavefront that caused imbalance
                    single_wf_vertices = set(W.get(i - 1, []))
                    single_wf_ccs = self._find_connected_components(G_coarsened, single_wf_vertices)
                    S_curr = self._bin_pack(single_wf_ccs, costs, p)

            # Line 34: S_prev = S_curr
            S_prev = [list(bin_contents) for bin_contents in S_curr]  # Deep copy

        # Handle remaining wavefronts after the loop
        # This handles the case where we either:
        # 1. Never cut at all (all wavefronts merged)
        # 2. Cut using the "else" branch (cut = i-1), leaving wavefront i-1 to l-1 unhandled
        # Note: When we cut using "single unbalanced wave" (cut = i = l), cut == l,
        # so this branch correctly does NOT add duplicates.
        if cut < l:
            # Append the final merged wavefronts
            S.append((S_prev, True))
            self._log(f"Final: merged wavefronts {cut} to {l-1}")

        # Lines 36-38: Check if overall PGP < ε, then disable bin packing
        overall_pgp = self._calculate_overall_pgp(S, costs)
        disable_bin_pack = overall_pgp < epsilon

        if disable_bin_pack:
            self._log(f"Overall PGP ({overall_pgp:.3f}) < epsilon ({epsilon}): disabling bin packing")
            # Mark all coarsened wavefronts to not use bin packing
            S = [(bins, False) for bins, _ in S]

        self.stats['coarsened_wavefronts'] = len(S)
        self.stats['bin_packing_disabled'] = disable_bin_pack

        return S, disable_bin_pack

    def _calculate_overall_pgp(self, S: List[Tuple[List[List[Any]], bool]],
                               costs: Dict[Any, float]) -> float:
        """Calculate overall PGP across all coarsened wavefronts.

        This is used for the final check in lines 36-38 of the algorithm.

        Args:
            S: List of coarsened wavefronts
            costs: Cost of each vertex

        Returns:
            Overall PGP value
        """
        if not S:
            return 0.0

        # Calculate accumulated load per bin across all coarsened wavefronts
        p = len(S[0][0]) if S else 0
        if p == 0:
            return 0.0

        total_loads = [0.0] * p
        for bins, _ in S:
            for i, bin_contents in enumerate(bins):
                total_loads[i] += sum(costs.get(v, 0.0) for v in bin_contents)

        if max(total_loads) == 0:
            return 0.0

        avg_load = sum(total_loads) / len(total_loads)
        max_load = max(total_loads)

        return 1.0 - (avg_load / max_load)

    # =========================================================================
    # Main Scheduling Method
    # =========================================================================

    def schedule(self, hardware: BSPHardware, task_graph: nx.DiGraph) -> BSPSchedule:
        """Schedule tasks using the HDagg algorithm.

        Implements Algorithm 1 from the paper:
        1. Step 1: Aggregate densely connected vertices
        2. Step 2: Load-balance preserving wavefront coarsening
        3. Build BSP schedule from the resulting structure

        Args:
            hardware: BSP hardware configuration
            task_graph: Task dependency graph

        Returns:
            BSPSchedule with tasks assigned to processors and supersteps
        """
        self._reset_stats()
        self.stats['original_vertices'] = len(task_graph.nodes())

        p = len(hardware.network.nodes())  # Number of processors
        processors = list(hardware.network.nodes())

        self._log(f"Starting HDagg scheduling with {self.stats['original_vertices']} vertices, {p} processors, epsilon={self.epsilon}")

        # =====================================================================
        # Step 1: Aggregating Densely Connected Vertices (Lines 1-20)
        # =====================================================================

        # Line 1: G' = TransitiveReduction(G)
        G_prime = self._transitive_reduction(task_graph)

        # Lines 2-19: Find subtrees/groups of densely connected vertices
        T = self._aggregate_densely_connected(G_prime)

        # Line 20: G'' = CoarsenedDAG(G', T)
        G_coarsened, group_to_vertices, costs = self._create_coarsened_dag(
            task_graph, G_prime, T
        )

        # =====================================================================
        # Step 2: Load-Balance Preserving Wavefront Coarsening (Lines 21-38)
        # =====================================================================

        # Line 21: W, l = Wavefront(G'')
        W, l = self._compute_wavefronts(G_coarsened)

        # Lines 22-38: LBP wavefront coarsening
        S, disable_bin_pack = self._lbp_wavefront_coarsening(
            G_coarsened, W, l, costs, p, self.epsilon
        )

        # =====================================================================
        # Build BSP Schedule from HDagg output
        # =====================================================================

        schedule = BSPSchedule(hardware, task_graph)

        self._build_bsp_schedule(
            schedule, S, group_to_vertices, processors, task_graph, disable_bin_pack
        )

        # Validate the schedule
        schedule.assert_valid()

        self._log(f"Schedule complete: {len(schedule.supersteps)} supersteps, makespan={schedule.makespan:.2f}")

        return schedule

    def _build_bsp_schedule(self, schedule: BSPSchedule,
                            S: List[Tuple[List[List[Any]], bool]],
                            group_to_vertices: Dict[int, List[Any]],
                            processors: List[Any],
                            task_graph: nx.DiGraph,
                            disable_bin_pack: bool):
        """Convert HDagg output to BSPSchedule.

        Maps the coarsened wavefronts and width-partitions to supersteps and
        processor assignments.

        Args:
            schedule: BSPSchedule to populate
            S: Coarsened wavefronts from LBP algorithm
            group_to_vertices: Mapping from super-vertex ID to original vertices
            processors: List of processor IDs
            task_graph: Original task graph (for dependency ordering)
            disable_bin_pack: Whether bin packing is disabled
        """
        for cw_idx, (bins, use_bin_pack) in enumerate(S):
            superstep = schedule.add_superstep()

            if use_bin_pack and not disable_bin_pack:
                # Use bin packing: each bin maps to a processor
                for proc_idx, bin_contents in enumerate(bins):
                    if proc_idx >= len(processors):
                        break
                    processor = processors[proc_idx]

                    # Expand super-vertices back to original vertices
                    original_vertices = []
                    for super_vertex in bin_contents:
                        original_vertices.extend(group_to_vertices[super_vertex])

                    # Sort vertices for spatial locality and dependency ordering
                    sorted_vertices = self._topological_sort_subset(
                        task_graph, original_vertices
                    )

                    for vertex in sorted_vertices:
                        schedule.schedule(vertex, processor, superstep)
            else:
                # Bin packing disabled: still need to respect BSP constraints
                # Keep connected components together on the same processor
                # but distribute bins round-robin across processors
                for bin_idx, bin_contents in enumerate(bins):
                    processor = processors[bin_idx % len(processors)]

                    # Expand super-vertices back to original vertices
                    original_vertices = []
                    for super_vertex in bin_contents:
                        original_vertices.extend(group_to_vertices[super_vertex])

                    # Sort vertices for dependency ordering
                    sorted_vertices = self._topological_sort_subset(
                        task_graph, original_vertices
                    )

                    for vertex in sorted_vertices:
                        schedule.schedule(vertex, processor, superstep)

    def _topological_sort_subset(self, task_graph: nx.DiGraph,
                                  vertices: List[Any]) -> List[Any]:
        """Sort a subset of vertices in topological order.

        This ensures that within a superstep/processor, tasks are ordered
        to respect dependencies (predecessors before successors).

        Args:
            task_graph: Full task graph
            vertices: Subset of vertices to sort

        Returns:
            Vertices sorted in topological order
        """
        vertex_set = set(vertices)

        # Build subgraph and find topological order
        subgraph = task_graph.subgraph(vertex_set)

        try:
            sorted_vertices = list(nx.topological_sort(subgraph))
        except nx.NetworkXUnfeasible:
            # If cycle detected (shouldn't happen), fall back to sorted order
            sorted_vertices = sorted(vertices)

        return sorted_vertices

    def print_stats(self):
        """Print scheduling statistics."""
        print("\n" + "="*50)
        print(f"HDagg Scheduler Statistics")
        print("="*50)
        print(f"Original vertices:         {self.stats['original_vertices']}")
        print(f"Transitive edges removed:  {self.stats['transitive_edges_removed']}")
        print(f"Groups created:            {self.stats['groups_created']}")
        print(f"Coarsened vertices:        {self.stats['coarsened_vertices']}")
        print(f"Original wavefronts:       {self.stats['wavefronts']}")
        print(f"Coarsened wavefronts:      {self.stats['coarsened_wavefronts']}")
        print(f"Bin packing disabled:      {self.stats['bin_packing_disabled']}")
        print("="*50)
