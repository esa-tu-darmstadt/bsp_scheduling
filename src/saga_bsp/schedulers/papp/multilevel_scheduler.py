"""Multilevel Scheduler - Section 4.5 from Papp et al. 2024

This implements the multilevel coarsen-solve-refine scheduling approach:
1. Coarsening: Contract DAG to 15% or 30% of original size
2. Solving: Apply base scheduler (BSPg/Source) to coarsened DAG
3. Uncoarsening + Refinement: Gradually uncoarsen with HC refinement

Reference: Section 4.5, Appendix A.5 (pages 17-18), Figure 4
"""

from typing import Dict, Set, List, Optional, Tuple, Union
import copy
import networkx as nx

from saga_bsp.optimization.ilp_solvers import ILPcs

from ..base import BSPScheduler
from ...schedule import BSPSchedule, BSPHardware
from .bspg_scheduler import BSPgScheduler
from .source_scheduler import SourceScheduler
from .coarsening import DAGCoarsener, ContractionRecord
from ...optimization.hill_climbing import HillClimbing, HCcs


class MultilevelScheduler(BSPScheduler):
    """Multilevel scheduling algorithm from Papp et al. 2024.

    Three-phase approach:
    1. Coarsening: Contract DAG to target fraction of original size
    2. Solving: Apply base scheduler to coarsened DAG
    3. Uncoarsening + Refinement: Gradually expand with local optimization

    This approach is particularly effective for high-communication scenarios
    where nodes need to be grouped together to minimize communication costs.
    """

    def __init__(
        self,
        base_scheduler: Optional[BSPScheduler] = None,
        coarsening_ratios: List[float] = [0.15, 0.30],
        hc_interval: int = 5,
        hc_max_steps: int = 100,
        use_ilp: bool = False,
        incremental_coarsening: bool = True,
        verbose: bool = False
    ):
        """
        Args:
            base_scheduler: Scheduler to use for coarsened DAG (default: BSPgScheduler)
            coarsening_ratios: Target ratios to try (picks best result)
            hc_interval: Uncoarsening steps between HC refinements
            hc_max_steps: Max HC iterations per refinement
            use_ilp: Whether to use ILP optimization at the end
            incremental_coarsening: Use incremental maintenance of contractable edges
                                    during coarsening. This is our optimization (not from the
                                    paper) that avoids recomputing all contractable edges after
                                    each contraction. Instead, it tracks which edges could be
                                    affected and only rechecks those. Provides ~40x speedup
                                    on large graphs. Results may differ from standard due to
                                    unspecified tie-breaking in edge selection (both are valid
                                    per the paper's algorithm).
            verbose: Print progress information
        """
        super().__init__()
        self.name = "Multilevel"
        self.base_scheduler = base_scheduler or BSPgScheduler(optimized=True)
        self.coarsening_ratios = coarsening_ratios
        self.hc_interval = hc_interval
        self.hc_max_steps = hc_max_steps
        self.use_ilp = use_ilp
        self.incremental_coarsening = incremental_coarsening
        self.verbose = verbose
        self.stats = {
            'coarsening_time': 0.0,
            'solving_time': 0.0,
            'refinement_time': 0.0,
            'initial_nodes': 0,
            'coarsened_nodes': 0,
            'final_cost': 0.0
        }

    def schedule(self, hardware: BSPHardware, task_graph: nx.DiGraph) -> BSPSchedule:
        """Schedule tasks using multilevel approach."""
        import time

        self.stats['initial_nodes'] = len(task_graph.nodes())

        # Try multiple coarsening ratios and pick the best
        best_schedule = None
        best_cost = float('inf')

        for ratio in self.coarsening_ratios:
            if self.verbose:
                print(f"\nMultilevel: Trying coarsening ratio {ratio}")

            schedule = self._schedule_with_ratio(hardware, task_graph, ratio)

            if schedule is not None and schedule.makespan < best_cost:
                best_cost = schedule.makespan
                best_schedule = schedule

        if best_schedule is None:
            # Fallback to direct scheduling if multilevel fails
            if self.verbose:
                print("Multilevel: Falling back to direct scheduling")
            best_schedule = self.base_scheduler.schedule(hardware, task_graph)

        self.stats['final_cost'] = best_schedule.makespan
        best_schedule.assert_valid()
        return best_schedule

    def _schedule_with_ratio(
        self,
        hardware: BSPHardware,
        task_graph: nx.DiGraph,
        ratio: float
    ) -> Optional[BSPSchedule]:
        """Perform multilevel scheduling with a specific coarsening ratio."""
        import time

        # Phase 1: Coarsening
        start_time = time.time()
        coarsener = DAGCoarsener(incremental=self.incremental_coarsening)
        coarsened_graph = coarsener.coarsen(task_graph, target_ratio=ratio)
        coarsening_time = time.time() - start_time

        if self.verbose:
            print(f"  Coarsened: {len(task_graph.nodes())} -> {len(coarsened_graph.nodes())} nodes")

        self.stats['coarsened_nodes'] = len(coarsened_graph.nodes())

        # Phase 2: Solve on coarsened DAG
        start_time = time.time()
        try:
            coarse_schedule = self.base_scheduler.schedule(hardware, coarsened_graph)
        except Exception as e:
            if self.verbose:
                print(f"  Solving failed: {e}")
            return None
        solving_time = time.time() - start_time

        if self.verbose:
            print(f"  Coarse schedule cost: {coarse_schedule.makespan:.2f}")

        # Phase 3: Uncoarsen and refine
        start_time = time.time()
        schedule = self._uncoarsen_and_refine(
            coarse_schedule, coarsener, task_graph, hardware
        )
        refinement_time = time.time() - start_time

        self.stats['coarsening_time'] = coarsening_time
        self.stats['solving_time'] = solving_time
        self.stats['refinement_time'] = refinement_time

        if self.verbose:
            print(f"  Final schedule cost: {schedule.makespan:.2f}")

        return schedule

    def _uncoarsen_and_refine(
        self,
        coarse_schedule: BSPSchedule,
        coarsener: DAGCoarsener,
        original_graph: nx.DiGraph,
        hardware: BSPHardware
    ) -> BSPSchedule:
        """Gradually uncoarsen the schedule with refinement steps."""
        # Build node -> (processor, superstep) mapping from coarse schedule
        node_assignments: Dict[str, Tuple[str, int]] = {}
        for task_name in coarse_schedule.task_graph.nodes():
            instances = coarse_schedule.get_all_instances(task_name)
            if instances:
                task = instances[0]
                node_assignments[task_name] = (task.proc, task.superstep.index)

        current_graph = coarse_schedule.task_graph.copy()
        steps_since_refinement = 0

        # Process contractions in reverse order
        while coarsener.num_contractions > 0:
            # Uncoarsen a batch of steps
            batch_size = min(self.hc_interval, coarsener.num_contractions)
            current_graph, records = coarsener.uncoarsen_n_steps(
                current_graph, original_graph, n=batch_size
            )

            # Extend node_assignments for newly uncontracted nodes
            for record in records:
                # The contracted node inherits the assignment of the absorbing node
                if record.absorbing_node in node_assignments:
                    node_assignments[record.contracted_node] = node_assignments[record.absorbing_node]

            steps_since_refinement += batch_size

            # Apply refinement after every hc_interval steps
            if steps_since_refinement >= self.hc_interval:
                # Build schedule for current graph
                current_schedule = self._build_schedule_from_assignments(
                    hardware, current_graph, node_assignments
                )

                # Apply HC refinement
                hc = HillClimbing(max_iterations=self.hc_max_steps, verbose=False)
                current_schedule = hc.optimize(current_schedule, time_limit=60.0)

                # Update node_assignments from refined schedule
                node_assignments = {}
                for task_name in current_graph.nodes():
                    instances = current_schedule.get_all_instances(task_name)
                    if instances:
                        task = instances[0]
                        node_assignments[task_name] = (task.proc, task.superstep.index)

                steps_since_refinement = 0

                if self.verbose:
                    print(f"    Refined at {len(current_graph.nodes())} nodes, "
                          f"cost: {current_schedule.makespan:.2f}")

        # Build final schedule on original graph
        final_schedule = self._build_schedule_from_assignments(
            hardware, original_graph, node_assignments
        )

        # TODO: Bring back in!
        # # Final optimization
        # hccs = HCcs(verbose=self.verbose)
        # final_schedule = hccs.optimize(final_schedule)
        # if self.use_ilp:
        #     ilpcs = ILPcs(verbose=self.verbose)
        #     final_schedule = ilpcs.optimize(final_schedule)

        # Merge supersteps
        final_schedule.merge_supersteps()

        return final_schedule

    def _build_schedule_from_assignments(
        self,
        hardware: BSPHardware,
        task_graph: nx.DiGraph,
        node_assignments: Dict[str, Tuple[str, int]]
    ) -> BSPSchedule:
        """Build a BSP schedule from node->(processor, superstep) assignments.

        Handles cases where the assignment might be incomplete or need adjustment.
        """
        schedule = BSPSchedule(hardware, task_graph)

        # Find the maximum superstep index needed
        max_ss_idx = 0
        for node in task_graph.nodes():
            if node in node_assignments:
                _, ss_idx = node_assignments[node]
                max_ss_idx = max(max_ss_idx, ss_idx)
            else:
                # Unassigned nodes - will need to determine their superstep
                pass

        # Create supersteps
        for _ in range(max_ss_idx + 1):
            schedule.add_superstep()

        # Assign nodes to processors and supersteps
        # Process in topological order to handle dependencies
        processors = list(hardware.network.nodes())

        for node in nx.topological_sort(task_graph):
            if node in node_assignments:
                proc, ss_idx = node_assignments[node]

                # Validate: ensure dependencies are satisfied
                valid_ss_idx = self._find_valid_superstep(
                    node, proc, ss_idx, schedule, task_graph
                )

                # Ensure enough supersteps exist
                while len(schedule.supersteps) <= valid_ss_idx:
                    schedule.add_superstep()

                superstep = schedule.supersteps[valid_ss_idx]
                schedule.schedule(node, proc, superstep)
            else:
                # Node was not in the coarsened graph - assign it
                # Find earliest valid superstep based on predecessors
                proc, ss_idx = self._find_best_assignment(
                    node, schedule, task_graph, processors
                )

                while len(schedule.supersteps) <= ss_idx:
                    schedule.add_superstep()

                superstep = schedule.supersteps[ss_idx]
                schedule.schedule(node, proc, superstep)

        # Merge supersteps
        schedule.merge_supersteps()

        return schedule

    def _find_valid_superstep(
        self,
        node: str,
        proc: str,
        target_ss_idx: int,
        schedule: BSPSchedule,
        task_graph: nx.DiGraph
    ) -> int:
        """Find a valid superstep index for a node, respecting dependencies.

        Returns the target index if valid, or a later index if needed.
        """
        min_valid_ss = 0

        for pred in task_graph.predecessors(node):
            pred_instances = schedule.get_all_instances(pred)
            if pred_instances:
                pred_task = pred_instances[0]
                pred_ss_idx = pred_task.superstep.index
                pred_proc = pred_task.proc

                if pred_proc == proc:
                    # Same processor - can be in same superstep
                    min_valid_ss = max(min_valid_ss, pred_ss_idx)
                else:
                    # Different processor - must be in later superstep
                    min_valid_ss = max(min_valid_ss, pred_ss_idx + 1)

        return max(target_ss_idx, min_valid_ss)

    def _find_best_assignment(
        self,
        node: str,
        schedule: BSPSchedule,
        task_graph: nx.DiGraph,
        processors: List[str]
    ) -> Tuple[str, int]:
        """Find the best (processor, superstep) for an unassigned node."""
        # Find the earliest valid superstep and best processor
        min_ss_idx = 0
        best_proc = processors[0]
        best_comm_cost = float('inf')

        for pred in task_graph.predecessors(node):
            pred_instances = schedule.get_all_instances(pred)
            if pred_instances:
                pred_task = pred_instances[0]
                pred_ss_idx = pred_task.superstep.index
                pred_proc = pred_task.proc

                # Update minimum superstep
                min_ss_idx = max(min_ss_idx, pred_ss_idx + 1)

                # Prefer processor with most predecessors (reduces communication)
                # This is a simplified heuristic

        # Find processor with most predecessors on it
        proc_counts: Dict[str, int] = {p: 0 for p in processors}
        for pred in task_graph.predecessors(node):
            pred_instances = schedule.get_all_instances(pred)
            if pred_instances:
                pred_proc = pred_instances[0].proc
                proc_counts[pred_proc] = proc_counts.get(pred_proc, 0) + 1

        if any(proc_counts.values()):
            best_proc = max(proc_counts.keys(), key=lambda p: proc_counts[p])

        # If no predecessors, use round-robin based on current load
        if min_ss_idx == 0 and not list(task_graph.predecessors(node)):
            # Source node - distribute evenly
            load = {p: 0 for p in processors}
            for ss in schedule.supersteps:
                for p, tasks in ss.tasks.items():
                    load[p] = load.get(p, 0) + len(tasks)
            best_proc = min(processors, key=lambda p: load.get(p, 0))

        return best_proc, min_ss_idx
