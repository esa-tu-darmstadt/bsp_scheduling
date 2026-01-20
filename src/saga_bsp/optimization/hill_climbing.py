"""Hill Climbing Optimization - Section 4.3 from Papp et al. 2024

This implements:
- HC: Hill climbing for schedule optimization (node moves)
- HCcs: Hill climbing for communication scheduling optimization

Reference: Section 4.3, Appendix A.3 (pages 15-16)
"""

from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Tuple
import time
import networkx as nx

from ..schedule import BSPSchedule, BSPHardware, BSPTask, Superstep


@dataclass
class MoveCandidate:
    """Represents a potential node move."""
    node: str
    from_proc: str
    from_superstep_idx: int
    to_proc: str
    to_superstep_idx: int
    cost_delta: float = 0.0


class HillClimbing:
    """Hill climbing optimizer for BSP schedules.

    Algorithm:
    1. Start from initial schedule
    2. For each node v at (processor p, superstep s), consider moves to:
       - Any processor p' ≠ p in same superstep s
       - Any processor in supersteps (s-1) or (s+1)
    3. Accept move if it decreases total cost
    4. Repeat until local minimum or max iterations

    Uses lazy communication schedule where values are sent in the last
    possible superstep before they are needed.
    """

    def __init__(self, max_iterations: int = 1000, verbose: bool = False):
        """
        Args:
            max_iterations: Maximum number of improvement iterations
            verbose: Print progress information
        """
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.stats = {
            'iterations': 0,
            'improvements': 0,
            'initial_cost': 0.0,
            'final_cost': 0.0,
            'time_elapsed': 0.0
        }

    def optimize(self, schedule: BSPSchedule, time_limit: float = 300.0) -> BSPSchedule:
        """Optimize a BSP schedule using hill climbing.

        Args:
            schedule: Initial BSP schedule to optimize
            time_limit: Maximum time in seconds (default 5 minutes)

        Returns:
            Optimized BSP schedule
        """
        start_time = time.time()
        self.stats['initial_cost'] = schedule.makespan

        current_cost = schedule.makespan
        iteration = 0
        improvements = 0

        processors = list(schedule.hardware.network.nodes())

        while iteration < self.max_iterations:
            if time.time() - start_time > time_limit:
                if self.verbose:
                    print(f"HC: Time limit reached after {iteration} iterations")
                break

            # Greedy approach: apply the FIRST improving move found
            # Per Papp et al. 2024, Section A.3: "we have applied the former,
            # greedy variant of the approach in our experiments"
            found_improvement = False

            # Iterate through all tasks
            for task_name in schedule.task_graph.nodes():
                if found_improvement:
                    break

                instances = schedule.get_all_instances(task_name)
                if not instances:
                    continue

                # Consider only the first instance (no duplication assumed)
                task = instances[0]
                current_proc = task.proc
                current_ss_idx = task.superstep.index

                # Generate candidate moves
                candidates = self._generate_move_candidates(
                    task_name, current_proc, current_ss_idx, schedule, processors
                )

                for candidate in candidates:
                    # Check if move is valid
                    if not self._is_valid_move(candidate, schedule):
                        continue

                    # Evaluate cost impact
                    improvement = self._evaluate_move(candidate, schedule, current_cost)

                    if improvement > 0:
                        # Greedy: apply first improving move immediately
                        self._apply_move(candidate, schedule)
                        current_cost -= improvement
                        improvements += 1
                        found_improvement = True

                        if self.verbose:
                            print(f"HC iteration {iteration}: cost {current_cost:.2f} "
                                  f"(improved by {improvement:.2f})")
                        break

            if not found_improvement:
                # Local minimum reached
                if self.verbose:
                    print(f"HC: Local minimum reached after {iteration} iterations")
                break

            iteration += 1

        self.stats['iterations'] = iteration
        self.stats['improvements'] = improvements
        self.stats['final_cost'] = schedule.makespan
        self.stats['time_elapsed'] = time.time() - start_time

        return schedule

    def _generate_move_candidates(
        self,
        task_name: str,
        current_proc: str,
        current_ss_idx: int,
        schedule: BSPSchedule,
        processors: List[str]
    ) -> List[MoveCandidate]:
        """Generate all candidate moves for a task.

        Moves considered:
        - Same superstep, different processor
        - Previous superstep (s-1), any processor
        - Next superstep (s+1), any processor
        """
        candidates = []

        # Same superstep, different processor
        for proc in processors:
            if proc != current_proc:
                candidates.append(MoveCandidate(
                    node=task_name,
                    from_proc=current_proc,
                    from_superstep_idx=current_ss_idx,
                    to_proc=proc,
                    to_superstep_idx=current_ss_idx
                ))

        # Previous superstep (if exists)
        if current_ss_idx > 0:
            for proc in processors:
                candidates.append(MoveCandidate(
                    node=task_name,
                    from_proc=current_proc,
                    from_superstep_idx=current_ss_idx,
                    to_proc=proc,
                    to_superstep_idx=current_ss_idx - 1
                ))

        # Next superstep (if exists or can be created)
        if current_ss_idx < len(schedule.supersteps) - 1:
            for proc in processors:
                candidates.append(MoveCandidate(
                    node=task_name,
                    from_proc=current_proc,
                    from_superstep_idx=current_ss_idx,
                    to_proc=proc,
                    to_superstep_idx=current_ss_idx + 1
                ))

        return candidates

    def _is_valid_move(self, move: MoveCandidate, schedule: BSPSchedule) -> bool:
        """Check if a move is valid (preserves dependencies).

        A move is valid if after the move:
        1. All predecessors are either:
           - In an earlier superstep, OR
           - In the same superstep on the same processor
        2. All successors are either:
           - In a later superstep, OR
           - In the same superstep on the same processor
        """
        task_graph = schedule.task_graph
        target_ss_idx = move.to_superstep_idx
        target_proc = move.to_proc

        # Check if target superstep exists
        if target_ss_idx >= len(schedule.supersteps):
            return False

        target_superstep = schedule.supersteps[target_ss_idx]

        # Check predecessors
        for pred_name in task_graph.predecessors(move.node):
            pred_instances = schedule.get_all_instances(pred_name)
            if not pred_instances:
                return False

            valid = False
            for pred in pred_instances:
                pred_ss_idx = pred.superstep.index
                pred_proc = pred.proc

                if pred_ss_idx < target_ss_idx:
                    valid = True
                    break
                elif pred_ss_idx == target_ss_idx and pred_proc == target_proc:
                    # Same superstep, same processor - check ordering
                    # Predecessor must come before in task list
                    # (For simplicity, we assume append to end, so this is invalid
                    # unless predecessor is already scheduled earlier)
                    valid = True
                    break

            if not valid:
                return False

        # Check successors
        for succ_name in task_graph.successors(move.node):
            succ_instances = schedule.get_all_instances(succ_name)
            if not succ_instances:
                # Successor not yet scheduled - OK
                continue

            valid = False
            for succ in succ_instances:
                succ_ss_idx = succ.superstep.index
                succ_proc = succ.proc

                if succ_ss_idx > target_ss_idx:
                    valid = True
                    break
                elif succ_ss_idx == target_ss_idx and succ_proc == target_proc:
                    valid = True
                    break

            if not valid:
                return False

        return True

    def _evaluate_move(
        self,
        move: MoveCandidate,
        schedule: BSPSchedule,
        current_cost: float
    ) -> float:
        """Evaluate the cost improvement of a move.

        Returns positive value if move improves (reduces) cost.
        """
        # Create copy and apply move
        test_schedule = schedule.copy()
        self._apply_move(move, test_schedule)

        new_cost = test_schedule.makespan
        improvement = current_cost - new_cost

        return improvement

    def _apply_move(self, move: MoveCandidate, schedule: BSPSchedule) -> None:
        """Apply a move to the schedule."""
        # Find and remove the task from current location
        instances = schedule.get_all_instances(move.node)
        if not instances:
            return

        task = instances[0]
        schedule.unschedule(task)

        # Schedule in new location
        target_superstep = schedule.supersteps[move.to_superstep_idx]
        schedule.schedule(move.node, move.to_proc, target_superstep)


class HCcs:
    """Hill climbing optimizer for communication scheduling.

    Optimizes only the communication schedule Γ while keeping
    processor assignment π and superstep assignment τ fixed.

    For each communication (v, p1, p2, s), tries different s' in range [τ(v), s0-1]
    where τ(v) is when v is computed and s0 is when v is first needed on p2.
    """

    def __init__(self, max_iterations: int = 1000, verbose: bool = False):
        """
        Args:
            max_iterations: Maximum number of improvement iterations
            verbose: Print progress information
        """
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.stats = {
            'iterations': 0,
            'improvements': 0,
            'initial_cost': 0.0,
            'final_cost': 0.0,
            'time_elapsed': 0.0
        }

    def optimize(self, schedule: BSPSchedule, time_limit: float = 30.0) -> BSPSchedule:
        """Optimize communication scheduling using hill climbing.

        Args:
            schedule: BSP schedule to optimize
            time_limit: Maximum time in seconds

        Returns:
            Optimized BSP schedule

        Raises:
            NotImplementedError: BSPSchedule does not support explicit communication scheduling
        """
        raise NotImplementedError(
            "HCcs requires explicit communication scheduling support in BSPSchedule.\n"
            "\n"
            "The paper's HCcs algorithm (Section 4.3, Appendix A.3) optimizes the communication "
            "schedule Γ while keeping processor assignment π and superstep assignment τ fixed. "
            "For each communication (v, p₁, p₂, s), it tries moving to different supersteps "
            "s' ∈ [τ(v), s₀-1] where τ(v) is when v is computed and s₀ is when v is first needed.\n"
            "\n"
            "However, BSPSchedule uses implicit/lazy communication scheduling where data is "
            "automatically sent in the last possible superstep before needed. To implement HCcs, "
            "BSPSchedule would need to be extended to:\n"
            "  1. Track explicit communication schedule Γ = set of (node, src_proc, dst_proc, superstep)\n"
            "  2. Allow communications to be scheduled in any valid superstep range\n"
            "  3. Recompute exchange times based on explicit Γ rather than implicit dependencies"
        )


def optimize_with_hill_climbing(
    schedule: BSPSchedule,
    hc_time_limit: float = 270.0,  # 90% of 5 minutes
    hccs_time_limit: float = 30.0,  # 10% of 5 minutes
    hc_max_iterations: int = 1000,
    verbose: bool = False
) -> BSPSchedule:
    """Apply HC followed by HCcs optimization.

    This is the combined optimization as used in the Papp et al. pipeline.

    Args:
        schedule: Initial BSP schedule
        hc_time_limit: Time limit for HC (default 4.5 minutes)
        hccs_time_limit: Time limit for HCcs (default 30 seconds)
        hc_max_iterations: Max iterations for HC
        verbose: Print progress

    Returns:
        Optimized BSP schedule
    """
    # Apply HC
    hc = HillClimbing(max_iterations=hc_max_iterations, verbose=verbose)
    schedule = hc.optimize(schedule, time_limit=hc_time_limit)

    if verbose:
        print(f"HC: {hc.stats['improvements']} improvements, "
              f"cost {hc.stats['initial_cost']:.2f} -> {hc.stats['final_cost']:.2f}")

    # Apply HCcs
    hccs = HCcs(verbose=verbose)
    schedule = hccs.optimize(schedule, time_limit=hccs_time_limit)

    if verbose:
        print(f"HCcs: cost {hccs.stats['initial_cost']:.2f} -> {hccs.stats['final_cost']:.2f}")

    return schedule
