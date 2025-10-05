"""Fill-in/Append BSP Scheduler with hole-filling and earliest-finish placement.

This scheduler supports three priority modes:
- HEFT mode: Uses upward rank for task prioritization (default)
- CPOP mode: Uses upward + downward rank for task prioritization
- DS mode: Uses dominantsequence (static upward + dynamic downward rank)

Implements a two-phase placement strategy in strict priority order:
1. Fill-in: Try to place tasks in holes of existing supersteps
2. Earliest-finish: If no hole available, evaluate two strategies and pick earliest:
   a. Append to last superstep on each processor
   b. Create new superstep at end and place on best processor
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List
import logging
from queue import PriorityQueue
from .base import BSPScheduler
from .. import draw_bsp_gantt
from ..schedule import BSPSchedule, BSPHardware, Superstep, BSPTask
import networkx as nx
from .delaymodel.priorities import upward_rank, cpop_ranks, calculate_dynamic_downward_rank

logger = logging.getLogger(__name__)


@dataclass(order=True)
class PrioritizedTask:
    """Task with priority for queue ordering."""
    priority: float
    task: str

    def __post_init__(self):
        # Make priority negative so higher ranks are processed first
        self.priority = -self.priority


@dataclass
class PlacementCandidate:
    """Represents a candidate placement for a task."""
    finish_time: float
    strategy: str  # "fill-in", "append", or "new"
    processor: str
    superstep: Superstep


class FillInAppendBSPScheduler(BSPScheduler):
    """Fill-in/Append BSP scheduler with hole-filling and earliest-finish placement.

    This scheduler:
    1. Computes task priority using HEFT (upward rank) or CPOP (upward + downward rank)
    2. Processes tasks in priority order
    3. For each task, tries strategies in order:
       - Phase 1: Fill-in (find holes in existing supersteps)
       - Phase 2: Earliest-finish (evaluate append vs new superstep, pick earliest)
    """

    def __init__(self, verbose: bool = False, draw_after_each_step: bool = False,
                 priority_mode: str = "heft", optimize_merging: bool = False):
        """Initialize the Fill-in/Append BSP scheduler.

        Args:
            verbose: Enable detailed output to console
            draw_after_each_step: Enable drawing Gantt chart after each scheduling step
            priority_mode: Priority calculation mode - "heft" (upward rank only), "cpop" (upward + downward rank),
                          or "ds" (dominantsequence: static upward + dynamic downward rank)
            optimize_merging: Enable second phase optimization that iteratively merges adjacent supersteps
                            for maximum makespan reduction
        """
        super().__init__()
        self.priority_mode = priority_mode.lower()
        self.name = "FillInAppend+" + self.priority_mode.capitalize()
        self.verbose = verbose
        self.draw_after_each_step = draw_after_each_step
        self.optimize_merging = optimize_merging

        if self.priority_mode not in ["heft", "cpop", "ds"]:
            raise ValueError(f"Invalid priority_mode: {priority_mode}. Must be 'heft', 'cpop', or 'ds'")

        # Initialize statistics
        self._reset_stats()

    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"  {message}")

    def _reset_stats(self):
        """Reset scheduler statistics."""
        self.stats = {
            'tasks_scheduled': 0,
            'fill_in_placements': 0,
            'append_placements': 0,
            'new_superstep_placements': 0,
            'supersteps_created': 0,
            'supersteps_merged': 0,
            'optimization_eliminations': 0,
            'optimization_iterations': 0
        }

    def schedule(self, hardware: BSPHardware, task_graph: nx.DiGraph) -> BSPSchedule:
        """Schedule tasks using fill-in and earliest-finish placement strategy.

        Args:
            hardware: BSP hardware configuration
            task_graph: Task dependency graph

        Returns:
            Optimized BSP schedule
        """
        self._reset_stats()

        # Initialize schedule
        schedule = BSPSchedule(hardware, task_graph)

        # Compute task priorities based on selected mode
        if self.priority_mode == "heft":
            rank = upward_rank(hardware.network, task_graph)
        elif self.priority_mode == "cpop":
            rank = cpop_ranks(hardware.network, task_graph)
        elif self.priority_mode == "ds":
            # For dominantsequence, we need static upward ranks + dynamic recalculation
            static_upward_ranks = upward_rank(hardware.network, task_graph)
            rank = {}  # Will be recalculated dynamically

        # Initialize priority queue and ready tasks set
        queue = PriorityQueue()
        ready_tasks = set()

        # Add tasks with no dependencies to queue
        for task in task_graph.nodes:
            if task_graph.in_degree(task) == 0:
                if self.priority_mode == "ds":
                    # For DS mode, calculate initial priority
                    dynamic_downward = calculate_dynamic_downward_rank(task, task_graph, schedule)
                    task_priority = static_upward_ranks[task] + dynamic_downward
                    rank[task] = task_priority
                else:
                    task_priority = rank[task]

                queue.put(PrioritizedTask(task_priority, task))
                ready_tasks.add(task)

                if self.verbose:
                    self._log(f"Task {task} added to initial queue with {self.priority_mode} rank {task_priority}")

        # Create initial superstep if needed
        if not schedule.supersteps:
            schedule.add_superstep()
            self.stats['supersteps_created'] = 1

        # Process tasks in priority order
        task_num = 0
        while not queue.empty():
            task_item = queue.get()
            task_name = task_item.task

            if self.verbose:
                self._log(f"\n#{task_num}: Processing task {task_name} with {self.priority_mode} rank {rank[task_name]}")

            # Try placement strategies in priority order
            placed = False

            # Phase 1: Try fill-in strategy
            candidate = self._try_fill_in(task_name, schedule, hardware, task_graph)
            if candidate is not None:
                schedule.schedule(task_name, candidate.processor, candidate.superstep)
                self.stats['fill_in_placements'] += 1
                placed = True
                if self.verbose:
                    self._log(f"  Placed using fill-in strategy on processor {candidate.processor}, "
                               f"superstep {candidate.superstep.index}, finish={candidate.finish_time:.2f}")

            # Phase 2: Evaluate append vs new superstep, pick earliest finish
            if not placed:
                candidate = self._find_earliest_finish_placement(task_name, schedule, hardware, task_graph)

                # If the candidate requires creating a new superstep, do it now
                if candidate.superstep is None:
                    candidate.superstep = schedule.add_superstep()
                    self.stats['supersteps_created'] += 1

                schedule.schedule(task_name, candidate.processor, candidate.superstep)

                if candidate.strategy == "append":
                    self.stats['append_placements'] += 1
                    if self.verbose:
                        self._log(f"  Placed using append strategy on processor {candidate.processor}, "
                                   f"superstep {candidate.superstep.index}, finish={candidate.finish_time:.2f}")
                else:  # "new"
                    self.stats['new_superstep_placements'] += 1
                    if self.verbose:
                        self._log(f"  Placed using new superstep strategy on processor {candidate.processor}, "
                                   f"superstep {candidate.superstep.index}, finish={candidate.finish_time:.2f}")

                placed = True

            # Try to merge with previous superstep if beneficial
            if candidate.superstep.index > 0 and schedule.merge_supersteps(candidate.superstep.index - 1) > 0:
                self.stats['supersteps_merged'] += 1
                if self.verbose:
                    self._log(f"  Merged supersteps {candidate.superstep.index-1} and {candidate.superstep.index} after placement")

            self.stats['tasks_scheduled'] += 1

            # Add newly ready successors to queue
            for successor in task_graph.successors(task_name):
                if all(schedule.task_scheduled(pred) for pred in task_graph.predecessors(successor)):
                    if successor not in ready_tasks:
                        ready_tasks.add(successor)

            # For DS mode, recalculate priorities for all ready tasks after placing a task
            if self.priority_mode == "ds":
                # Clear current queue and recalculate priorities
                new_queue = PriorityQueue()
                for ready_task in ready_tasks:
                    if not schedule.task_scheduled(ready_task):
                        dynamic_downward = calculate_dynamic_downward_rank(ready_task, task_graph, schedule)
                        task_priority = static_upward_ranks[ready_task] + dynamic_downward
                        rank[ready_task] = task_priority
                        new_queue.put(PrioritizedTask(task_priority, ready_task))

                        if self.verbose:
                            self._log(f"  Task {ready_task} priority recalculated: DS rank {task_priority}")
                queue = new_queue
            else:
                # For HEFT and CPOP modes, just add new ready tasks
                for successor in task_graph.successors(task_name):
                    if (all(schedule.task_scheduled(pred) for pred in task_graph.predecessors(successor)) and
                        not schedule.task_scheduled(successor)):
                        queue.put(PrioritizedTask(rank[successor], successor))
                        if self.verbose:
                            mode_str = {"heft": "HEFT", "cpop": "CPOP"}[self.priority_mode]
                            self._log(f"  Task {successor} became ready with {mode_str} rank {rank[successor]}")

            # Optionally draw Gantt chart after each step
            if self.draw_after_each_step:
                draw_bsp_gantt(schedule, title=f"After #{task_num}: Scheduling {task_name}")
            task_num += 1

        # Merge supersteps where possible
        merges = schedule.merge_supersteps()
        self.stats['supersteps_merged'] += merges
        if self.verbose and merges > 0:
            self._log(f"\nMerged {merges} superstep(s) to reduce synchronization overhead")

        # Second phase: Optimize by iteratively eliminating supersteps for maximum makespan reduction
        if self.optimize_merging:
            from ..optimization import optimize_superstep_elimination
            initial_supersteps = len(schedule.supersteps)
            schedule = optimize_superstep_elimination(schedule, verbose=self.verbose)
            optimization_eliminations = initial_supersteps - len(schedule.supersteps)
            self.stats['optimization_eliminations'] = optimization_eliminations
            if self.verbose and optimization_eliminations > 0:
                self._log(f"Optimization phase: eliminated {optimization_eliminations} superstep(s) to reduce makespan")

        # Validate the final schedule
        schedule.assert_valid()
        return schedule

    def _calculate_task_duration(self, task_name: str, processor: str,
                                hardware: BSPHardware, task_graph: nx.DiGraph) -> float:
        """Calculate computation duration for a task on a processor.

        Args:
            task_name: Task to calculate duration for
            processor: Target processor
            hardware: Hardware configuration
            task_graph: Task dependency graph

        Returns:
            Task duration (computation time only)
        """
        task_weight = task_graph.nodes[task_name]['weight']
        processor_speed = hardware.network.nodes[processor]['weight']
        return task_weight / processor_speed

    def _calculate_communication_time(self, task_name: str, processor: str,
                                     superstep: Superstep, schedule: BSPSchedule,
                                     hardware: BSPHardware, task_graph: nx.DiGraph) -> float:
        """Calculate communication time for predecessors not on same processor.

        Since we don't use task duplication, each predecessor has exactly one instance.
        Communication is needed if the predecessor is on a different processor and in
        a previous superstep.

        Args:
            task_name: Task to calculate communication for
            processor: Target processor
            superstep: Target superstep
            schedule: Current schedule
            hardware: Hardware configuration
            task_graph: Task dependency graph

        Returns:
            Total communication time needed
        """
        comm_time = 0.0

        for pred_name in task_graph.predecessors(task_name):
            if not schedule.task_scheduled(pred_name):
                continue  # Predecessor not scheduled yet

            # Get the single instance of the predecessor
            pred_instance = schedule.get_single_instance(pred_name)

            # Check if communication is needed
            # No communication needed if:
            # 1. Predecessor is on same processor (regardless of superstep)
            if pred_instance.proc == processor:
                # Same processor - no communication needed
                continue

            # Different processor - need communication if predecessor is in earlier superstep
            if pred_instance.superstep.index < superstep.index:
                # Calculate communication time
                edge_weight = task_graph.edges[(pred_name, task_name)]['weight']

                if hardware.network.has_edge(pred_instance.proc, processor):
                    network_speed = hardware.network.edges[pred_instance.proc, processor]['weight']
                    comm_time = comm_time + edge_weight / network_speed
                else:
                    # No network connection available
                    return float('inf')

        return comm_time

    def _try_fill_in(self, task_name: str, schedule: BSPSchedule,
                    hardware: BSPHardware, task_graph: nx.DiGraph) -> Optional[PlacementCandidate]:
        """Try to place task in a hole of an existing superstep.

        Args:
            task_name: Task to place
            schedule: Current schedule
            hardware: Hardware configuration
            task_graph: Task dependency graph

        Returns:
            PlacementCandidate if successful, None otherwise
        """
        best_candidate = None
        best_finish_time = float('inf')

        # Check all supersteps for holes
        for superstep in schedule.supersteps:
            for processor in hardware.network.nodes:
                # Check if dependencies allow scheduling here
                if not schedule.can_be_scheduled_in(task_name, superstep, processor):
                    continue

                # Calculate task duration
                task_duration = self._calculate_task_duration(task_name, processor, hardware, task_graph)

                # Current end time of processor in this superstep
                proc_end_time = superstep.compute_phase_start(processor) + superstep.compute_time(processor)

                # Would the task fit in a hole?
                task_finish_time = proc_end_time + task_duration
                superstep_end_time = superstep.end_time

                # Check if there's a hole (task fits without extending superstep)
                if task_finish_time <= superstep_end_time * 1.0001:  # Small tolerance
                    if task_finish_time < best_finish_time:
                        best_finish_time = task_finish_time
                        best_candidate = PlacementCandidate(
                            finish_time=task_finish_time,
                            strategy="fill-in",
                            processor=processor,
                            superstep=superstep
                        )

                        if self.verbose:
                            hole_size = superstep_end_time - proc_end_time
                            self._log(f"    Found fill-in opportunity: superstep {superstep.index}, "
                                       f"proc {processor}, hole_size={hole_size:.2f}, finish={task_finish_time:.2f}")

        return best_candidate

    def _find_earliest_finish_placement(self, task_name: str, schedule: BSPSchedule,
                                       hardware: BSPHardware, task_graph: nx.DiGraph) -> PlacementCandidate:
        """Find placement with earliest finish time by evaluating append and new superstep strategies.

        Args:
            task_name: Task to place
            schedule: Current schedule
            hardware: Hardware configuration
            task_graph: Task dependency graph

        Returns:
            PlacementCandidate with earliest finish time
        """
        candidates: List[PlacementCandidate] = []

        # Strategy A: Append to last superstep on each processor
        candidates.extend(self._evaluate_append_strategy(task_name, schedule, hardware, task_graph))

        # Strategy B: Create new superstep at end
        candidates.extend(self._evaluate_new_superstep_strategy(task_name, schedule, hardware, task_graph))

        # Select candidate with earliest finish time
        if not candidates:
            raise ValueError(f"No valid placement found for task {task_name}")

        best_candidate = min(candidates, key=lambda c: c.finish_time)
        return best_candidate

    def _evaluate_append_strategy(self, task_name: str, schedule: BSPSchedule,
                                  hardware: BSPHardware, task_graph: nx.DiGraph) -> List[PlacementCandidate]:
        """Evaluate appending task to last superstep on each processor.

        Args:
            task_name: Task to place
            schedule: Current schedule
            hardware: Hardware configuration
            task_graph: Task dependency graph

        Returns:
            List of PlacementCandidates for append strategy
        """
        candidates = []

        # For each processor, find the last superstep that contains tasks on that processor
        for processor in hardware.network.nodes:
            last_superstep = None

            # Find last superstep with tasks on this processor
            for superstep in reversed(schedule.supersteps):
                if processor in superstep.tasks and len(superstep.tasks[processor]) > 0:
                    last_superstep = superstep
                    break

            # If processor has no tasks yet, use the first superstep
            if last_superstep is None:
                last_superstep = schedule.supersteps[0]

            # Check if we can schedule in this superstep
            if not schedule.can_be_scheduled_in(task_name, last_superstep, processor):
                continue

            # Calculate task duration including communication
            task_duration = self._calculate_task_duration(task_name, processor, hardware, task_graph)
            comm_time = self._calculate_communication_time(task_name, processor, last_superstep,
                                                          schedule, hardware, task_graph)

            if comm_time == float('inf'):
                continue  # No network connection

            # Calculate finish time if appended
            proc_compute_end = last_superstep.compute_phase_start(processor) + last_superstep.compute_time(processor)
            task_finish_time = proc_compute_end + task_duration

            candidates.append(PlacementCandidate(
                finish_time=task_finish_time,
                strategy="append",
                processor=processor,
                superstep=last_superstep
            ))

            if self.verbose:
                self._log(f"    Append strategy: proc {processor}, superstep {last_superstep.index}, "
                           f"finish={task_finish_time:.2f}")

        return candidates

    def _evaluate_new_superstep_strategy(self, task_name: str, schedule: BSPSchedule,
                                        hardware: BSPHardware, task_graph: nx.DiGraph) -> List[PlacementCandidate]:
        """Evaluate creating new superstep at end and placing on each processor.

        Args:
            task_name: Task to place
            schedule: Current schedule
            hardware: Hardware configuration
            task_graph: Task dependency graph

        Returns:
            List of PlacementCandidates for new superstep strategy
        """
        candidates = []

        # Calculate what the new superstep's timing would be
        # (without actually creating it yet)
        new_superstep_start = schedule.makespan if schedule.supersteps else 0.0

        # Evaluate each processor in the hypothetical new superstep
        for processor in hardware.network.nodes:
            # Check if dependencies are satisfied (all must be scheduled already)
            can_schedule = True
            for pred_name in task_graph.predecessors(task_name):
                if not schedule.task_scheduled(pred_name):
                    can_schedule = False
                    break

            if not can_schedule:
                continue

            # Calculate task duration
            task_duration = self._calculate_task_duration(task_name, processor, hardware, task_graph)

            # Calculate communication time
            comm_time = 0.0
            for pred_name in task_graph.predecessors(task_name):
                pred_instance = schedule.get_single_instance(pred_name)
                if pred_instance.proc != processor:
                    edge_weight = task_graph.edges[(pred_name, task_name)]['weight']
                    if hardware.network.has_edge(pred_instance.proc, processor):
                        network_speed = hardware.network.edges[pred_instance.proc, processor]['weight']
                        comm_time += edge_weight / network_speed
                    else:
                        comm_time = float('inf')
                        break

            if comm_time == float('inf'):
                continue  # No network connection

            # Calculate finish time in new superstep
            # New superstep starts with sync + exchange + computation
            task_finish_time = (new_superstep_start +
                              hardware.sync_time +
                              comm_time +
                              task_duration)

            # Create a placeholder that we'll replace with actual superstep if chosen
            # We use a marker object to indicate this needs to be created
            candidates.append(PlacementCandidate(
                finish_time=task_finish_time,
                strategy="new",
                processor=processor,
                superstep=None  # Will create when this candidate is chosen
            ))

            if self.verbose:
                self._log(f"    New superstep strategy: proc {processor}, "
                           f"finish={task_finish_time:.2f}")

        return candidates

    def print_stats(self):
        """Print scheduling statistics."""
        print("\n" + "="*50)
        mode_str = {"heft": "HEFT", "cpop": "CPOP", "ds": "DominantSequence"}[self.priority_mode]
        print(f"Fill-in/Append BSP Scheduler Statistics ({mode_str} priority mode)")
        print("="*50)
        print(f"Tasks scheduled:         {self.stats['tasks_scheduled']}")
        print(f"Placement strategies:")
        print(f"  - Fill-in:            {self.stats['fill_in_placements']}")
        print(f"  - Append:             {self.stats['append_placements']}")
        print(f"  - New superstep:      {self.stats['new_superstep_placements']}")
        print(f"Superstep operations:")
        print(f"  - Created:            {self.stats['supersteps_created']}")
        print(f"  - Merged:             {self.stats['supersteps_merged']}")
        if self.optimize_merging:
            print(f"Optimization phase:")
            print(f"  - Eliminations:       {self.stats['optimization_eliminations']}")
        print("="*50)