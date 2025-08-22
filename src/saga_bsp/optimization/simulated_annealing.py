import logging
import math
import random
import networkx as nx
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type
from ..schedule import BSPSchedule, BSPTask


logger = logging.getLogger(__name__)


@dataclass
class SimulatedAnnealingIteration:
    """Data class for simulated annealing iteration results"""
    iteration: int
    temperature: float
    current_energy: float
    neighbor_energy: float
    best_energy: float
    accept_probability: float
    accepted: bool
    action: 'ScheduleAction'
    action_applied: bool


class ScheduleAction(ABC):
    """Abstract base class for schedule modification actions"""
    
    @abstractmethod
    def is_feasible(self, schedule: BSPSchedule, task_instance: BSPTask) -> bool:
        """Check if this action is feasible for the given task instance in the schedule"""
        pass
    
    @abstractmethod
    def apply(self, schedule: BSPSchedule, task_instance: BSPTask) -> bool:
        """Apply the action to the schedule. Returns True if successful"""
        pass
    
    @abstractmethod
    def get_possible_targets(self, task_instance: BSPTask) -> List:
        """Get list of possible targets for this action given a task instance"""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """String representation of the action"""
        pass


class MoveTaskToSuperstep(ScheduleAction):
    """Action to move a task instance to a different superstep"""
    
    def __init__(self, target_superstep_idx: int):
        self.target_superstep_idx = target_superstep_idx
    
    def is_feasible(self, schedule: BSPSchedule, task_instance: BSPTask) -> bool:
        """Check if moving task instance to target superstep maintains precedence constraints"""
        if self.target_superstep_idx < 0 or self.target_superstep_idx >= len(schedule.supersteps):
            return False
            
        if task_instance.superstep.index == self.target_superstep_idx:
            return False  # Already in target superstep
        
        task_node = task_instance.node
        target_processor = task_instance.proc
        target_superstep = schedule.supersteps[self.target_superstep_idx]
        
        # Check if task already exists on this processor in target superstep (duplicate conflict)
        for task in target_superstep.tasks.get(target_processor, []):
            if task.node == task_node:
                return False  # Can't have same task twice on same processor in same superstep
        
        # Check precedence constraints
        # Predecessors must be in earlier supersteps OR same superstep on same processor
        # (we'll handle ordering by inserting at end when moving to earlier superstep)
        for pred_node in schedule.task_graph.predecessors(task_node):
            pred_instances = schedule.task_mapping[pred_node]
            if pred_instances:
                # Check if any predecessor instance is valid
                has_valid_predecessor = False
                for pred_inst in pred_instances:
                    if pred_inst.superstep.index < self.target_superstep_idx:
                        has_valid_predecessor = True
                        break
                    elif (pred_inst.superstep.index == self.target_superstep_idx and
                          pred_inst.proc == target_processor and
                          self.target_superstep_idx < task_instance.superstep.index):
                        # Predecessor in same target superstep on same processor
                        # This is OK if we're moving from later superstep (will be inserted after)
                        has_valid_predecessor = True
                        break
                
                if not has_valid_predecessor:
                    return False
        
        # For successors: they must be in later supersteps OR same superstep on same processor
        # (we'll handle ordering by inserting at beginning when moving to later superstep)
        for succ_node in schedule.task_graph.successors(task_node):
            succ_instances = schedule.task_mapping[succ_node]
            current_task_instances = schedule.task_mapping[task_node]
            
            for succ_inst in succ_instances:
                # For each successor instance, check if it will have a valid predecessor
                # after we move this instance
                will_have_valid_pred = False
                
                # Check remaining instances of current task (excluding the one we're moving)
                for curr_inst in current_task_instances:
                    if curr_inst != task_instance and curr_inst.superstep.index < succ_inst.superstep.index:
                        will_have_valid_pred = True
                        break
                
                # Check if moved instance would still satisfy this successor
                if self.target_superstep_idx < succ_inst.superstep.index:
                    will_have_valid_pred = True
                elif (self.target_superstep_idx == succ_inst.superstep.index and 
                      succ_inst.proc == target_processor and
                      self.target_superstep_idx > task_instance.superstep.index):
                    # Moving to same superstep as successor on same processor
                    # This is OK if we're moving from earlier superstep (will be inserted before)
                    will_have_valid_pred = True
                
                if not will_have_valid_pred:
                    return False
        
        return True
    
    def apply(self, schedule: BSPSchedule, task_instance: BSPTask) -> bool:
        """Move task instance to the target superstep on the same processor"""
        if not self.is_feasible(schedule, task_instance):
            return False
        
        old_processor = task_instance.proc
        task_node = task_instance.node
        
        # Remove from current superstep
        schedule.unschedule(task_instance)
        
        # Add to target superstep at appropriate position
        target_superstep = schedule.supersteps[self.target_superstep_idx]
        
        # Find the correct position to maintain dependencies
        position = self._find_insertion_position(
            schedule, task_node, old_processor, target_superstep
        )
        
        target_superstep.schedule_task(task_node, old_processor, position)
        
        return True
    
    def _find_insertion_position(self, schedule: BSPSchedule, task_node: str, 
                                 processor: str, target_superstep) -> Optional[int]:
        """Find the correct insertion position to maintain dependencies.
        
        Returns:
            Position index, or None to append at end
        """
        existing_tasks = target_superstep.tasks.get(processor, [])
        if not existing_tasks:
            return None  # Empty list, append at end
        
        # Find latest predecessor and earliest successor positions
        latest_pred_pos = -1
        earliest_succ_pos = len(existing_tasks)
        
        for i, existing_task in enumerate(existing_tasks):
            existing_node = existing_task.node
            
            # Check if existing task is a predecessor
            if existing_node in schedule.task_graph.predecessors(task_node):
                latest_pred_pos = max(latest_pred_pos, i)
            
            # Check if existing task is a successor
            if existing_node in schedule.task_graph.successors(task_node):
                earliest_succ_pos = min(earliest_succ_pos, i)
        
        # Insert after all predecessors and before all successors
        insert_pos = latest_pred_pos + 1
        
        if insert_pos > earliest_succ_pos:
            # This shouldn't happen if feasibility check is correct
            # But let's be safe and insert at the earliest valid position
            insert_pos = earliest_succ_pos
        
        if insert_pos >= len(existing_tasks):
            return None  # Append at end
        
        return insert_pos
    
    def get_possible_targets(self, task_instance: BSPTask) -> List[int]:
        """Get list of feasible superstep indices for this task instance"""
        schedule = task_instance.schedule
        targets = []
        for i in range(len(schedule.supersteps)):
            if i != task_instance.superstep.index:
                temp_action = MoveTaskToSuperstep(i)
                if temp_action.is_feasible(schedule, task_instance):
                    targets.append(i)
        return targets
    
    def __str__(self) -> str:
        return f"MoveTaskToSuperstep(target={self.target_superstep_idx})"


class MoveTaskToProcessor(ScheduleAction):
    """Action to move a task instance to a different processor in the same superstep"""
    
    def __init__(self, target_processor: str):
        self.target_processor = target_processor
    
    def is_feasible(self, schedule: BSPSchedule, task_instance: BSPTask) -> bool:
        """Check if moving task instance to target processor is feasible"""
        if task_instance.proc == self.target_processor:
            return False  # Already on target processor
            
        # Check if target processor exists in hardware network
        if self.target_processor not in schedule.hardware.network.nodes:
            return False
        
        current_superstep_index = task_instance.superstep.index
        task_node = task_instance.node
        
        # Check for dependencies that would be violated by the move
        # 1. Check predecessors - cannot have cross-processor dependencies in same superstep
        for pred_node in schedule.task_graph.predecessors(task_node):
            pred_instances = schedule.task_mapping[pred_node]
            
            for pred_inst in pred_instances:
                if pred_inst.superstep.index == current_superstep_index:
                    # Predecessor is in same superstep
                    if pred_inst.proc == task_instance.proc:
                        # Predecessor on same processor - would need duplication
                        return False
                    elif pred_inst.proc != self.target_processor:
                        # Predecessor on different processor than target - invalid in BSP
                        return False
        
        # 2. Check successors - cannot have cross-processor dependencies in same superstep
        for succ_node in schedule.task_graph.successors(task_node):
            succ_instances = schedule.task_mapping[succ_node]
            
            for succ_inst in succ_instances:
                if succ_inst.superstep.index == current_superstep_index:
                    # Successor is in same superstep
                    if succ_inst.proc == task_instance.proc:
                        # Successor on same processor - would leave it without predecessor
                        return False
                    elif succ_inst.proc != self.target_processor:
                        # Successor on different processor than target - invalid in BSP
                        return False
        
        return True
    
    def apply(self, schedule: BSPSchedule, task_instance: BSPTask) -> bool:
        """Move task instance to the target processor in the same superstep"""
        if not self.is_feasible(schedule, task_instance):
            return False
        
        old_superstep = task_instance.superstep
        
        # Remove from current position
        schedule.unschedule(task_instance)
        
        # Add to same superstep but different processor
        schedule.schedule(task_instance.node, self.target_processor, old_superstep)
        
        return True
    
    def get_possible_targets(self, task_instance: BSPTask) -> List[str]:
        """Get list of feasible processors for this task instance"""
        schedule = task_instance.schedule
        current_processor = task_instance.proc
        
        targets = []
        for processor in schedule.hardware.network.nodes:
            if processor != current_processor:
                targets.append(processor)
        
        return targets
    
    def __str__(self) -> str:
        return f"MoveTaskToProcessor(target={self.target_processor})"


class DuplicateAndMoveTask(ScheduleAction):
    """Action to move a task instance to a different processor by duplicating required dependencies"""
    
    def __init__(self, target_processor: str):
        self.target_processor = target_processor
    
    def is_feasible(self, schedule: BSPSchedule, task_instance: BSPTask) -> bool:
        """Check if duplication and move is feasible"""
        if task_instance.proc == self.target_processor:
            return False  # Already on target processor
            
        # Check if target processor exists in hardware network
        if self.target_processor not in schedule.hardware.network.nodes:
            return False
        
        # This action is more permissive - it can handle intra-superstep dependencies
        # by duplicating the required tasks
        return True
    
    def apply(self, schedule: BSPSchedule, task_instance: BSPTask) -> bool:
        """Move task instance to target processor by duplicating dependencies if needed"""
        if not self.is_feasible(schedule, task_instance):
            return False
        
        current_superstep = task_instance.superstep
        task_node = task_instance.node
        
        # Find all tasks that need to be duplicated (including transitive dependencies)
        tasks_to_duplicate = self._get_required_duplicates(
            schedule, task_node, task_instance.proc, current_superstep
        )
        
        # Duplicate required tasks to target processor in dependency order
        for dup_task_node in tasks_to_duplicate:
            schedule.schedule(dup_task_node, self.target_processor, current_superstep)
        
        # Now move the main task
        schedule.unschedule(task_instance)
        schedule.schedule(task_node, self.target_processor, current_superstep)
        
        return True
    
    def _get_required_duplicates(self, schedule: BSPSchedule, task_node: str, 
                                 current_processor: str, current_superstep) -> List[str]:
        """Get all tasks that need duplication (including transitive dependencies).
        
        Returns tasks in topological order to maintain dependencies when duplicating.
        """
        current_superstep_index = current_superstep.index
        to_duplicate = set()
        to_check = [task_node]
        
        # Find all tasks that need duplication using DFS
        while to_check:
            current = to_check.pop()
            
            for pred_node in schedule.task_graph.predecessors(current):
                pred_instances = schedule.task_mapping[pred_node]
                
                # Check if predecessor has an instance on current processor in current superstep
                pred_on_current_proc = None
                pred_on_target_proc = None
                
                for pred_inst in pred_instances:
                    if (pred_inst.superstep.index == current_superstep_index and 
                        pred_inst.proc == current_processor):
                        pred_on_current_proc = pred_inst
                    if (pred_inst.superstep.index == current_superstep_index and 
                        pred_inst.proc == self.target_processor):
                        pred_on_target_proc = pred_inst
                
                # If predecessor is on current processor but not on target processor, need to duplicate
                if pred_on_current_proc and not pred_on_target_proc:
                    if pred_node not in to_duplicate:
                        to_duplicate.add(pred_node)
                        to_check.append(pred_node)  # Check its dependencies too
        
        # Return in topological order to ensure dependencies are duplicated first
        if not to_duplicate:
            return []
        
        # Create subgraph of tasks to duplicate
        subgraph = schedule.task_graph.subgraph(to_duplicate)
        return list(nx.topological_sort(subgraph))
    
    def get_possible_targets(self, task_instance: BSPTask) -> List[str]:
        """Get list of feasible processors for this task instance"""
        schedule = task_instance.schedule
        current_processor = task_instance.proc
        
        targets = []
        for processor in schedule.hardware.network.nodes:
            if processor != current_processor:
                targets.append(processor)
        
        return targets
    
    def __str__(self) -> str:
        return f"DuplicateAndMoveTask(target={self.target_processor})"


class MergeSupersteps(ScheduleAction):
    """Action to merge two adjacent supersteps"""
    
    def __init__(self, first_superstep_idx: int):
        """Merge superstep at first_superstep_idx with the next one"""
        self.first_superstep_idx = first_superstep_idx
    
    def is_feasible(self, schedule: BSPSchedule, task_instance: BSPTask) -> bool:
        """Check if merging is feasible"""
        # Check bounds
        if (self.first_superstep_idx < 0 or 
            self.first_superstep_idx >= len(schedule.supersteps) - 1):
            return False
        
        first_ss = schedule.supersteps[self.first_superstep_idx]
        second_ss = schedule.supersteps[self.first_superstep_idx + 1]
        
        # Check if any task in second superstep depends on a task in first superstep
        # If so, they must be on different processors or properly ordered
        for proc2, tasks2 in second_ss.tasks.items():
            for task2 in tasks2:
                for pred_name in schedule.task_graph.predecessors(task2.node):
                    pred_instances = schedule.task_mapping[pred_name]
                    
                    # Check if any predecessor is in first superstep
                    for pred_inst in pred_instances:
                        if pred_inst.superstep.index == self.first_superstep_idx:
                            # Dependency exists between the two supersteps
                            if pred_inst.proc == proc2:
                                # Same processor - can merge if we maintain order
                                continue
                            else:
                                # Different processors - can always merge
                                continue
        
        return True
    
    def apply(self, schedule: BSPSchedule, task_instance: BSPTask) -> bool:
        """Merge the two supersteps"""
        if not self.is_feasible(schedule, task_instance):
            return False
        
        first_ss = schedule.supersteps[self.first_superstep_idx]
        second_ss = schedule.supersteps[self.first_superstep_idx + 1]
        
        # Move all tasks from second superstep to first, maintaining dependency order
        for proc, tasks in second_ss.tasks.items():
            for task in tasks[:]:  # Create copy since we're modifying
                # Find correct insertion position in first superstep
                position = self._find_merge_position(schedule, task, first_ss, proc)
                
                # Remove from second superstep (removes from both superstep and task_mapping)
                schedule.unschedule(task)
                
                # Add to first superstep at correct position (adds to both superstep and task_mapping)
                schedule.schedule(task.node, proc, first_ss, position)
        
        # Remove the now-empty second superstep
        schedule.supersteps.pop(self.first_superstep_idx + 1)
        
        # Important: After removing a superstep, all subsequent supersteps have shifted indices
        # but our BSPTask objects still reference the old superstep objects, which is correct
        # since they were moved to the first superstep above
        
        return True
    
    def _find_merge_position(self, schedule: BSPSchedule, task: BSPTask, 
                           target_superstep, processor: str) -> Optional[int]:
        """Find correct position to insert task when merging"""
        existing_tasks = target_superstep.tasks.get(processor, [])
        if not existing_tasks:
            return None  # Append at end
        
        # Find where to insert based on dependencies
        latest_pred_pos = -1
        earliest_succ_pos = len(existing_tasks)
        
        for i, existing_task in enumerate(existing_tasks):
            existing_node = existing_task.node
            task_node = task.node
            
            # Check if existing task is a predecessor of task being merged
            if existing_node in schedule.task_graph.predecessors(task_node):
                latest_pred_pos = max(latest_pred_pos, i)
            
            # Check if existing task is a successor of task being merged
            if existing_node in schedule.task_graph.successors(task_node):
                earliest_succ_pos = min(earliest_succ_pos, i)
        
        # Insert after all predecessors and before all successors
        insert_pos = latest_pred_pos + 1
        
        if insert_pos > earliest_succ_pos:
            insert_pos = earliest_succ_pos
        
        if insert_pos >= len(existing_tasks):
            return None  # Append at end
        
        return insert_pos
    
    def get_possible_targets(self, task_instance: BSPTask) -> List[int]:
        """Get list of feasible superstep pairs to merge"""
        schedule = task_instance.schedule
        targets = []
        
        for i in range(len(schedule.supersteps) - 1):
            temp_action = MergeSupersteps(i)
            if temp_action.is_feasible(schedule, task_instance):
                targets.append(i)
        
        return targets
    
    def __str__(self) -> str:
        return f"MergeSupersteps(first={self.first_superstep_idx})"


class BSPSimulatedAnnealing:
    """Simulated Annealing optimizer for BSP schedules"""
    
    def __init__(self,
                 max_iterations: int = 1000,
                 max_temp: float = 100.0,
                 min_temp: float = 0.1,
                 cooling_rate: float = 0.99,
                 action_types: Optional[List[Type[ScheduleAction]]] = None):
        """Initialize simulated annealing optimizer
        
        Args:
            max_iterations: Maximum number of iterations
            max_temp: Initial temperature
            min_temp: Minimum temperature (stopping condition)
            cooling_rate: Temperature cooling factor (0 < rate < 1)
            action_types: List of action types to use for optimization
        """
        self.max_iterations = max_iterations
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        
        if action_types is None:
            # Temporarily exclude MergeSupersteps until bugs are fixed
            self.action_types = [MoveTaskToSuperstep, MoveTaskToProcessor]
        else:
            self.action_types = action_types
        
        self.iterations: List[SimulatedAnnealingIteration] = []
    
    def get_energy(self, schedule: BSPSchedule) -> float:
        """Calculate energy (cost) of a schedule - using makespan"""
        return schedule.makespan
    
    def get_random_task_instance(self, schedule: BSPSchedule) -> Optional[BSPTask]:
        """Get a random task instance from the schedule"""
        if not schedule.task_mapping:
            return None
        
        # Collect all task instances
        all_instances = []
        for task_instances in schedule.task_mapping.values():
            all_instances.extend(task_instances)
        
        if not all_instances:
            return None
            
        return random.choice(all_instances)
    
    def generate_neighbor(self, schedule: BSPSchedule) -> Optional[Tuple[BSPSchedule, ScheduleAction]]:
        """Generate a neighbor schedule by applying a random feasible action"""
        # Try multiple times to find a feasible action
        for _ in range(50):  # Avoid infinite loops
            task_instance = self.get_random_task_instance(schedule)
            if task_instance is None:
                continue
            
            # Choose random action type
            ActionType = random.choice(self.action_types)
            
            # Get possible targets for this action with this task instance
            if ActionType == MoveTaskToSuperstep:
                temp_action = ActionType(0)  # Dummy for getting targets
                targets = temp_action.get_possible_targets(task_instance)
                if targets:
                    target = random.choice(targets)
                    action = ActionType(target)
                else:
                    continue
            elif ActionType == MoveTaskToProcessor:
                temp_action = ActionType("")  # Dummy for getting targets
                targets = temp_action.get_possible_targets(task_instance)
                if targets:
                    target = random.choice(targets)
                    action = ActionType(target)
                else:
                    continue
            elif ActionType == DuplicateAndMoveTask:
                temp_action = ActionType("")  # Dummy for getting targets
                targets = temp_action.get_possible_targets(task_instance)
                if targets:
                    target = random.choice(targets)
                    action = ActionType(target)
                else:
                    continue
            elif ActionType == MergeSupersteps:
                temp_action = ActionType(0)  # Dummy for getting targets
                targets = temp_action.get_possible_targets(task_instance)
                if targets:
                    target = random.choice(targets)
                    action = ActionType(target)
                else:
                    continue
            else:
                continue  # Unknown action type
            
            # Create copy and apply action
            neighbor_schedule = schedule.copy()
            # Need to find the corresponding task instance in the copied schedule
            copied_task_instance = self.find_corresponding_instance(neighbor_schedule, task_instance)
            if copied_task_instance and action.apply(neighbor_schedule, copied_task_instance):
                # Validate the generated schedule
                is_valid, errors = neighbor_schedule.is_valid()
                if is_valid:
                    return neighbor_schedule, action
                else:
                    # Invalid schedule generated, skip this action
                    logger.error(f"Generated invalid schedule with {action}: {errors}")
                    continue
        
        return None, None  # Could not find feasible neighbor
    
    def find_corresponding_instance(self, copied_schedule: BSPSchedule, original_instance: BSPTask) -> Optional[BSPTask]:
        """Find the corresponding task instance in a copied schedule"""
        task_name = original_instance.node
        original_proc = original_instance.proc
        original_superstep_idx = original_instance.superstep.index
        original_index_in_proc = original_instance.index
        
        copied_instances = copied_schedule.task_mapping[task_name]
        
        for copied_instance in copied_instances:
            if (copied_instance.proc == original_proc and
                copied_instance.superstep.index == original_superstep_idx and
                copied_instance.index == original_index_in_proc):
                return copied_instance
        
        return None
    
    def optimize(self, initial_schedule: BSPSchedule) -> BSPSchedule:
        """Run simulated annealing optimization on the given schedule"""
        logger.info("Starting BSP simulated annealing optimization")
        
        # Initialize
        current_schedule = initial_schedule.copy()
        current_energy = self.get_energy(current_schedule)
        
        best_schedule = current_schedule.copy()
        best_energy = current_energy
        
        temp = self.max_temp
        iteration = 0
        
        # Clear previous iterations
        self.iterations = []
        
        logger.info(f"Initial energy (makespan): {current_energy:.2f}")
        
        while iteration < self.max_iterations and temp > self.min_temp:
            log_prefix = f"[Iter {iteration}/{self.max_iterations} | Temp {temp:.2f} | Energy {current_energy:.2f} | Best {best_energy:.2f}]"
            
            # Generate neighbor
            neighbor_schedule, action = self.generate_neighbor(current_schedule)
            
            if neighbor_schedule is None:
                # Could not find feasible neighbor, record failed iteration
                iteration_data = SimulatedAnnealingIteration(
                    iteration=iteration,
                    temperature=temp,
                    current_energy=current_energy,
                    neighbor_energy=current_energy,
                    best_energy=best_energy,
                    accept_probability=0.0,
                    accepted=False,
                    action=action,
                    action_applied=False
                )
                self.iterations.append(iteration_data)
                temp *= self.cooling_rate
                iteration += 1
                continue
            
            neighbor_energy = self.get_energy(neighbor_schedule)
            
            # Calculate acceptance probability
            energy_diff = neighbor_energy - current_energy
            if energy_diff <= 0:
                # Better or equal solution - always accept
                accept_probability = 1.0
            else:
                # Worse solution - accept with probability
                accept_probability = math.exp(-energy_diff / temp)
            
            accepted = random.random() < accept_probability
            
            # Record iteration
            iteration_data = SimulatedAnnealingIteration(
                iteration=iteration,
                temperature=temp,
                current_energy=current_energy,
                neighbor_energy=neighbor_energy,
                best_energy=best_energy,
                accept_probability=accept_probability,
                accepted=accepted,
                action=action,
                action_applied=True
            )
            self.iterations.append(iteration_data)
            
            if accepted:
                current_schedule = neighbor_schedule
                current_energy = neighbor_energy
                
                # Update best solution if needed
                if neighbor_energy < best_energy:
                    best_schedule = neighbor_schedule.copy()
                    best_energy = neighbor_energy
                    logger.info(f"{log_prefix} New best energy: {best_energy:.2f}")
                
                status = "better" if energy_diff <= 0 else "worse"
                logger.debug(f"{log_prefix} Accepted {status} neighbor (energy={neighbor_energy:.2f}, prob={accept_probability:.3f})")
            else:
                logger.debug(f"{log_prefix} Rejected neighbor (energy={neighbor_energy:.2f}, prob={accept_probability:.3f})")
            
            # Cool down
            temp *= self.cooling_rate
            iteration += 1
        
        logger.info(f"Optimization completed. Best energy: {best_energy:.2f}, Initial energy: {self.get_energy(initial_schedule):.2f}")
        
        is_valid, errors = best_schedule.is_valid()
        if not is_valid:
            logger.error(f"Best schedule is not valid after optimization! Errors: {errors}")
            return initial_schedule
        return best_schedule
    
    def get_optimization_stats(self) -> dict:
        """Get statistics about the optimization process"""
        if not self.iterations:
            return {}
        
        total_iterations = len(self.iterations)
        accepted_iterations = sum(1 for it in self.iterations if it.accepted)
        applied_iterations = sum(1 for it in self.iterations if it.action_applied)
        
        initial_energy = self.iterations[0].current_energy
        final_energy = self.iterations[-1].best_energy
        
        return {
            'total_iterations': total_iterations,
            'accepted_iterations': accepted_iterations,
            'applied_iterations': applied_iterations,
            'acceptance_rate': accepted_iterations / applied_iterations if applied_iterations > 0 else 0,
            'initial_energy': initial_energy,
            'final_energy': final_energy,
            'improvement': initial_energy - final_energy,
            'improvement_percent': (initial_energy - final_energy) / initial_energy * 100 if initial_energy > 0 else 0
        }
        
    def print_optimization_stats(self):
        """Print optimization statistics"""
        stats = self.get_optimization_stats()
        if not stats:
            logger.warning("No optimization statistics available.")
            return
        
        logger.info("Optimization Statistics:")
        logger.info(f"  Total iterations: {stats['total_iterations']}")
        logger.info(f"  Accepted iterations: {stats['accepted_iterations']}")
        logger.info(f"  Applied iterations: {stats['applied_iterations']}")
        logger.info(f"  Acceptance rate: {stats['acceptance_rate']:.2%}")
        logger.info(f"  Initial energy: {stats['initial_energy']:.2f}")
        logger.info(f"  Final energy: {stats['final_energy']:.2f}")
        logger.info(f"  Improvement: {stats['improvement']:.2f} ({stats['improvement_percent']:.1f}%)")