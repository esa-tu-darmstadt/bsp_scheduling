import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type
from ..schedule import BSPSchedule


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
    def is_feasible(self, schedule: BSPSchedule, task_node: str) -> bool:
        """Check if this action is feasible for the given task in the schedule"""
        pass
    
    @abstractmethod
    def apply(self, schedule: BSPSchedule, task_node: str) -> bool:
        """Apply the action to the schedule. Returns True if successful"""
        pass
    
    @abstractmethod
    def get_possible_targets(self, schedule: BSPSchedule, task_node: str) -> List:
        """Get list of possible targets for this action"""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """String representation of the action"""
        pass


class MoveTaskToSuperstep(ScheduleAction):
    """Action to move a task to a different superstep"""
    
    def __init__(self, target_superstep_idx: int):
        self.target_superstep_idx = target_superstep_idx
    
    def is_feasible(self, schedule: BSPSchedule, task_node: str) -> bool:
        """Check if moving task to target superstep maintains precedence constraints"""
        if task_node not in schedule.task_mapping:
            return False
        
        if self.target_superstep_idx < 0 or self.target_superstep_idx >= len(schedule.supersteps):
            return False
            
        task = schedule.task_mapping[task_node]
        if task.superstep.index == self.target_superstep_idx:
            return False  # Already in target superstep
        
        # Check precedence constraints
        # All predecessors must be in earlier supersteps
        for pred_node in schedule.task_graph.predecessors(task_node):
            if pred_node in schedule.task_mapping:
                pred_task = schedule.task_mapping[pred_node]
                if pred_task.superstep.index >= self.target_superstep_idx:
                    return False
        
        # All successors must be in later supersteps or can be moved later
        for succ_node in schedule.task_graph.successors(task_node):
            if succ_node in schedule.task_mapping:
                succ_task = schedule.task_mapping[succ_node]
                if succ_task.superstep.index <= self.target_superstep_idx:
                    return False
        
        return True
    
    def apply(self, schedule: BSPSchedule, task_node: str) -> bool:
        """Move task to the target superstep on the same processor"""
        if not self.is_feasible(schedule, task_node):
            return False
        
        task = schedule.task_mapping[task_node]
        old_processor = task.proc
        
        # Remove from current superstep
        schedule.unschedule(task)
        
        # Add to target superstep
        target_superstep = schedule.supersteps[self.target_superstep_idx]
        schedule.schedule(task_node, old_processor, target_superstep)
        
        return True
    
    def get_possible_targets(self, schedule: BSPSchedule, task_node: str) -> List[int]:
        """Get list of feasible superstep indices for this task"""
        targets = []
        for i in range(len(schedule.supersteps)):
            if i != schedule.task_mapping[task_node].superstep.index:
                temp_action = MoveTaskToSuperstep(i)
                if temp_action.is_feasible(schedule, task_node):
                    targets.append(i)
        return targets
    
    def __str__(self) -> str:
        return f"MoveTaskToSuperstep(target={self.target_superstep_idx})"


class MoveTaskToProcessor(ScheduleAction):
    """Action to move a task to a different processor in the same superstep"""
    
    def __init__(self, target_processor: str):
        self.target_processor = target_processor
    
    def is_feasible(self, schedule: BSPSchedule, task_node: str) -> bool:
        """Check if moving task to target processor is feasible"""
        if task_node not in schedule.task_mapping:
            return False
        
        task = schedule.task_mapping[task_node]
        if task.proc == self.target_processor:
            return False  # Already on target processor
            
        # Check if target processor exists in hardware network
        if self.target_processor not in schedule.hardware.network.nodes:
            return False
        
        # Check for intra-superstep dependencies
        # Task can only be moved if it doesn't depend on tasks in the same superstep
        current_superstep_index = task.superstep.index
        
        for pred_node in schedule.task_graph.predecessors(task_node):
            if pred_node in schedule.task_mapping:
                pred_task = schedule.task_mapping[pred_node]
                # If predecessor is in same superstep, task cannot be moved
                # (would require duplication of predecessor)
                if pred_task.superstep.index == current_superstep_index:
                    return False
        
        return True
    
    def apply(self, schedule: BSPSchedule, task_node: str) -> bool:
        """Move task to the target processor in the same superstep"""
        if not self.is_feasible(schedule, task_node):
            return False
        
        task = schedule.task_mapping[task_node]
        old_superstep = task.superstep
        
        # Remove from current position
        schedule.unschedule(task)
        
        # Add to same superstep but different processor
        schedule.schedule(task_node, self.target_processor, old_superstep)
        
        return True
    
    def get_possible_targets(self, schedule: BSPSchedule, task_node: str) -> List[str]:
        """Get list of feasible processors for this task"""
        if task_node not in schedule.task_mapping:
            return []
        
        task = schedule.task_mapping[task_node]
        current_processor = task.proc
        
        targets = []
        for processor in schedule.hardware.network.nodes:
            if processor != current_processor:
                targets.append(processor)
        
        return targets
    
    def __str__(self) -> str:
        return f"MoveTaskToProcessor(target={self.target_processor})"


class DuplicateAndMoveTask(ScheduleAction):
    """Action to move a task to a different processor by duplicating required dependencies"""
    
    def __init__(self, target_processor: str):
        self.target_processor = target_processor
    
    def is_feasible(self, schedule: BSPSchedule, task_node: str) -> bool:
        """Check if duplication and move is feasible"""
        if task_node not in schedule.task_mapping:
            return False
        
        task = schedule.task_mapping[task_node]
        if task.proc == self.target_processor:
            return False  # Already on target processor
            
        # Check if target processor exists in hardware network
        if self.target_processor not in schedule.hardware.network.nodes:
            return False
        
        # This action is more permissive - it can handle intra-superstep dependencies
        # by duplicating the required tasks
        return True
    
    def apply(self, schedule: BSPSchedule, task_node: str) -> bool:
        """Move task to target processor by duplicating dependencies if needed"""
        if not self.is_feasible(schedule, task_node):
            return False
        
        task = schedule.task_mapping[task_node]
        current_superstep = task.superstep
        current_superstep_index = current_superstep.index
        
        # Find all predecessors in the same superstep that need to be duplicated
        tasks_to_duplicate = []
        for pred_node in schedule.task_graph.predecessors(task_node):
            if pred_node in schedule.task_mapping:
                pred_task = schedule.task_mapping[pred_node]
                if pred_task.superstep.index == current_superstep_index and pred_task.proc != self.target_processor:
                    tasks_to_duplicate.append(pred_node)
        
        # Duplicate required tasks to target processor
        for dup_task_node in tasks_to_duplicate:
            # Only duplicate if not already on target processor
            dup_task = schedule.task_mapping[dup_task_node]
            if dup_task.proc != self.target_processor:
                # Schedule duplicate on target processor in same superstep
                schedule.schedule(dup_task_node, self.target_processor, current_superstep)
        
        # Now move the main task
        schedule.unschedule(task)
        schedule.schedule(task_node, self.target_processor, current_superstep)
        
        return True
    
    def get_possible_targets(self, schedule: BSPSchedule, task_node: str) -> List[str]:
        """Get list of feasible processors for this task"""
        if task_node not in schedule.task_mapping:
            return []
        
        task = schedule.task_mapping[task_node]
        current_processor = task.proc
        
        targets = []
        for processor in schedule.hardware.network.nodes:
            if processor != current_processor:
                targets.append(processor)
        
        return targets
    
    def __str__(self) -> str:
        return f"DuplicateAndMoveTask(target={self.target_processor})"


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
            self.action_types = [MoveTaskToSuperstep, MoveTaskToProcessor]
        else:
            self.action_types = action_types
        
        self.iterations: List[SimulatedAnnealingIteration] = []
    
    def get_energy(self, schedule: BSPSchedule) -> float:
        """Calculate energy (cost) of a schedule - using makespan"""
        return schedule.makespan
    
    def get_random_task(self, schedule: BSPSchedule) -> Optional[str]:
        """Get a random task from the schedule"""
        if not schedule.task_mapping:
            return None
        return random.choice(list(schedule.task_mapping.keys()))
    
    def generate_neighbor(self, schedule: BSPSchedule) -> Optional[Tuple[BSPSchedule, ScheduleAction]]:
        """Generate a neighbor schedule by applying a random feasible action"""
        # Try multiple times to find a feasible action
        for _ in range(50):  # Avoid infinite loops
            task_node = self.get_random_task(schedule)
            if task_node is None:
                continue
            
            # Choose random action type
            ActionType = random.choice(self.action_types)
            
            # Get possible targets for this action
            if ActionType == MoveTaskToSuperstep:
                temp_action = ActionType(0)  # Dummy for getting targets
                targets = temp_action.get_possible_targets(schedule, task_node)
                if targets:
                    target = random.choice(targets)
                    action = ActionType(target)
                else:
                    continue
            elif ActionType == MoveTaskToProcessor:
                temp_action = ActionType("")  # Dummy for getting targets
                targets = temp_action.get_possible_targets(schedule, task_node)
                if targets:
                    target = random.choice(targets)
                    action = ActionType(target)
                else:
                    continue
            else:
                continue  # Unknown action type
            
            # Create copy and apply action
            neighbor_schedule = schedule.copy()
            if action.apply(neighbor_schedule, task_node):
                return neighbor_schedule, action
        
        return None, None  # Could not find feasible neighbor
    
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