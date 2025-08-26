from functools import cache
import logging
import math
import random
import networkx as nx
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict
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
    action: 'Action'
    action_applied: bool


class Action(ABC):
    """Base class for a specific action to be applied to a schedule"""
    
    @abstractmethod
    def apply(self, schedule: BSPSchedule) -> bool:
        """Apply this specific action to the schedule. Returns True if successful."""
        pass
    
    @abstractmethod
    def describe(self) -> str:
        """Human-readable description of what this action does"""
        pass
    
    def __str__(self) -> str:
        return self.describe()


class ActionGenerator(ABC):
    """Base class for generating possible actions based on schedule analysis"""
    
    @abstractmethod
    def generate_actions(self, schedule: BSPSchedule, max_actions: int = 10) -> List[Action]:
        """Analyze schedule and generate list of beneficial actions"""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Name of this generator for logging purposes"""
        pass


class MoveTaskAction(Action):
    """Concrete action to move a specific task instance to a different location"""
    
    def __init__(self, task_name: str, from_proc: str, from_superstep: int, 
                 to_proc: str, to_superstep: int, new_makespan: float = 0.0):
        self.task_name = task_name
        self.from_proc = from_proc
        self.from_superstep = from_superstep
        self.to_proc = to_proc
        self.to_superstep = to_superstep
        self.new_makespan = new_makespan
    
    def apply(self, schedule: BSPSchedule) -> bool:
        """Move the task to the target location"""
        # Find the task instance
        task_instance = None
        for instance in schedule.task_mapping[self.task_name]:
            if (instance.proc == self.from_proc and 
                instance.superstep.index == self.from_superstep):
                task_instance = instance
                break
        
        if not task_instance:
            logger.debug(f"Task instance {self.task_name} not found at {self.from_proc}:{self.from_superstep}")
            return False
        
        # Check if target superstep exists
        if self.to_superstep >= len(schedule.supersteps):
            logger.debug(f"Target superstep {self.to_superstep} does not exist")
            return False
        
        target_superstep = schedule.supersteps[self.to_superstep]
        
        # Check precedence constraints for the move
        if not self._check_precedence_valid(schedule, task_instance, target_superstep):
            logger.debug(f"Move would violate precedence constraints")
            return False
        
        # Remove from current location
        schedule.unschedule(task_instance)
        
        # Determine insertion position based on superstep direction
        if self.to_superstep < self.from_superstep:
            # Moving to earlier superstep - append to end (after all existing tasks)
            position = None
        else:
            # Moving to later superstep - insert at beginning (before all existing tasks)  
            position = 0
        
        # Add to target location at correct position
        schedule.schedule(self.task_name, self.to_proc, target_superstep, position)
        
        return True
    
    def _check_precedence_valid(self, schedule: BSPSchedule, task_instance: BSPTask, 
                                target_superstep) -> bool:
        """Check if move maintains precedence constraints"""
        task_name = task_instance.node
        target_superstep_idx = target_superstep.index
        
        # Check predecessors
        for pred_name in schedule.task_graph.predecessors(task_name):
            pred_instances = schedule.task_mapping.get(pred_name, [])
            if not pred_instances:
                continue
                
            valid_pred = False
            for pred in pred_instances:
                if pred.superstep.index < target_superstep_idx:
                    valid_pred = True
                    break
                elif (pred.superstep.index == target_superstep_idx and 
                      pred.proc == self.to_proc):
                    # Same processor, same superstep - need to check ordering
                    valid_pred = True
                    break
            
            if not valid_pred:
                return False
        
        # Check successors
        for succ_name in schedule.task_graph.successors(task_name):
            succ_instances = schedule.task_mapping.get(succ_name, [])
            
            for succ in succ_instances:
                # Each successor needs a valid predecessor after the move
                will_have_valid_pred = False
                
                # Check if this move will still satisfy the successor
                if target_superstep_idx < succ.superstep.index:
                    will_have_valid_pred = True
                elif (target_superstep_idx == succ.superstep.index and 
                      succ.proc == self.to_proc):
                    # Same processor, same superstep - will be ordered correctly
                    will_have_valid_pred = True
                else:
                    # Check if there are other instances that satisfy this successor
                    for other_instance in schedule.task_mapping[task_name]:
                        if (other_instance != task_instance and 
                            other_instance.superstep.index < succ.superstep.index):
                            will_have_valid_pred = True
                            break
                
                if not will_have_valid_pred:
                    return False
        
        return True
    
    def describe(self) -> str:
        return (f"Move {self.task_name} from {self.from_proc}:SS{self.from_superstep} "
                f"to {self.to_proc}:SS{self.to_superstep} (new_makespan: {self.new_makespan:.2f})")
    
    @staticmethod
    def calculate_makespan_after_move(schedule: BSPSchedule, task_name: str, 
                                      from_proc: str, from_superstep: int,
                                      to_proc: str, to_superstep: int) -> float:
        """Calculate the makespan if this move were applied"""
        # Create a copy and apply the move
        test_schedule = schedule.copy()
        
        # Find the task instance and create the action
        task_instance = None
        for instance in test_schedule.task_mapping[task_name]:
            if (instance.proc == from_proc and instance.superstep.index == from_superstep):
                task_instance = instance
                break
        
        if not task_instance:
            return schedule.makespan  # No change if task not found
        
        # Create and apply the move action
        move_action = MoveTaskAction(task_name, from_proc, from_superstep, to_proc, to_superstep)
        success = move_action.apply(test_schedule)
        
        if success:
            return test_schedule.makespan
        else:
            return schedule.makespan  # No change if move failed


class MergeSuperstepsAction(Action):
    """Action to merge two adjacent supersteps"""
    
    def __init__(self, first_superstep_idx: int, new_makespan: float = 0.0):
        self.first_superstep_idx = first_superstep_idx
        self.new_makespan = new_makespan
    
    def apply(self, schedule: BSPSchedule) -> bool:
        """Merge the two supersteps"""
        if self.first_superstep_idx >= len(schedule.supersteps) - 1:
            return False
        
        first_ss = schedule.supersteps[self.first_superstep_idx]
        second_ss = schedule.supersteps[self.first_superstep_idx + 1]
        
        # Move all tasks from second to first superstep
        for proc, tasks in second_ss.tasks.items():
            for task in tasks[:]:  # Copy list since we're modifying
                schedule.unschedule(task)
                schedule.schedule(task.node, proc, first_ss)
        
        # Remove the now-empty second superstep
        schedule.supersteps.pop(self.first_superstep_idx + 1)
        
        return True
    
    def describe(self) -> str:
        return (f"Merge supersteps {self.first_superstep_idx} and {self.first_superstep_idx + 1} "
                f"(new_makespan: {self.new_makespan:.2f})")


class LoadBalancingGenerator(ActionGenerator):
    """Generates actions to balance load across processors in each superstep"""
    
    def name(self) -> str:
        return "LoadBalancing"
    
    def generate_actions(self, schedule: BSPSchedule, max_actions: int = 10) -> List[Action]:
        actions = []
        
        # Analyze each superstep for load imbalance
        for ss_idx, superstep in enumerate(schedule.supersteps):
            processor_loads = self._calculate_processor_loads(superstep)
            
            if not processor_loads:
                continue
            
            # Find idle and overloaded processors
            max_load = max(processor_loads.values())
            min_load = min(processor_loads.values())
            avg_load = sum(processor_loads.values()) / len(processor_loads)
            
            # Skip if empty superstep
            if avg_load == 0:
                continue
            
            # Skip if reasonably balanced
            if (max_load - min_load) / avg_load <= 0.1:
                continue
            
            # Find source processors (overloaded) and target processors (underloaded)
            overloaded = [(p, load) for p, load in processor_loads.items() 
                          if load > avg_load + 0.5]
            underloaded = [(p, load) for p, load in processor_loads.items() 
                           if load < avg_load - 0.5]
            
            # Generate move actions from overloaded to underloaded
            for source_proc, source_load in overloaded:
                # Find movable tasks on overloaded processor
                tasks = superstep.tasks.get(source_proc, [])
                
                for task in tasks:
                    # Option 1: Move to underloaded processor in same superstep
                    for target_proc, target_load in underloaded:
                        if self._can_move_task(schedule, task, target_proc, ss_idx):
                            new_makespan = MoveTaskAction.calculate_makespan_after_move(
                                schedule, task.node, source_proc, ss_idx, target_proc, ss_idx
                            )
                            action = MoveTaskAction(
                                task.node, source_proc, ss_idx,
                                target_proc, ss_idx, new_makespan
                            )
                            actions.append(action)
                            
                            if len(actions) >= max_actions:
                                return actions
                    
                    # Option 2: Move to different superstep on same processor
                    valid_supersteps = self._get_valid_supersteps_for_task(
                        schedule, task, source_proc
                    )
                    
                    for target_ss_idx in valid_supersteps:
                        if target_ss_idx != ss_idx:  # Don't move to same superstep
                            new_makespan = MoveTaskAction.calculate_makespan_after_move(
                                schedule, task.node, source_proc, ss_idx, source_proc, target_ss_idx
                            )
                            action = MoveTaskAction(
                                task.node, source_proc, ss_idx,
                                source_proc, target_ss_idx, new_makespan
                            )
                            actions.append(action)
                            
                            if len(actions) >= max_actions:
                                return actions
        
        return actions
    
    def _calculate_processor_loads(self, superstep) -> Dict[str, float]:
        """Calculate the load (total execution time) on each processor"""
        # Get all processors from the hardware, not just ones with tasks
        schedule = superstep.schedule
        all_processors = list(schedule.hardware.network.nodes)
        
        loads = {}
        for proc in all_processors:
            tasks = superstep.tasks.get(proc, [])
            loads[proc] = sum(task.duration for task in tasks) if tasks else 0.0
        return loads
    
    def _can_move_task(self, schedule: BSPSchedule, task: BSPTask, 
                       target_proc: str, superstep_idx: int) -> bool:
        """Check if a task can be moved to a different processor in same superstep"""
        task_name = task.node
        
        # Check if dependencies allow this move
        for pred_name in schedule.task_graph.predecessors(task_name):
            pred_instances = schedule.task_mapping.get(pred_name, [])
            for pred in pred_instances:
                if pred.superstep.index == superstep_idx and pred.proc != target_proc:
                    # Would create cross-processor dependency in same superstep
                    return False
        
        for succ_name in schedule.task_graph.successors(task_name):
            succ_instances = schedule.task_mapping.get(succ_name, [])
            for succ in succ_instances:
                if succ.superstep.index == superstep_idx and succ.proc != target_proc:
                    # Would create cross-processor dependency in same superstep
                    return False
        
        return True
    
    def _get_valid_supersteps_for_task(self, schedule: BSPSchedule, task: BSPTask, 
                                       processor: str) -> List[int]:
        """Get list of superstep indices where this task can be placed on the given processor"""
        valid_supersteps = []
        task_name = task.node
        
        for ss_idx in range(len(schedule.supersteps)):
            # Check if moving to this superstep maintains precedence constraints
            valid = True
            
            # Check predecessors - must be in earlier supersteps or same superstep on same processor
            for pred_name in schedule.task_graph.predecessors(task_name):
                pred_instances = schedule.task_mapping.get(pred_name, [])
                if not pred_instances:
                    continue
                
                has_valid_pred = False
                for pred in pred_instances:
                    if pred.superstep.index < ss_idx:
                        has_valid_pred = True
                        break
                    elif pred.superstep.index == ss_idx and pred.proc == processor:
                        has_valid_pred = True
                        break
                
                if not has_valid_pred:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check successors - must be in later supersteps or same superstep on same processor
            for succ_name in schedule.task_graph.successors(task_name):
                succ_instances = schedule.task_mapping.get(succ_name, [])
                
                for succ in succ_instances:
                    # Check if moving this task would leave successor without valid predecessor
                    will_have_valid_pred = False
                    
                    # Check if moved task would still satisfy this successor
                    if ss_idx < succ.superstep.index:
                        will_have_valid_pred = True
                    elif ss_idx == succ.superstep.index and succ.proc == processor:
                        will_have_valid_pred = True
                    else:
                        # Check other instances of current task
                        for other_instance in schedule.task_mapping[task_name]:
                            if (other_instance != task and 
                                other_instance.superstep.index < succ.superstep.index):
                                will_have_valid_pred = True
                                break
                    
                    if not will_have_valid_pred:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                valid_supersteps.append(ss_idx)
        
        return valid_supersteps


class SuperstepMergeGenerator(ActionGenerator):
    """Generates actions to merge adjacent supersteps with low utilization"""
    
    def name(self) -> str:
        return "SuperstepMerge"
    
    def generate_actions(self, schedule: BSPSchedule, max_actions: int = 10) -> List[Action]:
        actions = []
        
        # Look for adjacent supersteps that can be merged
        for i in range(len(schedule.supersteps) - 1):
            first_ss = schedule.supersteps[i]
            second_ss = schedule.supersteps[i + 1]
            
            # Calculate utilization
            first_util = self._calculate_utilization(first_ss, schedule.hardware)
            second_util = self._calculate_utilization(second_ss, schedule.hardware)
            
            # Consider merging if combined utilization would be reasonable
            combined_tasks = sum(len(tasks) for tasks in first_ss.tasks.values())
            combined_tasks += sum(len(tasks) for tasks in second_ss.tasks.values())
            num_processors = len(schedule.hardware.network.nodes)
            
            avg_tasks_per_proc = combined_tasks / num_processors
            
            # Only merge if it won't create severe imbalance
            if avg_tasks_per_proc <= 5:  # Threshold for reasonable load
                if self._can_merge(schedule, i):
                    benefit = first_ss.sync_time + first_ss.exchange_time  # Save sync/exchange overhead
                    action = MergeSuperstepsAction(i, benefit)
                    actions.append(action)
                    
                    if len(actions) >= max_actions:
                        return actions
        
        return actions
    
    def _calculate_utilization(self, superstep, hardware) -> float:
        """Calculate processor utilization in a superstep"""
        num_processors = len(hardware.network.nodes)
        active_processors = len([p for p, tasks in superstep.tasks.items() if tasks])
        return active_processors / num_processors if num_processors > 0 else 0
    
    def _can_merge(self, schedule: BSPSchedule, first_idx: int) -> bool:
        """Check if two supersteps can be merged without violating constraints"""
        first_ss = schedule.supersteps[first_idx]
        second_ss = schedule.supersteps[first_idx + 1]
        
        # Check for dependencies between supersteps
        for proc2, tasks2 in second_ss.tasks.items():
            for task2 in tasks2:
                for pred_name in schedule.task_graph.predecessors(task2.node):
                    pred_instances = schedule.task_mapping.get(pred_name, [])
                    
                    for pred in pred_instances:
                        if pred.superstep.index == first_idx:
                            # Dependency exists between supersteps
                            if pred.proc != proc2:
                                # Different processors - would violate BSP
                                return False
        
        return True


class CriticalPathOptimizer(ActionGenerator):
    """Generates actions to optimize tasks on the critical path"""
    
    def name(self) -> str:
        return "CriticalPath"
    
    def generate_actions(self, schedule: BSPSchedule, max_actions: int = 10) -> List[Action]:
        actions = []
        
        # Find critical path through the schedule
        critical_tasks = self._find_critical_path(schedule)
        
        if not critical_tasks:
            return actions
        
        # Try to move critical tasks earlier
        for task_name in critical_tasks:
            instances = schedule.task_mapping.get(task_name, [])
            
            for instance in instances:
                current_ss_idx = instance.superstep.index
                
                # Try moving to earlier supersteps
                for target_ss_idx in range(current_ss_idx):
                    target_ss = schedule.supersteps[target_ss_idx]
                    
                    # Try each processor in the target superstep
                    for target_proc in schedule.hardware.network.nodes:
                        if self._can_move_to(schedule, instance, target_proc, target_ss_idx):
                            new_makespan = MoveTaskAction.calculate_makespan_after_move(
                                schedule, task_name, instance.proc, current_ss_idx, target_proc, target_ss_idx
                            )
                            action = MoveTaskAction(
                                task_name, instance.proc, current_ss_idx,
                                target_proc, target_ss_idx, new_makespan
                            )
                            actions.append(action)
                            
                            if len(actions) >= max_actions:
                                return actions
        
        return actions
    
    @cache
    def _find_critical_path(self, schedule: BSPSchedule) -> List[str]:
        """Find the critical path considering both computation and communication times"""
        task_graph = schedule.task_graph
        avg_comp_speed = schedule.hardware.avg_computation_speed
        avg_comm_speed = schedule.hardware.avg_communication_speed
        
        try:
            # Get topological ordering
            topo_order = list(nx.topological_sort(task_graph))
            
            # distances[node] = longest path distance TO this node
            distances = {node: 0.0 for node in task_graph.nodes()}
            predecessors = {node: None for node in task_graph.nodes()}
            
            # For each node in topological order
            for node in topo_order:
                # Node execution time
                node_weight = task_graph.nodes[node].get('weight', 1.0)
                node_exec_time = node_weight / avg_comp_speed if avg_comp_speed > 0 else node_weight
                
                # Find best predecessor
                best_distance = node_exec_time  # Just the node itself
                best_pred = None
                
                for pred in task_graph.predecessors(node):
                    # Communication time from predecessor
                    edge_weight = task_graph.edges[pred, node].get('weight', 1.0)
                    comm_time = edge_weight / avg_comm_speed if avg_comm_speed > 0 else edge_weight
                    
                    # Total distance through this predecessor
                    candidate_distance = distances[pred] + comm_time + node_exec_time
                    
                    if candidate_distance > best_distance:
                        best_distance = candidate_distance
                        best_pred = pred
                
                distances[node] = best_distance
                predecessors[node] = best_pred
            
            # Find the node with maximum distance (end of critical path)
            if not distances:
                return []
                
            critical_end = max(distances.keys(), key=lambda x: distances[x])
            
            # Reconstruct path
            path = []
            current = critical_end
            while current is not None:
                path.append(current)
                current = predecessors[current]
            
            path.reverse()
            return path
            
        except Exception as e:
            logger.debug(f"Error finding critical path: {e}")
            return []
    
    def _can_move_to(self, schedule: BSPSchedule, task_instance: BSPTask,
                     target_proc: str, target_ss_idx: int) -> bool:
        """Check if task can be moved to target location"""
        task_name = task_instance.node
        
        # Check predecessors
        for pred_name in schedule.task_graph.predecessors(task_name):
            pred_instances = schedule.task_mapping.get(pred_name, [])
            
            valid_pred = False
            for pred in pred_instances:
                if pred.superstep.index < target_ss_idx:
                    valid_pred = True
                    break
                elif (pred.superstep.index == target_ss_idx and 
                      pred.proc == target_proc):
                    valid_pred = True
                    break
            
            if pred_instances and not valid_pred:
                return False
        
        # Check successors
        for succ_name in schedule.task_graph.successors(task_name):
            succ_instances = schedule.task_mapping.get(succ_name, [])
            
            for succ in succ_instances:
                if target_ss_idx > succ.superstep.index:
                    return False
                elif (target_ss_idx == succ.superstep.index and 
                      target_proc != succ.proc):
                    return False
        
        return True


class BSPSimulatedAnnealing:
    """Simulated Annealing optimizer for BSP schedules using action generators"""
    
    def __init__(self,
                 max_iterations: int = 1000,
                 max_temp: float = 100.0,
                 min_temp: float = 0.1,
                 cooling_rate: float = 0.99,
                 action_generators: Optional[List[ActionGenerator]] = None):
        """Initialize simulated annealing optimizer
        
        Args:
            max_iterations: Maximum number of iterations
            max_temp: Initial temperature
            min_temp: Minimum temperature (stopping condition)
            cooling_rate: Temperature cooling factor (0 < rate < 1)
            action_generators: List of action generators to use
        """
        self.max_iterations = max_iterations
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        
        if action_generators is None:
            self.action_generators = [
                LoadBalancingGenerator(),
                SuperstepMergeGenerator(),
                CriticalPathOptimizer()
            ]
        else:
            self.action_generators = action_generators
        
        self.iterations: List[SimulatedAnnealingIteration] = []
    
    def get_energy(self, schedule: BSPSchedule) -> float:
        """Calculate energy (cost) of a schedule - using makespan"""
        return schedule.makespan
    
    def generate_neighbor(self, schedule: BSPSchedule, temperature: float) -> Optional[Tuple[BSPSchedule, Action]]:
        """Generate a neighbor schedule by applying an action"""
        # Collect all possible actions from generators
        all_actions = []
        
        for generator in self.action_generators:
            try:
                actions = generator.generate_actions(schedule, max_actions=20)
                all_actions.extend(actions)
                logger.debug(f"{generator.name()} generated {len(actions)} actions")
                for action in actions:
                    logger.debug(f"  - {action.describe()}")
            except Exception as e:
                logger.warning(f"Error in {generator.name()}: {e}")
        
        if not all_actions:
            logger.debug("No actions generated")
            return None, None
        
        # Select action (could be weighted by makespan improvement or random)
        action = self.select_action(all_actions, temperature, schedule.makespan)
        
        # Apply action to create neighbor
        neighbor = schedule.copy()
        
        try:
            if action.apply(neighbor):
                # Validate the neighbor schedule
                is_valid, errors = neighbor.is_valid()
                if is_valid:
                    return neighbor, action
                else:
                    logger.debug(f"Action created invalid schedule: {errors}")
            else:
                logger.debug(f"Action failed to apply: {action}")
        except Exception as e:
            logger.debug(f"Error applying action: {e}")
        
        return None, None
    
    def select_action(self, actions: List[Action], temperature: float, current_makespan: float) -> Action:
        """Select an action from the list based on makespan improvement"""
        if not actions:
            return None
        
        # At high temperature, more random selection
        # At low temperature, prefer actions with better makespan
        if temperature > self.max_temp * 0.5:
            # High temperature - random selection
            return random.choice(actions)
        else:
            # Lower temperature - weighted by makespan improvement
            improvements = []
            for action in actions:
                new_makespan = getattr(action, 'new_makespan', current_makespan)
                improvement = current_makespan - new_makespan  # Positive = better
                improvements.append(max(0.01, improvement + 1.0))  # Shift to ensure positive weights
            
            total_weight = sum(improvements)
            
            if total_weight == 0:
                return random.choice(actions)
            
            # Random weighted selection
            r = random.random() * total_weight
            cumsum = 0
            for action, weight in zip(actions, improvements):
                cumsum += weight
                if r <= cumsum:
                    return action
            
            return actions[-1]
    
    def optimize(self, initial_schedule: BSPSchedule) -> BSPSchedule:
        """Run simulated annealing optimization on the given schedule"""
        logger.info("Starting BSP simulated annealing optimization (v2)")
        
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
            neighbor_schedule, action = self.generate_neighbor(current_schedule, temp)
            
            if neighbor_schedule is None:
                # Could not find feasible neighbor
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
                    logger.info(f"{log_prefix} New best: {best_energy:.2f} via {action.describe()}")
                
                logger.debug(f"{log_prefix} Accepted: {action.describe()}")
            else:
                logger.debug(f"{log_prefix} Rejected: {action.describe()}")
            
            # Cool down
            temp *= self.cooling_rate
            iteration += 1
        
        logger.info(f"Optimization completed. Best energy: {best_energy:.2f}, Initial energy: {self.get_energy(initial_schedule):.2f}")
        
        # Final validation
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
    
    def draw_energy_history(self, title="Simulated Annealing Energy History"):
        """Draw the energy history during optimization"""
        import matplotlib.pyplot as plt
        
        if not self.iterations:
            logger.warning("No optimization data available to plot")
            return
        
        # Extract data
        iterations = [it.iteration for it in self.iterations]
        current_energies = [it.current_energy for it in self.iterations]
        best_energies = [it.best_energy for it in self.iterations]
        temperatures = [it.temperature for it in self.iterations]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot energy
        ax1.plot(iterations, current_energies, 'b-', alpha=0.7, label='Current Energy')
        ax1.plot(iterations, best_energies, 'r-', linewidth=2, label='Best Energy')
        ax1.set_ylabel('Energy (Makespan)')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot temperature
        ax2.plot(iterations, temperatures, 'g-', alpha=0.8, label='Temperature')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Temperature')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig