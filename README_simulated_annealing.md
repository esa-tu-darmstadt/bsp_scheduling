# BSP Simulated Annealing Optimization

This document describes the simulated annealing optimization system for BSP schedules.

## Overview

The simulated annealing system optimizes BSP schedules by iteratively applying random modifications and accepting or rejecting them based on energy changes and temperature. The goal is to minimize the makespan (total execution time) of the BSP schedule.

## Key Components

### 1. Actions (Schedule Modifications)

The system provides two built-in actions:

- **MoveTaskToSuperstep**: Moves a task to a different superstep while maintaining precedence constraints
- **MoveTaskToProcessor**: Moves a task to a different processor within the same superstep

### 2. Extensible Framework

New actions can be easily added by inheriting from `ScheduleAction`:

```python
from saga_bsp.optimization import ScheduleAction

class CustomAction(ScheduleAction):
    def is_feasible(self, schedule, task_node):
        # Check if action is feasible
        pass
    
    def apply(self, schedule, task_node):
        # Apply the action
        pass
    
    def get_possible_targets(self, schedule, task_node):
        # Return possible targets for this action
        pass
    
    def __str__(self):
        return "CustomAction()"
```

### 3. Simulated Annealing Algorithm

The `BSPSimulatedAnnealing` class implements the core optimization logic:

- Temperature cooling schedule
- Energy-based acceptance probability
- Random neighbor generation
- Optimization statistics tracking

## Usage

### Basic Usage

```python
from saga_bsp.schedule import BSPSchedule, BSPHardware
from saga_bsp.optimization import BSPSimulatedAnnealing
import networkx as nx

# Create hardware and task graph
hardware = BSPHardware(network=network, sync_time=1.0)
# ... create task_graph and initial_schedule ...

# Create optimizer
optimizer = BSPSimulatedAnnealing(
    max_iterations=1000,
    max_temp=100.0,
    min_temp=0.1,
    cooling_rate=0.99
)

# Run optimization
optimized_schedule = optimizer.optimize(initial_schedule)

# Get statistics
stats = optimizer.get_optimization_stats()
print(f"Improvement: {stats['improvement_percent']:.1f}%")
```

### Custom Actions

```python
from saga_bsp.optimization import BSPSimulatedAnnealing, MoveTaskToSuperstep

# Use custom action types
optimizer = BSPSimulatedAnnealing(
    action_types=[MoveTaskToSuperstep],  # Only use superstep moves
    max_iterations=500
)
```

## Implementation Details

### Precedence Constraints

The system ensures that BSP precedence constraints are maintained:

1. Tasks can only be moved to supersteps where all their predecessors are in earlier supersteps
2. Tasks can only be moved to supersteps where all their successors are in later supersteps
3. Tasks within the same superstep cannot have dependencies between them

### Energy Function

The energy function uses the schedule's makespan:
- Lower makespan = lower energy = better solution
- The algorithm minimizes energy (makespan)

### Temperature Schedule

- Initial temperature: `max_temp`
- Cooling: `temp *= cooling_rate` each iteration  
- Stopping condition: `temp < min_temp` or `iteration >= max_iterations`

### Acceptance Probability

- Better solutions (lower energy): Always accepted
- Worse solutions: Accepted with probability `exp(-energy_diff/temp)`

## Test Results

The test shows the system working correctly:

- **Individual Actions**: Both `MoveTaskToSuperstep` and `MoveTaskToProcessor` work as expected with proper feasibility checking
- **Full Optimization**: The simulated annealing algorithm runs successfully with proper statistics tracking
- **Extensibility**: Custom actions can be easily created and integrated
- **Constraint Validation**: Precedence constraints are properly enforced

## Files Created

1. `src/saga_bsp/optimization/__init__.py` - Package initialization
2. `src/saga_bsp/optimization/simulated_annealing.py` - Main implementation
3. `test_simulated_annealing.py` - Comprehensive test suite
4. `README_simulated_annealing.md` - This documentation

The implementation is fully functional and extensible, allowing for easy addition of new optimization actions while maintaining BSP scheduling constraints.