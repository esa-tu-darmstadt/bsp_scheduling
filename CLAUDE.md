# CLAUDE.md - BSP Scheduling Framework Documentation

## Overview

The bsp_scheduling framework extends SAGA (Scheduling Algorithms Gathered) to support **Bulk Synchronous Parallel (BSP)** scheduling, a fundamentally different paradigm from the asynchronous scheduling implemented in the core SAGA library.

## The BSP Scheduling Problem

### Basic Problem
BSP scheduling addresses the challenge of mapping computational tasks (represented as a DAG) onto parallel processors that operate in synchronized phases called **supersteps**. Unlike asynchronous scheduling where tasks can execute as soon as dependencies are satisfied, BSP enforces a rigid structure of alternating computation and communication phases.

### BSP Execution Model
Each superstep consists of three phases executed in strict order:

1. **Synchronization Phase**: All processors synchronize (barrier synchronization)
2. **Exchange Phase**: Processors exchange data needed for upcoming computations
3. **Computation Phase**: Processors execute assigned tasks independently

Tasks can only communicate across processor boundaries between supersteps, not during computation phases.

## Key Differences from Async Scheduling

### Asynchronous Scheduling (SAGA)
- **Execution**: Tasks execute as soon as dependencies are satisfied
- **Communication**: Can happen at any time between dependent tasks
- **Synchronization**: No global barriers; processors work independently
- **Data Structure**: Simple mapping of processors to task lists with start/end times
- **Optimization Goal**: Minimize total makespan through efficient task placement

### BSP Scheduling (bsp_scheduling)
- **Execution**: Tasks grouped into synchronized supersteps
- **Communication**: Only between supersteps during exchange phase
- **Synchronization**: Global barriers between supersteps with associated cost
- **Data Structure**: Hierarchical schedule with supersteps containing per-processor task lists
- **Optimization Goal**: Balance computation within supersteps while minimizing synchronization overhead

## Core Data Structures

### BSPHardware
Encapsulates the hardware configuration for BSP execution:
```python
@dataclass
class BSPHardware:
    network: nx.Graph  # Processor network topology
    sync_time: float   # Cost of synchronization barrier
```

### BSPTask
Extended task representation with BSP-specific timing:
- Links to parent superstep and schedule
- Tracks relative timing within superstep computation phase
- Maintains absolute timing including sync/exchange overhead

### Superstep
Container for tasks executing in parallel:
- Manages tasks across multiple processors
- Calculates phase timings (sync, exchange, compute)
- Enforces BSP communication constraints
- Exchange time determined by inter-processor data dependencies

### BSPSchedule
Complete BSP schedule representation:
- List of supersteps
- Task mapping supporting duplication
- Validation of BSP constraints
- Methods for schedule manipulation (splitting supersteps, etc.)

### AsyncSchedule
Wrapper around SAGA's async schedule format for easier conversion:
- Implements MutableMapping interface
- Provides structured access to processor→task mappings

## Challenges in BSP Scheduling

### 1. Synchronization Overhead
- Global barriers add significant overhead
- Must balance computation distribution to minimize idle time
- Trade-off between parallelism and synchronization costs

### 2. Dependency Constraints
- Tasks can only use data from previous supersteps
- Exception: Same-processor tasks in same superstep can share data directly
- Requires careful superstep assignment to respect dependencies

### 3. Load Balancing
- Superstep duration determined by slowest processor
- Uneven load distribution causes processor idle time
- Must consider both computation and communication costs

### 4. Exchange Time Calculation
- Communication only happens between supersteps
- Multiple dependencies may require different exchange times
- Network topology affects communication costs

### 5. Schedule Optimization
- NP-hard problem with multiple objectives
- Minimize number of supersteps vs. minimize makespan
- Task duplication can reduce communication but increases computation

## Implementation Status

### ✅ Completed Components

#### Core Infrastructure
- **BSPSchedule**: Full implementation with superstep management
- **BSPTask**: Complete with timing calculations
- **Superstep**: Supports all BSP phases and timing
- **BSPHardware**: Network and synchronization modeling

#### Schedulers
- **HeftBSPScheduler**: BSP adaptation of HEFT algorithm
- **AsyncToBSPScheduler**: Converts async schedules to BSP with multiple strategies:
  - Eager: Maximize tasks per superstep
  - Level-based: One task per processor per superstep  
  - Earliest-finishing-next: Greedy selection with optional backfilling

#### Conversion Utilities
- **async_to_bsp**: Three conversion strategies with optional SA optimization
- **bsp_to_async**: Export BSP schedules to SAGA format

#### Optimization
- **Simulated Annealing**: Two implementations (v1 and v2) for schedule optimization
- Actions: Task movement, swapping, superstep merging/splitting

#### Utilities
- **Visualization**: Gantt charts, heatmaps, superstep breakdowns
- **Validation**: Schedule constraint checking

### 🚧 Partial/Experimental
- **GraphCore Hardware**: Specific hardware modeling for IPU architectures
- **SAGA Wrapper**: Integration with existing SAGA schedulers

### ❌ Not Implemented
- **MSA Scheduler**: Ignore

## Key Algorithms and Strategies

### Conversion Strategies

#### 1. Eager Strategy
- Schedules maximum tasks per superstep
- Respects inter-processor dependencies
- Tends to create fewer, larger supersteps

#### 2. Level-Based Strategy  
- One task per processor per superstep
- Creates many small supersteps
- High synchronization overhead

#### 3. Earliest-Finishing-Next Strategy
- Greedy selection of earliest completing task
- Optional backfilling within threshold
- Balances superstep size and count

### Optimization Techniques

#### Simulated Annealing
- Explores schedule space through local modifications
- Actions include task moves, swaps, superstep operations
- Temperature-based acceptance of worse solutions

## Usage Examples

### Creating a BSP Schedule
```python
from bsp_scheduling.schedule import BSPSchedule, BSPHardware

hardware = BSPHardware(network=network_graph, sync_time=10.0)
schedule = BSPSchedule(hardware, task_graph)
superstep = schedule.add_superstep()
superstep.schedule_task("task1", "proc0")
```

### Converting Async to BSP
```python
from bsp_scheduling.conversion import convert_async_to_bsp

bsp_schedule = convert_async_to_bsp(
    hardware, 
    task_graph,
    async_schedule,
    strategy="earliest-finishing-next",
    backfill_threshold_percent=0.05,
    optimize_sa=True
)
```

### Using BSP Schedulers
```python
from bsp_scheduling.schedulers import HeftBSPScheduler

scheduler = HeftBSPScheduler(verbose=True)
schedule = scheduler.schedule(hardware, task_graph)
```

## Performance Considerations

### Makespan Components
1. **Computation Time**: Maximum processor computation per superstep
2. **Synchronization Time**: Fixed cost × number of supersteps
3. **Exchange Time**: Data transfer between processors

### Optimization Targets
- Minimize total makespan