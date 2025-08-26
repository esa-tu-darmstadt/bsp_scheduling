# SAGA-BSP: Bulk Synchronous Parallel Scheduling Framework

SAGA-BSP extends [SAGA (Scheduling Algorithms Gathered)](https://github.com/saga-scheduling/saga) to support **Bulk Synchronous Parallel (BSP)** scheduling, enabling efficient mapping of task graphs onto parallel processors that operate in synchronized phases.

## Installation

```bash
# Clone the repository
git clone ...
cd saga-bsp

# Install in development mode (installs all core dependencies)
pip install -e .

# Optional: Install development dependencies
pip install -e ".[dev]"

# Optional: Install visualization dependencies
pip install -e ".[visualization]"
```

## Quick Start

```python
import networkx as nx
from saga_bsp.schedule import BSPSchedule, BSPHardware
from saga_bsp.schedulers import HeftBSPScheduler

# Create hardware configuration
network = nx.complete_graph(4)  # 4 fully-connected processors
hardware = BSPHardware(network=network, sync_time=10.0)

# Load or create your task graph
task_graph = nx.DiGraph()
# ... add tasks and dependencies ...

# Schedule using List-BSP
scheduler = ListBSPScheduler(verbose=True)
schedule = scheduler.schedule(hardware, task_graph)

# Visualize the result
from saga_bsp.visualization import plot_gantt_chart
plot_gantt_chart(schedule)

import matplotlib.pyplot as plt
plt.show()
```

## Features

### Core Components

- **BSP Schedule Management**: Hierarchical schedule representation with supersteps
- **Multiple Scheduling Algorithms**: HEFT-BSP, List-based, and conversion from async schedules
- **Schedule Optimization**: Simulated annealing with configurable actions
- **Visualization Tools**: Gantt charts, heatmaps, and superstep breakdowns
- **SAGA Integration**: Seamless conversion between BSP and async schedules

### Scheduling Algorithms

#### List-Based BSP Scheduler (WIP)
HEFT-like scheduler that can split supersteps or append tasks to existing ones:
```python
from saga_bsp.schedulers import ListBSPScheduler

scheduler = ListBSPScheduler(
    verbose=True,  # Enable detailed logging
    draw_after_each_step=False  # Optionally visualize after each task placement
)
schedule = scheduler.schedule(hardware, task_graph)
```

The scheduler evaluates multiple placement strategies for each task:
- Appending to existing supersteps
- Creating new supersteps at dependency-ready times
- Automatically chooses the strategy that minimizes makespan

#### HEFT-BSP Scheduler (Experimental, non-working)
Experimental adaptation of the Heterogeneous Earliest Finish Time algorithm for BSP:
```python
from saga_bsp.schedulers import HeftBSPScheduler

scheduler = HeftBSPScheduler(verbose=True)
schedule = scheduler.schedule(hardware, task_graph)  # Note: Still under development
```

#### Async to BSP Conversion
Convert existing asynchronous schedules to BSP with multiple strategies:
```python
from saga_bsp.conversion import convert_async_to_bsp

bsp_schedule = convert_async_to_bsp(
    hardware, 
    task_graph,
    async_schedule,
    strategy="earliest-finishing-next",  # or "eager", "level-based"
    backfill_threshold_percent=0.05,
    optimize_sa=True  # Apply simulated annealing optimization
)
```

### Schedule Optimization

Optimize existing schedules using simulated annealing:
```python
from saga_bsp.optimization import SimulatedAnnealingV2

optimizer = SimulatedAnnealingV2(
    initial_temp=100,
    cooling_rate=0.95,
    min_temp=0.1,
    max_iterations_without_improvement=500
)

optimized_schedule = optimizer.optimize(schedule, task_graph, hardware)
```

### Visualization

Generate various visualizations to analyze schedule performance:
```python
from saga_bsp.visualization import (
    plot_gantt_chart,
    plot_processor_utilization_heatmap,
    plot_superstep_breakdown
)

# Gantt chart with superstep boundaries
plot_gantt_chart(schedule)

# Processor utilization heatmap
plot_processor_utilization_heatmap(schedule)

# Superstep timing breakdown
plot_superstep_breakdown(schedule)
```

## BSP Execution Model

BSP scheduling divides computation into synchronized **supersteps**, each consisting of three phases:

1. **Synchronization Phase**: All processors synchronize at a barrier
2. **Exchange Phase**: Processors exchange data needed for upcoming computations  
3. **Computation Phase**: Processors execute assigned tasks independently

Key constraints:
- Tasks can only communicate across processors between supersteps
- Tasks on the same processor within a superstep can share data directly
- Superstep duration is determined by the slowest processor

## API Reference

### BSPHardware
```python
BSPHardware(network: nx.Graph, sync_time: float)
```
Defines the hardware configuration with processor network topology and synchronization cost.

### BSPSchedule
```python
BSPSchedule(hardware: BSPHardware, task_graph: nx.DiGraph)
```
Main schedule container managing supersteps and task assignments.

Key methods:
- `add_superstep()`: Create a new superstep
- `get_task(task_name)`: Retrieve a scheduled task
- `validate()`: Check schedule validity
- `makespan()`: Calculate total execution time

### Superstep
Container for tasks executing in parallel within a BSP phase.

Key methods:
- `schedule_task(task_name, processor)`: Assign task to processor
- `get_tasks_on_processor(processor)`: Get tasks for specific processor
- `exchange_time()`: Calculate data exchange duration
- `computation_time()`: Get maximum computation across processors

## Examples

### Converting SAGA Async Schedule
```python
from saga import Task, Hardware
from saga.schedulers import HEFTScheduler
from saga_bsp.conversion import convert_async_to_bsp
from saga_bsp.schedule import BSPHardware

# Create async schedule using SAGA
saga_hw = Hardware(num_processors=4)
saga_scheduler = HEFTScheduler()
async_schedule = saga_scheduler.schedule(task_graph, saga_hw)

# Convert to BSP
bsp_hw = BSPHardware(network=nx.complete_graph(4), sync_time=10.0)
bsp_schedule = convert_async_to_bsp(
    bsp_hw,
    task_graph, 
    async_schedule,
    strategy="earliest-finishing-next",
    optimize_sa=True
)
```

### Custom Task Graph
```python
import networkx as nx
from saga import Task

# Create task graph
G = nx.DiGraph()

# Add tasks with computation costs
tasks = {
    "A": Task("A", flops=100),
    "B": Task("B", flops=150),
    "C": Task("C", flops=200),
    "D": Task("D", flops=120)
}

for name, task in tasks.items():
    G.add_node(name, weight=task.flops)

# Add dependencies with communication costs
G.add_edge("A", "B", weight=50)  # 50 bytes of data
G.add_edge("A", "C", weight=30)
G.add_edge("B", "D", weight=40)
G.add_edge("C", "D", weight=60)

# Schedule
scheduler = HeftBSPScheduler()
schedule = scheduler.schedule(hardware, G)
```

## Testing

Run the test suite:
```bash
# All tests
pytest

# Specific test file
pytest tests/test_schedule.py

# With coverage
pytest --cov=saga_bsp
```

<!-- ## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request -->

<!-- ## License

This project is licensed under the MIT License - see the LICENSE file for details. -->

<!-- ## Citation

If you use SAGA-BSP in your research, please cite:
```bibtex
@software{saga_bsp,
  title = {SAGA-BSP: Bulk Synchronous Parallel Scheduling Framework},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/saga-bsp}
}
``` -->

<!-- ## Acknowledgments

- Built on top of [SAGA](https://github.com/saga-scheduling/saga)
- Inspired by BSP model research in parallel computing
- Special thanks to contributors and maintainers -->