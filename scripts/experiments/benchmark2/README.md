# Benchmark2: Independent BSP Scheduling Benchmark

This is a complete rewrite of the BSP scheduling benchmark system, designed to be completely independent from SAGA's infrastructure. It uses WfCommons directly for workflow generation and implements the full IPU hardware model.

## Features

- **WfCommons Integration**: Direct workflow generation using WfCommons recipes
- **IPU Hardware Model**: Complete implementation of IPU topology with proper tile-to-tile connections
- **Task Graph Caching**: Efficient caching system with metadata for CCR adaptation
- **Scheduler Support**: All current BSP schedulers plus the async HeftBusyCommScheduler
- **Advanced Visualizations**: Box plots and heatmaps with special delay model visualization
- **Modular Design**: Split into focused modules for maintainability

## Architecture

```
benchmark2/
├── main.py                 # Main benchmark script with CLI
├── dataset_generator.py    # WfCommons-based dataset generation with caching
├── hardware_ipu.py        # Complete IPU hardware implementation
├── schedulers.py          # Scheduler configuration and management
├── benchmark_runner.py    # Benchmark execution and result management
├── visualizations.py      # Box plot and heatmap generators
└── README.md             # This file
```

## Hardware Model

The IPU hardware model implements the complete topology:

- **4 tiles per island** - connected at 128 bit/cycle
- **16 islands per column** - connected at 64 bit/cycle
- **23 columns per IPU** - connected at 32 bit/cycle
- **Multiple IPUs** - connected at 8 bit/cycle

**Key Feature**: ALL tiles are connected to ALL other tiles, with connection speeds determined by their hierarchical relationship.

## Dataset Generation

Uses WfCommons recipes directly:
- `blast`, `bwa`, `cycles`, `epigenomics`, `genome`, `montage`, `seismology`, `soykb`, `srasearch`
- 50 variations per recipe with random task counts
- Task graphs cached with metadata (avg task weight, avg edge weight) for CCR adaptation
- Networks NOT cached (generated fresh for each benchmark run)

## Schedulers

Currently enabled schedulers matching the existing benchmark:

### Async/Delay Model
- **HeftBusyCommScheduler**: Async scheduler with busy communication

### BSP Models
- **HEFT-BSP-EarliestNext**: HEFT converted with earliest-finishing-next strategy
- **HEFT-BSP-Eager**: HEFT converted with eager strategy
- **BALSScheduler-HEFT**: Native BSP scheduler with HEFT priority
- **BALSScheduler-CPoP**: Native BSP scheduler with CPoP priority
- **BCSHScheduler-NoEFT**: BCSH scheduler without EFT
- **BCSHScheduler-EFT**: BCSH scheduler with EFT

## Visualizations

### Box Plots
- Per-dataset makespan distributions
- Combined multi-dataset view
- Special marking for delay model schedulers (red borders)

### Heatmaps
- Makespan and makespan ratio heatmaps
- Scheduler reordering (delay model first)
- Visual separator between delay and BSP models
- Color-coded annotations

The heatmap places HeftBusyCommScheduler first with a visual separator (red/black line) to clearly distinguish it as the delay/async model from the BSP models.

## Usage

### Basic Usage
```bash
python main.py
```

### Advanced Options
```bash
# Run with more parallel jobs and overwrite existing results
python main.py --num-jobs 4 --overwrite

# Benchmark specific datasets and schedulers
python main.py --datasets epigenomics montage --schedulers HeftBusyCommScheduler HEFT-BSP-EarliestNext

# Limit to first 10 instances per dataset (default is 5)
python main.py --max-instances 10

# Run full benchmark for paper (all instances, may take a long time)
python main.py --max-instances 0

# Only generate visualizations from existing results
python main.py --skip-generation --skip-benchmarking
```

### Step-by-Step
```bash
# Generate datasets only
python main.py --skip-benchmarking

# Run benchmarks only (using cached data)
python main.py --skip-generation

# Generate visualizations from existing results
python main.py --skip-generation --skip-benchmarking
```

## Dependencies

- `wfcommons`: Workflow generation
- `saga_bsp`: BSP scheduling framework
- `networkx`: Graph operations
- `pandas`, `numpy`: Data processing
- `matplotlib`, `seaborn`: Visualizations
- `pickle`: Caching

## Output Structure

```
benchmark2/
├── data/                   # Cached datasets
│   ├── blast_cached.pkl
│   ├── blast_metadata.json
│   └── ...
├── results/                # Benchmark results
│   ├── blast.csv
│   ├── montage.csv
│   └── ...
└── output/
    ├── boxplots/          # Box plot visualizations
    └── heatmaps/          # Heatmap visualizations
```

## Key Implementation Details

1. **Fully Connected IPU Model**: Every tile connects to every other tile with appropriate speeds
2. **WfCommons Direct Integration**: No SAGA dependency for workflow generation
3. **Metadata Caching**: Enables easy CCR adaptation without regenerating graphs
4. **Special Visualization**: Clear distinction between delay and BSP models in heatmaps
5. **Modular Design**: Easy to extend with new schedulers or visualizations