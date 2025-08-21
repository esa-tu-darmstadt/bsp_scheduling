# Saga BSP Extension

Bulk Synchronous Parallel (BSP) scheduling extension for the Saga framework.

## Overview

This package extends the Saga scheduling framework to support BSP (Bulk Synchronous Parallel) architectures, particularly those used in specialized hardware like Graphcore's IPU (Intelligence Processing Unit).

## BSP Model

The BSP model organizes computation into supersteps, each containing three phases:
1. **Sync**: Global synchronization barrier
2. **Exchange**: Inter-processor data communication  
3. **Compute**: Local computation on each processor

This implementation follows the IPU's BSP execution model where all processors execute in lockstep through these phases.

## Prerequisites

We recommend using a virtual environment for this package.

## Installation

```bash
git clone <this-repo>
cd saga-bsp
pip install -e .
```

For development with testing dependencies:
```bash
pip install -e .[dev]
```

For visualization dependencies:
```bash
pip install -e .[vis]
```
