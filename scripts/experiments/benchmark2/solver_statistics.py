"""
Solver statistics reporting for BSP benchmarking.

This module aggregates per-solver metrics across all datasets and writes
human‑readable summaries to a text file.
"""

from __future__ import annotations

import pathlib
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from benchmark_runner import BenchmarkRunner
from schedulers import (
    get_scheduler_display_name,
    get_ordered_scheduler_names,
)


def _format_float(x: float | int | None) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "-"
    if isinstance(x, int):
        return str(x)
    return f"{x:.3f}"


def _compute_series_stats(series: pd.Series) -> Dict[str, float]:
    if series.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std(ddof=1)),
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "p75": float(series.quantile(0.75)),
        "max": float(series.max()),
    }


def write_solver_statistics(resultsdir: pathlib.Path, outpath: pathlib.Path) -> None:
    """Compute and write per‑solver makespan ratio and runtime statistics.

    Produces two sections:
    1) Overall (all runs) — each row is a run; datasets with more runs have more weight.
    2) Dataset‑averaged — median per dataset, then stats across datasets (equal weight per dataset).
    3) Scheduler Runtime Statistics — min/max/median/mean for runtime across all runs.
    """
    runner = BenchmarkRunner(schedulers={})
    df = runner.load_results(resultsdir)

    outpath.parent.mkdir(parents=True, exist_ok=True)

    with outpath.open("w", encoding="utf-8") as f:
        f.write("BSP Scheduling Benchmark — Solver Statistics\n")
        f.write(f"Results directory: {resultsdir}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        if df.empty:
            f.write("No results found.\n")
            return

        # Ensure expected columns
        required_cols = {"scheduler", "dataset", "makespan_ratio"}
        missing = required_cols - set(df.columns)
        if missing:
            f.write(f"Missing required columns in results: {sorted(missing)}\n")
            return

        # Order schedulers for stable reporting
        schedulers = df["scheduler"].unique().tolist()
        ordered = get_ordered_scheduler_names(schedulers)

        datasets = sorted(df["dataset"].unique().tolist())
        f.write(f"Datasets: {len(datasets)} | Schedulers: {len(ordered)} | Total runs: {len(df)}\n\n")

        # Section 1: Overall (all runs)
        f.write("[Overall — All Runs]\n")
        for name in ordered:
            disp = get_scheduler_display_name(name)
            series = df.loc[df["scheduler"] == name, "makespan_ratio"]
            stats = _compute_series_stats(series)

            f.write(f"- {disp} ({name})\n")
            f.write(f"  count:  {_format_float(stats['count'])}\n")
            f.write(f"  mean:   {_format_float(stats['mean'])}\n")
            f.write(f"  median: {_format_float(stats['median'])}\n")
            f.write(f"  std:    {_format_float(stats['std'])}\n")
            f.write(f"  min:    {_format_float(stats['min'])}\n")
            f.write(f"  p25:    {_format_float(stats['p25'])}\n")
            f.write(f"  p75:    {_format_float(stats['p75'])}\n")
            f.write(f"  max:    {_format_float(stats['max'])}\n")
        f.write("\n")

        # Section 2: Dataset-averaged (equal weight per dataset)
        f.write("[Dataset‑Averaged — Median per Dataset]\n")
        for name in ordered:
            disp = get_scheduler_display_name(name)
            # Median within each dataset for the solver, then aggregate across datasets
            per_ds = (
                df.loc[df["scheduler"] == name]
                .groupby("dataset")["makespan_ratio"]
                .median()
            )
            stats = _compute_series_stats(per_ds)

            f.write(f"- {disp} ({name})\n")
            f.write(f"  datasets: {_format_float(stats['count'])}\n")
            f.write(f"  mean:     {_format_float(stats['mean'])}\n")
            f.write(f"  median:   {_format_float(stats['median'])}\n")
            f.write(f"  std:      {_format_float(stats['std'])}\n")
            f.write(f"  min:      {_format_float(stats['min'])}\n")
            f.write(f"  p25:      {_format_float(stats['p25'])}\n")
            f.write(f"  p75:      {_format_float(stats['p75'])}\n")
            f.write(f"  max:      {_format_float(stats['max'])}\n")
        f.write("\n")

        # Section 3: Scheduler Runtime Statistics
        if "scheduler_runtime_s" in df.columns:
            f.write("[Scheduler Runtime Statistics (seconds)]\n")
            for name in ordered:
                disp = get_scheduler_display_name(name)
                runtime_series = df.loc[df["scheduler"] == name, "scheduler_runtime_s"]
                runtime_stats = _compute_series_stats(runtime_series)

                f.write(f"- {disp} ({name})\n")
                f.write(f"  count:  {_format_float(runtime_stats['count'])}\n")
                f.write(f"  mean:   {_format_float(runtime_stats['mean'])}\n")
                f.write(f"  median: {_format_float(runtime_stats['median'])}\n")
                f.write(f"  std:    {_format_float(runtime_stats['std'])}\n")
                f.write(f"  min:    {_format_float(runtime_stats['min'])}\n")
                f.write(f"  p25:    {_format_float(runtime_stats['p25'])}\n")
                f.write(f"  p75:    {_format_float(runtime_stats['p75'])}\n")
                f.write(f"  max:    {_format_float(runtime_stats['max'])}\n")
        else:
            f.write("[Scheduler Runtime Statistics]\n")
            f.write("No scheduler runtime data available.\n")

        # Done

