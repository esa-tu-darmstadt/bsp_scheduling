# Barrier-Aware Task Scheduling for Bulk-Synchronous Parallel Architectures

Code accompanying the ICS'26 paper

> **Task Scheduling for Bulk-Synchronous Parallel Architectures**
> Tim Noack and Andreas Koch
> TU Darmstadt
> Proceedings of the 2026 International Conference on Supercomputing (ICS '26)

The repo contains the BALS scheduler introduced in the paper, reimplementations of the baseline and prior-work schedulers it is compared against, and the benchmark harness that produces the figures and tables in the evaluation.

## Install

```bash
git clone <repo-url> bsp_scheduling
cd bsp_scheduling
python -m venv .venv && source .venv/bin/activate
pip install -e '.[benchmark,dev]'
```

The core install pulls saga (async scheduling primitives), wfcommons (workflow DAG generation), pycapnp (SPN parsing), and matplotlib. The `benchmark` extra adds `rich`, `pandas`, and `seaborn` for the experiment scripts; `dev` adds `pytest` and `pytest-cov`.

## Reproducing the paper results

All figures and the summary table in the paper come from `scripts/experiments/benchmark2`. End-to-end reproduction:

```bash
cd scripts/experiments/benchmark2
python main.py --num-jobs=32 --max-instances=0
```

- `--max-instances=0` runs every dataset instance (the paper uses the full set; the default cap is 5 per dataset).
- `--num-jobs=32` runs schedulers in parallel; lower this if you have fewer cores.

First-time runs spend some time generating workflow and SPN datasets under `data/`; subsequent runs reuse the cache. The benchmark itself takes around 7 minutes for on our 2x64 core CPU.

Outputs appear under `scripts/experiments/benchmark2/visualizations/`:

| path                                               | contents                                                |
| -------------------------------------------------- | ------------------------------------------------------- |
| `boxplots/boxplot_aggregated.{png,pdf}`            | aggregated makespan-ratio boxplot across all datasets   |
| `boxplots/boxplot_combined.{png,pdf}`              | per-dataset boxplots in one figure                      |
| `heatmaps/heatmap_makespan_ratio.{png,pdf}`        | per-dataset / per-scheduler heatmap                     |
| `tables/summary_table.tex`, `tables/summary_stats.tex` | LaTeX table of mean / median / stddev and `\newcommand` stats |

For per-option details see `scripts/experiments/benchmark2/README.md`.

## Repository layout

```
src/bsp_scheduling/
  schedulers/bals.py           BALS (introduced in the paper)
  schedulers/bcsh.py           BCSH baseline
  schedulers/hdagg.py          HDagg baseline
  schedulers/papp/             Papp et al. 2024 (BSPg, Source, Multilevel)
  schedulers/delaymodel/       HEFT variants
  conversion/                  async-schedule -> BSP conversion strategies
  optimization/                superstep elimination, hill climbing, ILP stubs
  task_graphs/                 WfCommons and SPN task-graph generators
  utils/visualization.py       Gantt-chart helpers

scripts/experiments/benchmark2/  paper benchmark harness
scripts/playground.py            small standalone example

tests/                           pytest suite
```

## Tests

```bash
pytest
```

73 tests are expected to pass, with 4 strict `xfail` markers tracking the ILP and HCcs stubs (see `tests/test_papp_schedulers.py`).

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{noack2026bals,
  author    = {Noack, Tim and Koch, Andreas},
  title     = {Barrier-Aware Task Scheduling for Bulk-Synchronous Parallel Architectures},
  booktitle = {Proceedings of the 2026 International Conference on Supercomputing},
  series    = {ICS '26},
  year      = {2026},
  location  = {Belfast, Northern Ireland},
  publisher = {ACM},
}
```

## License

MIT.