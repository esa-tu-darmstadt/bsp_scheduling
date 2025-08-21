import logging
import pathlib
from saga.pisa import run_experiments
import saga_bsp as bsp
from saga.schedulers import (BILScheduler, CpopScheduler, DuplexScheduler,
                             ETFScheduler, FastestNodeScheduler, FCPScheduler,
                             FLBScheduler, GDLScheduler, HeftScheduler,
                             MaxMinScheduler, MCTScheduler, METScheduler,
                             MinMinScheduler, OLBScheduler, WBAScheduler)

thisdir = pathlib.Path(__file__).parent

logging.basicConfig(level=logging.INFO)


def run(output_path: pathlib.Path, sync_time: int):
    """Run first set of experiments."""
    scheduler_pairs = [
        (("HEFT-BSP-EarliestNext", bsp.SagaSchedulerWrapper(bsp.AsyncToBSPScheduler(HeftScheduler(), 'earliest-finishing-next'), sync_time=sync_time)), 
         ("HeftScheduler", HeftScheduler())),
        (("HEFT-BSP-Eager", bsp.SagaSchedulerWrapper(bsp.AsyncToBSPScheduler(HeftScheduler(), 'eager'), sync_time=sync_time)), 
         ("HeftScheduler", HeftScheduler()))
    ]
    
    run_experiments(
        scheduler_pairs=scheduler_pairs,
        max_iterations=1000,
        num_tries=10,
        max_temp=10,
        min_temp=0.1,
        cooling_rate=0.99,
        skip_existing=False,
        output_path=output_path
    )


def main():
    logging.basicConfig(level=logging.INFO)

    resultsdir = thisdir.joinpath("results")
    sync_time = 0

    run(resultsdir, sync_time)

    # Visualize results
    from postprocessing import visualize_results
    visualize_results(resultsdir, sync_time)

if __name__ == "__main__":
    main()