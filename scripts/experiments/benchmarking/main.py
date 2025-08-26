import logging
import pathlib
import sys
from typing import List, Optional

from saga.data import Dataset
from saga.schedulers import (
    HeftScheduler, CpopScheduler, ETFScheduler, 
    MCTScheduler, METScheduler, MaxMinScheduler, MinMinScheduler
)
from saga.scheduler import Scheduler
import saga_bsp as bsp
from saga_bsp.schedulers import ListBSPScheduler

from prepare import load_dataset, prepare_datasets

thisdir = pathlib.Path(__file__).parent.resolve()

class TrimmedDataset(Dataset):
    def __init__(self, dataset: Dataset, max_instances: int):
        super().__init__(dataset.name)
        self.dataset = dataset
        self.max_instances = max_instances

    def __len__(self):
        return min(len(self.dataset), self.max_instances)
    
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        return self.dataset[index]

def get_async_schedulers() -> List[Scheduler]:
    """Get a list of SAGA async schedulers for BSP conversion.
    
    Returns:
        List[Scheduler]: List of SAGA schedulers to convert to BSP
    """
    schedulers = [
        HeftScheduler(),
        CpopScheduler(), 
        # ETFScheduler(),
        # MCTScheduler(),
        # METScheduler(),
        # MaxMinScheduler(),
        # MinMinScheduler()
    ]
    return schedulers


def get_conversion_strategies():
    """Get BSP conversion strategies to evaluate."""
    return [
        ("eager", None),
        # ("level-based", None), 
        ("earliest-finishing-next", None),
        ("earliest-finishing-next", 0.20),  # 20% backfill
    ]


def create_schedulers(sync_time: float = 1.0) -> List[Scheduler]:
    """Create a list of BSP schedulers to evaluate.
    
    Args:
        sync_time: BSP synchronization overhead time
    """
    schedulers = []
    
    # First, we add native async schedulers (without any BSP)
    schedulers.extend([HeftScheduler(), CpopScheduler()])
    
    # Then, we add async-to-BSP conversion schedulers
    for async_scheduler in [HeftScheduler()]:
        schedulers.append(bsp.SagaSchedulerWrapper(bsp.AsyncToBSPScheduler(
                    async_scheduler=async_scheduler,
                    strategy="earliest-finishing-next"
                ), sync_time=sync_time))
        schedulers.append(bsp.SagaSchedulerWrapper(bsp.AsyncToBSPScheduler(
                    async_scheduler=async_scheduler,
                    strategy="eager"
                ), sync_time=sync_time))
        # schedulers.append(bsp.SagaSchedulerWrapper(bsp.AsyncToBSPScheduler(
        #             async_scheduler=async_scheduler,
        #             strategy="earliest-finishing-next",
        #             optimize_sa=True,
        #             sa_max_iterations=50,
        #             sa_cooling_rate=0.95
        #         ), sync_time=sync_time))
    schedulers.append(bsp.SagaSchedulerWrapper(ListBSPScheduler(), sync_time=sync_time, preprocess=True))
        # for strategy, backfill_threshold in get_conversion_strategies():
        #     schedulers.append(
        #         bsp.SagaSchedulerWrapper(bsp.AsyncToBSPScheduler(
        #             async_scheduler=async_scheduler,
        #             strategy=strategy,
        #             backfill_threshold_percent=backfill_threshold
        #         ), sync_time=sync_time)
        #     )
            
    # Finally, we add native BSP schedulers
    # TODO: Add native BSP schedulers if available
   
    return schedulers


def evaluate_dataset(datadir: pathlib.Path,
                     schedulers: List[Scheduler],
                     resultsdir: pathlib.Path,
                     dataset_name: str,
                     sync_time: float = 1.0,
                     max_instances: int = 0,
                     num_jobs: int = 1,
                     overwrite: bool = False):
    """Evaluate a dataset with BSP schedulers.
    
    Args:
        datadir: Directory containing the dataset
        schedulers: List of schedulers to evaluate
        resultsdir: Directory to save results
        dataset_name: Name of the dataset
        sync_time: BSP synchronization overhead time
        max_instances: Maximum instances to evaluate (0 = all)
        num_jobs: Number of parallel jobs
        overwrite: Whether to overwrite existing results
    """
    logging.info("Evaluating BSP dataset %s with sync_time=%.2f", dataset_name, sync_time)
    
    savepath = resultsdir.joinpath(f"{dataset_name}_bsp_sync{sync_time:.1f}.csv")
    if savepath.exists() and not overwrite:
        logging.info("Results already exist. Skipping.")
        return
        
    dataset = load_dataset(datadir, dataset_name)
    if max_instances > 0 and len(dataset) > max_instances:
        dataset = TrimmedDataset(dataset, max_instances)

    logging.info("Running comparison for %d schedulers.", len(schedulers))
    comparison = dataset.compare(schedulers, num_jobs=num_jobs)

    logging.info("Saving results.")
    df_comp = comparison.to_df()
    savepath.parent.mkdir(exist_ok=True, parents=True)
    df_comp.to_csv(savepath)
    logging.info("Saved results to %s.", savepath)


def run_experiment(datadir: pathlib.Path,
                       resultsdir: pathlib.Path,
                       dataset: str = None,
                       sync_time: float = 1.0,
                       num_jobs: int = 1,
                       trim: int = 0,
                       overwrite: bool = False):
    """Run BSP benchmarking experiment.
    
    Args:
        datadir: Directory containing datasets
        resultsdir: Directory to save results
        dataset: Specific dataset name (None = all datasets)
        sync_time: BSP synchronization overhead time
        num_jobs: Number of parallel jobs
        trim: Maximum instances per dataset (0 = no limit)
        mixed_comparison: Include both async and BSP schedulers
        overwrite: Whether to overwrite existing results
    """
    resultsdir.mkdir(parents=True, exist_ok=True)
    
    schedulers = create_schedulers(sync_time=sync_time)
    
    default_datasets = [path.stem for path in datadir.glob("*.json")]
    dataset_names = [dataset] if dataset else default_datasets
    
    for dataset_name in dataset_names:
        evaluate_dataset(
            datadir=datadir,
            resultsdir=resultsdir,
            dataset_name=dataset_name,
            sync_time=sync_time,
            max_instances=trim,
            num_jobs=num_jobs,
            schedulers=schedulers,
            overwrite=overwrite
        )


def main():
    logging.basicConfig(level=logging.INFO)
    
    # Increase recursion limit for large task graphs
    sys.setrecursionlimit(100000) 
    
    # # Apply WFCommons compatibility patch
    # print("Applying WFCommons compatibility patch...")
    # from wfcommons_patch import patch_wfcommons
    # patch_wfcommons()
    
    datadir = thisdir.joinpath("data")
    outputdir = thisdir.joinpath("output")
    resultsdir = thisdir.joinpath("results")
    
    # Prepare datasets using SAGA's infrastructure
    prepare_datasets(savedir=datadir, skip_existing=True)
    
    # Run BSP-only comparison
    run_experiment(
        datadir=datadir, 
        resultsdir=resultsdir, 
        sync_time=1.0,
        num_jobs=10, 
        trim=5,  # Limit to 10 instances for faster testing
        overwrite=False
    )
    
    # Run analysis on the results
    from postprocessing import run_bsp_analysis
    run_bsp_analysis(resultsdir=resultsdir, outputdir=outputdir)
    
    


if __name__ == "__main__":
    main()