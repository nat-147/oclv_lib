import logging
import os
from pathlib import Path
from typing import Callable

import torch
from omegaconf import DictConfig

from data.datasets import hydra_to_dataset_args, make_dataloaders
from models.base_model import BaseModel
from models.utils import load_model
from utils.logging import setup_logging
from utils.utils import check_common_args, load_config, now, set_all_seeds
from velocity_extraction.data import ContinuousShiftingSpritesloops
from velocity_extraction.utils import get_dataset_dataloader


@torch.no_grad()
def eval_shared(
    config: DictConfig,
    run_eval: Callable[[DictConfig, DictConfig, torch.utils.data.DataLoader, BaseModel, Path], None],
    eval_name: str,
    get_batch_size: Callable[[DictConfig], int],
):
    """Utility function for simple evaluation of a trained checkpoint.

    This is a stripped-down counterpart of the `evaluation.shared.eval_shared` helper from the
    original `ocl+velocity` project: oclv_lib only ever trains on a single dataset config per run
    (no dataset variants, see the project README), so there is no variant-type loop and no
    dataset-variant loading here. The argument `run_eval` defines what the evaluation consists of
    (computing metrics, saving qualitative visualizations, ...).
    """
    check_common_args(config)

    # Resolve checkpoint path to an absolute path: if relative, changing the working directory
    # below would break it.
    original_checkpoint_path = config.checkpoint_path
    checkpoint_path = Path(config.checkpoint_path).resolve()
    config.checkpoint_path = str(checkpoint_path)

    os.chdir(checkpoint_path)
    setup_logging(log_fname=f"eval_{eval_name}_{now()}.log")
    if original_checkpoint_path != config.checkpoint_path:
        logging.info(
            f"Checkpoint path '{original_checkpoint_path}' was resolved to "
            f"'{config.checkpoint_path}'"
        )

    # Load the config the checkpoint was trained with, then overwrite the device with the one
    # requested for evaluation.
    checkpoint_config = load_config(checkpoint_path)
    checkpoint_config.device = config.device

    set_all_seeds(config.seed)

    # Build the held-out dataloader for the dataset the checkpoint was trained on.
    if checkpoint_config.dataset.name == "contshiftingsprites":
        dataconfig = dict(
            dataset_class=ContinuousShiftingSpritesloops,
            dataset_args=hydra_to_dataset_args(checkpoint_config),
            dataloader_args=dict(
                batch_size=get_batch_size(config),
                pin_memory=True,
                num_workers=checkpoint_config.num_workers,
                persistent_workers=checkpoint_config.persistent_workers,
                prefetch_factor=checkpoint_config.prefetch_factor,
            ),
        )
        _, _, _, dataloader = get_dataset_dataloader(dataconfig)
    else:
        *_, dataloader = make_dataloaders(
            dataset_config=checkpoint_config.dataset,
            data_sizes=checkpoint_config.data_sizes[:2],
            batch_size=get_batch_size(config),
            num_workers=checkpoint_config.num_workers,
        )

    # Always evaluate the best checkpoint saved during training.
    model = load_model(checkpoint_config, checkpoint_path, best_epoch=True)
    model.eval()

    results_path = checkpoint_path / "evaluation" / eval_name
    results_path.mkdir(exist_ok=config.overwrite, parents=True)

    logging.info("Starting evaluation")
    run_eval(checkpoint_config, config, dataloader, model, results_path)
    logging.info("Evaluation completed")
