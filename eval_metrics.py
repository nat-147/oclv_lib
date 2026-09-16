import json
import logging
from copy import deepcopy
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from evaluation.metrics.metrics_evaluator_vs_acc import MetricsEvaluator
from evaluation.shared import eval_shared
from models.base_model import BaseModel
from models.utils import ForwardPass
from utils.logging import logging_wrapper
from utils.paths import CONFIG
from utils.viz import save_images_as_grid


@hydra.main(config_path=str(CONFIG), config_name="eval_metrics")
@logging_wrapper  # logging is set up in eval_shared()
def main(config: DictConfig):
    # Override config if debug mode
    if config.debug:
        config.batch_size = 4

    assert isinstance(config.batch_size, int) and config.batch_size > 0

    eval_shared(
        config=config,
        run_eval=run_eval,
        eval_name="metrics",
        get_batch_size=lambda config: config.batch_size,
    )
    logging.info("Evaluation completed.")


def run_eval(
    checkpoint_config: DictConfig,
    config: DictConfig,
    dataloader: DataLoader,
    model: BaseModel,
    results_path: Path,
):
    # Save some example images.
    example_images = next(iter(dataloader))["image"]
    save_images_as_grid(
        images=example_images[:16, 0] if example_images.dim() == 5 else example_images[:16],
        path=results_path / "examples",
    )

    eval_step = ForwardPass(model, config.device)

    _, eval_results = MetricsEvaluator(
        _forward_pass=eval_step,
        dataloader=dataloader,
        loss_terms=list(config.loss_terms),
        skip_background=True,
        device=config.device,
    ).eval(model)

    results = []
    for metric, value in eval_results.items():
        results.append(
            {
                "train_config.uuid": checkpoint_config.uuid,
                "eval_config": _cleanup_eval_config(config),
                "results": {
                    "metric_name": metric,
                    "metric_value": value,
                },
            }
        )
    # Save results dict
    with open(results_path / "results.json", "w") as fp:
        json.dump(results, fp, indent=2)


def _cleanup_eval_config(config):
    """Removes unnecessary data from the eval config for saving."""
    eval_config_dict = deepcopy(OmegaConf.to_container(config))
    for key in ["loss_terms", "overwrite", "debug"]:
        del eval_config_dict[key]
    return eval_config_dict


if __name__ == "__main__":
    main()
