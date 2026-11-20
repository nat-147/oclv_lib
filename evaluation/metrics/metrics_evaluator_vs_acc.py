import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from torch import Tensor

import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from torch.utils.data import DataLoader

from data.datasets import MultiObjectDataset
from evaluation.metrics.ari import ari as ari_

from evaluation.metrics.segmentation_covering import segmentation_covering
from models.base_model import BaseModel
from models.utils import ForwardPass
from utils.utils import dict_tensor_mean
from utils.viz import make_recon_img, image_to_masks


from accelerate import Accelerator


def dict_tensor_gather_mean(accelator: Accelerator, data: Dict[str, List[Tensor]]) -> Dict[str, float]:
    output = {}
    for key, value in data.items():
        values = accelator.gather(torch.stack(value))
        output[key] = values.mean().item()
    return output


_DEFAULT_METRICS = [
    "image_ari",
    "video_ari",
    "image_mbo",
    "video_mbo",
    "mse",
]


from evaluation.metrics.metrics import (
    ImageARI,
    VideoARI,
    ImageIoU,
    VideoIoU,
)


def evaluate_metrics(
    pred_masks,   # [B,T,K,H,W] one-hot prediction
    gt_masks,     # [B,T,M,H,W] binary GT masks
):
    """
    Args
    ----
    pred_masks : torch.Tensor
        Shape [B,T,K,H,W]
        One-hot slot assignments.

    gt_masks : torch.Tensor
        Shape [B,T,M,H,W]
        Binary ground-truth instance masks.
        Channel 0 may optionally be background.

    Returns
    -------
    dict
    """

    image_ari = ImageARI(
        video_input=True,
        ignore_background=True,
        pred_key="pred_mask",
        true_key="true_mask",
    )

    video_ari = VideoARI(
        ignore_background=True,
        pred_key="pred_mask",
        true_key="true_mask",
    )

    image_mbo = ImageIoU(
        video_input=True,
        ignore_background=True,
        matching="overlap",
        pred_key="pred_mask",
        true_key="true_mask",
    )

    video_mbo = VideoIoU(
        ignore_background=True,
        matching="overlap",
        pred_key="pred_mask",
        true_key="true_mask",
    )

    image_ari.update(
        true_mask=gt_masks,
        pred_mask=pred_masks,
    )

    video_ari.update(
        true_mask=gt_masks,
        pred_mask=pred_masks,
    )

    image_mbo.update(
        true_mask=gt_masks,
        pred_mask=pred_masks,
    )

    video_mbo.update(
        true_mask=gt_masks,
        pred_mask=pred_masks,
    )

    results = {
        "image_ari": image_ari.compute(),
        "video_ari": video_ari.compute(),
        "image_mbo": image_mbo.compute(),
        "video_mbo": video_mbo.compute()
    }

    return results


@dataclass
class MetricsEvaluator:
    _forward_pass: ForwardPass
    dataloader: DataLoader
    loss_terms: List[str]
    skip_background: bool
    device: str
    metrics: List[str] = field(default_factory=lambda: _DEFAULT_METRICS)
    
    num_bg_objects: int = field(init=False)
    num_ignored_objects: int = field(init=False)

    accelerator: Accelerator = field(default=None)


    def __post_init__(self):
        # This should be a MultiObjectDataset.
        dataset: MultiObjectDataset = self.dataloader.dataset  # type: ignore
        self.num_bg_objects = dataset.num_background_objects
        self.num_ignored_objects = self.num_bg_objects if self.skip_background else 0

        if self.accelerator is None:
            self.accelerator = Accelerator()

    @torch.no_grad()
    def _eval_step(self, engine: Engine, batch: dict):
        # step=None → model uses its full vel-loss weight; a training step applies the lambda warmup schedule
        if getattr(self, "_model_step", None) is not None:
            batch, output = self._forward_pass(batch, step=self._model_step)
        else:
            batch, output = self._forward_pass(batch)
        B, T, C, H, W = batch["image"].shape      
        K = output["mask"].shape[-4]

        if C > 3:
            batch["image"], batch["mask"] = torch.split(batch["image"], 3, dim=2)
            batch["mask"] = torch.nn.functional.one_hot(
                            batch["mask"].view(B, T, H, W).long(),
                            num_classes=K
                            ).permute(
                                0, 1, 4, 2, 3
                            ).float()

        # Compute metrics
        if len(_DEFAULT_METRICS) > 1:
            # One-hot masks
            true_mask = batch.get("mask", 
                image_to_masks(
                batch["image"].view(B*T,3,H,W), 
                max_num_objects=K
                ).view(B,T,-1,H,W).cuda()
                )
            pred_mask = output["mask"].view(B, T, -1, H, W)
            pred_mask = torch.nn.functional.one_hot(
                pred_mask.argmax(dim=2),
                num_classes=K
            ).permute(
                0, 1, 4, 2, 3
            )
            metrics_values = evaluate_metrics(pred_mask, true_mask)
        else:
            metrics_values = None

        
        if "reconstruction" in output:
            reconstruction = output["reconstruction"]
        else:
            reconstruction = make_recon_img(output["slot"].view(-1,K,3,H,W), output["mask"].view(-1,K,1,H,W)).view(B,T,3,H,W).clamp(0.0, 1.0)
       
        mse_full = (batch["image"].cuda() - reconstruction) ** 2
        mse = mse_full.mean()


        # Collect loss values from model output
        loss_values = {}
        for loss_term in self.loss_terms:
            loss_values[loss_term] = output[loss_term]

        # Return with shape (batch_size, )
        results = dict(
            mse = mse,
            **loss_values
        )
        if metrics_values is not None:
            results.update(metrics_values)

        return results
    
    @torch.no_grad()
    def eval(
        self, model: BaseModel, steps: Optional[int] = None, model_step: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """model_step: training iteration passed to model.forward(step=...) so the vel-loss lambda
        warmup applies to the validation loss too (only loss terms change, not ARI/mBO/MSE)."""
        self._model_step = model_step
        engine = Engine(self._eval_step)
        results = {name: [] for name in self.metrics + list(self.loss_terms)}

        @engine.on(Events.ITERATION_COMPLETED)
        def accumulate_metrics(engine):
            for name in self.metrics + list(self.loss_terms):
                batch_results = engine.state.output[name]
                if batch_results.dim() == 0:  # scalar
                    batch_size = engine.state.batch["image"].shape[0]
                    batch_results = [batch_results] * batch_size
                    
                results[name].extend(batch_results)

        if sys.stdout.isatty():
            ProgressBar().attach(
                engine,
                output_transform=lambda o: {k: v for k, v in o.items()},
            )

        #steps = min(len(self.dataloader), steps) 
        engine.run(self.dataloader, 1, steps)


        # Split results into losses and metrics
        losses = {k: results[k] for k in self.loss_terms}
        metrics = {k: results[k] for k in self.metrics}

        return dict_tensor_gather_mean(self.accelerator, losses), dict_tensor_gather_mean(self.accelerator, metrics)
