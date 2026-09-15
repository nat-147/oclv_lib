import functools
import logging
import sys
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import List, Optional, Union
import wandb

import torch
import numpy as np
from ignite.engine import Engine
from ignite.engine.events import CallableEventWithFilter
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from models.base_model import MANDATORY_FIELDS
from models.nn_utils import (
    global_norm,
    grad_global_norm,
    group_global_norm,
    group_grad_global_norm,
)
from utils.utils import ExitResubmitException, filter_dict
from utils.viz import apply_color_map, make_recon_img, image_to_masks

from velocity_extraction.viz import save_figures_ocvelocity


@dataclass
class TBLogger:
    working_dir: Path
    trainer: Engine
    model: nn.Module
    dataloaders: List
    loss_terms: List[str]
    scalar_params: List[str]
    event_images: CallableEventWithFilter
    event_loss: CallableEventWithFilter
    event_stats: CallableEventWithFilter
    event_parameters: CallableEventWithFilter
    param_groups: Optional[List[str]] = None
    num_images: int = 3

    use_wandb: bool = False
    wandb_run: wandb = None

    def __post_init__(self):
        self.writer = SummaryWriter(str(self.working_dir))

        # Wandb use check
        if self.use_wandb:
            if self.wandb_run is None:
                self.wandb_run = wandb
            self._wandb_buffer = {}
            self.epoch = self.trainer.state.iteration / len(self.dataloaders[0])

        # Attach logging events to trainer.
        add_event = self.trainer.add_event_handler
        add_event(self.event_loss, self._log_train_losses)
        add_event(self.event_loss, self._log_scalar_params)
        add_event(self.event_stats, self._log_stats)
        add_event(self.event_parameters, self._log_params)
        add_event(self.event_parameters, self._log_grouped_params)
        add_event(self.event_images, self._log_images)

    # =========================================================
    # add_scalar / add_image override
    # =========================================================

    def add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

        if self.wandb_run is not None:
            if isinstance(value, torch.Tensor):
                value = value.item()
            self._wandb_buffer[tag] = value

    def add_image(self, tag, img, step):
        self.writer.add_image(tag, img, step)

        if self.wandb_run is not None:
            self._wandb_buffer[tag] = wandb.Image(img.float())

    def add_images(self, tag, imgs, step):
        self.writer.add_images(tag, imgs, step)

        if self.wandb_run is not None:
            self._wandb_buffer[tag] = [
                wandb.Image(i.float()) for i in imgs
            ]

    # =========================================================

    @torch.no_grad()
    def log_dict(self, metrics: dict, iteration_num: int, group_name: str):
        for metric_name in metrics:
            self.add_scalar(
                f"{group_name}/{metric_name}",
                metrics[metric_name],
                iteration_num,
            )

    @torch.no_grad()
    def _log_images(self, engine):
        n_img = self.num_images
        batch, out = engine.state.output
        step = engine.state.iteration
            
        if batch["image"].dim() == 5: # (B,T,C,H,W)
            B, T, C, H, W = batch["image"].shape
            K = out["mask"].shape[-4]

            if C > 3:
                batch["image"], batch["mask"] = torch.split(batch["image"], 3, dim=2)
                batch["mask"] = torch.nn.functional.one_hot(
                                batch["mask"].view(B, T, H, W).long(),
                                num_classes=K
                                ).permute(
                                    0, 1, 4, 2, 3
                                ).float()
                batch["mask"] = batch["mask"].view(B, T, K, 1, H, W)

            if "x" in out:
                batch["image"] = out["x"]
            if "true_mask" in out:
                batch["mask"] = out["true_mask"]
                
            t_idx = np.random.randint(T-1)
  
            batch.update({k: v[:, t_idx] for k, v in batch.items()}) 
            out.update({k: v[:, t_idx] for k, v in out.items() if v.dim() > 1})

        if not "mask" in batch:
            batch["mask"] = image_to_masks(batch["image"], max_num_objects=out["slot"].shape[1])

        if "reconstruction" in out:
            recon_img = out["reconstruction"][:n_img]
        else:
            recon_img = make_recon_img(out["slot"][:n_img], out["mask"][:n_img])

        sqrt_nrow = 1 #int(sqrt(n_img))

        x = batch["image"][:n_img]
        x_recon = _flatten_slots(torch.stack([x, recon_img], dim=1), nrow=sqrt_nrow)
        self.add_image("input-reconstruction", x_recon.clamp(0.0, 1.0), step)

        slot = _flatten_slots(out["slot"][:n_img], sqrt_nrow)
        self.add_image("slot", slot.clamp(0.0, 1.0), step)

        flat_mask = _flatten_slots(batch["mask"][:n_img], sqrt_nrow)
        mask = make_grid(flat_mask, nrow=sqrt_nrow).float()
        self.add_image("mask: true", mask, step)

        flat_pred_mask = _flatten_slots(out["mask"][:n_img], sqrt_nrow)
        pred_mask = make_grid(flat_pred_mask, nrow=sqrt_nrow)
        self.add_image("mask: pred", pred_mask, step)

        mask_segmap, pred_mask_segmap = _compute_segmentation_mask(batch, n_img, out)
        self.add_images("segmentation: true", mask_segmap, step)
        self.add_images("segmentation: pred", pred_mask_segmap, step)

        if self.use_wandb:
            self.wandb_run.log(self._wandb_buffer, step=engine.state.iteration)
            self._wandb_buffer = {}
    

        if 'loop_loss' in out: # ocl-velocity model
            train_dataset, test_dataset = (dataloader.dataset for dataloader in self.dataloaders)
            train_dataset.data_to_ocl = False
            test_dataset.data_to_ocl = False
            save_figures_ocvelocity(
                    self.wandb_run, 
                    self.model,
                    train_dataset, test_dataset,
                    pred_space_plot=True,
                    pred_images_plot=True,
                    pred_loops_plot=True,
                    step=engine.state.iteration
                )
            train_dataset.data_to_ocl = True
            test_dataset.data_to_ocl = True
       
        
    @torch.no_grad()
    def _log_train_losses(self, engine):
        batch, output = engine.state.output
        self.log_dict(
            filter_dict(output, allow_list=self.loss_terms, inplace=False),
            engine.state.iteration,
            "train losses",
        )

        epoch = self.trainer.state.iteration / len(self.dataloaders[0])
        self.add_scalar("train losses/epoch", epoch, engine.state.iteration)
    
        if self.use_wandb:
            self.wandb_run.log(self._wandb_buffer, step=engine.state.iteration)
            self._wandb_buffer = {}

    @torch.no_grad()
    def _log_scalar_params(self, engine):
        batch, output = engine.state.output
        self.log_dict(
            filter_dict(output, allow_list=self.scalar_params, inplace=False),
            engine.state.iteration,
            "model params",
        )
        if self.use_wandb:
            self.wandb_run.log(self._wandb_buffer, step=engine.state.iteration)
            self._wandb_buffer = {}

    @torch.no_grad()
    def _log_stats(self, engine):
        batch, output = engine.state.output
        for metric_name in output:
            if metric_name in self.loss_terms:
                continue
            if metric_name in self.scalar_params:
                continue

            prefix = "model outputs"
            if metric_name not in MANDATORY_FIELDS:
                prefix += f" ({self.model.name})"

            self._log_tensor(engine, f"{prefix}/{metric_name}", output[metric_name])

        if self.use_wandb:
            self.wandb_run.log(self._wandb_buffer, step=engine.state.iteration)
            self._wandb_buffer = {}

    @torch.no_grad()
    def _log_params(self, engine):
        self.add_scalar(
            "param grad norms/global",
            grad_global_norm(self.model.parameters()),
            engine.state.iteration,
        )
        self.add_scalar(
            "param norms/global",
            global_norm(self.model.parameters()),
            engine.state.iteration,
        )

        if self.use_wandb:
            self.wandb_run.log(self._wandb_buffer, step=engine.state.iteration)
            self._wandb_buffer = {}

    @torch.no_grad()
    def _log_grouped_params(self, engine):
        if self.param_groups is None:
            return

        for name in self.param_groups:
            self.add_scalar(
                f"param grad norms/group: {name}",
                group_grad_global_norm(self.model, name),
                engine.state.iteration,
            )
            self.add_scalar(
                f"param norms/group: {name}",
                group_global_norm(self.model, name),
                engine.state.iteration,
            )

        if self.use_wandb:
            self.wandb_run.log(self._wandb_buffer, step=engine.state.iteration)
            self._wandb_buffer = {}
        

    @torch.no_grad()
    def _log_tensor(self, engine, name, tensor):
        if not isinstance(tensor, Tensor):
            return

        if tensor.numel() == 1:
            stats = ["item"]
        else:
            stats = ["min", "max", "mean"]

        for stat in stats:
            value = getattr(tensor, stat)()
            name_ = name if stat == "item" else f"{name} [{stat}]"
            self.add_scalar(name_, value, engine.state.iteration)


def _compute_segmentation_mask(batch, num_images, output):
    # [bs, ns, 1, H, W] to [bs, 1, H, W]
    mask_segmap = batch["mask"][:num_images].argmax(1)

    # [bs, ns, 1, H, W] to [bs, 1, H, W]
    pred_mask_segmap = output["mask"][:num_images].argmax(1)

    # If shape is [bs, H, W], turn it into [bs, 1, H, W]
    if mask_segmap.shape[1] != 1:
        mask_segmap = mask_segmap.unsqueeze(1)

    # If shape is [bs, H, W], turn it into [bs, 1, H, W]
    if pred_mask_segmap.shape[1] != 1:
        pred_mask_segmap = pred_mask_segmap.unsqueeze(1)

    mask_segmap = apply_color_map(mask_segmap)
    pred_mask_segmap = apply_color_map(pred_mask_segmap)
    return mask_segmap, pred_mask_segmap


def _flatten_slots(images: Tensor, nrow: int):
    image_lst = images.split(1, dim=0)
    image_lst = [
        make_grid(image.squeeze(0), nrow=images.shape[1]) for image in image_lst
    ]
    images = torch.stack(image_lst, dim=0)
    pad_value = 255 if isinstance(images, torch.LongTensor) else 1.0
    return make_grid(images, nrow=nrow, pad_value=pad_value, padding=4)


# ***** Printing utils


def log_tensor_stats(tensor: Tensor, name: str, prefix: str = ""):
    """Logs stats of a tensor."""
    if tensor.numel() == 0:
        return
    logging.debug(f"{prefix}{name} stats:")
    for stat_name in ["min", "max", "mean", "std"]:
        try:
            stat_value = getattr(tensor, stat_name)()
        except RuntimeError as e:
            logging.warning(f"Could not log stat '{stat_name}' of tensor '{name}': {e}")
        else:
            logging.debug(f"   {stat_name}: {stat_value}")
    logging.debug(f"   {name}.shape: {tensor.shape}")


def log_dict_stats(d: dict, prefix: str = ""):
    """Logs stats of tensors in a dict."""
    for k in d:
        if isinstance(d[k], Tensor):
            log_tensor_stats(d[k].float(), k, prefix=prefix)


def log_engine_stats(engine: Engine):
    """Logs stats of all tensors in an engine's state (inputs and outputs)."""
    batch, output = engine.state.output
    log_dict_stats(batch, "[input] ")
    log_dict_stats(output, "[output] ")


# ***** Logging


class PaddingFilter(logging.Filter):
    def __init__(self, pad_len: int = 8, pad_char: str = " "):
        super().__init__()
        assert len(pad_char) == 1
        self.pad_len = pad_len
        self.pad_char = pad_char

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str) and "\n" in record.msg:
            parts = record.msg.split("\n")
            padding = self.pad_char * self.pad_len
            record.msg = f"\n{padding}".join(parts)
        return super().filter(record)


class IgniteFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str) and "terminating due to exception" in record.msg:
            return False
        return super().filter(record)


def filter_ignite_logging():
    engine_logger = logging.getLogger("ignite.engine.engine.Engine")
    engine_logger.addFilter(IgniteFilter())
    engine_logger.setLevel(logging.WARNING)  # Lower this for debugging


def set_logger(
    level: Optional[Union[int, str]] = logging.INFO,
    log_dir: Optional[Path] = None,
    log_fname: Optional[str] = None,
    capture_warnings: bool = True,
):
    """Sets up the default logger.

    Args:
        level: logging level.
        log_dir: log directory. If None (default), it defaults to `${CWD}/logs`.
        log_fname: log file name. If None (default), logging to file is disabled.
        capture_warnings: captures UserWarnings from the warnings package.
    """

    logging.captureWarnings(capture_warnings)

    def formatting_wrapper(format_):
        return f"[{format_}] %(message)s"

    prefix = "%(levelname)s:%(filename)s:%(lineno)s"
    logging.basicConfig(
        format=formatting_wrapper(prefix),
        level=level,
    )

    logging.root.addFilter(PaddingFilter())  # `logging.root` is the default logger

    # Skip logging to file.
    if log_fname is None:
        logging.info("Completed default logger setup. Logging to file disabled.")
        return

    # Else, setup logging to file below.
    if log_dir is None:
        log_dir = Path.cwd() / "logs"
    if not log_dir.exists():
        logging.info(f"Required log dir {log_dir} does not exist: will be created")
        log_dir.mkdir(parents=True)
    elif not log_dir.is_dir():  # exists but is a file
        raise FileExistsError(
            f"Required log dir {log_dir} exists and is not a directory."
        )
    log_path = log_dir / log_fname
    formatter = logging.Formatter(fmt=formatting_wrapper("%(asctime)s " + prefix))
    # If the file exists and is not empty, append a separator to denote different runs.
    if log_path.exists() and log_path.stat().st_size > 0:
        with open(log_path, "a") as fh:
            fh.write("\n======\n\n")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logging.root.addHandler(fh)  # `logging.root` is the default logger
    logging.info(f"Completed default logger setup. Logging to file: {log_path}")


def setup_logging(
    level: Optional[Union[int, str]] = logging.INFO,
    log_dir: Optional[Path] = None,
    log_fname: Optional[str] = None,
):
    """Sets up the default logger and silences most ignite logging.

    Args:
        level: logging level.
        log_dir: log directory. If None (default), it defaults to `${CWD}/logs`.
        log_fname: log file name. If None (default), logging to file is disabled.
    """
    set_logger(level, log_dir, log_fname)
    filter_ignite_logging()


def logging_wrapper(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        except ExitResubmitException:
            # Customize this depending on the job scheduler. E.g., this works for HTCondor.
            sys.exit(3)
        except BaseException as e:
            logging.exception(e)
            raise e

    return wrapper
