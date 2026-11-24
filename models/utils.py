import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, Dict, List

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from ignite.engine import Engine
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from models.base_model import BaseModel

import itertools


import accelerate
from accelerate import Accelerator

@dataclass
class ForwardPass:
    model: BaseModel
    device: Union[torch.device, str]
    preprocess_fn: Optional[Callable] = None
    def __call__(self, batch: dict, step: int = None) -> Tuple[dict, dict]:
        for key in batch.keys():
            batch[key] = batch[key].to(self.device, non_blocking=True)
        if self.preprocess_fn is not None:
            batch = self.preprocess_fn(batch)

        if step is not None:
            output = self.model(batch["image"], step=step)
        else:
            output = self.model(batch["image"])
        
        return batch, output
    




def single_forward_pass(
    model: BaseModel, dataloader: DataLoader, device: Union[torch.device, str]
) -> Tuple[dict, dict]:
    eval_step = ForwardPass(model, device)
    evaluator = Engine(lambda e, b: eval_step(b, step=0))
    evaluator.run(dataloader, 1, 1)
    batch, output = evaluator.state.output
    return batch, output


class TrainCheckpointHandler:
    def __init__(
        self, checkpoint_path: Union[str, Path], device: Union[torch.device, str]
    ):
        if isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)
        self.checkpoint_train_path = checkpoint_path / "train_checkpoint.pt"
        self.model_path = checkpoint_path / "model.pt"
        self.train_yaml_path = checkpoint_path / "train_state.yaml"
        self.device = device

    def save_checkpoint(self, state_dicts: dict, best_metrics: dict):
        """Saves a checkpoint.

        If the state contains the key "model", the model parameters are saved
        separately to model.pt, and they are not saved to the checkpoint file.
        """
        if "model" in state_dicts:
            logging.info(f"Saving model to {self.model_path}")
            torch.save(state_dicts["model"], self.model_path)
            del state_dicts["model"]  # do not include model (avoid duplicating)
        
        torch.save(state_dicts, self.checkpoint_train_path)
            

        # Save train state (duplicate info from main checkpoint)
        trainer_state = state_dicts["trainer"]
        with open(self.train_yaml_path, "w") as f:
            train_state = {
                "step": trainer_state["iteration"],
                "max_step": trainer_state["epoch_length"],
                "best_model": dict(best_metrics)
            }
            yaml.safe_dump(train_state, f)

    def load_checkpoint(self, objects: dict):
        """Loads checkpoint into the provided dictionary."""

        # Load checkpoint without model
        state = torch.load(self.checkpoint_train_path, self.device)
        for varname in state:
            logging.debug(f"Loading checkpoint: variable name '{varname}'")
            objects[varname].load_state_dict(state[varname])

        # Load model
        if "model" in objects:
            logging.debug(f"Loading checkpoint: model")
            model_state_dict = torch.load(self.model_path, self.device)
            objects["model"].load_state_dict(model_state_dict)



def linear_warmup_exp_decay(
    warmup_steps: Optional[int] = None,
    exp_decay_rate: Optional[float] = None,
    exp_decay_steps: Optional[int] = None,
) -> Callable[[int], float]:
    assert (exp_decay_steps is None) == (exp_decay_rate is None)
    use_exp_decay = exp_decay_rate is not None
    if warmup_steps is not None:
        assert warmup_steps > 0

    def lr_lambda(step):
        multiplier = 1.0
        if warmup_steps is not None and step < warmup_steps:
            multiplier *= step / warmup_steps
        if use_exp_decay:
            multiplier *= exp_decay_rate ** (step / exp_decay_steps)
        return multiplier

    return lr_lambda


def load_model(
    config: DictConfig, checkpoint_path: Union[Path, str], model_args: dict = None, best_epoch: bool = False
) -> BaseModel:
    """Instantiates model from config and loads it from checkpoint."""
    if model_args is None:
        model_args = {}
    model: BaseModel = hydra.utils.instantiate(config.model, **model_args)
    model.to(config.device)
    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)
    model_path = checkpoint_path / "best_model.pt" if best_epoch else checkpoint_path / "model.pt" 
    model.load_state_dict(torch.load(model_path, config.device))
    return model


def infer_model_type(model_name: str) -> str:
    if model_name.startswith("baseline_vae"):
        return "distributed"
    if model_name in [
        "slot-attention",
        "monet",
        "monet-velocity",
        "sa-velocity",
        "tf-velocity",
        "genesis",
        "space",
        "monet-big-decoder",
        "slot-attention-big-decoder",
    ]:
        return "object-centric"
    raise ValueError(f"Could not infer model type for model '{model_name}'")

def save_best_model(checkpoint_path: Union[Path, str], state_dicts: dict):
    model_path = checkpoint_path / "best_model.pt"
    logging.info(f"Saving model to {model_path}")
    torch.save(state_dicts["model"], model_path)

def load_best_model(checkpoint_path: Union[Path, str]):
    train_yaml_path = checkpoint_path / "train_state.yaml"
    with open(train_yaml_path, "r") as f:
        train_state = yaml.safe_load(f)
        
    return train_state["best_model"]


class HungarianMatcher(nn.Module):
    def __init__(self, num_slots):
        super().__init__()

        perms = torch.tensor(
            list(itertools.permutations(range(num_slots))),
            dtype=torch.long
        )

        self.register_buffer("perms", perms)
        self.prev_slots = None

    def reset(self):
        self.prev_slots = None

    def forward(self, slots):

        if self.prev_slots is None:
            self.prev_slots = slots.detach()
            return slots

        matched = self.match(self.prev_slots, slots)

        self.prev_slots = matched.detach()

        return matched

    def step(self, slots):
        """Like forward, but also returns the permutation (B, N) with matched = slots[b, perm[b]]."""
        B, N, _ = slots.shape
        if self.prev_slots is None:
            self.prev_slots = slots.detach()
            return slots, torch.arange(N, device=slots.device).expand(B, N)
        perm = self.best_perm(self.prev_slots, slots)
        matched = torch.gather(slots, 1, perm.unsqueeze(-1).expand_as(slots))
        self.prev_slots = matched.detach()
        return matched, perm

    def best_perm(self, prev_slots, curr_slots):
        """Permutation (B, N) of curr_slots that minimises the summed cosine distance to prev_slots."""
        N = curr_slots.shape[1]
        prev = F.normalize(prev_slots, dim=-1)
        curr = F.normalize(curr_slots, dim=-1)
        cost = 1 - torch.bmm(prev, curr.transpose(1, 2))          # (B, N, N)
        perms = self.perms.to(cost.device)
        rows = torch.arange(N, device=cost.device)
        perm_cost = cost[:, rows[None, :], perms].sum(-1)          # (B, N!)
        return perms[perm_cost.argmin(-1)]                         # (B, N)

    def match(self, prev_slots, curr_slots):

        B, N, D = curr_slots.shape

        prev = F.normalize(prev_slots, dim=-1)
        curr = F.normalize(curr_slots, dim=-1)

        cost = 1 - torch.bmm(prev, curr.transpose(1, 2))
        # cost : (B,N,N)

        perms = self.perms.to(cost.device)

        # cost[:, row=i, col=perm[i]]
        rows = torch.arange(N, device=cost.device)

        perm_cost = cost[:, rows[None,None,:], perms[None,:,:]]
        # (B,N!,N)

        perm_cost = perm_cost.sum(-1)
        # (B,N!)

        best = perm_cost.argmin(-1).squeeze(-1)

        best_perm = perms[best]

        matched = torch.gather(
            curr_slots,
            1,
            best_perm.unsqueeze(-1).expand(-1,-1,D)
        )

        return matched