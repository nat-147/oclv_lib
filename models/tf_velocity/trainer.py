from dataclasses import dataclass
from math import sqrt
from typing import List, Optional, Callable

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from models.base_model import BaseModel
from models.base_trainer_accerelate import BaseTrainer
from models.nn_utils import init_xavier_

from models.tf_velocity.utilities.setup_utils import get_model, LrScheduler
import argparse


def linear_warmup_exp_decay(
    lr_scheduler : LrScheduler,
    lr_scheduler_2 : Optional[LrScheduler],
    data: str
) -> Callable[[int], float]:
    # lr_scheduler_2 is None -> single-phase warmup + exp decay (clevrtex, or alpha == beta == 0)
    assert lr_scheduler is not None
    
    def lr_lambda(step):
        glob_it = step

        # update learning rate
        multiplier = lr_scheduler.get_current_lambda(glob_it)
        
        # lr scheduler for sprites and clevr6
        if data != "clevrtex" and lr_scheduler_2 is not None:
            if glob_it > lr_scheduler.warmup_steps:
                if glob_it < 30000:
                    multiplier = 1
                if glob_it > 30000:
                    multiplier = lr_scheduler_2.get_current_lambda(glob_it)
        
        return multiplier

    return lr_lambda


@dataclass
class TransformerTrainer(BaseTrainer):

    use_exp_decay: bool = False
    exp_decay_rate: Optional[float] = None
    exp_decay_steps: Optional[int] = None
    use_warmup_lr: bool = False
    warmup_steps: Optional[int] = None

    # overrides for the LR scheduler hyperparameters (see defaults in _post_init)
    lr_scheduler_args: Optional[dict] = None

    @property
    def loss_terms(self) -> List[str]:
        loss_terms = ["loss", "x_recon_loss", "interac_loss", "kl_loss"]
        if self.accelerator.unwrap_model(self.model).use_vel_extractor:
            loss_terms.extend(['loop_loss', 'z_pred_loss', 'shortcut_loss', 'isometry_loss'])
                     
        return loss_terms

    @property
    def param_groups(self) -> List[str]:
        return ["autoencoder"]

    def _post_init(self, model: BaseModel, dataloaders: List[DataLoader]):
        super()._post_init(model, dataloaders)
        self._init_model()

        # defaults below can be overridden via `trainer.lr_scheduler_args` in the config.
        # data/alpha/beta default to the values used by the model's autoencoder.
        model_args = self.accelerator.unwrap_model(self.model).args
        args = dict(
            data = model_args.data,
            lr = 5e-4,
            warmup_lr_steps = 30000,
            decay_rate = 0.1,
            decay_steps = 500000,
            num_iters = 500000,
            alpha = model_args.alpha,
            beta = model_args.beta,
        )
        if self.lr_scheduler_args is not None:
            overrides = dict(self.lr_scheduler_args)
            unknown = set(overrides) - set(args)
            if unknown:
                raise ValueError(f"Unknown lr_scheduler_args: {sorted(unknown)}")
            args.update(overrides)
        args = argparse.Namespace(**args)

        # Set unused variables to None, necessary for LR scheduler setup
        if (args.alpha > 0 or args.beta > 0) and args.data != "clevrtex":
            lr_scheduler = LrScheduler(args.lr, args.warmup_lr_steps, args.decay_rate, args.decay_steps)
            lr_scheduler_2 = LrScheduler(1e-4, 30000, args.decay_rate, args.decay_steps, prev_lr=args.lr)
        else:
            lr_scheduler = LrScheduler(args.lr, args.warmup_lr_steps, args.decay_rate, args.decay_steps)
            lr_scheduler_2 = None
        
        lr_scheduler = LambdaLR(
            self.optimizers[0],
            lr_lambda=linear_warmup_exp_decay(
                lr_scheduler, lr_scheduler_2, args.data
            ),
        )
        self.lr_schedulers.append(lr_scheduler)
        
        

    @torch.no_grad()
    def _init_model(self):
        model = self.accelerator.unwrap_model(self.model)
        
        init_xavier_(model)
    
