from dataclasses import dataclass
from math import sqrt
from typing import List, Optional

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from models.base_model import BaseModel
from models.base_trainer_accerelate import BaseTrainer
from models.nn_utils import init_xavier_
from models.utils import linear_warmup_exp_decay


@dataclass
class SlotAttentionTrainer(BaseTrainer):

    use_exp_decay: bool = False
    exp_decay_rate: Optional[float] = None
    exp_decay_steps: Optional[int] = None
    use_warmup_lr: bool = False
    warmup_steps: Optional[int] = None

    @property
    def loss_terms(self) -> List[str]:
        loss_terms = ["loss", "x_recon_loss"]
        if self.accelerator.unwrap_model(self.model).use_vel_extractor:
            loss_terms.extend(['loop_loss', 'z_pred_loss', 'shortcut_loss', 'isometry_loss'])
                     
        return loss_terms

    @property
    def param_groups(self) -> List[str]:
        return ["encoder", "decoder", "slot_attention"]

    def _post_init(self, model: BaseModel, dataloaders: List[DataLoader]):
        super()._post_init(model, dataloaders)
        self._init_model()

        # Set unused variables to None, necessary for LR scheduler setup
        if not self.use_exp_decay:
            self.exp_decay_steps = self.exp_decay_rate = None
        if not self.use_warmup_lr:
            self.warmup_steps = None
        
        lr_scheduler = LambdaLR(
            self.optimizers[0],
            lr_lambda=linear_warmup_exp_decay(
                self.warmup_steps, self.exp_decay_rate, self.exp_decay_steps
            ),
        )
        self.lr_schedulers.append(lr_scheduler)
        
        

    @torch.no_grad()
    def _init_model(self):
        model = self.accelerator.unwrap_model(self.model)
        
        init_xavier_(model)
    
        torch.nn.init.zeros_(model.slot_attention.gru.bias_ih)
        torch.nn.init.zeros_(model.slot_attention.gru.bias_hh)
        torch.nn.init.orthogonal_(model.slot_attention.gru.weight_hh)
        if model.slot_init_type == 'gaussian':
            limit = sqrt(6.0 / (1 + model.slot_attention.dim))
            torch.nn.init.uniform_(model.slot_attention.slots_mu, -limit, limit)
            torch.nn.init.uniform_(model.slot_attention.slots_log_sigma, -limit, limit)
