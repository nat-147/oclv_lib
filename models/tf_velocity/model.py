from dataclasses import dataclass, field
from math import sqrt
from typing import Dict, List, Literal, Optional, Tuple, Union

from omegaconf import DictConfig

import torch, copy, math
from torch import Tensor, nn

from models.base_model import BaseModel
from models.nn_utils import get_conv_output_shape, make_sequential_from_config
from models.shared.nn import PositionalEmbedding

from velocity_extraction.utils import mse_loss_fn, cosine_dissim

import argparse
from models.tf_velocity.utilities.setup_utils import get_model, LrScheduler
from models.tf_velocity.models import autoencoder, transformer
from models.tf_velocity.eval.eval_metrics import compute_decoder_jacobian, get_jac_interaction, get_recon_loss

from utils.viz import make_recon_img, image_to_masks

from accelerate import Accelerator

import torch.nn.functional as F
import itertools

from models.utils import HungarianMatcher

from velocity_extraction.model import VelocityExtractor


@dataclass(eq=False, repr=False)
class TFVelocityAE(BaseModel):

    vel_loss_weights : Union[dict, DictConfig] 
    lambda_warmup_steps_interval: List[int]

    temporal_slot_init: bool = False
    input_channels: int = 3

    use_vel_extractor: bool = True
    hungarian_matching: bool = True
    use_input_pairs: bool = False
    use_lambda_warmup: bool = False

    # overrides for the autoencoder hyperparameters (see defaults in __post_init__)
    autoencoder_args: Optional[Union[dict, DictConfig]] = None

    # compute masks / jacobian images every N steps (should match trainer.logimages_steps)
    logimages_steps: int = 5000

    vel_extractor: VelocityExtractor = field(init=False)
    
    autoencoder: autoencoder = field(init=False)

 
    def __post_init__(self):
        super().__post_init__()
        self.loss_fn = nn.MSELoss(reduction='none')

        self.accelerator = Accelerator()

    
        # create model (defaults below can be overridden via `model.autoencoder_args` in the config)
        args = dict(
            data = 'sprites',
            slot_dim = 64,
            encoder = "transformer",
            decoder = "transformer",
            alpha = .05,
            beta = .05,
            query_dim = 180,
            proj_dim = 516,
            num_dec_layers = 2,
            warmup_alpha_steps = 30000,
            sigma = None,
        )
        if self.autoencoder_args is not None:
            overrides = dict(self.autoencoder_args)
            unknown = set(overrides) - set(args)
            if unknown:
                raise ValueError(f"Unknown autoencoder_args: {sorted(unknown)}")
            args.update(overrides)

        # num_slots always follows model.num_slots (= dataset.max_num_objects)
        args['num_slots'] = self.num_slots

        if args['sigma'] is None:
            args['sigma'] = 5. if (args['alpha'] > 0 or args['beta'] > 0) else 1.

        args = argparse.Namespace(**args)

        self.autoencoder = get_model(args)
        self.args = args

        # the first conditional is the lr scheduling used in the paper for sprites and clevr6, the second is for clevrtex
        self.alpha_scheduler = LrScheduler(args.alpha, args.warmup_alpha_steps, .2, 250000)

        if self.use_vel_extractor:
            self.vel_loss_weights = dict(self.vel_loss_weights)
            self.vel_extractor = VelocityExtractor(slot_dim=args.slot_dim, num_slots=self.num_slots, v_dim=2, loss_weights=self.vel_loss_weights, use_input_pairs=self.use_input_pairs)
        
            if self.hungarian_matching:
                self.hungarian_matcher = HungarianMatcher(args.num_slots)

        self.accelerator = Accelerator()

    @property
    def slot_size(self) -> int:
        return self.latent_size

    def forward(self, x: Tensor, step: int = None) -> dict:
        # input: (B, T, 3, H, W)
        # Forward pass through recurrent attention network.
       
        B, T, C, H, W = x.shape
        if C > 3:
            x, m  = torch.split(x, 3, dim=2)
        else:
            m = None

        glob_it = 0

        # set global iteration to load iterafftion if we're loading a model
        if step is not None:
            glob_it = step
    
        # get recfonstructed latents and observations
        if self.temporal_slot_init:
            slot_init = None
            zhs, xhs, mus = [], [], []
            for t in range(T):
                zh, xh, enc_out = self.autoencoder(x[:,t], slot_init)
                zhs.append(zh)
                xhs.append(xh)
                mus.append(enc_out[:, :, 0:self.args.slot_dim])
                slot_init = enc_out.detach()
            zh, xh, mu = torch.stack(zhs, dim=1), torch.stack(xhs, dim=1), torch.stack(mus, dim=1)
            _, _, K, D = zh.shape
        else:
            x = x.flatten(0, 1)
            zh, xh, enc_out = self.autoencoder(x)
            mu = enc_out[:, :, 0:self.args.slot_dim]
            _, K, D = zh.shape

        # reconstruction loss
        recon_loss = self.args.sigma * get_recon_loss(x.reshape(-1,3,H,W), xh.reshape(-1,3,H,W))
        loss = recon_loss.clone()

        # defaults for when beta == 0 / alpha == 0 (or non-transformer decoder) so logging still works
        kl_loss = torch.zeros_like(recon_loss)
        interac_loss = torch.zeros_like(recon_loss)
        
        # kl loss
        if self.args.beta > 0:
            beta = self.args.beta
            kl_loss = beta * self.autoencoder.dkl
            loss += kl_loss

        # interaction loss
        if self.args.decoder == "transformer" and self.args.alpha > 0:
            interac_dec_cross_att = self.autoencoder.decoder.transformer.compute_interaction()
            if self.training and glob_it < self.args.warmup_alpha_steps:
                # get current value for alpha during warming phase
                alpha = self.alpha_scheduler.get_current_lr(glob_it)
            else:
                alpha = self.args.alpha

            interac_loss = alpha * interac_dec_cross_att
            loss += interac_loss

        #Velocity Extraction
        if self.use_vel_extractor:
            if self.args.beta > 0:
                slots = mu
            else:
                slots = zh

            slots = slots.view(B, T, K, D)
            
            if  self.hungarian_matching:
                slots_matched = slots.clone()
                
                self.hungarian_matcher.reset()
                for t in range(T):
                    slots_matched[:,t] = self.hungarian_matcher(slots[:,t])
                slots = slots_matched


            vel_losses, pred_slots = self.vel_extractor(slots)
            vel_loss = sum(vel_losses.values()).squeeze()

            vel_weight = self.vel_loss_weights["vel_loss_weight_lambda"]


            if self.use_lambda_warmup and step is not None:
                warmup_start, warmup_end = self.lambda_warmup_steps_interval

                vel_weight *= min(
                    max(0, step - warmup_start) / (warmup_end - warmup_start),
                    1
                )

            vel_losses.update({k: vel_weight*v for k, v in vel_losses.items()})

            loss += vel_weight * vel_loss 
        
        else:
            slots = zh

        results = {
            "loss": loss,  # scalar
            "representation": zh.view(B, T, K, -1),  # (B, T, slots, latent dim)
            "reconstruction": xh.view(B, T, 3, H, W),  # (B, T, 3, H, W)
            "x_recon_loss": recon_loss, # scalar
            "interac_loss" : interac_loss,
            "kl_loss" : kl_loss,
        }
        
        # step is None -> evaluation (MetricsEvaluator needs "mask"); otherwise only on image-logging steps
        if  step is None or (step % self.logimages_steps == 0):
            slot_norm_jacobian = compute_decoder_jacobian(self.args, self.autoencoder, slots.view(-1, K, D), norm=True).cuda()

            #One-hot masks
            if m is not None:
                true_masks = torch.nn.functional.one_hot(
                m.view(B, T, H, W).long(),
                num_classes=K,
                ).permute(
                    0, 1, 4, 2, 3
                ).float()
            else: 
                true_masks = image_to_masks(
                x.view(B*T,3,H,W), 
                max_num_objects=K,
                ).cuda()

            true_masks = true_masks.view(-1, K, H, W)
            background_mask = true_masks[:, 1:, :, :].sum(1).unsqueeze(1).repeat(1, K, 1, 1)
            
            # zero out all background pixels for slot-wise jacobians
            slot_norm_jacobian = (slot_norm_jacobian * background_mask)
            slot_norm_jacobian = slot_norm_jacobian.view(B, T, K, 1, H, W)

            results.update({
                "mask": slot_norm_jacobian,  # (B, T, slots, 1, H, W)
                "slot": xh.view(B, T, 1, 3, H, W).expand(B, T, K, 3, H, W) * slot_norm_jacobian,  # (B, T, slots, 3, H, W)
                "x" : x.view(B, T, 3, H, W),
                "true_mask" : true_masks.view(B, T, K, 1, H, W),
            })
        
            
            # gather across processes only for training-time image logging. In evaluation each rank
            # computes metrics on its own batch (MetricsEvaluator gathers the metrics itself), so
            # gathering here would mismatch output (B*world) vs batch (B) -> one_hot index out of bounds.
            if self.accelerator is not None and self.training:
                results.update({k: self.accelerator.gather(v) for k, v in results.items() if v.dim() > 0})

        if self.use_vel_extractor:
            results.update(vel_losses)

        return results





