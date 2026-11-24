from dataclasses import dataclass, field
from typing import Tuple, Union, List

import torch
import torch.distributions as dists
from omegaconf import DictConfig
from torch import Tensor, nn
import torch.nn.functional as F

from models.base_model import BaseModel
from models.shared.encoder_decoder import BroadcastDecoderNet, EncoderNet
from models.shared.unet import UNet
from models.utils import HungarianMatcher

from velocity_extraction.model import VelocityExtractor


@dataclass(eq=False, repr=False)
class MonetVelocity(BaseModel):
    latent_size: int

    num_blocks_unet: int
    beta_kl: float
    gamma: float

    encoder_params: DictConfig
    decoder_params: DictConfig

    vel_loss_weights : Union[dict, DictConfig]
    lambda_warmup_steps_interval: List[int]

    input_channels: int = 3
    bg_sigma: float = 0.09
    fg_sigma: float = 0.11

    prior_mean: float = 0.0
    prior_std: float = 1.0

    use_vel_extractor: bool = True
    use_input_pairs: bool = False
    hungarian_matching: bool = False
    use_lambda_warmup: bool = False
    recon_from_pred: bool = False

    vel_extractor: VelocityExtractor = field(init=False)

    encoder: EncoderNet = field(init=False)
    decoder: BroadcastDecoderNet = field(init=False)


    def __post_init__(self):
        super().__post_init__()
        self.attention = AttentionNet(self.input_channels, self.num_blocks_unet)
        self.encoder_params.update(width=self.width, height=self.height)
        self.encoder = EncoderNet(**self.encoder_params)
        self.decoder = BroadcastDecoderNet(**self.decoder_params)
        self.prior_dist = dists.Normal(self.prior_mean, self.prior_std)

        if self.use_vel_extractor:
            self.vel_loss_weights = dict(self.vel_loss_weights)
            self.vel_extractor = VelocityExtractor(slot_dim=self.latent_size, num_slots=self.num_slots, v_dim=2, loss_weights=self.vel_loss_weights, use_input_pairs=self.use_input_pairs)

            if self.hungarian_matching:
                self.hungarian_matcher = HungarianMatcher(self.num_slots-1)

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


        x = x.flatten(start_dim=0, end_dim=1) # (B*T, 3, H, W)
        log_masks = self._attention_process(x)  # (B*T, K = num slots, H, W)
        masks = log_masks.exp()

        # Component VAE forward pass.
        zs, kl_zs, slot_means = self._encode(x, log_masks) 

        if self.hungarian_matching:
            # Match foreground slots (1..K-1) frame to frame and apply the same permutation to every per-slot
            # quantity (z, masks, means) so slot k keeps its own mask. The loss is permutation invariant, so
            # matching only changes which slot index an object gets over time (and the velocity input / VideoARI).
            K = zs.shape[1]
            zs_bt = zs.unflatten(0, (B, T))  # (B, T, K, D)
            perm = torch.arange(K, device=zs.device).repeat(B, T, 1)  # (B, T, K)
            self.hungarian_matcher.reset()
            for t in range(T):
                _, p = self.hungarian_matcher.step(zs_bt[:, t, 1:])
                perm[:, t, 1:] = p + 1

            def permute_slots(v: Tensor) -> Tensor:  # v: (B*T, K, ...)
                vb = v.unflatten(0, (B, T))
                idx = perm.view(B, T, K, *([1] * (vb.dim() - 3))).expand_as(vb)
                return torch.gather(vb, 2, idx).flatten(0, 1)

            zs = permute_slots(zs)
            slot_means = permute_slots(slot_means)
            log_masks = permute_slots(log_masks)
            masks = log_masks.exp()

        slots_recon, masks_pred = self._decode(zs)


        # Overall (scalar) negative log likelihood: -log p(x|z).
        neg_log_pxs = self._compute_likelihood(x, slots_recon, log_masks)

        # KL for reconstructed masks.
        mask_kl = self._compute_mask_kl(masks, masks_pred)


        loss = neg_log_pxs + self.beta_kl * kl_zs + self.gamma * mask_kl 
        
        # Velocity Extraction
        if self.use_vel_extractor:
            slots = zs.unflatten(0, (B,T))
            slots = slots[:,:,1:,:]

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

        output = {
            "loss": loss,  # scalar
            "mask": masks.unsqueeze(2),  # masks from inference (B*T, slots, 1, H, W)
            "slot": slots_recon,  # (B*T, num slots, 3, H, W)
            "representation": slot_means,  # (B*T, slots, latent dim)
            #
            "z": zs, # (B*T, K, D)
            "neg_log_p_x": neg_log_pxs, # scalar
            "kl_mask": mask_kl, # scalar
            "kl_latent": kl_zs,  # scalar
            "mask_pred": masks_pred,  # (B*T, num slots, H, W)
        }
        output.update({k: v.unflatten(0, (B,T)) for k, v in output.items() if v.dim() > 0})
        if self.use_vel_extractor:
            output.update(vel_losses)

        return output

    def _attention_process(self, x: Tensor) -> Tensor:
        scope_shape = list(x.shape)
        scope_shape[1] = 1
        log_scope = torch.zeros(scope_shape, device=x.device)
        log_masks = []
        for i in range(self.num_slots - 1):
            # Mask and scope: (B, 1, H, W)
            log_mask, log_scope = self.attention(x, log_scope)
            log_masks.append(log_mask)
        log_masks.append(log_scope)
        log_masks = torch.cat(log_masks, dim=1)  # (B, num slots, H, W)
        return log_masks

    def _compute_likelihood(
        self, x: Tensor, slots: Tensor, log_masks: Tensor
    ) -> Tensor:
        sigma = torch.full(
            [
                self.num_slots,
            ],
            self.fg_sigma,
            device=x.device,
        )
        sigma[0] = self.bg_sigma
        sigma = sigma.reshape([1, self.num_slots, 1, 1, 1])
        dist = dists.Normal(slots, sigma)
        log_pxs_masked = log_masks.unsqueeze(2) + dist.log_prob(x.unsqueeze(1))

        # Global negative log likelihood p(x|z): scalar.
        neg_log_pxs = -log_pxs_masked.logsumexp(dim=1).mean(dim=0).sum()

        return neg_log_pxs

    def _compute_mask_kl(self, masks: Tensor, masks_pred: Tensor) -> Tensor:
        bs = len(masks)
        d_masks = self._make_mask_distribution(masks)
        d_masks_pred = self._make_mask_distribution(masks_pred)
        kl_masks = dists.kl_divergence(d_masks, d_masks_pred)
        kl_masks = kl_masks.sum() / bs
        return kl_masks

    @staticmethod
    def _make_mask_distribution(masks: Tensor) -> dists.Distribution:
        flat_masks = masks.permute(0, 2, 3, 1).flatten(0, 2)  # (B*H*W, n_slots)
        flat_masks = flat_masks.clamp_min(1e-5)
        d_masks = dists.Categorical(probs=flat_masks)
        return d_masks

    def _encode(self, x: Tensor, log_masks: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # x: (B, 3, H, W) -> (B, num slots, 3, H, W)
        # log_masks: (B, num slots, H, W) -> (B, num slots, 1, H, W)
        x = x.unsqueeze(1).repeat(1, self.num_slots, 1, 1, 1)
        log_masks = log_masks.unsqueeze(2)

        # Encoder input: (B * num slots, RGB+mask, H, W).
        encoder_input = torch.cat([x, log_masks], 2).flatten(0, 1)

        # Encode and reshape parameters to (B, num slots, latent dim).
        mean, log_sigma = self.encoder(encoder_input).chunk(2, dim=1)
        sigma = log_sigma.exp()
        mean = mean.unflatten(0, [x.shape[0], self.num_slots])  # (B, num_slots, D)
        sigma = sigma.unflatten(0, [x.shape[0], self.num_slots])  # (B, num_slots, D)

        # Return mean, sample, and KL.
        latent_normal = dists.Normal(mean, sigma)
        kl_z = dists.kl_divergence(latent_normal, self.prior_dist)
        kl_z = kl_z.sum(dim=[1, 2]).mean(0)  # sum over latent dimensions and slots
        z = latent_normal.rsample()  # (B, num_slots, D)
        return z, kl_z, mean

    def _decode(self, zs: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = zs.shape[0]
        zs = zs.flatten(0, 1)  # (B * num slots, D)
        decoder_output = self.decoder(zs)

        # (B * num slots, 3, H, W)
        slots_recon = decoder_output[:, :3].sigmoid()
        # (B, num slots, 3, H, W)
        slots_recon = slots_recon.unflatten(0, [batch_size, self.num_slots])

        # (B * num slots, 1, H, W)
        mask_pred = decoder_output[:, 3:]
        # (B, num slots, H, W)
        mask_pred = mask_pred.unflatten(0, [batch_size, self.num_slots]).squeeze(2)
        mask_pred = mask_pred.softmax(dim=1)

        return slots_recon, mask_pred


class AttentionNet(nn.Module):
    def __init__(self, input_channels: int, num_blocks: int):
        super().__init__()
        self.unet = UNet(input_channels, num_blocks)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: Tensor, log_scope: Tensor) -> Tuple[Tensor, Tensor]:
        inp = torch.cat((x, log_scope), 1)
        logits = self.unet(inp)
        # logits = self.last_conv(logits)
        log_alpha = self.log_softmax(logits)
        log_mask = log_scope + log_alpha[:, 0:1]  # log (scope * alpha)
        log_scope = log_scope + log_alpha[:, 1:2]  # log (scope * (1-alpha))
        return log_mask, log_scope