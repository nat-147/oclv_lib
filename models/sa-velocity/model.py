from dataclasses import dataclass, field
from math import sqrt
from typing import Dict, List, Literal, Optional, Tuple, Union

from omegaconf import DictConfig

import torch
from torch import Tensor, nn

from models.base_model import BaseModel
from models.nn_utils import get_conv_output_shape, make_sequential_from_config
from models.shared.nn import PositionalEmbedding
from models.utils import HungarianMatcher

from velocity_extraction.model import VelocityExtractor


class EncoderConfig(Dict):
    channels: List[int]
    kernels: List[int]
    strides: List[int]
    paddings: List[int]
    width: int
    height: int
    input_channels: int = 3


class DecoderConfig(Dict):
    conv_tranposes: List[bool]
    channels: List[int]
    kernels: List[int]
    strides: List[int]
    paddings: List[int]
    width: int
    height: int
    input_channels: int = 3


class Encoder(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        channels: List[int] = (32, 32, 32, 32),
        kernels: List[int] = (5, 5, 5, 5),
        strides: List[int] = (1, 1, 1, 1),
        paddings: List[int] = (2, 2, 2, 2),
        input_channels: int = 3,
        batchnorms: List[bool] = tuple([False] * 4),
    ):
        super().__init__()
        assert len(kernels) == len(strides) == len(paddings) == len(channels)
        self.conv_bone = make_sequential_from_config(
            input_channels,
            channels,
            kernels,
            batchnorms,
            False,
            paddings,
            strides,
            "relu",
            try_inplace_activation=True,
        )
        output_channels = channels[-1]
        output_width, output_height = get_conv_output_shape(
            width, height, kernels, paddings, strides
        )
        self.pos_embedding = PositionalEmbedding(
            output_width, output_height, output_channels
        )
        self.lnorm = nn.LayerNorm(output_channels)
        self.conv_1x1 = [
            nn.Conv1d(output_channels, output_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(output_channels, output_channels, kernel_size=1),
        ]
        self.conv_1x1 = nn.Sequential(*self.conv_1x1)

    def forward(self, x: Tensor) -> Tensor:
        conv_output = self.conv_bone(x)
        conv_output = self.pos_embedding(conv_output)
        conv_output = conv_output.flatten(2, 3).permute(0, 2, 1)  # bs x c x (w * h)
        conv_output = self.lnorm(conv_output).permute(0, 2, 1)
        return self.conv_1x1(conv_output)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        width: int,
        height: int,
        channels: List[int] = (32, 32, 32, 4),
        kernels: List[int] = (5, 5, 5, 3),
        strides: List[int] = (1, 1, 1, 1),
        paddings: List[int] = (2, 2, 2, 1),
        output_paddings: List[int] = (0, 0, 0, 0),
        conv_transposes: List[bool] = tuple([False] * 4),
        activations: List[str] = tuple(["relu"] * 4),
    ):
        super().__init__()
        self.conv_bone = []
        assert len(channels) == len(kernels) == len(strides) == len(paddings)
        if conv_transposes:
            assert len(channels) == len(output_paddings)

        self.pos_embedding = PositionalEmbedding(width, height, input_channels)
        self.width = width
        self.height = height

        self.conv_bone = make_sequential_from_config(
            input_channels,
            channels,
            kernels,
            False,
            False,
            paddings,
            strides,
            activations,
            output_paddings,
            conv_transposes,
            try_inplace_activation=True,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.pos_embedding(x)
        output = self.conv_bone(x)
        img, mask = output[:, :3], output[:, -1:]
        return img, mask


class SlotAttentionModule(nn.Module):
    def __init__(self, num_slots, channels_enc, dim, iters, eps=1e-8, hidden_dim=128, init_type='gaussian'):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim**-0.5
        self.init_type = init_type

        if init_type=='gaussian':
            self.slots_mu = nn.Parameter(torch.empty(1, 1, dim))
            self.slots_log_sigma = nn.Parameter(torch.empty(1, 1, dim))
            with torch.no_grad():
                limit = sqrt(6.0 / (1 + dim))
                torch.nn.init.uniform_(self.slots_mu, -limit, limit)
                torch.nn.init.uniform_(self.slots_log_sigma, -limit, limit)
        elif init_type=='learned':
            self.slots_init = nn.Parameter(torch.randn(1, self.num_slots, dim))
                    
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(channels_enc, dim, bias=False)
        self.to_v = nn.Linear(channels_enc, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

        self.norm_input = nn.LayerNorm(channels_enc, eps=0.001)
        self.norm_slots = nn.LayerNorm(dim, eps=0.001)
        self.norm_pre_ff = nn.LayerNorm(dim, eps=0.001)
        self.dim = dim

    def forward(self, inputs: Tensor, slots_init: Optional[Tensor] = None, batch_traj: Optional[Tuple[int]] = None) -> Tensor:
        if batch_traj is not None:
            B, T = batch_traj
        else:
            B = inputs.shape[0]

        if slots_init is None:
            if batch_traj is not None:
                slots = slots.unsqueeze(1).expand(B, T, -1, -1).flatten(0, 1)

            if self.init_type == 'gaussian':
                mu = self.slots_mu.expand(B, self.num_slots, -1)
                sigma = self.slots_log_sigma.expand(B, self.num_slots, -1).exp()
                slots = mu + sigma * torch.randn_like(sigma) 
            elif self.init_type == 'learned':   
                slots = self.slots_init.expand(B, -1, -1)

            iters = self.iters[0]
        else :
            slots = slots_init
            iters = self.iters[1]

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum("bjd,bij->bid", v, attn)

            slots = self.gru(
                updates.reshape(-1, self.dim), slots_prev.reshape(-1, self.dim)
            )
            slots = slots.reshape(-1, self.num_slots, self.dim)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


@dataclass(eq=False, repr=False)
class SAVelocityAE(BaseModel):
    latent_size: int

    vel_loss_weights : Union[dict, DictConfig] 
    lambda_warmup_steps_interval: List[int]

    encoder_params: EncoderConfig
    decoder_params: DecoderConfig
    attention_iters: List[int]
    slot_init_type: str = 'gaussian'
    temporal_prop: bool = False
    input_channels: int = 3
    eps: float = 1e-8
    mlp_size: int = 128
    
    w_broadcast: Union[int, Literal["dataset"]] = "dataset"
    h_broadcast: Union[int, Literal["dataset"]] = "dataset"

    use_vel_extractor: bool = True
    use_input_pairs: bool = False
    hungarian_matching: bool = True
    use_lambda_warmup: bool = False
    

    vel_extractor: VelocityExtractor = field(init=False)
    
    encoder: Encoder = field(init=False)
    decoder: Decoder = field(init=False)


    def __post_init__(self):
        super().__post_init__()
        self.loss_fn = nn.MSELoss(reduction='mean')
        if self.w_broadcast == "dataset":
            self.w_broadcast = self.width
        if self.h_broadcast == "dataset":
            self.h_broadcast = self.height
        self.encoder_params.update(
            width=self.width, height=self.height, input_channels=self.input_channels
        )
        self.encoder = Encoder(**self.encoder_params)
        self.slot_attention = SlotAttentionModule(
            self.num_slots,
            self.encoder_params["channels"][-1],
            self.latent_size,
            self.attention_iters,
            self.eps,
            self.mlp_size,
            self.slot_init_type,
        )

        self.decoder_params.update(
            width=self.w_broadcast,
            height=self.h_broadcast,
            input_channels=self.latent_size,
        )
        self.decoder = Decoder(**self.decoder_params)

        if self.use_vel_extractor:
            self.vel_loss_weights = dict(self.vel_loss_weights)
            self.vel_extractor = VelocityExtractor(slot_dim=self.latent_size, num_slots=self.num_slots, v_dim=2, loss_weights=self.vel_loss_weights, use_input_pairs=self.use_input_pairs)
            if self.hungarian_matching:
                self.hungarian_matcher = HungarianMatcher(self.num_slots)

        '''
        self.predictor = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.Linear(self.latent_size, 2*self.latent_size),
            nn.ReLU(2*self.latent_size),
            nn.Linear(2*self.latent_size, self.latent_size)
        )
        '''
        
        
        

    @property
    def slot_size(self) -> int:
        return self.latent_size

    def spatial_broadcast(self, slot: Tensor) -> Tensor:
        slot = slot.unsqueeze(-1).unsqueeze(-1)
        return slot.repeat(1, 1, self.w_broadcast, self.h_broadcast)

    def encoder_process(self, x: Tensor) -> Tensor:
        B, T = x.shape[:2]
        encoded = self.encoder(x.flatten(0, 1))
        encoded = encoded.permute(0, 2, 1)
        
        if self.temporal_prop:
            encoded = encoded.unflatten(0, (B, T))
            slots_init = None
            zs = []
            for t in range(T):    
                z = self.slot_attention(encoded[:, t], slots_init) # B, K, D
                zs.append(z)
                slots_init = z.detach()

            zs = torch.stack(zs, dim=1)
        else:
            zs = self.slot_attention(encoded).unflatten(0, (B, T))

        return zs

    def decoder_process(self, slot: Tensor) -> Tensor:
        slots = slot.reshape(-1, self.latent_size)
        slots = self.spatial_broadcast(slots)
        img_slots, masks = self.decoder(slots)
        img_slots = img_slots.view(*slot.shape[:-1], 3, self.height, self.width)
        masks = masks.view(*slot.shape[:-1], 1, self.height, self.width)
        masks = masks.softmax(dim=2)

        recon_slots_masked = img_slots * masks
        recon_img = recon_slots_masked.sum(dim=2)

        return recon_img, img_slots, masks


    def forward(self, x: Tensor, step: int = None) -> dict:
        # input: (B, T, 3, H, W)
        # Forward pass through recurrent attention network.
        B, T, C, H, W = x.shape
        if C > 3:
            x, m  = torch.split(x, 3, dim=2)
        else:
            m = None

        zs = self.encoder_process(x)

        if  self.hungarian_matching:
            zs_matched = zs.clone()
            self.hungarian_matcher.reset()
            for t in range(T):
                zs_matched[:,t] = self.hungarian_matcher(zs[:,t])
            zs = zs_matched
        
        recon_img, img_slots, masks = self.decoder_process(zs)
        recon_loss = self.loss_fn(x, recon_img)

        loss = recon_loss.clone()


        if self.use_vel_extractor:
            #Velocity Extraction
            slots = zs
            vel_losses, pred_zs = self.vel_extractor(slots)
            vel_weight = self.vel_loss_weights["vel_loss_weight_lambda"]

            vel_loss = sum(vel_losses.values()).squeeze()
            
            if self.use_lambda_warmup and step is not None:
                warmup_start, warmup_end = self.lambda_warmup_steps_interval

                vel_weight *= min(
                    max(0, step - warmup_start) / (warmup_end - warmup_start),
                    1
                )

            loss += vel_weight * vel_loss 
                    
            vel_losses.update({k: vel_weight*v for k, v in vel_losses.items()})

        recon_slots_output = img_slots

        
        results = {
            "x" : x.view(B, T, 3, H, W),
            "loss": loss,  # scalar
            "mask": masks,  # (B, T, slots, 1, H, W)
            "slot": recon_slots_output,  # (B, T, slots, 3, H, W)
            "representation": zs,  # (B, T, slots, latent dim)
            #
            "reconstruction": recon_img,  # (B, T, 3, H, W)
            "x_recon_loss": recon_loss,
        }
        if self.use_vel_extractor:
            results.update(vel_losses)

        return results


