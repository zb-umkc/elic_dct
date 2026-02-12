"""
A simple test algorithm to rewrite the network
"""
import math
import time
import sys
from thop import profile
from ptflops import get_model_complexity_info

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.layers import trunc_normal_
from compressai.layers import (
    AttentionBlock,
    conv3x3,
    CheckerboardMaskedConv2d, conv1x1
)
from compressai.models.google import CompressionModel, GaussianConditional
from compressai.ops import quantize_ste as ste_round
from compressai.models.utils import conv, deconv, update_registered_buffers

from ELICUtils.encoding.rle import *


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class ResidualBottleneckBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv1 = conv1x1(in_ch, in_ch//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_ch//2, in_ch//2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(in_ch//2, in_ch)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        out = out + identity
        return out


class Quantizer():
    def quantize(self, inputs, quantize_type="noise"):
        if quantize_type == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs
        elif quantize_type == "ste":
            return torch.round(inputs) - inputs.detach() + inputs
        else:
            return torch.round(inputs)
        

# class LearnableGroupSizes(nn.Module):
#     def __init__(self, G):
#         super().__init__()
#         self.G = G
#         self.logits = nn.Parameter(torch.zeros(G))

#     def forward(self):
#         p = torch.softmax(self.logits, dim=0)
#         return p
    

# class SoftChannelGrouping(nn.Module):
#     def __init__(self, M, temperature=0.01):
#         super().__init__()
#         self.M = M
#         self.temperature = temperature

#     def forward(self, proportions):
#         """
#         proportions: tensor of shape [G]
#         returns: assignment matrix of shape [M, G]
#         """
#         G = proportions.size(0)

#         # cumulative boundaries
#         boundaries = torch.cumsum(proportions, dim=0)  # [p1, p1+p2, ...]
#         boundaries = torch.cat([torch.tensor([0.0], device=proportions.device),
#                                 boundaries], dim=0)  # shape [G+1]

#         # normalized channel positions
#         channel_pos = torch.linspace(0, 1, self.M, device=proportions.device)  # [M]

#         # Soft indicator per group
#         W = []
#         for i in range(G):
#             left = boundaries[i]
#             right = boundaries[i+1]

#             # Smooth step functions (low temp -> steeper curve)
#             left_mask  = torch.sigmoid((channel_pos - left)  / self.temperature)
#             right_mask = torch.sigmoid((channel_pos - right) / self.temperature)

#             group_mask = left_mask - right_mask  # [M]
#             W.append(group_mask)

#         W = torch.stack(W, dim=1)  # [M, G]

#         # Normalize so each channel sums to 1 across groups
#         W = W / (W.sum(dim=1, keepdim=True) + 1e-6)

#         return W
    

class STEGroupSizes(nn.Module):
    """
    Learnable contiguous partition of latent channels using STE.
    """
    def __init__(self, num_channels=320, num_groups=5, group_size_init=None):
        super().__init__()
        self.C = num_channels
        self.G = num_groups
        
        # Learnable logits for softmax proportions
        if group_size_init is not None:
            groups = torch.tensor(group_size_init)
            assert self.C == groups.sum()
            assert self.G == len(groups)
            probs = groups/self.C
            self.logits = nn.Parameter(torch.log(probs) - torch.log(probs).mean())
        else:
            self.logits = nn.Parameter(torch.zeros(self.G))

    def soft_intervals(self, pos, bounds, sharpness=500.0):
        """
        pos: channel positions [C] normalized 0..1
        bounds: cumulative boundaries [G]
        Returns soft membership: [G, C]
        """
        C = pos.shape[0]
        G = bounds.shape[0]

        # Start and end of each interval
        start = torch.cat([torch.tensor([0.0], device=bounds.device), bounds[:-1]])
        end   = bounds

        # Soft "inside the interval" indicator using sigmoid edges
        left  = torch.sigmoid(sharpness * (pos[None, :] - start[:, None]))
        right = torch.sigmoid(-sharpness * (pos[None, :] - end[:, None]))
        soft = left * right
        soft = soft / (soft.sum(dim=0, keepdim=True) + 1e-8)

        return soft  # shape [G, C]

    def forward(self):
        """
        Output:
            mask: [G, C] -- straight-through hard assignment mask
            soft: [G, C] -- soft assignment (useful for debugging)
        """
        # Proportions
        probs = F.softmax(self.logits, dim=0)  # shape [G]
        bounds = torch.cumsum(probs, dim=0)    # cumulative
        
        # Channel positions [C] in [0, 1]
        pos = torch.linspace(0, 1, self.C, device=self.logits.device)

        # Soft memberships
        soft = self.soft_intervals(pos, bounds)  # [G, C]

        # Hard memberships (one-hot)
        hard_idx = torch.argmax(soft, dim=0)     # [C]
        hard = F.one_hot(hard_idx, num_classes=self.G).float().t()  # [G, C]
        group_sizes = torch.bincount(hard_idx, minlength=self.G)

        # Straight-through estimator:
        mask = hard + soft - soft.detach()       # Hard forward, soft backward
        # print(mask.shape)
        # print(mask[0])
        # print(mask[1])
        # print(mask[2])

        return mask, soft, group_sizes


class SAREliC(CompressionModel):
    def __init__(
        self,
        N=192,
        M=320,
        num_slices=5,
        groups=[0, 16, 16, 32, 64, 192], # 16,16,32,64,M-128
        input_channels=3,
        **kwargs,
    ):
        """ELIC 2022; uneven channel groups with checkerboard context.

        Context model from [He2022], with minor simplifications.
        Based on modified attention model architecture from [Cheng2020].

        .. note::

            This implementation contains some differences compared to the
            original [He2022] paper. For instance, the implemented context
            model only uses the first and the most recently decoded channel
            groups to predict the current channel group. In contrast, the
            original paper uses all previously decoded channel groups.

        [He2022]: `"ELIC: Efficient Learned Image Compression with
        Unevenly Grouped Space-Channel Contextual Adaptive Coding"
        <https://arxiv.org/abs/2203.10886>`_, by Dailan He, Ziming Yang,
        Weikun Peng, Rui Ma, Hongwei Qin, and Yan Wang, CVPR 2022.

        [Cheng2020]: `"Learned Image Compression with Discretized Gaussian
        Mixture Likelihoods and Attention Modules"
        <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun,
        Masaru Takeuchi, and Jiro Katto, CVPR 2020.

        ## Please See from compressai.models.sensetime import Elic2022Official, Elic2022Chandelier
        - Includes official implementations and chandelier implementations with added comments and explanations

        Args:
             N (int): Number of main network channels
             M (int): Number of latent space channels
             num_slices (int): Number of slices/groups
             groups (list[int]): Number of channels in each channel group
        """
        super().__init__(entropy_bottleneck_channels=N)

        assert len(groups) == num_slices + 1
        assert sum(groups) == M
        assert groups[0] == 0

        self.N = int(N)
        self.M = int(M)
        self.num_slices = num_slices
        self.groups = groups
        self.input_channels = input_channels
        # self.proportions_module = LearnableGroupSizes(self.num_slices)
        # self.assignment_module = SoftChannelGrouping(M=self.M, temperature=0.01)
        self.grouping_module = STEGroupSizes(
            num_channels=self.M,
            num_groups=self.num_slices,
            group_size_init=[16, 16, 32, 64, 192],
        )

        self.g_a = nn.Sequential(
            conv(self.input_channels*4*4, N, kernel_size=5, stride=1),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            #conv(N, N, stride=1),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            AttentionBlock(N),
            conv(N, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            conv(N, M),
            AttentionBlock(M),
        )

        #NOTE: We may want to replace the deconv with a interpolation layer see https://distill.pub/2016/deconv-checkerboard/ 
        self.g_s = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            deconv(N, N),
            AttentionBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            deconv(N, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            deconv(N, self.input_channels),
        )

        self.h_a = nn.Sequential(
            conv3x3(M, N),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N * 3 // 2),
            nn.ReLU(inplace=True),
            conv3x3(N * 3 // 2, 2 * M),
        )

        # In [He2022], this is labeled "g_ch^(k)".
        # channel_context 
        self.cc_transforms = nn.ModuleList(
            nn.Sequential(
                conv(
                    # Input: first group, and most recently decoded group.
                    self.M + (i > 1) * self.M,
                    # self.groups[1] + (i > 1) * self.groups[i],
                    224,
                    stride=1,
                    kernel_size=5,
                ),
                nn.ReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=5),
                nn.ReLU(inplace=True),
                conv(128, self.M * 2, stride=1, kernel_size=5),
                # conv(128, self.groups[i + 1] * 2, stride=1, kernel_size=5),
            )
            for i in range(1, num_slices)
        )  ## from https://github.com/tensorflow/compression/blob/master/models/ms2020.py

        # In [He2022], this is labeled "g_sp^(k)".
        # spatial_context
        self.context_prediction = nn.ModuleList(
            CheckerboardMaskedConv2d(
                self.M, self.M * 2, kernel_size=5, padding=2, stride=1
                # self.groups[i], self.groups[i] * 2, kernel_size=5, padding=2, stride=1
            )
            for i in range(1, num_slices + 1)
        )  ## from https://github.com/JiangWeibeta/Checkerboard-Context-Model-for-Efficient-Learned-Image-Compression/blob/main/version2/layers/CheckerboardContext.py

        # In [He2022], this is labeled "Param Aggregation".
        self.ParamAggregation = nn.ModuleList(
            nn.Sequential(
                conv1x1(
                    # Input: spatial context, channel context, and hyper params.
                    (self.M * 2) + ((i > 1) * self.M * 2) + (self.M * 2),
                    # self.groups[i] * 2 + (i > 1) * self.groups[i] * 2 + M * 2,
                    self.M * 2,
                ),
                nn.ReLU(inplace=True),
                conv1x1(self.M * 2, 512),
                nn.ReLU(inplace=True),
                conv1x1(512, self.M * 2),
                # conv1x1(512, self.groups[i] * 2),
            )
            for i in range(1, num_slices + 1)
        )  ##from checkboard "Checkerboard Context Model for Efficient Learned Image Compression"" gepç½ç»åæ°

        self.quantizer = Quantizer()

        self.gaussian_conditional = GaussianConditional(None)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def apply_group_mask(self, y, mask):
        """
        y: [B, C, H, W]
        mask: [G, C]
        Returns:
            groups: list of G tensors, each [B, C, H, W]
        """
        B, C, H, W = y.shape
        G = mask.shape[0]

        groups = []
        for g in range(G):
            w = mask[g].view(1, C, 1, 1)   # broadcast
            groups.append(y * w)
        return groups

    def forward(self, x, mode_quant="ste"):
        y = self.g_a(x)

        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        if mode_quant == "ste":
            z_offset = self.entropy_bottleneck._get_medians()
            z_hat = ste_round(z - z_offset) + z_offset

        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)

        # # Soft group assignment for latents
        # B, M, H, W = y.shape
        # proportions = self.proportions_module()  # [G]
        # assignments = self.assignment_module(proportions)  # [M, G]

        mask, soft, group_sizes = self.grouping_module()
        y_slices = self.apply_group_mask(y, mask)  # list of G tensors
        
        # y_slices = []
        anchor_slices = []
        non_anchor_slices = []
        for y_i in y_slices:
            # # Expand assignment weights to broadcast over spatial dims
            # w = assignments[:, i].view(1, M, 1, 1)  # shape [1,M,1,1]

            # # Weighted mask (effectively zeroes out contributions of channels not in group i)
            # y_i = y * w
            # y_slices.append(y_i)

            # Extract anchor and non-anchors for each channel group:
            anchor_i = torch.zeros_like(y_i)
            non_anchor_i = torch.zeros_like(y_i)
            self._copy(anchor_i, y_i, "anchor") # Copies latent values for anchor positions only
            self._copy(non_anchor_i, y_i, "non_anchor") # Same for non-anchors
            anchor_slices.append(anchor_i)
            non_anchor_slices.append(non_anchor_i)

        y_likelihood = []
        y_hat_slices = []
        y_hat_slices_for_gs = []

        for slice_index, y_slice in enumerate(y_slices):
            support = self._calculate_support(
                slice_index, y_hat_slices, latent_means, latent_scales
            )

            # print(f"slice_index: {slice_index}")
            # print(f"support: {support.shape}")
            # print(f"y_hat_i: {y_slice.shape}, anchor: {anchor_slices[slice_index].shape}, non_anchor: {non_anchor_slices[slice_index].shape}")

            y_hat_i, y_hat_for_gs_i, y_likelihood_i = self._checkerboard_forward(
                [y_slice, anchor_slices[slice_index], non_anchor_slices[slice_index]],
                slice_index,
                support,
                mode_quant=mode_quant,
            )

            y_hat_slices.append(y_hat_i)
            y_hat_slices_for_gs.append(y_hat_for_gs_i)
            y_likelihood.append(y_likelihood_i)

        y_likelihoods = torch.cat(y_likelihood, dim=1)
        y_hat = torch.stack(y_hat_slices_for_gs, dim=0).sum(dim=0)  # [B,C,H,W]

        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
                "z": z_likelihoods,
            },
            "group_sizes": group_sizes,
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, x):
        y_enc_start = time.time()
        y = self.g_a(x)
        y_enc = time.time() - y_enc_start

        z_enc_start = time.time()
        z = self.h_a(y)
        z_enc = time.time() - z_enc_start
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        z_dec_start = time.time()
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)
        z_dec = time.time() - z_dec_start

        y_slices = torch.split(y, self.groups[1:], dim=1)

        slice_times = {0: [], 1: [], 2: [], 3: [], 4: []}
        final_slice_times = {}
        
        iterations = 1
        for it in range(iterations+1):
            y_strings = []
            y_hat_slices = []

            for slice_index, y_slice in enumerate(y_slices):
                slice_start = time.time()
                support = self._calculate_support(
                    slice_index, y_hat_slices, latent_means, latent_scales
                )

                y_hat_i, y_strings_i = self._checkerboard_codec(
                    [y_slice.clone(), y_slice.clone()],
                    slice_index,
                    support,
                    y_shape=y.shape[-2:],
                    mode="compress",
                )

                y_hat_slices.append(y_hat_i)
                y_strings.append(y_strings_i)

                # Warmup
                if it > 0:
                    slice_times[slice_index].append(time.time() - slice_start)

        for slice_index in slice_times.keys():
            final_slice_times[slice_index] = {}
            final_slice_times[slice_index]["mean"] = np.mean(slice_times[slice_index])
            final_slice_times[slice_index]["std"] = np.std(slice_times[slice_index])

        strings = [y_strings, z_strings]

        y = y.cpu().numpy()
        y_hat = torch.cat(y_hat_slices, dim=1).squeeze(0).cpu().numpy()
        z_hat = z_hat.squeeze(0).cpu().numpy()

        out_enc = {
            "strings": strings,
            "shape": z.size()[-2:],
            "y": y,
            "y_hat": y_hat,
            "z_hat": z_hat,
            "time": {
                "y_enc": y_enc,
                "z_enc": z_enc,
                "z_dec": z_dec,
                "slices": final_slice_times,
            },
        }

        return out_enc


    def decompress(self, strings, shape, out_enc):
        assert isinstance(strings, list) and len(strings) == 2
        [y_strings, z_strings] = strings

        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)

        slice_times = {0: [], 1: [], 2: [], 3: [], 4: []}
        final_slice_times = {}
        
        iterations = 1
        for it in range(iterations+1):
            y_hat_slices = []

            for slice_index in range(len(self.groups) - 1):
                slice_start = time.time()
                support = self._calculate_support(
                    slice_index, y_hat_slices, latent_means, latent_scales
                )

                y_hat_i, _ = self._checkerboard_codec(
                    y_strings[slice_index],
                    slice_index,
                    support,
                    y_shape=(shape[0] * 4, shape[1] * 4),
                    mode="decompress",
                )

                y_hat_slices.append(y_hat_i)

                # Warmup
                if it > 0:
                    slice_times[slice_index].append(time.time() - slice_start)

        for slice_index in slice_times.keys():
            final_slice_times[slice_index] = {}
            final_slice_times[slice_index]["mean"] = np.mean(slice_times[slice_index])
            final_slice_times[slice_index]["std"] = np.std(slice_times[slice_index])

        y_hat = torch.cat(y_hat_slices, dim=1)

        y_dec_start = time.time()
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        y_dec = time.time() - y_dec_start

        out_dec = {
            "x_hat": x_hat,
            "y_hat": y_hat,
            "time": {
                "y_dec": y_dec,
                "slices": final_slice_times,
            }
        }

        return out_dec

    def inference(self, x, mode_quant="ste"):
        y_enc_start = time.time()
        y = self.g_a(x)
        y_enc = time.time() - y_enc_start

        z_enc_start = time.time()
        z = self.h_a(y)
        z_enc = time.time() - z_enc_start

        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        if mode_quant == "ste":
            z_offset = self.entropy_bottleneck._get_medians()
            z_hat = ste_round(z - z_offset) + z_offset

        z_dec_start = time.time()
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)
        z_dec = time.time() - z_dec_start

        # Extract anchor and non-anchors for each channel group:
        anchor = torch.zeros_like(y)
        non_anchor = torch.zeros_like(y)
        self._copy(anchor, y, "anchor")
        self._copy(non_anchor, y, "non_anchor")
        anchor_split = torch.split(anchor, self.groups[1:], dim=1)
        non_anchor_split = torch.split(non_anchor, self.groups[1:], dim=1)

        y_slices = torch.split(y, self.groups[1:], dim=1)

        y_likelihood = []
        y_hat_slices = []
        params_start = time.time()

        for slice_index, y_slice in enumerate(y_slices):
            support = self._calculate_support(
                slice_index, y_hat_slices, latent_means, latent_scales
            )

            y_hat_i, _, y_likelihood_i = self._checkerboard_forward(
                [y_slice, anchor_split[slice_index], non_anchor_split[slice_index]],
                slice_index,
                support,
                mode_quant=mode_quant,
            )

            y_hat_slices.append(y_hat_i)
            y_likelihood.append(y_likelihood_i)

        params_time = time.time() - params_start
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        y_hat = torch.cat(y_hat_slices, dim=1)
        y_dec_start = time.time()
        x_hat = self.g_s(y_hat)
        y_dec = time.time() - y_dec_start

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "time": {
                "y_enc": y_enc,
                "y_dec": y_dec,
                "z_enc": z_enc,
                "z_dec": z_dec,
                "params": params_time,
            },
        }

    def _apply_quantizer(self, y, means, mode_quant):
        if mode_quant == "noise":
            quantized = self.quantizer.quantize(y, "noise")
            quantized_for_g_s = self.quantizer.quantize(y, "ste")
        elif mode_quant == "ste":
            quantized = self.quantizer.quantize(y - means, "ste") + means
            quantized_for_g_s = self.quantizer.quantize(y - means, "ste") + means
        else:
            raise ValueError(f"Unknown quantization mode: {mode_quant}")
        return quantized, quantized_for_g_s

    def _calculate_support(
        self, slice_index, y_hat_slices, latent_means, latent_scales
    ):
        if slice_index == 0:
            return torch.concat([latent_means, latent_scales], dim=1)

        cc_means, cc_scales = self.cc_transforms[slice_index - 1](
            y_hat_slices[0]
            if slice_index == 1
            else torch.concat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
        ).chunk(2, 1)

        return torch.concat([cc_means, cc_scales, latent_means, latent_scales], dim=1)

    def _checkerboard_forward(self, y_input, slice_index, support, mode_quant):
        y, y_anchor, y_non_anchor = y_input
        # NOTE: y == y_anchor + y_non_anchor

        B, C, H, W = y.shape
        means = torch.zeros_like(y)
        scales = torch.zeros_like(y)

        y_anchor_hat, y_anchor_hat_for_gs = self._checkerboard_forward_step(
            y_anchor,
            slice_index,
            support,
            means,
            scales,
            ctx_params=torch.zeros((B, C * 2, H, W), device=support.device),
            mode_quant=mode_quant,
            mode="anchor",
        )

        y_non_anchor_hat, y_non_anchor_hat_for_gs = self._checkerboard_forward_step(
            y_non_anchor,
            slice_index,
            support,
            means,
            scales,
            ctx_params=self.context_prediction[slice_index](y_anchor_hat),
            mode_quant=mode_quant,
            mode="non_anchor",
        )

        y_hat = y_anchor_hat + y_non_anchor_hat
        y_hat_for_gs = y_anchor_hat_for_gs + y_non_anchor_hat_for_gs

        # Entropy estimation
        idx = torch.nonzero(y[0, :, 0, 0], as_tuple=True)[0]
        y_sub = y[:, idx, :, :]
        means_sub = means[:, idx, :, :]
        scales_sub = scales[:, idx, :, :]
        _, y_likelihood = self.gaussian_conditional(y_sub, scales_sub, means=means_sub)

        return y_hat, y_hat_for_gs, y_likelihood

    def _checkerboard_forward_step(
        self, y, slice_index, support, means, scales, ctx_params, mode_quant, mode
    ):
        means_new, scales_new = self.ParamAggregation[slice_index](
            torch.concat([ctx_params, support], dim=1)
        ).chunk(2, 1)

        self._copy(means, means_new, mode)
        self._copy(scales, scales_new, mode)

        y_hat, y_hat_for_gs = self._apply_quantizer(y, means_new, mode_quant)

        self._keep_only(y_hat, mode)
        self._keep_only(y_hat_for_gs, mode)

        return y_hat, y_hat_for_gs

    def _checkerboard_codec(self, y_input, slice_index, support, y_shape, mode):
        y_anchor_input, y_non_anchor_input = y_input

        # NOTE: y.shape == (B, C, H, W)
        B, *_ = support.shape
        C = self.groups[slice_index + 1]
        H, W = y_shape

        anchor_strings, y_anchor_hat = self._checkerboard_codec_step(
            y_anchor_input,
            slice_index,
            support,
            ctx_params=torch.zeros((B, C * 2, H, W), device=support.device),
            mode_codec=mode,
            mode_step="anchor",
        )

        non_anchor_strings, y_non_anchor_hat = self._checkerboard_codec_step(
            y_non_anchor_input,
            slice_index,
            support,
            ctx_params=self.context_prediction[slice_index](y_anchor_hat),
            mode_codec=mode,
            mode_step="non_anchor",
        )

        y_hat = y_anchor_hat + y_non_anchor_hat
        y_strings = [anchor_strings, non_anchor_strings]

        return y_hat, y_strings

    def _checkerboard_codec_step(
        self, y_input, slice_index, support, ctx_params, mode_codec, mode_step
    ):
        means_new, scales_new = self.ParamAggregation[slice_index](
            torch.concat([ctx_params, support], dim=1)
        ).chunk(2, 1)

        device = means_new.device
        decode_shape = means_new.shape
        B, C, H, W = decode_shape
        encode_shape = (B, C, H, W // 2)

        means = torch.zeros(encode_shape, device=device)
        scales = torch.zeros(encode_shape, device=device)
        self._unembed(means, means_new, mode_step)
        self._unembed(scales, scales_new, mode_step)

        indexes = self.gaussian_conditional.build_indexes(scales)

        if mode_codec == "compress":
            y = y_input
            y_encode = torch.zeros(encode_shape, device=device)
            self._unembed(y_encode, y, mode_step)
            strings = self.gaussian_conditional.compress(y_encode, indexes, means=means)

        elif mode_codec == "decompress":
            strings = y_input

        quantized = self.gaussian_conditional.decompress(strings, indexes, means=means)
        y_decode = torch.zeros(decode_shape, device=device)
        self._embed(y_decode, quantized, mode_step)

        return strings, y_decode

    def _copy(self, dest, src, mode):
        """Copy pixels in the current mode (i.e. anchor / non-anchor)."""
        if mode == "anchor":
            dest[:, :, 0::2, 0::2] = src[:, :, 0::2, 0::2]
            dest[:, :, 1::2, 1::2] = src[:, :, 1::2, 1::2]
        elif mode == "non_anchor":
            dest[:, :, 0::2, 1::2] = src[:, :, 0::2, 1::2]
            dest[:, :, 1::2, 0::2] = src[:, :, 1::2, 0::2]

    def _unembed(self, dest, src, mode):
        """Compactly extract pixels for the given mode.

        src                     dest

        â  â¡ â  â¡                 â  â            â¡ â¡
        â¡ â  â¡ â        --->      â  â            â¡ â¡
        â  â¡ â  â¡                 â  â            â¡ â¡
                                anchor        non-anchor
        """
        if mode == "anchor":
            dest[:, :, 0::2, :] = src[:, :, 0::2, 0::2]
            dest[:, :, 1::2, :] = src[:, :, 1::2, 1::2]
        elif mode == "non_anchor":
            dest[:, :, 0::2, :] = src[:, :, 0::2, 1::2]
            dest[:, :, 1::2, :] = src[:, :, 1::2, 0::2]

    def _embed(self, dest, src, mode):
        """Insert pixels for the given mode.

        src                                   dest

        â  â            â¡ â¡                     â  â¡ â  â¡
        â  â            â¡ â¡           --->      â¡ â  â¡ â 
        â  â            â¡ â¡                     â  â¡ â  â¡
        anchor        non-anchor
        """
        if mode == "anchor":
            dest[:, :, 0::2, 0::2] = src[:, :, 0::2, :]
            dest[:, :, 1::2, 1::2] = src[:, :, 1::2, :]
        elif mode == "non_anchor":
            dest[:, :, 0::2, 1::2] = src[:, :, 0::2, :]
            dest[:, :, 1::2, 0::2] = src[:, :, 1::2, :]

    def _keep_only(self, y, mode):
        """Keep only pixels in the current mode, and zero out the rest."""
        if mode == "anchor":
            # Zero the non-anchors:
            y[:, :, 0::2, 1::2] = 0
            y[:, :, 1::2, 0::2] = 0
        elif mode == "non_anchor":
            # Zero the anchors:
            y[:, :, 0::2, 0::2] = 0
            y[:, :, 1::2, 1::2] = 0


if __name__ == "__main__":
    import compressai
    import matplotlib.pyplot as plt
    import os

    from ELICUtilis.datasets.utils import SarIQDataset
    from option_NGA import args
    from torch.utils.data import DataLoader
    from dct_fast import ImageDCT
    

    block_size = 4
    dct = ImageDCT(block_size)

    dataset = SarIQDataset(os.path.join('/home/pmc4p/', args.validation_dataset), train=False, data_type=args.datatype, min_val=args.min_val, max_val=args.max_val)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    img = next(iter(loader)).cuda()

    img = dct.dct_2d(img)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    compressai.set_entropy_coder(args.entropy_coder)
    #data       = torch.load(args.test_model)
    #state_dict = load_state_dict(data['state_dict'])
    model_cls  = SAREliC(N=args.N, M=args.M, input_channels=args.inputchannels).to(device)
    #model_cls.load_state_dict(state_dict)
    model      = model_cls.eval()
    #model = SAREliC(N=192, M=320, input_channels=args.inputchannels).cuda()

    flops, params = get_model_complexity_info(model, (32, 64, 64), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)
    flops, params = profile(model, (img,))
    print('flops: ', flops, 'params: ', params)

    # out = model(img)
    # print(out["x_hat"].shape)

    # avg_time = 0
    # for _ in range(10):
    #     out = model.compress(img)
    #     avg_time += out["time"]["y_enc"] + out["time"]["y_dec"] + out["time"]["z_enc"] + out["time"]["z_dec"] + out["time"]["params"]
    # print("Time: ", avg_time / 10)