"""Spatial and sparse-product blocks for CASPR-Net."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from .activations import build_activation
from .reparameterization import CasprDenseStep, CasprSparseStep, fold_branches


class CasprSpatialGroup(nn.Module):
    """Parallel depthwise branches that fold to one dense 3x3 operator."""

    def __init__(self, channels: int, num_branches: int = 2,
                 kernel_size: int = 3, metric_enabled: bool = True,
                 metric_update_interval: int = 32):
        super().__init__()
        if num_branches < 1 or kernel_size % 2 == 0:
            raise ValueError("num_branches must be positive and kernel_size must be odd")
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.branches = nn.ModuleList(
            CasprDenseStep(channels, kernel_size, metric_enabled=metric_enabled,
                           metric_update_interval=metric_update_interval)
            for _ in range(num_branches)
        )
        self.deploy_conv: nn.Conv2d | None = None

    @property
    def is_reparameterized(self) -> bool:
        return self.deploy_conv is not None

    def forward(self, x: Tensor) -> Tensor:
        if self.deploy_conv is not None:
            return self.deploy_conv(x)
        padded = F.pad(x, (self.padding,) * 4)
        output = self.branches[0](padded)
        for branch in self.branches[1:]:
            output = output + branch(padded)
        return output

    @torch.no_grad()
    def reparameterize(self) -> "CasprSpatialGroup":
        if self.deploy_conv is not None:
            return self
        if self.training:
            raise RuntimeError("Call eval() before reparameterize()")
        weight, bias = fold_branches(self.branches)
        self.deploy_conv = nn.Conv2d(
            self.channels, self.channels, self.kernel_size, padding=self.padding,
            groups=self.channels, bias=True, device=weight.device, dtype=weight.dtype)
        self.deploy_conv.weight.copy_(weight)
        self.deploy_conv.bias.copy_(bias)
        self.branches = nn.ModuleList()
        return self


class CasprSparseLevel(nn.Module):
    """A reparameterizable block-diagonal channel factor."""

    def __init__(self, channels: int, group_size: int = 4,
                 num_branches: int = 2, metric_enabled: bool = True,
                 metric_update_interval: int = 32):
        super().__init__()
        if channels % group_size:
            raise ValueError(
                f"channels={channels} must be divisible by group_size={group_size}")
        if num_branches < 1:
            raise ValueError("num_branches must be positive")
        self.channels = channels
        self.group_size = group_size
        self.groups = channels // group_size
        self.branches = nn.ModuleList(
            CasprSparseStep(channels, group_size, metric_enabled=metric_enabled,
                            metric_update_interval=metric_update_interval)
            for _ in range(num_branches)
        )
        self.deploy_conv: nn.Conv2d | None = None

    @property
    def is_reparameterized(self) -> bool:
        return self.deploy_conv is not None

    def forward(self, x: Tensor) -> Tensor:
        if self.deploy_conv is not None:
            return self.deploy_conv(x)
        output = self.branches[0](x)
        for branch in self.branches[1:]:
            output = output + branch(x)
        return output

    @torch.no_grad()
    def reparameterize(self) -> "CasprSparseLevel":
        if self.deploy_conv is not None:
            return self
        if self.training:
            raise RuntimeError("Call eval() before reparameterize()")
        weight, bias = fold_branches(self.branches)
        self.deploy_conv = nn.Conv2d(
            self.channels, self.channels, 1, groups=self.groups, bias=True,
            device=weight.device, dtype=weight.dtype)
        self.deploy_conv.weight.copy_(weight)
        self.deploy_conv.bias.copy_(bias)
        self.branches = nn.ModuleList()
        return self


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    """Parameter-free perfect shuffle between sparse factors."""
    batch, channels, height, width = x.shape
    if channels % groups:
        raise ValueError("channels must be divisible by shuffle groups")
    x = x.reshape(batch, groups, channels // groups, height, width)
    return x.transpose(1, 2).contiguous().reshape(batch, channels, height, width)


class CasprSparseProduct(nn.Module):
    """Product of sparse channel factors with inter-level perfect shuffles."""

    def __init__(self, channels: int, group_size: int = 4,
                 levels: int | None = None, num_branches: int = 2,
                 metric_enabled: bool = True, metric_update_interval: int = 32):
        super().__init__()
        if group_size <= 1:
            raise ValueError("group_size must be greater than one")
        if channels % group_size:
            raise ValueError("channels must be divisible by group_size")
        if levels is None:
            levels = 1
        self.channels = channels
        self.group_size = group_size
        self.level_count = int(levels)
        self.shuffle_groups = channels // group_size
        self.levels = nn.ModuleList(
            CasprSparseLevel(
                channels, group_size=group_size, num_branches=num_branches,
                metric_enabled=metric_enabled,
                metric_update_interval=metric_update_interval)
            for _ in range(self.level_count)
        )

    def forward(self, x: Tensor) -> Tensor:
        for index, level in enumerate(self.levels):
            if index:
                x = channel_shuffle(x, self.shuffle_groups)
            x = level(x)
        return x

    def reparameterize(self) -> "CasprSparseProduct":
        for level in self.levels:
            level.reparameterize()
        return self


class CasprChannelMixer(nn.Module):
    """Two global sparse products separated by exactly one activation."""

    def __init__(self, channels: int, group_size: int, levels: int | None,
                 num_branches: int, activation: str, metric_enabled: bool,
                 metric_update_interval: int):
        super().__init__()
        kwargs = dict(
            channels=channels, group_size=group_size, levels=levels,
            num_branches=num_branches, metric_enabled=metric_enabled,
            metric_update_interval=metric_update_interval)
        self.pre = CasprSparseProduct(**kwargs)
        self.activation = build_activation(activation)
        self.post = CasprSparseProduct(**kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.pre(x))
        x = channel_shuffle(x, self.pre.shuffle_groups)
        return self.post(x)

    def reparameterize(self) -> "CasprChannelMixer":
        self.pre.reparameterize()
        self.post.reparameterize()
        return self


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep) / keep


class CasprBlock(nn.Module):
    def __init__(self, channels: int, group_size: int = 4,
                 sparse_levels: int | None = None, spatial_branches: int = 2,
                 channel_branches: int = 2, activation: str = "relu",
                 drop_path: float = 0.0, layer_scale_init: float = 1e-5,
                 metric_enabled: bool = True, metric_update_interval: int = 32):
        super().__init__()
        self.spatial = CasprSpatialGroup(
            channels, num_branches=spatial_branches,
            metric_enabled=metric_enabled,
            metric_update_interval=metric_update_interval)
        self.channel_mixer = CasprChannelMixer(
            channels, group_size=group_size, levels=sparse_levels,
            num_branches=channel_branches, activation=activation,
            metric_enabled=metric_enabled,
            metric_update_interval=metric_update_interval)
        self.layer_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), layer_scale_init))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        update = self.channel_mixer(self.spatial(x)) * self.layer_scale
        return x + self.drop_path(update)

    def reparameterize(self) -> "CasprBlock":
        self.spatial.reparameterize()
        self.channel_mixer.reparameterize()
        return self


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, groups: int = 1,
                 activation: str | None = "relu"):
        modules: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if activation is not None:
            modules.append(build_activation(activation))
        super().__init__(*modules)

    @property
    def is_reparameterized(self) -> bool:
        return isinstance(self[1], nn.Identity)

    @torch.no_grad()
    def reparameterize(self) -> "ConvNormAct":
        if self.is_reparameterized:
            return self
        if self.training:
            raise RuntimeError("Call eval() before ConvNormAct.reparameterize()")
        self[0] = fuse_conv_bn_eval(self[0], self[1])
        self[1] = nn.Identity()
        return self


class EfficientStem(nn.Sequential):
    """Low-MAC two-step reduction preserving a local convolutional prior."""

    def __init__(self, in_channels: int, out_channels: int, activation: str,
                 stem_channels: int = 16):
        super().__init__(
            ConvNormAct(in_channels, stem_channels, 3, stride=2,
                        activation=activation),
            ConvNormAct(stem_channels, stem_channels, 3, stride=2,
                        groups=stem_channels, activation=None),
            ConvNormAct(stem_channels, out_channels, 1, activation=activation),
        )


def transition_groups(in_channels: int, out_channels: int,
                      target_group_size: int) -> int:
    """Choose a hardware-valid group count near the requested input group size."""
    common = math.gcd(in_channels, out_channels)
    target = max(1, round(in_channels / target_group_size))
    divisors = [value for value in range(1, common + 1) if common % value == 0]
    # Prefer fewer groups on ties to retain more cross-channel capacity.
    return min(divisors, key=lambda value: (abs(value - target), value))


class SparseDownsample(nn.Module):
    """Depthwise reduction followed by a sparse cross-stage projection."""

    def __init__(self, in_channels: int, out_channels: int, activation: str,
                 target_group_size: int = 8):
        super().__init__()
        self.groups = transition_groups(
            in_channels, out_channels, target_group_size)
        self.depthwise = ConvNormAct(
            in_channels, in_channels, 3, stride=2, groups=in_channels,
            activation=activation)
        self.transition = ConvNormAct(
            in_channels, out_channels, 1, groups=self.groups, activation=None)

    def forward(self, x: Tensor) -> Tensor:
        x = self.transition(self.depthwise(x))
        return channel_shuffle(x, self.groups)
