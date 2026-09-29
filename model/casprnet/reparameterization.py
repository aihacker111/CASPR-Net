"""Exact branch folding and curvature metrics for CASPR-Net."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ActivationAwareKernelMetric(nn.Module):
    """Kronecker preconditioner for a depthwise spatial kernel."""

    def __init__(self, channels: int, kernel_size: int, momentum: float = 0.95,
                 damping: float = 1e-3, update_interval: int = 32,
                 max_samples: int = 256, eigenvalue_floor: float = 1e-2,
                 eigenvalue_ceiling: float = 1e2, enabled: bool = True):
        super().__init__()
        dimension = kernel_size * kernel_size
        identity = torch.eye(dimension).expand(channels, -1, -1).clone()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dimension = dimension
        self.momentum = float(momentum)
        self.damping = float(damping)
        self.update_interval = int(update_interval)
        self.max_samples = int(max_samples)
        self.eigenvalue_floor = float(eigenvalue_floor)
        self.eigenvalue_ceiling = float(eigenvalue_ceiling)
        self.enabled = bool(enabled)
        self.register_buffer("patch_metric", identity)
        self.register_buffer("output_metric", torch.ones(channels))
        self.register_buffer("patch_inverse", identity.clone())
        self.register_buffer("output_inverse", torch.ones(channels))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))
        self._observe_this_step = False

    @torch.no_grad()
    def _stable_inverse(self, matrix: Tensor) -> Tensor:
        matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix.float())
        eigenvalues = eigenvalues.clamp(self.eigenvalue_floor, self.eigenvalue_ceiling)
        inverse = eigenvectors @ torch.diag_embed(eigenvalues.reciprocal())
        inverse = inverse @ eigenvectors.transpose(-1, -2)
        inverse = inverse / inverse.diagonal(dim1=-2, dim2=-1).mean(-1).view(-1, 1, 1)
        return inverse.to(dtype=matrix.dtype)

    @torch.no_grad()
    def observe_input(self, x: Tensor) -> None:
        if not self.enabled or not self.training:
            self._observe_this_step = False
            return
        step = int(self.num_updates.item())
        self._observe_this_step = step % self.update_interval == 0
        self.num_updates.add_(1)
        if not self._observe_this_step:
            return
        patches = F.unfold(x[:1].detach().float(), self.kernel_size, padding=0)
        patches = patches.reshape(self.channels, self.dimension, -1)
        if patches.shape[-1] > self.max_samples:
            indices = torch.linspace(0, patches.shape[-1] - 1, self.max_samples,
                                     device=patches.device).long()
            patches = patches.index_select(-1, indices)
        covariance = patches @ patches.transpose(-1, -2)
        covariance.div_(max(1, patches.shape[-1]))
        identity = torch.eye(self.dimension, device=covariance.device,
                             dtype=covariance.dtype).expand(self.channels, -1, -1)
        covariance.add_(identity, alpha=self.damping)
        self.patch_metric.lerp_(covariance.to(self.patch_metric.dtype), 1.0 - self.momentum)
        self.patch_inverse.copy_(self._stable_inverse(self.patch_metric))

    @torch.no_grad()
    def observe_output_gradient(self, gradient: Tensor) -> Tensor:
        if not self.enabled or not self.training or not self._observe_this_step:
            return gradient
        curvature = gradient.detach().float().square().mean(dim=(0, 2, 3))
        self.output_metric.lerp_(curvature.add(self.damping).to(self.output_metric.dtype),
                                 1.0 - self.momentum)
        inverse = self.output_metric.clamp(self.eigenvalue_floor,
                                           self.eigenvalue_ceiling).reciprocal()
        self.output_inverse.copy_(inverse / inverse.mean())
        return gradient

    def precondition(self, gradient: Tensor) -> Tensor:
        if not self.enabled:
            return gradient
        dtype = gradient.dtype
        flat = gradient.float().reshape(self.channels, self.dimension)
        flat = torch.bmm(flat.unsqueeze(1), self.patch_inverse.float()).squeeze(1)
        flat.mul_(self.output_inverse.float().unsqueeze(1))
        return flat.reshape_as(gradient).to(dtype=dtype)


class ActivationAwareSparseMetric(nn.Module):
    r"""Diagonal Kronecker metric restricted to existing sparse edges."""

    def __init__(self, channels: int, group_size: int, momentum: float = 0.95,
                 damping: float = 1e-3, update_interval: int = 32,
                 enabled: bool = True):
        super().__init__()
        if channels % group_size:
            raise ValueError("channels must be divisible by group_size")
        self.channels = channels
        self.group_size = group_size
        self.groups = channels // group_size
        self.momentum = float(momentum)
        self.damping = float(damping)
        self.update_interval = int(update_interval)
        self.enabled = bool(enabled)
        self.register_buffer("input_metric", torch.ones(channels))
        self.register_buffer("output_metric", torch.ones(channels))
        self.register_buffer("input_inverse", torch.ones(channels))
        self.register_buffer("output_inverse", torch.ones(channels))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))
        self._observe_this_step = False

    @torch.no_grad()
    def _normalized_inverse(self, value: Tensor) -> Tensor:
        inverse = value.clamp_min(self.damping).rsqrt()
        return inverse / inverse.mean().clamp_min(1e-12)

    @torch.no_grad()
    def observe_input(self, x: Tensor) -> None:
        if not self.enabled or not self.training:
            self._observe_this_step = False
            return
        step = int(self.num_updates.item())
        self._observe_this_step = step % self.update_interval == 0
        self.num_updates.add_(1)
        if not self._observe_this_step:
            return
        moment = x.detach().float().square().mean(dim=(0, 2, 3)).add(self.damping)
        self.input_metric.lerp_(moment.to(self.input_metric.dtype), 1.0 - self.momentum)
        self.input_inverse.copy_(self._normalized_inverse(self.input_metric))

    @torch.no_grad()
    def observe_output_gradient(self, gradient: Tensor) -> Tensor:
        if not self.enabled or not self.training or not self._observe_this_step:
            return gradient
        moment = gradient.detach().float().square().mean(dim=(0, 2, 3)).add(self.damping)
        self.output_metric.lerp_(moment.to(self.output_metric.dtype), 1.0 - self.momentum)
        self.output_inverse.copy_(self._normalized_inverse(self.output_metric))
        return gradient

    def precondition(self, gradient: Tensor) -> Tensor:
        if not self.enabled:
            return gradient
        dtype = gradient.dtype
        grad = gradient.float().reshape(self.groups, self.group_size, self.group_size)
        input_scale = self.input_inverse.float().reshape(self.groups, self.group_size)
        output_scale = self.output_inverse.float().reshape(self.groups, self.group_size)
        grad = grad * output_scale.unsqueeze(-1) * input_scale.unsqueeze(-2)
        return grad.reshape_as(gradient).to(dtype=dtype)


class CasprDenseStep(nn.Module):
    """One train-time depthwise spatial branch."""

    def __init__(self, channels: int, kernel_size: int = 3,
                 metric_enabled: bool = True, metric_update_interval: int = 32):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=0,
                              groups=channels, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.metric = ActivationAwareKernelMetric(
            channels, kernel_size, update_interval=metric_update_interval,
            enabled=metric_enabled)
        self.conv.weight.register_hook(self.metric.precondition)

    @property
    def weight(self) -> nn.Parameter:
        return self.conv.weight

    def forward(self, x: Tensor) -> Tensor:
        self.metric.observe_input(x)
        y = self.conv(x)
        if y.requires_grad and self.metric.enabled:
            y.register_hook(self.metric.observe_output_gradient)
        return self.norm(y)

    @torch.no_grad()
    def fused_operator(self) -> tuple[Tensor, Tensor]:
        scale = self.norm.weight / torch.sqrt(self.norm.running_var + self.norm.eps)
        return (self.conv.weight * scale.reshape(-1, 1, 1, 1),
                self.norm.bias - self.norm.running_mean * scale)


class CasprSparseStep(nn.Module):
    """One grouped 1x1 branch with curvature-aligned optimization."""

    def __init__(self, channels: int, group_size: int,
                 metric_enabled: bool = True, metric_update_interval: int = 32):
        super().__init__()
        self.channels = channels
        self.group_size = group_size
        self.groups = channels // group_size
        self.conv = nn.Conv2d(channels, channels, 1, groups=self.groups, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.metric = ActivationAwareSparseMetric(
            channels, group_size, update_interval=metric_update_interval,
            enabled=metric_enabled)
        self.conv.weight.register_hook(self.metric.precondition)

    @property
    def weight(self) -> nn.Parameter:
        return self.conv.weight

    def forward(self, x: Tensor) -> Tensor:
        self.metric.observe_input(x)
        y = self.conv(x)
        if y.requires_grad and self.metric.enabled:
            y.register_hook(self.metric.observe_output_gradient)
        return self.norm(y)

    @torch.no_grad()
    def fused_operator(self) -> tuple[Tensor, Tensor]:
        scale = self.norm.weight / torch.sqrt(self.norm.running_var + self.norm.eps)
        return (self.conv.weight * scale.reshape(-1, 1, 1, 1),
                self.norm.bias - self.norm.running_mean * scale)


@torch.no_grad()
def fold_branches(branches: nn.ModuleList) -> tuple[Tensor, Tensor]:
    if not branches:
        raise ValueError("Cannot fold an empty branch set")
    operators = [branch.fused_operator() for branch in branches]
    weight = torch.stack([operator[0] for operator in operators]).sum(0)
    bias = torch.stack([operator[1] for operator in operators]).sum(0)
    return weight, bias
