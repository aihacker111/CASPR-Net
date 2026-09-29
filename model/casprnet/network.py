"""Hierarchical CASPR-Net backbone and ImageNet classification head."""
from __future__ import annotations

from copy import deepcopy
from typing import Sequence

import torch
from torch import Tensor, nn

from .activations import build_activation
from .blocks import (CasprBlock, CasprSparseLevel, CasprSpatialGroup,
                     ConvNormAct, EfficientStem, SparseDownsample)


class ClassificationHead(nn.Module):
    def __init__(self, channels: int, num_classes: int, distillation: bool):
        super().__init__()
        self.distillation = distillation
        self.fc = nn.Linear(channels, num_classes) if num_classes > 0 else nn.Identity()
        self.dist_fc = nn.Linear(channels, num_classes) if distillation and num_classes > 0 else None

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        logits = self.fc(x)
        if self.dist_fc is None:
            return logits
        distillation_logits = self.dist_fc(x)
        if self.training and not torch.jit.is_scripting():
            return logits, distillation_logits
        return (logits + distillation_logits) * 0.5


class CasprNet(nn.Module):
    """Curvature-Aligned Sparse Product Reparameterization Network."""

    def __init__(
        self,
        widths: Sequence[int] = (48, 96, 192, 320),
        depths: Sequence[int] = (2, 2, 6, 2),
        group_sizes: int | Sequence[int] = 4,
        sparse_levels: int | None | Sequence[int | None] = 1,
        spatial_branches: int | Sequence[int] = (2, 2, 3, 3),
        channel_branches: int | Sequence[int] = 2,
        activation: str = "relu",
        head_dim: int = 1024,
        stem_channels: int = 16,
        transition_group_size: int = 8,
        in_chans: int = 3,
        num_classes: int = 1000,
        drop_path_rate: float = 0.05,
        layer_scale_init: float = 1e-5,
        distillation: bool = False,
        global_pool: str = "avg",
        deploy: bool = False,
        metric_enabled: bool = True,
        metric_update_interval: int = 32,
        **_: object,
    ):
        super().__init__()
        stage_count = len(widths)
        if len(depths) != stage_count:
            raise ValueError("widths and depths must have equal length")
        if global_pool != "avg":
            raise ValueError("CasprNet supports global_pool='avg' only")
        if head_dim < widths[-1]:
            raise ValueError("head_dim must be at least the final stage width")

        def per_stage(value, name: str):
            if isinstance(value, (int, type(None))):
                return (value,) * stage_count
            if len(value) != stage_count:
                raise ValueError(f"{name} needs one value per stage")
            return tuple(value)

        group_sizes = per_stage(group_sizes, "group_sizes")
        sparse_levels = per_stage(sparse_levels, "sparse_levels")
        spatial_branches = per_stage(spatial_branches, "spatial_branches")
        channel_branches = per_stage(channel_branches, "channel_branches")
        for width, group_size in zip(widths, group_sizes):
            if group_size is None or width % group_size:
                raise ValueError("Every width must be divisible by its group size")

        self.widths = tuple(widths)
        self.depths = tuple(depths)
        self.num_classes = num_classes
        self.backbone_features = widths[-1]
        self.num_features = head_dim
        self.global_pool = global_pool
        self.metric_enabled = bool(metric_enabled)

        self.stem = EfficientStem(
            in_chans, widths[0], activation, stem_channels=stem_channels)
        total_blocks = sum(depths)
        drop_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        block_index = 0
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        stage_values = zip(widths, depths, group_sizes, sparse_levels,
                           spatial_branches, channel_branches)
        for stage_index, (width, depth, group_size, levels,
                          spatial_count, channel_count) in enumerate(stage_values):
            blocks = []
            for _ in range(depth):
                blocks.append(CasprBlock(
                    width,
                    group_size=group_size,
                    sparse_levels=levels,
                    spatial_branches=spatial_count,
                    channel_branches=channel_count,
                    activation=activation,
                    drop_path=drop_rates[block_index],
                    layer_scale_init=layer_scale_init,
                    metric_enabled=metric_enabled,
                    metric_update_interval=metric_update_interval,
                ))
                block_index += 1
            self.stages.append(nn.Sequential(*blocks))
            if stage_index + 1 < stage_count:
                self.downsamples.append(SparseDownsample(
                    width,
                    widths[stage_index + 1],
                    activation,
                    target_group_size=transition_group_size,
                ))

        self.norm = nn.BatchNorm2d(self.backbone_features)
        self.head_projection = nn.Sequential(
            nn.Linear(self.backbone_features, self.num_features, bias=False),
            nn.LayerNorm(self.num_features),
            build_activation(activation),
        )
        self.head = ClassificationHead(self.num_features, num_classes, distillation)
        self.feature_info = [
            dict(num_chs=width, reduction=4 * (2 ** index), module=f"stages.{index}")
            for index, width in enumerate(widths)
        ]
        self.apply(self._init_weights)
        if deploy:
            self.eval()
            self.reparameterize()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_intermediates(self, x: Tensor) -> list[Tensor]:
        outputs = []
        x = self.stem(x)
        for index, stage in enumerate(self.stages):
            x = stage(x)
            outputs.append(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)
        return outputs

    def get_feature_maps(self, x: Tensor) -> list[Tensor]:
        return self.forward_intermediates(x)

    def forward_features(self, x: Tensor) -> Tensor:
        return self.norm(self.forward_intermediates(x)[-1])

    def forward_head(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        x = self.head_projection(x.mean(dim=(-2, -1)))
        return self.head(x)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        return self.forward_head(self.forward_features(x))

    @torch.no_grad()
    def reparameterize(self, fuse_qkv: bool = True,
                       verbose: bool = False) -> "CasprNet":
        del fuse_qkv
        if self.training:
            raise RuntimeError("Call eval() before CasprNet.reparameterize()")
        spatial_count = 0
        sparse_count = 0
        affine_count = 0
        for module in list(self.modules()):
            if isinstance(module, CasprSpatialGroup) and not module.is_reparameterized:
                module.reparameterize()
                spatial_count += 1
            elif isinstance(module, CasprSparseLevel) and not module.is_reparameterized:
                module.reparameterize()
                sparse_count += 1
            elif isinstance(module, ConvNormAct) and not module.is_reparameterized:
                module.reparameterize()
                affine_count += 1
        if verbose:
            print(f"CASPR fused {spatial_count} spatial groups and "
                  f"{sparse_count} sparse channel levels; "
                  f"folded {affine_count} Conv-BN pairs")
        return self

    switch_to_deploy = reparameterize

    def deploy_copy(self) -> "CasprNet":
        return deepcopy(self).eval().reparameterize()

    @torch.no_grad()
    def compare_reparameterization(self, inputs: Tensor, atol: float = 3e-5,
                                   rtol: float = 3e-5) -> dict[str, float | bool]:
        reference_model = deepcopy(self).eval()
        deployed_model = deepcopy(reference_model).reparameterize()
        reference = reference_model(inputs)
        deployed = deployed_model(inputs)
        if isinstance(reference, tuple) or isinstance(deployed, tuple):
            raise RuntimeError("Compare reparameterization in eval mode only")
        difference = (reference - deployed).abs()
        allowed = atol + rtol * reference.abs()
        return {
            "passed": bool(torch.all(difference <= allowed).item()),
            "max_abs_error": float(difference.max().item()),
            "mean_abs_error": float(difference.mean().item()),
            "max_rel_error": float(
                (difference / reference.abs().clamp_min(1e-12)).max().item()),
        }

    def get_classifier(self) -> nn.Module:
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: str = "avg") -> None:
        if global_pool != "avg":
            raise ValueError("CasprNet supports global_pool='avg' only")
        self.num_classes = num_classes
        self.head = ClassificationHead(self.num_features, num_classes,
                                       self.head.distillation)

    def no_weight_decay(self) -> set[str]:
        return {name for name, _ in self.named_parameters()
                if name.endswith("layer_scale")}
