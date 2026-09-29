"""timm registrations for the CASPR-Net family."""
from __future__ import annotations

from .network import CasprNet

try:
    from timm.models import register_model
except (ImportError, ModuleNotFoundError):
    try:
        from timm.models.registry import register_model
    except (ImportError, ModuleNotFoundError):
        def register_model(function):
            return function


def _create_casprnet(pretrained: bool, config: dict, **kwargs: object) -> CasprNet:
    if pretrained:
        raise ValueError("No pretrained CASPR-Net weights are available yet")
    for key in ("pretrained_cfg", "pretrained_cfg_overlay", "cache_dir"):
        kwargs.pop(key, None)
    return CasprNet(**{**config, **kwargs})


@register_model
def casprnet_n(pretrained: bool = False, **kwargs: object) -> CasprNet:
    return _create_casprnet(pretrained, dict(
        widths=(48, 96, 192, 320), depths=(2, 2, 6, 2),
        group_sizes=(8, 8, 8, 8), sparse_levels=(1, 1, 1, 1),
        spatial_branches=(2, 2, 3, 3), channel_branches=(2, 2, 2, 2),
        head_dim=1152, stem_channels=16, transition_group_size=8,
        drop_path_rate=0.05), **kwargs)


@register_model
def casprnet_t(pretrained: bool = False, **kwargs: object) -> CasprNet:
    return _create_casprnet(pretrained, dict(
        widths=(64, 128, 256, 384), depths=(2, 2, 8, 2),
        group_sizes=(8, 8, 8, 8), sparse_levels=(1, 1, 1, 1),
        spatial_branches=(2, 2, 3, 3), channel_branches=(2, 2, 2, 2),
        head_dim=1344, stem_channels=20, transition_group_size=8,
        drop_path_rate=0.10), **kwargs)


@register_model
def casprnet_s(pretrained: bool = False, **kwargs: object) -> CasprNet:
    return _create_casprnet(pretrained, dict(
        widths=(64, 128, 320, 512), depths=(3, 3, 10, 3),
        group_sizes=(8, 8, 8, 8), sparse_levels=(1, 1, 1, 1),
        spatial_branches=(2, 3, 3, 3), channel_branches=(2, 2, 2, 2),
        head_dim=1536, stem_channels=24, transition_group_size=8,
        drop_path_rate=0.15), **kwargs)
