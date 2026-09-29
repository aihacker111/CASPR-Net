"""timm registrations for the CASPR-Net family."""
from __future__ import annotations

from typing import Sequence

from .network import CasprNet

try:
    from timm.models import register_model
except (ImportError, ModuleNotFoundError):
    try:
        from timm.models.registry import register_model
    except (ImportError, ModuleNotFoundError):
        def register_model(function):
            return function


def coverage_group_sizes(
    widths: Sequence[int], alignment: int = 8
) -> tuple[int, ...]:
    """Choose efficient group widths with full two-factor coverage capacity.

    A CASPR channel mixer contains two grouped 1x1 factors separated by a
    perfect shuffle.  A group of width ``b`` can therefore reach at most
    ``b**2`` input channels after both factors.  Requiring ``b**2 >= C`` for a
    stage width ``C`` prevents a structural channel-coverage bottleneck.

    The smallest divisor satisfying that bound is selected, preferring an
    alignment-friendly divisor when one is available.
    """
    if alignment < 1:
        raise ValueError("alignment must be positive")
    result = []
    for width in widths:
        if width < 1:
            raise ValueError("stage widths must be positive")
        candidates = [
            size for size in range(1, width + 1)
            if width % size == 0 and size * size >= width
        ]
        aligned = [size for size in candidates if size % alignment == 0]
        result.append(min(aligned or candidates))
    return tuple(result)


def _create_casprnet(pretrained: bool, config: dict, **kwargs: object) -> CasprNet:
    if pretrained:
        raise ValueError("No pretrained CASPR-Net weights are available yet")
    for key in ("pretrained_cfg", "pretrained_cfg_overlay", "cache_dir"):
        kwargs.pop(key, None)
    return CasprNet(**{**config, **kwargs})


@register_model
def casprnet_n(pretrained: bool = False, **kwargs: object) -> CasprNet:
    widths = (48, 96, 192, 320)
    return _create_casprnet(pretrained, dict(
        widths=widths, depths=(2, 2, 6, 2),
        group_sizes=coverage_group_sizes(widths), sparse_levels=(1, 1, 1, 1),
        spatial_branches=(2, 2, 3, 3), channel_branches=(2, 2, 2, 2),
        head_dim=1152, stem_channels=16, transition_group_size=8,
        drop_path_rate=0.05), **kwargs)


@register_model
def casprnet_t(pretrained: bool = False, **kwargs: object) -> CasprNet:
    widths = (64, 128, 256, 384)
    return _create_casprnet(pretrained, dict(
        widths=widths, depths=(2, 2, 8, 2),
        group_sizes=coverage_group_sizes(widths), sparse_levels=(1, 1, 1, 1),
        spatial_branches=(2, 2, 3, 3), channel_branches=(2, 2, 2, 2),
        head_dim=1344, stem_channels=32, transition_group_size=8,
        drop_path_rate=0.10), **kwargs)


@register_model
def casprnet_s(pretrained: bool = False, **kwargs: object) -> CasprNet:
    widths = (64, 128, 320, 512)
    return _create_casprnet(pretrained, dict(
        widths=widths, depths=(3, 3, 10, 3),
        group_sizes=coverage_group_sizes(widths), sparse_levels=(1, 1, 1, 1),
        spatial_branches=(2, 3, 3, 3), channel_branches=(2, 2, 2, 2),
        head_dim=1536, stem_channels=64, transition_group_size=8,
        drop_path_rate=0.15), **kwargs)
