"""timm registrations for the ER-Net family."""
from __future__ import annotations

from .network import ERNet

try:
    from timm.models import register_model
except (ImportError, ModuleNotFoundError):
    try:
        from timm.models.registry import register_model
    except (ImportError, ModuleNotFoundError):
        def register_model(function):
            return function


def _create_ernet(pretrained: bool, config: dict, **kwargs: object) -> ERNet:
    if pretrained:
        raise ValueError("No pretrained ER-Net weights are available yet")
    for key in ("pretrained_cfg", "pretrained_cfg_overlay", "cache_dir"):
        kwargs.pop(key, None)
    return ERNet(**{**config, **kwargs})


# Budget targets (224x224, deploy graph): n ~0.18 GMACs, t ~0.33 GMACs,
# s ~0.84 GMACs.  Widths are bond dimensions chi of the renormalisation flow;
# re-derive them per stage with tools/ernet_bond_dims.py once a proxy model is
# trained (truncation-error rule) instead of keeping the doubling default.
ERNET_CONFIGS = {
    "ernet_n": dict(widths=(24, 48, 96, 192), depths=(2, 2, 6, 2),
                    mlp_ratios=(1.5, 1.5, 2, 2), stem_channels=16,
                    head_dim=1024, drop_path_rate=0.05),
    "ernet_t": dict(widths=(32, 64, 128, 256), depths=(2, 2, 8, 2),
                    mlp_ratios=(1.5, 1.5, 2, 2), stem_channels=16,
                    head_dim=1280, drop_path_rate=0.10),
    "ernet_s": dict(widths=(48, 96, 192, 384), depths=(2, 3, 10, 2),
                    mlp_ratios=(1.5, 1.5, 2, 2), stem_channels=24,
                    head_dim=1536, drop_path_rate=0.15),
}


@register_model
def ernet_n(pretrained: bool = False, **kwargs: object) -> ERNet:
    return _create_ernet(pretrained, ERNET_CONFIGS["ernet_n"], **kwargs)


@register_model
def ernet_t(pretrained: bool = False, **kwargs: object) -> ERNet:
    return _create_ernet(pretrained, ERNET_CONFIGS["ernet_t"], **kwargs)


@register_model
def ernet_s(pretrained: bool = False, **kwargs: object) -> ERNet:
    return _create_ernet(pretrained, ERNET_CONFIGS["ernet_s"], **kwargs)
