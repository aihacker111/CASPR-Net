"""Activation builders used by CASPR-Net."""
from __future__ import annotations

from torch import nn


SUPPORTED_ACTIVATIONS = (
    "relu",
    "relu6",
    "gelu",
    "silu",
    "swish",
    "mish",
    "hardswish",
)


def build_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "relu6":
        return nn.ReLU6(inplace=False)
    if name == "gelu":
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU(inplace=False)
    if name == "mish":
        return nn.Mish(inplace=False)
    if name == "hardswish":
        return nn.Hardswish(inplace=False)
    raise ValueError(
        f"Unsupported activation '{name}'. Expected one of {SUPPORTED_ACTIVATIONS}."
    )
