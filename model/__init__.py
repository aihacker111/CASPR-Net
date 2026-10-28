"""Model registry.

ER-Net (Entanglement Renormalization Network) is the default backbone.
CASPR-Net is still registered so it can be trained as a baseline
(``--model casprnet_n``); it is no longer the default anywhere.
"""
from .ernet import ERNet, ernet_n, ernet_s, ernet_t
from .casprnet import (CasprNet, casprnet_n, casprnet_s, casprnet_t,
                       coverage_group_sizes)

__all__ = [
    "ERNet", "ernet_n", "ernet_t", "ernet_s",
    "CasprNet", "coverage_group_sizes",
    "casprnet_n", "casprnet_t", "casprnet_s",
]
