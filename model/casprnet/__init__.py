"""Public CASPR-Net API."""

from .activations import SUPPORTED_ACTIVATIONS, build_activation
from .blocks import (CasprBlock, CasprChannelMixer, CasprSparseLevel,
                     CasprSparseProduct, CasprSpatialGroup, channel_shuffle)
from .network import CasprNet
from .reparameterization import (ActivationAwareKernelMetric,
                                 ActivationAwareSparseMetric, CasprDenseStep,
                                 CasprSparseStep)
from .variants import casprnet_n, casprnet_s, casprnet_t

__all__ = [
    "SUPPORTED_ACTIVATIONS", "build_activation",
    "ActivationAwareKernelMetric", "ActivationAwareSparseMetric",
    "CasprDenseStep", "CasprSparseStep", "CasprSpatialGroup",
    "CasprSparseLevel", "CasprSparseProduct", "CasprChannelMixer",
    "CasprBlock", "CasprNet", "channel_shuffle",
    "casprnet_n", "casprnet_t", "casprnet_s",
]
