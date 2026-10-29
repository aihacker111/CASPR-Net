"""LightlyTrain wrapper for distilling a DINOv3 teacher into ER-Net.

LightlyTrain (https://docs.lightly.ai/train) calls, in order,
``forward_features`` -> ``forward_pool`` on the student and matches

* the pooled (global) features to the teacher's class/pooled token, and
* the spatial features, bilinearly resized to the teacher's patch grid
  (14 x 14 for DINOv3 ViT-B/16 at 224 px), to the teacher's patch tokens.

By default the distilled feature map fuses stage 3 (stride 16, already on the
teacher's grid) with stage 4 (stride 32, upsampled), so both the dense and the
global part of the loss reach every stage that a COCO FPN later uses.

The classification head is removed (``ERNet.remove_head()``): it plays no role
in distillation, so the exported checkpoint
(``exported_models/exported_last.pt``) holds backbone weights only.  Loading
it into a full ERNet with ``load_distilled_weights`` gives a freshly
initialised head for ImageNet fine-tuning.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .network import ERNet


class ERNetDistillWrapper(nn.Module):
    def __init__(self, model: ERNet, feature_levels: Sequence[int] = (3, 4)):
        super().__init__()
        levels = sorted({int(level) for level in feature_levels})
        if not levels or levels[0] < 1 or levels[-1] > len(model.widths):
            raise ValueError(f"feature_levels must be within 1..{len(model.widths)}")
        if not model.headless:
            model.remove_head()          # backbone only: nothing unused for DDP
        self._model = model
        self.feature_levels = tuple(levels)

    # ----------------------------------------------------- LightlyTrain API
    def get_model(self) -> ERNet:
        return self._model

    def forward_features(self, x: Tensor) -> Dict[str, Any]:
        stages = self._model.forward_intermediates(x)
        last = len(stages)
        maps = []
        for level in self.feature_levels:
            feature = stages[level - 1]
            if level == last:
                feature = self._model.norm(feature)   # same norm as classification
            maps.append(feature)
        size = maps[0].shape[-2:]                     # finest selected resolution
        maps = [m if m.shape[-2:] == size else
                F.interpolate(m, size=size, mode="bilinear", align_corners=False)
                for m in maps]
        return {"features": torch.cat(maps, dim=1)}

    def forward_pool(self, x: Dict[str, Any]) -> Dict[str, Any]:
        return {"pooled_features": F.adaptive_avg_pool2d(x["features"], 1)}

    def feature_dim(self) -> int:
        return sum(self._model.widths[level - 1] for level in self.feature_levels)

    def architecture_info(self) -> Dict[str, Any]:
        # Convolutions + BatchNorm only (no attention): LightlyTrain then uses
        # its convolutional recipe (e.g. small weight decay for conv weights).
        return {"model_type": "convolutional", "norm_type": "batchnorm"}

    # PyTorch module call (not used by LightlyTrain, handy for tests).
    def forward(self, x: Tensor) -> Tensor:
        return self.forward_pool(self.forward_features(x))["pooled_features"]


HEAD_PREFIXES = ("head.", "head_projection.")


def load_distilled_weights(model: ERNet, path: str, strict: bool = True) -> ERNet:
    """Load distilled backbone weights (``exported_last.pt`` or ``{'model': ...}``).

    Works for a headless ERNet and for a full ERNet: in the latter case only
    the classification head stays at its fresh initialisation.  With
    ``strict`` any other missing or unexpected key raises.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    problems = [k for k in missing if not k.startswith(HEAD_PREFIXES)] + list(unexpected)
    if strict and problems:
        raise RuntimeError(f"Distilled weights do not match the backbone: {problems[:10]}")
    return model
