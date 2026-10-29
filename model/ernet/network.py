"""ER-Net: Entanglement Renormalization Network backbone + ImageNet head.

``headless=True`` builds the backbone without the classification head (for
distillation or as a detection backbone).
"""
from __future__ import annotations

from copy import deepcopy
from typing import Sequence

import torch
from torch import Tensor, nn

from .activations import build_activation
from .layers import (ConvNormAct, Disentangler, ERBlock, ERStem,
                     GatedContraction, Isometry, RenormalizationStep,
                     build_downsample)


class ClassificationHead(nn.Module):
    def __init__(self, channels: int, num_classes: int, distillation: bool):
        super().__init__()
        self.distillation = distillation
        self.fc = nn.Linear(channels, num_classes) if num_classes > 0 else nn.Identity()
        self.dist_fc = (nn.Linear(channels, num_classes)
                        if distillation and num_classes > 0 else None)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        logits = self.fc(x)
        if self.dist_fc is None:
            return logits
        distillation_logits = self.dist_fc(x)
        if self.training and not torch.jit.is_scripting():
            return logits, distillation_logits
        return (logits + distillation_logits) * 0.5


class ERNet(nn.Module):
    """Entanglement Renormalization Network.

    Layout (input 224):

        stem   : 3x3/s2 conv -> [U(shift) -> W]               -> 56x56, C1
        stage i: N_i x ERBlock (gated contraction; optional brick-wall U)
        between: RenormalizationStep [U(shift) -> W]          -> /2, C_{i+1}
        head   : BN -> GAP -> Linear -> LN -> act -> FC

    Every resolution change is an entanglement-renormalization step
    (disentangle, then coarse-grain).  Ablations with an otherwise identical
    network: ``disentangle=False`` (tree coarse-graining),
    ``block_disentangle=True`` (extra brick-wall disentangler in every block;
    the default is one disentangler layer per scale, as in MERA) and
    ``renormalization="conv4x4"`` (the same 4x4/s2 receptive field as
    ``W U`` but as an unstructured conv, without the factorisation).
    """

    def __init__(
        self,
        widths: Sequence[int] = (32, 64, 128, 256),
        depths: Sequence[int] = (2, 2, 6, 2),
        mlp_ratios: float | Sequence[float] = 2.0,
        circuit_depth: int = 1,
        wavelet_init: bool = True,
        local_kernel: int = 3,
        disentangle: bool = True,
        block_disentangle: bool | Sequence[bool] = False,
        renormalization: str = "mera",
        activation: str = "relu",
        head_dim: int = 1024,
        stem_channels: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        drop_path_rate: float = 0.05,
        layer_scale_init: float = 1e-5,
        distillation: bool = False,
        global_pool: str = "avg",
        deploy: bool = False,
        headless: bool = False,
        **_: object,
    ):
        super().__init__()
        stage_count = len(widths)
        if len(depths) != stage_count:
            raise ValueError("widths and depths must have equal length")
        if global_pool != "avg":
            raise ValueError("ERNet supports global_pool='avg' only")
        if not headless and head_dim < widths[-1]:
            raise ValueError("head_dim must be at least the final stage width")

        def per_stage(value, name: str) -> tuple:
            if isinstance(value, (int, float)):
                return (value,) * stage_count
            if len(value) != stage_count:
                raise ValueError(f"{name} needs one value per stage")
            return tuple(value)

        mlp_ratios = per_stage(mlp_ratios, "mlp_ratios")
        block_disentangle = per_stage(block_disentangle, "block_disentangle")

        self.widths = tuple(widths)
        self.depths = tuple(depths)
        self.num_classes = num_classes
        self.backbone_features = widths[-1]
        self.num_features = head_dim
        self.global_pool = global_pool
        self.disentangle = bool(disentangle)

        self.stem = ERStem(in_chans, widths[0], stem_channels, activation,
                           circuit_depth=circuit_depth, disentangle=disentangle,
                           renormalization=renormalization,
                           wavelet_init=wavelet_init)
        drop_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        block_index = 0
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for stage_index, (width, depth, ratio, in_blocks) in enumerate(
                zip(widths, depths, mlp_ratios, block_disentangle)):
            blocks = []
            for depth_index in range(depth):
                blocks.append(ERBlock(
                    width, parity=depth_index % 2,
                    circuit_depth=circuit_depth, mlp_ratio=ratio,
                    activation=activation, local_kernel=local_kernel,
                    drop_path=drop_rates[block_index],
                    layer_scale_init=layer_scale_init,
                    disentangle=disentangle and bool(in_blocks)))
                block_index += 1
            self.stages.append(nn.Sequential(*blocks))
            if stage_index + 1 < stage_count:
                self.downsamples.append(build_downsample(
                    width, widths[stage_index + 1], circuit_depth,
                    disentangle, renormalization, wavelet_init))

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
        self.headless = False
        if headless:
            self.remove_head()
        self.apply(self._init_weights)
        for module in self.modules():
            if isinstance(module, (Isometry, Disentangler)):
                module.reset_parameters()
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

    # ------------------------------------------------------------------ forward
    def forward_intermediates(self, x: Tensor) -> list[Tensor]:
        """Feature maps at strides 4, 8, 16, 32 (for FPN / detection necks)."""
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

    # ---------------------------------------------------------------- deploy
    @torch.no_grad()
    def reparameterize(self, fuse_qkv: bool = True, verbose: bool = False,
                       fuse_renormalization: bool | int = 2,
                       split_gates: bool = True) -> "ERNet":
        """Collapse every disentangler circuit, fold BN and layer scales.

        ``fuse_renormalization`` switches renormalization steps to their
        polyphase deploy form (depthwise 4x4/s2 x4 + 1x1, exact, no
        DepthToSpace).  ``True`` converts every step, ``False`` none, an
        integer ``n`` (default 2) the ``n`` highest-resolution steps (stem and
        first transition), where DepthToSpace is most expensive and channels
        are few.  With the default the deploy model never has more
        parameters than the training model.  With ``split_gates``
        the fused value/gate conv of every gated contraction is cut into two
        convs so that no Split op remains in the deploy graph (exact).
        """
        del fuse_qkv
        if self.training:
            raise RuntimeError("Call eval() before ERNet.reparameterize()")
        circuits = folded = scales = 0
        for module in list(self.modules()):
            if isinstance(module, Disentangler) and not module.is_reparameterized:
                module.reparameterize()
                circuits += 1
            elif isinstance(module, ConvNormAct) and not module.is_reparameterized:
                module.reparameterize()
                folded += 1
        for module in self.modules():
            if isinstance(module, ERBlock) and module.layer_scale is not None:
                module.fold_layer_scale()
                scales += 1
        splits = 0
        if split_gates:
            for module in self.modules():
                if isinstance(module, GatedContraction) and not module.is_split:
                    module.split_expand()
                    splits += 1
        steps = 0
        renormalization_steps = [m for m in self.modules()
                                 if isinstance(m, RenormalizationStep)]
        if fuse_renormalization is True:
            to_fuse = len(renormalization_steps)
        elif fuse_renormalization is False:
            to_fuse = 0
        else:
            to_fuse = int(fuse_renormalization)
        for module in renormalization_steps[:to_fuse]:
            if not module.is_fused:
                module.fuse()
                steps += int(module.is_fused)
        if verbose:
            print(f"ER-Net collapsed {circuits} disentangler circuits into single "
                  f"unitaries; folded {folded} Conv-BN pairs and {scales} layer "
                  f"scales; {steps} renormalization steps to polyphase form; "
                  f"split {splits} gated contractions")
        return self

    switch_to_deploy = reparameterize

    def deploy_copy(self) -> "ERNet":
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

    # ----------------------------------------------------------- diagnostics
    @torch.no_grad()
    def isometry_defects(self) -> dict[str, float]:
        """How far each coarse-graining map has drifted from an isometry."""
        return {name: float(module.isometry_defect())
                for name, module in self.named_modules()
                if isinstance(module, Isometry)}

    # ----------------------------------------------------------- headless
    def remove_head(self) -> "ERNet":
        """Drop the classification head (head projection + FC).

        Used for self-supervised distillation and as a detection backbone: the
        model then returns pooled stage-4 features from ``forward`` and its
        state dict contains backbone weights only.  Loading such weights into
        a full ERNet initialises a fresh head (see ``load_distilled_weights``).
        """
        self.head_projection = nn.Identity()
        self.head = nn.Identity()
        self.num_features = self.backbone_features
        self.num_classes = 0
        self.headless = True
        return self

    # ----------------------------------------------------------------- timm
    def get_classifier(self) -> nn.Module:
        return nn.Identity() if self.headless else self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: str = "avg") -> None:
        if global_pool != "avg":
            raise ValueError("ERNet supports global_pool='avg' only")
        if self.headless:
            raise ValueError("Headless ERNet: build it with headless=False and load "
                             "the backbone weights to add a classifier")
        self.num_classes = num_classes
        self.head = ClassificationHead(self.num_features, num_classes,
                                       self.head.distillation)

    def no_weight_decay(self) -> set[str]:
        # Quaternions are normalised, so weight decay would only distort their
        # effective learning rate; layer scales follow the usual convention.
        return {name for name, _ in self.named_parameters()
                if name.endswith("layer_scale") or name.endswith("quaternions")}
