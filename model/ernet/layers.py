"""Building blocks of ER-Net (Entanglement Renormalization Network).

ER-Net follows the Multi-scale Entanglement Renormalization Ansatz (MERA):
every coarse-graining step first removes short-range correlations that sit
*across* the boundaries of the 2x2 blocks (disentangler ``U``) and only then
maps each 2x2 block to one coarse site (isometry ``W``).  Standard backbones
(stride-2 convs, patch merging) are tree tensor networks: they coarse-grain
without the disentangler.

Primitives
----------
``Disentangler``  per-channel SO(4) rotation of every 2x2 block of a
                  (possibly shifted) grid, parametrised by two unit
                  quaternions (exactly orthogonal, no matrix exponential).
``Isometry``      2x2 -> 1 coarse-graining (2x2/s2 conv + BN), initialised
                  as wavelet bands (Daubechies-4 with the disentangler).
``RenormalizationStep``  Disentangler(shifted grid) -> Isometry, i.e. one MERA
                  layer.  Used in the stem and between all stages.
``ERBlock``       same-scale layer: gated (multiplicative) channel contraction,
                  optionally preceded by a disentangler on alternating grid
                  parity (brick-wall circuit, ``block_disentangle=True``).

Deploy graph uses only: Pad, Slice, Conv (dense, depthwise), DepthToSpace,
Mul, Add, activation.

Quantum-inspired, not quantum computing: MERA, unitary gates, the MERA <->
wavelet correspondence and entanglement entropy fix the *structure* and the
*initialisation* of the weights; every operation is an ordinary classical
tensor operation and the deploy graph is plain convolutions.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from .activations import build_activation


def make_divisible(value: float, divisor: int = 8) -> int:
    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * value:
        rounded += divisor
    return rounded


def block_padding(height: int, width: int, parity: int) -> tuple[int, int, int, int]:
    """Zero padding (left, right, top, bottom) that tiles the map with 2x2 blocks.

    ``parity=0`` uses blocks starting at (0, 0); ``parity=1`` shifts the block
    grid by one pixel so that every block straddles a boundary of the parity-0
    grid.  Padding is an open boundary condition.
    """
    top = left = int(parity)
    bottom = (-(height + top)) % 2
    right = (-(width + left)) % 2
    return left, right, top, bottom


# ---------------------------------------------------------------------------
# Conv + BN with exact folding
# ---------------------------------------------------------------------------

class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1,
                 stride: int = 1, groups: int = 1,
                 activation: str | None = None, padding: int | None = None):
        if padding is None:
            padding = kernel_size // 2
        modules: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if activation is not None:
            modules.append(build_activation(activation))
        super().__init__(*modules)

    @property
    def conv(self) -> nn.Conv2d:
        return self[0]

    @property
    def is_reparameterized(self) -> bool:
        return isinstance(self[1], nn.Identity)

    @torch.no_grad()
    def reparameterize(self) -> "ConvNormAct":
        if self.is_reparameterized:
            return self
        if self.training:
            raise RuntimeError("Call eval() before ConvNormAct.reparameterize()")
        self[0] = fuse_conv_bn_eval(self[0], self[1])
        self[1] = nn.Identity()
        return self


# ---------------------------------------------------------------------------
# Disentangler (local unitary across block boundaries)
# ---------------------------------------------------------------------------

# MERA <-> wavelet correspondence (Evenbly & White, PRL 116, 140403, 2016):
# a shifted 2-site rotation at +pi/6 followed by an aligned rotation at -pi/12
# reproduces the Daubechies-4 low/high-pass filters exactly; a single aligned
# rotation at -pi/4 (no disentangler, i.e. a tree) is the Haar wavelet.
D4_DISENTANGLER_ANGLE = math.pi / 6
D4_ISOMETRY_ANGLE = -math.pi / 12
HAAR_ISOMETRY_ANGLE = -math.pi / 4


def rotation_2x2(angle: float) -> Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s], [s, c]])


def quaternion_left(q: Tensor) -> Tensor:
    """Matrix of ``x -> q * x`` for quaternions ``q = (a, b, c, d)``; (..., 4, 4)."""
    a, b, c, d = q.unbind(-1)
    return torch.stack([
        torch.stack([a, -b, -c, -d], -1),
        torch.stack([b, a, -d, c], -1),
        torch.stack([c, d, a, -b], -1),
        torch.stack([d, -c, b, a], -1),
    ], -2)


def quaternion_right(q: Tensor) -> Tensor:
    """Matrix of ``x -> x * q`` for quaternions ``q = (a, b, c, d)``; (..., 4, 4)."""
    a, b, c, d = q.unbind(-1)
    return torch.stack([
        torch.stack([a, -b, -c, -d], -1),
        torch.stack([b, a, d, -c], -1),
        torch.stack([c, -d, a, b], -1),
        torch.stack([d, c, -b, a], -1),
    ], -2)


class Disentangler(nn.Module):
    """Per-channel SO(4) gate on every 2x2 block (no channel grouping).

    The 4 values of a 2x2 block in one channel form a quaternion ``x``.  Every
    rotation of R^4 is ``x -> q_L x q_R`` with unit quaternions ``q_L, q_R``
    (SO(4) = SU(2) x SU(2) / Z2), so the gate is exactly orthogonal for any
    parameters, costs 8 numbers per channel and needs no matrix exponential.

    The disentangler only mixes *space* (across the block boundary on the
    shifted grid); channel mixing is left to the dense 1x1 layers that follow
    (an alternating, Trotter-like split of a space x channel gate).  It is
    independent of channel ordering and has no grouping hyper-parameter.

    A train-time circuit of depth k, ``U_k ... U_1``, is again one SO(4)
    rotation per channel and collapses on deploy.  Deploy op: depthwise
    2x2/s2 conv with channel multiplier 4, then DepthToSpace.
    """

    def __init__(self, channels: int, parity: int = 1, circuit_depth: int = 1,
                 init: str = "identity", init_noise: float = 0.05,
                 wavelet_angle: float = D4_DISENTANGLER_ANGLE, **_: object):
        super().__init__()
        if parity not in (0, 1):
            raise ValueError("parity must be 0 or 1")
        if circuit_depth < 1:
            raise ValueError("circuit_depth must be positive")
        if init not in ("wavelet", "identity", "random"):
            raise ValueError("init must be 'wavelet', 'identity' or 'random'")
        self.channels = channels
        self.parity = int(parity)
        self.circuit_depth = circuit_depth
        self.init = init
        self.init_noise = float(init_noise)
        self.wavelet_angle = float(wavelet_angle)
        # (k, 2, C, 4): k gates, (left, right) unit quaternions per channel.
        self.quaternions: nn.Parameter | None = nn.Parameter(
            torch.empty(circuit_depth, 2, channels, 4))
        self.deploy_conv: nn.Conv2d | None = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        if self.quaternions is None:
            return
        q = self.quaternions
        if self.init == "random":
            q.normal_()
            return
        q.zero_()
        q[..., 0] = 1.0                              # identity rotation
        if self.init == "wavelet":
            # Separable 2D rotation R(t) x R(t) on the block, which equals
            # L(cos t, sin t, 0, 0) R(cos t, 0, sin t, 0).  With t = pi/6 on the
            # shifted grid followed by the isometry at -pi/12 this is exactly
            # one analysis step of the Daubechies-4 wavelet
            # (Evenbly & White, PRL 116, 140403 (2016)).
            q[0, 0] = torch.tensor([math.cos(self.wavelet_angle),
                                    math.sin(self.wavelet_angle), 0.0, 0.0])
            q[0, 1] = torch.tensor([math.cos(self.wavelet_angle), 0.0,
                                    math.sin(self.wavelet_angle), 0.0])
        q.add_(torch.randn_like(q), alpha=self.init_noise)

    @property
    def is_reparameterized(self) -> bool:
        return self.deploy_conv is not None

    def unitary(self) -> Tensor:
        """Collapsed gate per channel, shape ``(C, 4, 4)``, float32."""
        if self.deploy_conv is not None:
            return self.deploy_conv.weight.reshape(self.channels, 4, 4)
        device_type = self.quaternions.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            q = F.normalize(self.quaternions.float(), dim=-1)
            gates = quaternion_left(q[:, 0]) @ quaternion_right(q[:, 1])
            matrix = gates[0]
            for gate in gates[1:]:
                matrix = gate @ matrix
        return matrix

    def _block_weight(self) -> Tensor:
        # (C, 4_out, 4_in) -> depthwise weight (4C, 1, 2, 2).  Site index is
        # dy * 2 + dx on both sides, matching the 2x2 kernel and pixel_shuffle.
        return self.unitary().reshape(self.channels * 4, 1, 2, 2)

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        left, right, top, bottom = block_padding(height, width, self.parity)
        # Symmetric one-pixel padding of the shifted grid is done natively by
        # the conv; only odd sizes need an explicit (asymmetric) pad.
        conv_padding = 0
        if (left, right, top, bottom) == (1, 1, 1, 1):
            conv_padding = 1
        elif left or right or top or bottom:
            x = F.pad(x, (left, right, top, bottom))
        if self.deploy_conv is not None:
            weight = self.deploy_conv.weight
        else:
            weight = self._block_weight().to(x.dtype)
        blocks = F.conv2d(x, weight, stride=2, padding=conv_padding,
                          groups=self.channels)
        x = F.pixel_shuffle(blocks, 2)
        if left or right or top or bottom:
            x = x[..., top:top + height, left:left + width]
        return x

    @torch.no_grad()
    def reparameterize(self) -> "Disentangler":
        if self.deploy_conv is not None:
            return self
        weight = self._block_weight().to(self.quaternions.dtype)
        self.deploy_conv = nn.Conv2d(
            self.channels, self.channels * 4, 2, stride=2,
            groups=self.channels, bias=False,
            device=weight.device, dtype=weight.dtype)
        self.deploy_conv.weight.copy_(weight)
        self.quaternions = None
        return self

    def extra_repr(self) -> str:
        return (f"channels={self.channels}, parity={self.parity}, "
                f"circuit_depth={self.circuit_depth}, init={self.init}")


# ---------------------------------------------------------------------------
# Isometry (2x2 -> 1 coarse-graining)
# ---------------------------------------------------------------------------

class Isometry(nn.Module):
    """Map each 2x2 block (``4 * C_in`` values) to one coarse site (``C_out``).

    Implemented as a 2x2 / stride-2 convolution, which is exactly
    space-to-depth followed by a ``4 C_in -> C_out`` matrix.  ``W`` is
    initialised with orthonormal rows (``W W^T = I`` when ``C_out <= 4 C_in``),
    i.e. the MERA isometry that keeps ``C_out`` directions of the block.
    BatchNorm follows and folds on deploy.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 wavelet_angle: float | None = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wavelet_angle = wavelet_angle
        self.projection = ConvNormAct(in_channels, out_channels, 2, stride=2,
                                      padding=0)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        weight = self.projection.conv.weight
        nn.init.orthogonal_(weight)
        if self.wavelet_angle is None:
            return
        # Wavelet bands of each input channel: band 0 = low-low, then the two
        # mixed bands, then high-high.  Output o takes band o // C_in of
        # channel o % C_in, so the first C_in outputs are the low-pass
        # (coarse) copies of every channel.  Rows are orthonormal.
        rotation = rotation_2x2(self.wavelet_angle)
        bands = torch.kron(rotation, rotation).to(weight)        # (4, 4)
        count = min(self.out_channels, 4 * self.in_channels)
        weight[:count].zero_()
        for out_index in range(count):
            band, channel = divmod(out_index, self.in_channels)
            weight[out_index, channel] = bands[band].reshape(2, 2)

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        left, right, top, bottom = block_padding(height, width, 0)
        if right or bottom:
            x = F.pad(x, (left, right, top, bottom))
        return self.projection(x)

    def isometry_defect(self) -> Tensor:
        """``||W W^T - I||_F / sqrt(C_out)`` of the raw (pre-BN) weight."""
        weight = self.projection.conv.weight.flatten(1)
        if weight.shape[0] > weight.shape[1]:
            weight = weight.t()
        gram = weight @ weight.t()
        eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        return (gram - eye).norm() / math.sqrt(gram.shape[0])


class RenormalizationStep(nn.Module):
    """One MERA layer: disentangle across block boundaries, then coarse-grain.

    Deploy forms (both exact):

    * factorised: ``U`` as depthwise 2x2/s2 conv (x4) + DepthToSpace + Slice,
      then ``W`` as 2x2/s2 conv;
    * polyphase (``fuse()``): ``U`` evaluated directly on the aligned blocks
      as a depthwise 4x4/s2 conv with channel multiplier 4 (output channel
      ``c * 4 + phase``), then ``W`` as a 1x1 conv on those ``4 C_in``
      phase channels.  No DepthToSpace / Slice / layout re-orders, and the
      weight count stays ``O(C_in)`` + ``4 C_in C_out`` - unlike collapsing
      ``W U`` into one dense 4x4 conv, which would need ``16 C_in C_out``.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 circuit_depth: int = 1, disentangle: bool = True,
                 wavelet_init: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.disentangler = (
            Disentangler(in_channels, parity=1, circuit_depth=circuit_depth,
                         init="wavelet" if wavelet_init else "random")
            if disentangle else nn.Identity())
        if not wavelet_init:
            angle = None
        elif disentangle:
            angle = D4_ISOMETRY_ANGLE          # MERA layer  -> Daubechies-4
        else:
            angle = HAAR_ISOMETRY_ANGLE        # tree layer  -> Haar
        self.isometry = Isometry(in_channels, out_channels, wavelet_angle=angle)
        self.phase_conv: nn.Conv2d | None = None
        self.mix_conv: nn.Conv2d | None = None

    @property
    def is_fused(self) -> bool:
        return self.phase_conv is not None

    def forward(self, x: Tensor) -> Tensor:
        if self.phase_conv is None:
            return self.isometry(self.disentangler(x))
        height, width = x.shape[-2:]
        # Odd sizes need one extra zero row/col (the isometry's end padding).
        extra_h, extra_w = height % 2, width % 2
        if extra_h or extra_w:
            x = F.pad(x, (0, extra_w, 0, extra_h))
        return self.mix_conv(self.phase_conv(x))

    @torch.no_grad()
    def fuse(self) -> "RenormalizationStep":
        """Switch to the polyphase deploy form (U, W already reparameterised)."""
        if self.phase_conv is not None or isinstance(self.disentangler, nn.Identity):
            return self
        if self.training:
            raise RuntimeError("Call eval() before RenormalizationStep.fuse()")
        if not (self.disentangler.is_reparameterized
                and self.isometry.projection.is_reparameterized):
            raise RuntimeError("Reparameterize the disentangler and isometry first")
        unitary = self.disentangler.unitary()            # (C, 4_out, 4_in)
        conv = self.isometry.projection.conv               # (C_out, C, 2, 2) + bias
        channels = self.in_channels
        # The aligned block (i, j) is read through a 4x4 window starting at
        # fine (2i - 1, 2j - 1).  Its phase (py, px) is site (1 - py, 1 - px)
        # of the shifted block occupying window rows/cols 2py..2py+1, 2px..2px+1.
        weight = unitary.new_zeros(channels, 4, 4, 4)
        for py in range(2):
            for px in range(2):
                site = (1 - py) * 2 + (1 - px)
                weight[:, py * 2 + px, 2 * py:2 * py + 2, 2 * px:2 * px + 2] = (
                    unitary[:, site].reshape(channels, 2, 2))
        self.phase_conv = nn.Conv2d(channels, 4 * channels, 4, stride=2,
                                    padding=1, groups=channels, bias=False,
                                    device=weight.device, dtype=weight.dtype)
        self.phase_conv.weight.copy_(weight.reshape(4 * channels, 1, 4, 4))
        self.mix_conv = nn.Conv2d(4 * channels, self.out_channels, 1, bias=True,
                                  device=weight.device, dtype=weight.dtype)
        self.mix_conv.weight.copy_(
            conv.weight.reshape(self.out_channels, 4 * channels, 1, 1))
        self.mix_conv.bias.copy_(conv.bias)
        self.disentangler = nn.Identity()
        self.isometry = nn.Identity()
        return self

    @torch.no_grad()
    def effective_kernel(self) -> tuple[Tensor, Tensor]:
        """The linear map W U as a dense 4x4/s2 kernel (analysis only)."""
        reference = next(p for p in self.parameters())
        channels = self.in_channels
        probes = torch.zeros(channels * 16, channels, 8, 8,
                             device=reference.device, dtype=reference.dtype)
        index = torch.arange(channels * 16, device=reference.device)
        offsets = index % 16
        probes[index, index // 16, 3 + offsets // 4, 3 + offsets % 4] = 1.0
        bias = self(probes[:1] * 0)[0, :, 2, 2]
        response = self(probes)[:, :, 2, 2] - bias
        weight = response.reshape(channels, 4, 4, self.out_channels)
        return weight.permute(3, 0, 1, 2).contiguous(), bias


class PlainDownsample(nn.Module):
    """Ablation: unstructured 4x4/s2 conv.

    ``W U`` of a ``RenormalizationStep`` is a 4x4/s2 linear map with a
    unitary/isometry structure; this ablation learns the same map class
    without that structure (``16 C_in C_out`` free weights)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = ConvNormAct(in_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        if height % 2 or width % 2:
            x = F.pad(x, (0, width % 2, 0, height % 2))
        return self.conv(x)


# ---------------------------------------------------------------------------
# Same-scale block
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep) / keep


class GatedContraction(nn.Module):
    """Multiplicative channel mixer: ``W_o [ (W_v x) * act(local(W_g x)) ]``.

    The elementwise product is a tensor contraction with a copy tensor, the
    natural nonlinearity of tensor-network models.  ``local_kernel`` adds an
    optional depthwise prior on the gate path (``0`` disables it, ablation).

    Training uses one fused ``W_v ; W_g`` conv followed by a channel split (a
    free view in PyTorch).  On deploy the split is removed by cutting the conv
    into two convs: identical math, but runtimes such as ONNX Runtime no
    longer have to materialise a Split and re-layout the tensor.
    """

    def __init__(self, channels: int, hidden: int, activation: str,
                 local_kernel: int = 3):
        super().__init__()
        self.hidden = hidden
        self.expand: ConvNormAct | None = ConvNormAct(channels, 2 * hidden, 1)
        self.value_conv: nn.Conv2d | None = None
        self.gate_conv: nn.Conv2d | None = None
        self.local = (ConvNormAct(hidden, hidden, local_kernel, groups=hidden)
                      if local_kernel else None)
        self.activation = build_activation(activation)
        self.project = ConvNormAct(hidden, channels, 1)

    @property
    def is_split(self) -> bool:
        return self.expand is None

    def forward(self, x: Tensor) -> Tensor:
        if self.expand is not None:
            value, gate = self.expand(x).chunk(2, dim=1)
        else:
            value, gate = self.value_conv(x), self.gate_conv(x)
        if self.local is not None:
            gate = self.local(gate)
        return self.project(value * self.activation(gate))

    @torch.no_grad()
    def split_expand(self) -> "GatedContraction":
        """Replace the fused expand conv by separate value / gate convs."""
        if self.expand is None:
            return self
        if not self.expand.is_reparameterized:
            raise RuntimeError("Fold BN of the expand conv first")
        conv = self.expand.conv
        convs = []
        for part in (slice(0, self.hidden), slice(self.hidden, 2 * self.hidden)):
            piece = nn.Conv2d(conv.in_channels, self.hidden, 1, bias=True,
                              device=conv.weight.device, dtype=conv.weight.dtype)
            piece.weight.copy_(conv.weight[part])
            piece.bias.copy_(conv.bias[part])
            convs.append(piece)
        self.value_conv, self.gate_conv = convs
        self.expand = None
        return self


class ERBlock(nn.Module):
    """Gated contraction (residual), optionally after a brick-wall disentangler."""

    def __init__(self, channels: int, parity: int,
                 circuit_depth: int = 1, mlp_ratio: float = 2.0,
                 activation: str = "relu", local_kernel: int = 3,
                 drop_path: float = 0.0, layer_scale_init: float = 1e-5,
                 disentangle: bool = True):
        super().__init__()
        self.disentangler = (
            Disentangler(channels, parity=parity, circuit_depth=circuit_depth,
                         init="identity")
            if disentangle else nn.Identity())
        hidden = make_divisible(channels * mlp_ratio, 8)
        self.mixer = GatedContraction(channels, hidden, activation, local_kernel)
        self.layer_scale: nn.Parameter | None = nn.Parameter(
            torch.full((1, channels, 1, 1), float(layer_scale_init)))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        x = self.disentangler(x)
        update = self.mixer(x)
        if self.layer_scale is not None:
            update = update * self.layer_scale
        return x + self.drop_path(update)

    @torch.no_grad()
    def fold_layer_scale(self) -> None:
        """Fold the layer scale into the (already BN-folded) output conv."""
        if self.layer_scale is None:
            return
        project = self.mixer.project
        if not project.is_reparameterized:
            raise RuntimeError("Fold BN of the output projection first")
        scale = self.layer_scale.reshape(-1)
        project.conv.weight.mul_(scale.reshape(-1, 1, 1, 1))
        project.conv.bias.mul_(scale)
        self.layer_scale = None


def build_downsample(in_channels: int, out_channels: int, circuit_depth: int,
                     disentangle: bool, renormalization: str,
                     wavelet_init: bool = True) -> nn.Module:
    if renormalization == "mera":
        return RenormalizationStep(in_channels, out_channels,
                                   circuit_depth=circuit_depth,
                                   disentangle=disentangle,
                                   wavelet_init=wavelet_init)
    if renormalization == "conv4x4":
        return PlainDownsample(in_channels, out_channels)
    raise ValueError("renormalization must be 'mera' or 'conv4x4'")


class ERStem(nn.Module):
    """Transitional layer (3x3/s2 conv) followed by one MERA layer (-> 1/4)."""

    def __init__(self, in_channels: int, out_channels: int, stem_channels: int,
                 activation: str, circuit_depth: int = 1,
                 disentangle: bool = True, renormalization: str = "mera",
                 wavelet_init: bool = True):
        super().__init__()
        self.embed = ConvNormAct(in_channels, stem_channels, 3, stride=2,
                                 activation=activation)
        self.renormalize = build_downsample(
            stem_channels, out_channels, circuit_depth, disentangle,
            renormalization, wavelet_init)

    def forward(self, x: Tensor) -> Tensor:
        return self.renormalize(self.embed(x))
