"""Measure ER-Net bond dimensions (stage widths) with the MERA truncation rule.

For every renormalization step (stem and stage transitions) this tool collects
the 2x2 block vectors that the isometry W sees (dimension 4*C_in) and computes
the spectrum of their covariance, i.e. the reduced density matrix of a block.

Reported per step:
  * S_tree : block entanglement entropy *without* the disentangler,
  * S_mera : block entropy *after* the disentangler (should be lower: this is
             exactly what the disentangler is for),
  * chi(eps): smallest number of directions keeping (1 - eps) of the spectral
             weight after the disentangler -> the width the next stage needs.

Typical use: train a short proxy (e.g. 30-100 epochs, ImageNet-100), run

    python tools/ernet_bond_dims.py --model ernet_n \
        --checkpoint checkpoints/proxy/checkpoint_best.pth \
        --data-path /path/to/imagenet/val --max-images 2000 --eps 1e-2 1e-3

and copy the suggested widths into model/ernet/variants.py.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import model  # noqa: E402,F401  (registers ER-Net with timm)
from model.ernet import ERNet, RenormalizationStep  # noqa: E402


class BlockSpectrum:
    """Streaming covariance of 2x2 block vectors."""

    def __init__(self) -> None:
        self.count = 0
        self.sum: Tensor | None = None
        self.outer: Tensor | None = None

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        height, width = x.shape[-2:]
        x = F.pad(x, (0, width % 2, 0, height % 2))
        blocks = F.pixel_unshuffle(x.float(), 2)          # (B, 4C, h, w)
        vectors = blocks.permute(0, 2, 3, 1).reshape(-1, blocks.shape[1]).double()
        if self.sum is None:
            dim = vectors.shape[1]
            self.sum = vectors.new_zeros(dim)
            self.outer = vectors.new_zeros(dim, dim)
        self.count += vectors.shape[0]
        self.sum += vectors.sum(0)
        self.outer += vectors.t() @ vectors

    def eigenvalues(self) -> Tensor:
        mean = self.sum / self.count
        covariance = self.outer / self.count - torch.outer(mean, mean)
        values = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
        return values / values.sum()


def entropy(p: Tensor) -> float:
    p = p[p > 0]
    return float(-(p * p.log()).sum())


def chi(p: Tensor, eps: float) -> int:
    cumulative = torch.cumsum(p, 0)
    return int(torch.searchsorted(cumulative, torch.tensor(1.0 - eps,
               dtype=cumulative.dtype)).item()) + 1


def round_up(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def build_loader(args: argparse.Namespace):
    if args.synthetic:
        print("WARNING: synthetic noise input, spectra are meaningless "
              "(smoke test only)")
        images = torch.randn(args.max_images, 3, args.input_size, args.input_size)
        return [(images[i:i + args.batch_size], None)
                for i in range(0, args.max_images, args.batch_size)]
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.Resize(int(args.input_size * 256 / 224)),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    generator = torch.Generator().manual_seed(0)
    indices = torch.randperm(len(dataset), generator=generator)[:args.max_images]
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    return torch.utils.data.DataLoader(subset, batch_size=args.batch_size,
                                       num_workers=args.num_workers)


def main() -> None:
    parser = argparse.ArgumentParser("ER-Net bond-dimension measurement")
    parser.add_argument("--model", default="ernet_n")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--data-path", default="")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=2000)
    parser.add_argument("--eps", type=float, nargs="+", default=[1e-2, 1e-3])
    parser.add_argument("--align", type=int, default=16,
                        help="Round suggested widths up to this multiple")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.synthetic and not args.data_path:
        parser.error("--data-path (ImageFolder directory) or --synthetic is required")

    import timm
    net: ERNet = timm.create_model(args.model, num_classes=1000)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = state.get("model_ema") or state.get("model") or state
        net.reset_classifier(state["head.fc.weight"].shape[0])
        print(net.load_state_dict(state, strict=False))
    net = net.to(args.device).eval()

    steps = [(name, module) for name, module in net.named_modules()
             if isinstance(module, RenormalizationStep)]
    tree = {name: BlockSpectrum() for name, _ in steps}
    mera = {name: BlockSpectrum() for name, _ in steps}
    def make_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[Tensor], output: Tensor) -> None:
            tree[name].update(inputs[0])
            mera[name].update(output)
        return hook

    handles = [step.disentangler.register_forward_hook(make_hook(name))
               for name, step in steps]

    seen = 0
    with torch.no_grad():
        for images, _ in build_loader(args):
            net(images.to(args.device))
            seen += images.shape[0]
            if seen >= args.max_images:
                break
    for handle in handles:
        handle.remove()

    header = f"{'step':<16}{'C_in':>6}{'4C_in':>7}{'C_out':>7}{'S_tree':>9}{'S_mera':>9}"
    header += "".join(f"{f'chi({eps:g})':>12}" for eps in args.eps)
    print(f"\nimages={seen}  input={args.input_size}")
    print(header)
    print("-" * len(header))
    suggestions = {eps: [] for eps in args.eps}
    for name, step in steps:
        p_tree = tree[name].eigenvalues()
        p_mera = mera[name].eigenvalues()
        row = (f"{name:<16}{step.in_channels:>6}{4 * step.in_channels:>7}"
               f"{step.out_channels:>7}{entropy(p_tree):>9.3f}{entropy(p_mera):>9.3f}")
        for eps in args.eps:
            value = chi(p_mera, eps)
            suggestions[eps].append(round_up(value, args.align))
            row += f"{value:>12d}"
        print(row)
    print("\nS is in nats; S_mera < S_tree means the disentangler removed "
          "short-range entanglement before coarse-graining.")
    for eps, widths in suggestions.items():
        print(f"suggested widths (eps={eps:g}, aligned to {args.align}): "
              f"{tuple(widths)}  [stem->stage1, stage1->2, 2->3, 3->4]")


if __name__ == "__main__":
    main()
