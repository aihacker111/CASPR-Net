"""Train a small CASPR-Net on synthetic images and verify exact folding.

This is a fast correctness experiment, not an accuracy benchmark. It updates
BatchNorm statistics and weights, then checks the complete validation set
before and after reparameterization.
"""
from __future__ import annotations

import argparse
import copy
import math
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model.casprnet import CasprNet, SUPPORTED_ACTIVATIONS


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_synthetic_dataset(
    num_samples: int,
    num_classes: int,
    image_size: int,
    seed: int,
) -> TensorDataset:
    """Create oriented frequency patterns with class-dependent colors."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(num_samples) % num_classes
    labels = labels[torch.randperm(num_samples, generator=generator)]
    coordinates = torch.linspace(-1.0, 1.0, image_size)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    images = torch.empty(num_samples, 3, image_size, image_size)

    for index, label_tensor in enumerate(labels):
        label = int(label_tensor)
        angle = math.pi * label / num_classes
        frequency = 1.0 + (label % 4)
        axis = math.cos(angle) * xx + math.sin(angle) * yy
        stripe = torch.sin(math.pi * frequency * axis)
        radial = torch.cos(
            math.pi * (1.0 + label // 4) * torch.sqrt(xx.square() + yy.square())
        )
        color = torch.tensor(
            [
                0.35 + 0.65 * ((label + 0) % 3 == 0),
                0.35 + 0.65 * ((label + 1) % 3 == 0),
                0.35 + 0.65 * ((label + 2) % 3 == 0),
            ]
        ).reshape(3, 1, 1)
        signal = color * (0.65 * stripe + 0.35 * radial).unsqueeze(0)
        noise = 0.12 * torch.randn(3, image_size, image_size, generator=generator)
        images[index] = (signal + noise).clamp(-1.0, 1.0)
    return TensorDataset(images, labels.long())


def build_debug_model(
    num_classes: int,
    activation: str,
    metric_enabled: bool = True,
) -> CasprNet:
    return CasprNet(
        widths=(16, 32, 64, 96),
        depths=(1, 1, 2, 1),
        group_sizes=(8, 8, 8, 8),
        sparse_levels=(1, 1, 1, 1),
        head_dim=128,
        stem_channels=8,
        transition_group_size=8,
        spatial_branches=(2, 2, 2, 2),
        activation=activation,
        num_classes=num_classes,
        drop_path_rate=0.0,
        metric_enabled=metric_enabled,
        metric_update_interval=1,
    )


@torch.no_grad()
def evaluate(
    network: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    network.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = network(images)
        total_loss += float(criterion(logits, labels))
        total_correct += int((logits.argmax(1) == labels).sum())
        total_samples += labels.numel()
    return total_loss / total_samples, 100.0 * total_correct / total_samples


@torch.no_grad()
def compare_models(
    reference: nn.Module,
    deployed: nn.Module,
    loader: DataLoader,
    device: torch.device,
    atol: float,
    rtol: float,
) -> dict[str, float | int | bool]:
    reference.eval()
    deployed.eval()
    max_abs = 0.0
    max_rel = 0.0
    absolute_sum = 0.0
    value_count = 0
    prediction_mismatches = 0
    all_close = True
    for images, _ in loader:
        images = images.to(device)
        before = reference(images)
        after = deployed(images)
        difference = (before - after).abs()
        allowed = atol + rtol * before.abs()
        all_close = all_close and bool(torch.all(difference <= allowed).item())
        max_abs = max(max_abs, float(difference.max()))
        max_rel = max(
            max_rel,
            float((difference / before.abs().clamp_min(1e-12)).max()),
        )
        absolute_sum += float(difference.sum())
        value_count += difference.numel()
        prediction_mismatches += int(before.argmax(1).ne(after.argmax(1)).sum())
    return {
        "passed": all_close and prediction_mismatches == 0,
        "max_abs_error": max_abs,
        "mean_abs_error": absolute_sum / value_count,
        "max_rel_error": max_rel,
        "prediction_mismatches": prediction_mismatches,
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


@torch.inference_mode()
def measure_latency(
    network: nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    """Measure per-batch latency after warm-up with device synchronization."""
    if warmup < 0 or iterations < 1:
        raise ValueError("latency warmup must be >= 0 and iterations must be >= 1")
    network.eval()
    for _ in range(warmup):
        network(inputs)
    synchronize(device)

    if device.type == "cuda":
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        for index in range(iterations):
            starts[index].record()
            network(inputs)
            ends[index].record()
        torch.cuda.synchronize(device)
        samples = torch.tensor(
            [start.elapsed_time(end) for start, end in zip(starts, ends)],
            dtype=torch.float64,
        )
    else:
        measurements = []
        for _ in range(iterations):
            synchronize(device)
            started = time.perf_counter()
            network(inputs)
            synchronize(device)
            measurements.append((time.perf_counter() - started) * 1000.0)
        samples = torch.tensor(measurements, dtype=torch.float64)

    mean_ms = float(samples.mean())
    batch_size = inputs.shape[0]
    return {
        "mean_ms": mean_ms,
        "median_ms": float(samples.median()),
        "p95_ms": float(torch.quantile(samples, 0.95)),
        "std_ms": float(samples.std(unbiased=False)),
        "throughput": batch_size * 1000.0 / mean_ms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("CASPR-Net synthetic train and fold verifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--val-samples", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--activation", choices=SUPPORTED_ACTIVATIONS, default="relu")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--atol", type=float, default=3e-5)
    parser.add_argument("--rtol", type=float, default=3e-5)
    parser.add_argument("--latency-batch-size", type=int, default=1)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--latency-iters", type=int, default=100)
    parser.add_argument(
        "--disable-metric",
        action="store_true",
        help="Ablate CASPR spatial and sparse-channel metric preconditioning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_set = make_synthetic_dataset(
        args.train_samples, args.num_classes, args.image_size, args.seed
    )
    val_set = make_synthetic_dataset(
        args.val_samples, args.num_classes, args.image_size, args.seed + 1
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    network = build_debug_model(
        args.num_classes,
        args.activation,
        metric_enabled=not args.disable_metric,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )

    print(
        f"model=CASPR-debug activation={args.activation} device={device} "
        f"metric={'off' if args.disable_metric else 'on'} "
        f"params={sum(p.numel() for p in network.parameters()) / 1e6:.3f}M"
    )
    for epoch in range(args.epochs):
        started = time.perf_counter()
        network.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = network(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * labels.numel()
            total_correct += int((logits.argmax(1) == labels).sum())
            total_samples += labels.numel()
        scheduler.step()
        val_loss, val_accuracy = evaluate(network, val_loader, device)
        print(
            f"epoch={epoch + 1:02d}/{args.epochs} "
            f"train_loss={running_loss / total_samples:.4f} "
            f"train_acc={100.0 * total_correct / total_samples:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.2f}% "
            f"time={time.perf_counter() - started:.2f}s"
        )

    reference = copy.deepcopy(network).eval()
    deployed = copy.deepcopy(reference).reparameterize(verbose=True)
    before_loss, before_accuracy = evaluate(reference, val_loader, device)
    after_loss, after_accuracy = evaluate(deployed, val_loader, device)
    comparison = compare_models(
        reference, deployed, val_loader, device, args.atol, args.rtol
    )
    if args.latency_batch_size < 1:
        raise ValueError("--latency-batch-size must be positive")
    latency_images = val_set.tensors[0][:args.latency_batch_size].to(device)
    latency_before = measure_latency(
        reference,
        latency_images,
        device,
        args.latency_warmup,
        args.latency_iters,
    )
    latency_after = measure_latency(
        deployed,
        latency_images,
        device,
        args.latency_warmup,
        args.latency_iters,
    )
    speedup = latency_before["mean_ms"] / latency_after["mean_ms"]
    latency_reduction = 100.0 * (
        1.0 - latency_after["mean_ms"] / latency_before["mean_ms"]
    )
    print("\nReparameterization equivalence")
    print(f"  before: loss={before_loss:.8f} accuracy={before_accuracy:.4f}%")
    print(f"  after : loss={after_loss:.8f} accuracy={after_accuracy:.4f}%")
    print(f"  max_abs_error={comparison['max_abs_error']:.3e}")
    print(f"  mean_abs_error={comparison['mean_abs_error']:.3e}")
    print(f"  max_rel_error={comparison['max_rel_error']:.3e}")
    print(f"  prediction_mismatches={comparison['prediction_mismatches']}")
    print(f"  result={'PASS' if comparison['passed'] else 'FAIL'}")

    print(
        f"\nLatency benchmark (batch={args.latency_batch_size}, "
        f"warmup={args.latency_warmup}, iterations={args.latency_iters})"
    )
    print(
        "  before: "
        f"mean={latency_before['mean_ms']:.4f}ms "
        f"median={latency_before['median_ms']:.4f}ms "
        f"p95={latency_before['p95_ms']:.4f}ms "
        f"throughput={latency_before['throughput']:.2f} images/s"
    )
    print(
        "  after : "
        f"mean={latency_after['mean_ms']:.4f}ms "
        f"median={latency_after['median_ms']:.4f}ms "
        f"p95={latency_after['p95_ms']:.4f}ms "
        f"throughput={latency_after['throughput']:.2f} images/s"
    )
    print(f"  speedup={speedup:.3f}x latency_reduction={latency_reduction:.2f}%")
    if not comparison["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
