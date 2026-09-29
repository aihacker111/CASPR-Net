"""Profile CASPR-Net parameters and MACs before or after reparameterization.

Conventions:
  * one multiply-accumulate (MAC) is reported as one operation;
  * Conv2d and Linear operations are counted;
  * normalization, activation, elementwise coupling and pooling are excluded;
  * all values are for one inference image.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

from model.casprnet import CasprNet, casprnet_n, casprnet_s, casprnet_t


MODEL_FACTORIES: dict[str, Callable[..., CasprNet]] = {
    "casprnet_n": casprnet_n,
    "casprnet_t": casprnet_t,
    "casprnet_s": casprnet_s,
}


@dataclass
class ModelProfile:
    name: str
    graph: str
    parameters: int
    trainable_parameters: int
    macs: int


@torch.no_grad()
def profile_model(
    name: str,
    in_channels: int,
    image_size: int,
    num_classes: int,
    device: torch.device,
    deploy: bool,
) -> ModelProfile:
    network = MODEL_FACTORIES[name](
        in_chans=in_channels,
        num_classes=num_classes,
        distillation=False,
    ).to(device).eval()
    if deploy:
        network.reparameterize()

    macs = 0
    handles = []

    def count_conv(module: nn.Conv2d, _inputs: tuple[Tensor], output: Tensor) -> None:
        nonlocal macs
        kernel_height, kernel_width = module.kernel_size
        macs += output.numel() * (
            kernel_height
            * kernel_width
            * module.in_channels
            // module.groups
        )

    def count_linear(module: nn.Linear, _inputs: tuple[Tensor], output: Tensor) -> None:
        nonlocal macs
        macs += output.numel() * module.in_features

    for module in network.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(count_conv))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(count_linear))

    dummy = torch.zeros(1, in_channels, image_size, image_size, device=device)
    network(dummy)
    for handle in handles:
        handle.remove()

    return ModelProfile(
        name=name,
        graph="deploy" if deploy else "train",
        parameters=sum(parameter.numel() for parameter in network.parameters()),
        trainable_parameters=sum(
            parameter.numel()
            for parameter in network.parameters()
            if parameter.requires_grad
        ),
        macs=macs,
    )


def print_profiles(
    profiles: list[ModelProfile],
    in_channels: int,
    image_size: int,
    num_classes: int,
) -> None:
    print(
        f"Input: 1 x {in_channels} x {image_size} x {image_size}; "
        f"classes={num_classes}"
    )
    print("Convention: 1 MAC = 1 operation; norm/activation/elementwise excluded")
    print()
    header = (
        f"{'Model':<12} {'Graph':<8} {'Params(M)':>10} "
        f"{'Trainable(M)':>13} {'MACs(G)':>10}"
    )
    print(header)
    print("-" * len(header))
    for profile in profiles:
        print(
            f"{profile.name:<12} {profile.graph:<8} "
            f"{profile.parameters / 1e6:>10.3f} "
            f"{profile.trainable_parameters / 1e6:>13.3f} "
            f"{profile.macs / 1e9:>10.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Profile CASPR-Net variants")
    parser.add_argument(
        "--model",
        choices=("all", *sorted(MODEL_FACTORIES)),
        default="all",
    )
    parser.add_argument("--preset", choices=("synthetic", "imagenet"), default="imagenet")
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--in-chans", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--deploy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = {
        "synthetic": dict(input_size=32, in_chans=3, num_classes=8),
        "imagenet": dict(input_size=224, in_chans=3, num_classes=1000),
    }[args.preset]
    image_size = args.input_size or preset["input_size"]
    in_channels = args.in_chans or preset["in_chans"]
    num_classes = args.num_classes or preset["num_classes"]
    device = torch.device(args.device)
    names = sorted(MODEL_FACTORIES) if args.model == "all" else [args.model]
    profiles = [
        profile_model(
            name,
            in_channels,
            image_size,
            num_classes,
            device,
            args.deploy,
        )
        for name in names
    ]
    print_profiles(profiles, in_channels, image_size, num_classes)


if __name__ == "__main__":
    main()
