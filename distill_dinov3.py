"""Distill a DINOv3 teacher into ER-Net with LightlyTrain (unlabeled images).

Typical use (ImageNet-1k train images, labels are ignored):

    python distill_dinov3.py --model ernet_n \
        --data /kaggle/working/imagenet/train --out runs/distill_ernet_n \
        --epochs 100 --batch-size 256 --devices 2

Outputs (in --out):
    exported_models/exported_last.pt   LightlyTrain export (backbone state dict)
    ernet_distilled.pth                {'model': backbone state_dict}, no
                                        classification head: COCO backbone, or
                                        main.py --finetune (fresh head)
Resume an interrupted run with the same arguments plus --resume.

The same command with --no-disentangle / --renormalization conv4x4 /
--no-wavelet-init distills the ablations with an identical recipe, which is
required for a fair architecture comparison.

Requires: pip install lightly-train  (AGPL-3.0; DINOv3 weights are under the
DINOv3 license and are downloaded automatically by LightlyTrain).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

import model as _registry  # noqa: F401  (registers ER-Net with timm)
from model.ernet import ERNet
from model.ernet.lightly import ERNetDistillWrapper, load_distilled_weights


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("DINOv3 -> ER-Net distillation (LightlyTrain)")
    p.add_argument("--model", default="ernet_n", help="ernet_n | ernet_t | ernet_s")
    p.add_argument("--data", required=True,
                   help="Image folder, e.g. ImageNet train/ (searched recursively)")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--teacher", default="dinov3/vitb16",
                   help="LightlyTrain teacher name, e.g. dinov3/vitb16, dinov3/vitl16, "
                        "dinov2/vitb14 (Apache-2.0)")
    p.add_argument("--teacher-weights", default=None,
                   help="Optional local teacher weights (no download)")
    p.add_argument("--method", default="distillation",
                   help="distillation (=v3, global + dense), distillationv1, distillationv2")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256, help="GLOBAL batch size")
    p.add_argument("--accum-steps", type=int, default=1,
                   help="Gradient accumulation (global batch per update = batch*accum)")
    p.add_argument("--devices", default="auto", help="'auto', a count, or '0,1'")
    p.add_argument("--num-workers", default="auto", help="per device")
    p.add_argument("--precision", default="auto",
                   help="auto = bf16-mixed on Ampere+ GPUs, 16-mixed on older (T4/P100)")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=None,
                   help="Override LightlyTrain's AdamW lr (default 5e-4, sqrt-scaled)")
    p.add_argument("--feature-levels", default="3,4",
                   help="Stages whose features are distilled (fused at the finest one)")
    p.add_argument("--images-per-class", type=int, default=0,
                   help="Use only N images per class folder (0 = all); faster epochs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true",
                   help="Resume an interrupted run (same --out and arguments)")
    p.add_argument("--overwrite", action="store_true")
    # ER-Net architecture / ablations (same flags as main.py)
    p.add_argument("--circuit-depth", type=int, default=1)
    p.add_argument("--no-disentangle", action="store_false", dest="disentangle")
    p.add_argument("--block-disentangle", action="store_true")
    p.add_argument("--no-wavelet-init", action="store_false", dest="wavelet_init")
    p.add_argument("--renormalization", default="mera", choices=["mera", "conv4x4"])
    p.add_argument("--local-kernel", type=int, default=3)
    p.add_argument("--activation", default="relu")
    return p.parse_args()


def resolve_precision(choice: str) -> str:
    if choice != "auto":
        return choice
    if not torch.cuda.is_available():
        return "32-true"
    # torch.cuda.is_bf16_supported() is also True for emulated bf16 (T4, P100),
    # which is slow; use native fp16 mixed precision there.
    major, _ = torch.cuda.get_device_capability(0)
    return "bf16-mixed" if major >= 8 else "16-mixed"


def subset_images(root: str, per_class: int, seed: int) -> list[str]:
    """All images, or a deterministic N per class folder, from the cached index."""
    from data.datasets import CachedImageFolder
    index = CachedImageFolder(root)
    by_class: dict[int, list[str]] = defaultdict(list)
    for path, target in index.samples:
        by_class[target].append(path)
    rng = random.Random(seed)
    chosen = []
    for target in sorted(by_class):
        paths = sorted(by_class[target])
        if per_class > 0:
            rng.shuffle(paths)
            paths = paths[:per_class]
        chosen.extend(paths)
    return chosen


def build_student(args: argparse.Namespace) -> ERNet:
    import timm
    # No classification head: it is not part of the distillation objective.
    return timm.create_model(
        args.model, headless=True, distillation=False,
        activation=args.activation, circuit_depth=args.circuit_depth,
        disentangle=args.disentangle, block_disentangle=args.block_disentangle,
        renormalization=args.renormalization, wavelet_init=args.wavelet_init,
        local_kernel=args.local_kernel, pretrained=False)


def main() -> int:
    args = parse_args()
    try:
        import lightly_train
    except ImportError:
        print("LightlyTrain is not installed: pip install lightly-train", file=sys.stderr)
        return 1

    student = build_student(args)
    levels = [int(v) for v in args.feature_levels.split(",") if v.strip()]
    wrapper = ERNetDistillWrapper(student, feature_levels=levels)

    # Always hand LightlyTrain an explicit file list from the cached index: with a
    # list it only checks extensions, whereas a directory is re-walked
    # (sequentially) at every start, including every --resume.
    data = subset_images(args.data, args.images_per_class, args.seed)
    print(f"Using {len(data)} images "
          f"({'all' if args.images_per_class <= 0 else args.images_per_class} per class)")

    method_args: dict = {"teacher": args.teacher}
    if args.teacher_weights:
        method_args["teacher_weights"] = args.teacher_weights
    devices: int | str = int(args.devices) if args.devices.isdigit() else args.devices
    workers: int | str = (int(args.num_workers) if str(args.num_workers).isdigit()
                          else args.num_workers)
    precision = resolve_precision(args.precision)
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Student {args.model} (no classification head): "
          f"{trainable / 1e6:.2f}M params, "
          f"distilled features = stages {levels} ({wrapper.feature_dim()} channels)")
    print(f"Teacher {args.teacher} | method {args.method} | precision {precision} | "
          f"global batch {args.batch_size} x accum {args.accum_steps} | "
          f"{args.epochs} epochs")

    kwargs = dict(
        out=args.out, data=data, model=wrapper, method=args.method,
        method_args=method_args, epochs=args.epochs, batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum_steps, devices=devices,
        num_workers=workers, precision=precision, seed=args.seed,
        transform_args={"image_size": (args.image_size, args.image_size)},
        resume_interrupted=args.resume, overwrite=args.overwrite,
    )
    if args.lr is not None:
        kwargs["optim_args"] = {"lr": args.lr}
    lightly_train.pretrain(**kwargs)

    # Only the main process writes the final checkpoint.
    if int(os.environ.get("LOCAL_RANK", 0)) != 0 or int(os.environ.get("RANK", 0)) != 0:
        return 0
    exported = Path(args.out) / "exported_models" / "exported_last.pt"
    if not exported.exists():
        print(f"Export not found at {exported}", file=sys.stderr)
        return 1
    fresh = build_student(args)
    load_distilled_weights(fresh, str(exported), strict=True)
    target = Path(args.out) / "ernet_distilled.pth"
    torch.save({"model": fresh.state_dict(), "args": vars(args)}, target)
    print(f"Distilled ER-Net backbone weights (no classification head): {target}")
    print("Next: COCO backbone via model.ernet.lightly.load_distilled_weights(), or "
          f"ImageNet fine-tune: DISTILLATION_TYPE=none ... --finetune {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
