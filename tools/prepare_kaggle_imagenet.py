"""Prepare Kaggle "ImageNet Object Localization Challenge" for classification.

The Kaggle copy of ILSVRC2012 keeps the training images in class folders but
stores the 50,000 validation images in one flat folder, with their labels in
``LOC_val_solution.csv`` (detection format: ``ImageId,PredictionString`` where
the prediction string is ``wnid x_min y_min x_max y_max [wnid ...]``).

``main.py --data-set IMNET`` (torchvision ImageFolder) needs::

    DATA_PATH/train/<wnid>/*.JPEG
    DATA_PATH/val/<wnid>/*.JPEG

This script builds that layout without touching the read-only input:

    OUT/train                      -> symlink to .../ILSVRC/Data/CLS-LOC/train
    OUT/val/<wnid>/<ImageId>.JPEG  -> symlink (or copy with --copy) of each val image

The class of a validation image is the wnid of its boxes (all boxes of an
ILSVRC2012 CLS-LOC val image share one class; the script checks this).  If
the CSV is missing it falls back to ``ILSVRC/Annotations/CLS-LOC/val/*.xml``.

Kaggle usage (internet not needed)::

    python tools/prepare_kaggle_imagenet.py \
        --kaggle-root /kaggle/input/imagenet-object-localization-challenge \
        --out /kaggle/working/imagenet
    DATA_PATH=/kaggle/working/imagenet bash train_ernet_imagenet.sh ernet_n
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ElementTree
from collections import Counter, defaultdict
from pathlib import Path

WNID = re.compile(r"^n\d{8}$")
DEFAULT_ROOTS = (
    "/kaggle/input/imagenet-object-localization-challenge",
    "/kaggle/input/imagenet-object-localization-challenge/imagenet-object-localization-challenge",
)


def find_layout(root: Path) -> dict[str, Path | None]:
    """Locate the Kaggle files under ``root`` without walking the image trees."""
    candidates = [root] + [p for p in root.iterdir() if p.is_dir()]
    candidates += [q for p in candidates[1:] if p.name != "ILSVRC"
                   for q in p.iterdir() if q.is_dir()]
    layout: dict[str, Path | None] = dict(csv=None, mapping=None, cls_loc=None,
                                         annotations=None)
    for base in candidates:
        if layout["csv"] is None and (base / "LOC_val_solution.csv").is_file():
            layout["csv"] = base / "LOC_val_solution.csv"
        if layout["mapping"] is None and (base / "LOC_synset_mapping.txt").is_file():
            layout["mapping"] = base / "LOC_synset_mapping.txt"
        data = base / "ILSVRC" / "Data" / "CLS-LOC"
        if layout["cls_loc"] is None and data.is_dir():
            layout["cls_loc"] = data
            annotations = base / "ILSVRC" / "Annotations" / "CLS-LOC" / "val"
            layout["annotations"] = annotations if annotations.is_dir() else None
    return layout


def read_mapping(path: Path | None) -> list[str]:
    if path is None:
        return []
    wnids = []
    with open(path) as handle:
        for line in handle:
            token = line.strip().split(" ", 1)[0]
            if WNID.match(token):
                wnids.append(token)
    return wnids


def labels_from_csv(path: Path) -> tuple[dict[str, str], list[str]]:
    labels, conflicts = {}, []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = row["ImageId"].strip()
            wnids = [t for t in row["PredictionString"].split() if WNID.match(t)]
            if not wnids:
                raise ValueError(f"No wnid in PredictionString of {image_id}")
            if len(set(wnids)) > 1:
                conflicts.append(image_id)
            labels[image_id] = Counter(wnids).most_common(1)[0][0]
    return labels, conflicts


def labels_from_xml(folder: Path) -> tuple[dict[str, str], list[str]]:
    labels, conflicts = {}, []
    for xml_file in sorted(folder.glob("*.xml")):
        names = [n.text.strip() for n in ElementTree.parse(xml_file).iter("name")
                 if n.text and WNID.match(n.text.strip())]
        if not names:
            continue
        if len(set(names)) > 1:
            conflicts.append(xml_file.stem)
        labels[xml_file.stem] = Counter(names).most_common(1)[0][0]
    return labels, conflicts


def link_or_copy(source: Path, target: Path, copy: bool) -> None:
    if target.is_symlink() or target.exists():
        return
    if copy:
        shutil.copy2(source, target)
    else:
        os.symlink(source, target)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Turn the Kaggle ImageNet localization data into "
                    "train/<wnid>, val/<wnid> folders for classification")
    parser.add_argument("--kaggle-root", default=None,
                        help="Folder of the Kaggle dataset (auto-detected on Kaggle)")
    parser.add_argument("--out", default="/kaggle/working/imagenet",
                        help="Output DATA_PATH (must be writable)")
    parser.add_argument("--copy", action="store_true",
                        help="Copy val images (~6.4 GB) instead of symlinking")
    parser.add_argument("--expect-classes", type=int, default=1000)
    parser.add_argument("--expect-val", type=int, default=50000)
    parser.add_argument("--force", action="store_true",
                        help="Rebuild OUT/val and the OUT/train link")
    args = parser.parse_args()

    root = Path(args.kaggle_root) if args.kaggle_root else next(
        (Path(p) for p in DEFAULT_ROOTS if Path(p).is_dir()), None)
    if root is None or not root.is_dir():
        print("Kaggle dataset folder not found; pass --kaggle-root", file=sys.stderr)
        return 1
    layout = find_layout(root)
    if layout["cls_loc"] is None:
        print(f"ILSVRC/Data/CLS-LOC not found under {root}", file=sys.stderr)
        return 1
    train_dir = layout["cls_loc"] / "train"
    val_dir = layout["cls_loc"] / "val"
    print(f"train images : {train_dir}")
    print(f"val images   : {val_dir}")

    # ---- labels of the validation images
    if layout["csv"] is not None:
        print(f"val labels   : {layout['csv']}")
        labels, conflicts = labels_from_csv(layout["csv"])
    elif layout["annotations"] is not None:
        print(f"val labels   : {layout['annotations']} (XML)")
        labels, conflicts = labels_from_xml(layout["annotations"])
    else:
        print("Neither LOC_val_solution.csv nor val XML annotations found",
              file=sys.stderr)
        return 1

    # ---- class list: train folders must define the same 1000 classes
    train_classes = sorted(e.name for e in os.scandir(train_dir)
                           if e.is_dir() and WNID.match(e.name))
    mapping = read_mapping(layout["mapping"])
    errors = []
    if len(train_classes) != args.expect_classes:
        errors.append(f"{len(train_classes)} train class folders, expected "
                      f"{args.expect_classes}")
    if mapping and sorted(mapping) != train_classes:
        errors.append("LOC_synset_mapping.txt does not match the train folders")
    unknown = sorted({w for w in labels.values()} - set(train_classes))
    if unknown:
        errors.append(f"{len(unknown)} val wnids have no train folder, e.g. {unknown[:3]}")
    if errors:
        for message in errors:
            print(f"ERROR: {message}", file=sys.stderr)
        return 1

    # ---- build OUT
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_link = out / "train"
    if train_link.is_symlink() and args.force:
        train_link.unlink()
    if not train_link.exists():
        os.symlink(train_dir, train_link)
    elif train_link.resolve() != train_dir.resolve():
        print(f"ERROR: {train_link} exists and is not a link to {train_dir} "
              "(use --force or another --out)", file=sys.stderr)
        return 1
    out_val = out / "val"
    if out_val.exists() and args.force:
        shutil.rmtree(out_val)
    for wnid in train_classes:              # every class folder, even if empty,
        (out_val / wnid).mkdir(parents=True, exist_ok=True)  # keeps indices aligned

    missing = []
    per_class: dict[str, int] = defaultdict(int)
    for index, (image_id, wnid) in enumerate(sorted(labels.items()), 1):
        source = val_dir / f"{image_id}.JPEG"
        if not source.is_file():
            missing.append(image_id)
            continue
        link_or_copy(source, out_val / wnid / source.name, args.copy)
        per_class[wnid] += 1
        if index % 10000 == 0:
            print(f"  {index}/{len(labels)} val images placed", flush=True)

    # ---- report
    total = sum(per_class.values())
    counts = [per_class.get(w, 0) for w in train_classes]
    print("\nSummary")
    print(f"  classes (train = val)   : {len(train_classes)}")
    print(f"  val images placed       : {total} ({'copied' if args.copy else 'symlinked'})")
    print(f"  images per val class    : min {min(counts)}, max {max(counts)}")
    print(f"  multi-class val images  : {len(conflicts)} (majority wnid used)")
    print(f"  labels without an image : {len(missing)}")
    ok = total == args.expect_val and not missing and min(counts) > 0
    if not ok:
        print("WARNING: the validation set is incomplete, check the counts above",
              file=sys.stderr)
    print(f"\nDATA_PATH={out}")
    print(f"Train with:  DATA_PATH={out} bash train_ernet_imagenet.sh ernet_n")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
