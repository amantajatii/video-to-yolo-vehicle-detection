"""Assemble the final training-ready YOLO dataset and validate it.

Reads the locked split (data/split_assignment.csv), copies images + reviewed labels into

    <out>/
      images/{train,val,test}/*.jpg
      labels/{train,val,test}/*.txt
      data.yaml
      dataset_stats.csv

and validates every label line (5 fields, class id in range, normalised coords in (0, 1]).
`data.yaml` deliberately has no `path:` key: Ultralytics then resolves the splits relative
to the folder containing data.yaml, so the dataset works wherever it is unzipped.

Usage:
    python scripts/build_dataset.py --labels data/labels_reviewed --out data/dataset
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml

from common import CLASS_NAMES

SPLITS = ("train", "val", "test")
MIN_IMAGES, MAX_IMAGES = 200, 500  # assignment requirement


def parse_label_file(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path.name}:{n}: expected 5 fields, got {len(parts)}")
        cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        if cls not in CLASS_NAMES:
            raise ValueError(f"{path.name}:{n}: unknown class id {cls}")
        if not all(0.0 <= v <= 1.0 for v in (cx, cy)) or not all(0.0 < v <= 1.0 for v in (w, h)):
            raise ValueError(f"{path.name}:{n}: coordinates out of range: {line}")
        boxes.append((cls, cx, cy, w, h))
    return boxes


def copy_split(assignment: pd.DataFrame, frames: Path, labels: Path, out: Path) -> list[dict]:
    stats = []
    for row in assignment.itertuples():
        label_src = labels / f"{Path(row.file).stem}.txt"
        if not label_src.exists():
            raise FileNotFoundError(f"Missing label for {row.file} in {labels}")
        boxes = parse_label_file(label_src)
        shutil.copy2(frames / row.file, out / "images" / row.split / row.file)
        shutil.copy2(label_src, out / "labels" / row.split / label_src.name)
        counts = {CLASS_NAMES[c]: sum(b[0] == c for b in boxes) for c in CLASS_NAMES}
        stats.append({"file": row.file, "video": row.video, "split": row.split, **counts})
    return stats


def write_data_yaml(out: Path) -> None:
    data = {"train": "images/train", "val": "images/val", "test": "images/test",
            "nc": len(CLASS_NAMES), "names": dict(CLASS_NAMES)}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def print_stats(stats: pd.DataFrame) -> None:
    names = list(CLASS_NAMES.values())
    per_split = stats.groupby("split").agg(images=("file", "size"), **{n: (n, "sum") for n in names})
    for n in names:
        per_split[f"img_with_{n}"] = stats.groupby("split")[n].apply(lambda s: int((s > 0).sum()))
    print(per_split.reindex(list(SPLITS)))
    print(f"\nTotal images: {len(stats)}   total boxes: {int(stats[names].sum().sum())}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assignment", type=Path, default=Path("data/split_assignment.csv"))
    p.add_argument("--frames", type=Path, default=Path("data/frames"))
    p.add_argument("--labels", type=Path, default=Path("data/labels_reviewed"))
    p.add_argument("--out", type=Path, default=Path("data/dataset"))
    return p


def main() -> None:
    args = build_parser().parse_args()
    assignment = pd.read_csv(args.assignment)
    assignment = assignment[assignment.split.isin(SPLITS)]
    if not MIN_IMAGES <= len(assignment) <= MAX_IMAGES:
        raise SystemExit(f"Dataset must have {MIN_IMAGES}-{MAX_IMAGES} images, got {len(assignment)}")

    if args.out.exists():
        shutil.rmtree(args.out)
    for kind in ("images", "labels"):
        for split in SPLITS:
            (args.out / kind / split).mkdir(parents=True)

    stats = pd.DataFrame(copy_split(assignment, args.frames, args.labels, args.out))
    write_data_yaml(args.out)
    stats.to_csv(args.out / "dataset_stats.csv", index=False)
    print_stats(stats)
    print(f"\nDataset written to {args.out} (labels from {args.labels})")


if __name__ == "__main__":
    main()
