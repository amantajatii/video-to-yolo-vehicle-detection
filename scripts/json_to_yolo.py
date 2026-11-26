"""Convert reviewed X-AnyLabeling (LabelMe-style) JSON annotations to YOLO txt labels.

Every image gets a label file (empty if it has no boxes). Boxes are clipped to the image,
and the script fails loudly on unknown labels or non-rectangle shapes so that review
mistakes cannot silently enter the dataset.

Usage:
    python scripts/json_to_yolo.py --json-dir data/frames --out data/labels_reviewed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import CLASS_NAMES

NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}
MIN_SIDE_PX = 1.0  # degenerate boxes (accidental clicks) are dropped


def shape_to_yolo(shape: dict, width: int, height: int, source: Path) -> str | None:
    if shape.get("shape_type") != "rectangle":
        raise ValueError(f"{source.name}: unsupported shape type {shape.get('shape_type')!r}")
    label = shape["label"].strip()
    if label not in NAME_TO_ID:
        raise ValueError(f"{source.name}: unknown label {label!r} (expected {list(NAME_TO_ID)})")
    xs = [min(max(p[0], 0.0), width) for p in shape["points"]]
    ys = [min(max(p[1], 0.0), height) for p in shape["points"]]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    if x2 - x1 < MIN_SIDE_PX or y2 - y1 < MIN_SIDE_PX:
        return None
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    return f"{NAME_TO_ID[label]} {cx:.6f} {cy:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"


def convert(json_path: Path) -> list[str]:
    data = json.loads(json_path.read_text())
    w, h = data["imageWidth"], data["imageHeight"]
    lines = [shape_to_yolo(s, w, h, json_path) for s in data.get("shapes", [])]
    return [line for line in lines if line is not None]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json-dir", type=Path, default=Path("data/frames"))
    p.add_argument("--images-dir", type=Path, default=Path("data/frames"))
    p.add_argument("--out", type=Path, default=Path("data/labels_reviewed"))
    args = p.parse_args()

    images = sorted(args.images_dir.glob("*.jpg"))
    missing = [img.name for img in images if not (args.json_dir / f"{img.stem}.json").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images have no JSON annotation, e.g. {missing[:5]}")

    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for img in images:
        lines = convert(args.json_dir / f"{img.stem}.json")
        (args.out / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        total += len(lines)
    print(f"Converted {len(images)} images, {total} boxes -> {args.out}")


if __name__ == "__main__":
    main()
