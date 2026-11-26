"""Dataset statistics for the report: counts per split, video and class, plus box-size distribution.

Reads a built YOLO dataset (output of build_dataset.py) and prints:
  - images per split and per source video (file names start with v1_/v2_/v3_)
  - boxes per split and class, boxes per video and class
  - COCO size buckets per split (small < 32^2, medium < 96^2, large >= 96^2, in original pixels)
  - median box width x height per class, boxes per image, median shortest box side

Usage:
    python scripts/dataset_stats.py --dataset data/dataset --out reports/dataset_stats.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from common import CLASS_NAMES

SPLITS = ("train", "val", "test")
AREA_LIMITS = (32**2, 96**2)


def read_boxes(dataset: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (images, boxes) tables with sizes in original pixels."""
    images, boxes = [], []
    for split in SPLITS:
        for img_path in sorted((dataset / "images" / split).glob("*.jpg")):
            image = cv2.imread(str(img_path))
            if image is None:
                raise RuntimeError(f"Cannot read image: {img_path}")
            h, w = image.shape[:2]
            video = img_path.stem.split("_")[0]
            images.append({"file": img_path.name, "split": split, "video": video})
            label = dataset / "labels" / split / f"{img_path.stem}.txt"
            for line in label.read_text().splitlines():
                if not line.strip():
                    continue
                cls, _, _, bw, bh = map(float, line.split())
                boxes.append({"split": split, "video": video, "class": CLASS_NAMES[int(cls)],
                              "w_px": bw * w, "h_px": bh * h})
    return pd.DataFrame(images), pd.DataFrame(boxes)


def size_bucket(area: float) -> str:
    return "small" if area < AREA_LIMITS[0] else "medium" if area < AREA_LIMITS[1] else "large"


def summarise(images: pd.DataFrame, boxes: pd.DataFrame) -> dict[str, pd.DataFrame | str]:
    boxes = boxes.assign(size=[size_bucket(a) for a in boxes.w_px * boxes.h_px])
    classes = list(CLASS_NAMES.values())
    per_split = pd.crosstab(boxes.split, boxes["class"]).reindex(index=list(SPLITS), columns=classes)
    per_split.insert(0, "images", images.groupby("split").size().reindex(list(SPLITS)))
    sizes = pd.crosstab(boxes.split, boxes["size"]).reindex(index=list(SPLITS), columns=["small", "medium", "large"])
    shortest = boxes[["w_px", "h_px"]].min(axis=1)
    medians = boxes.groupby("class")[["w_px", "h_px"]].median().round(0).astype(int).reindex(classes)
    counts = boxes["class"].value_counts().reindex(classes)
    notes = (f"boxes: {len(boxes)} | boxes per image: {len(boxes) / len(images):.1f} | "
             f"median shortest side: {shortest.median():.0f} px | small boxes: {(boxes['size'] == 'small').mean():.1%} | "
             f"class ratio mobil:truk:bus = {counts['mobil'] / counts['bus']:.0f} : "
             f"{counts['truk'] / counts['bus']:.1f} : 1")
    return {
        "Images and boxes per split": per_split,
        "Images per video and split": pd.crosstab(images.video, images.split).reindex(columns=list(SPLITS)),
        "Boxes per video and class": pd.crosstab(boxes.video, boxes["class"]).reindex(columns=classes),
        "Boxes per size bucket (COCO areas, original pixels)": sizes,
        "Median box size per class (px)": medians,
        "Overall": notes,
    }


def table_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown table (avoids the optional `tabulate` dependency of DataFrame.to_markdown)."""
    header = [df.index.name or ""] + [str(c) for c in df.columns]
    rows = [[str(i)] + [str(v) for v in row] for i, row in zip(df.index, df.itertuples(index=False))]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def to_markdown(sections: dict) -> str:
    parts = ["# Dataset statistics", ""]
    for title, value in sections.items():
        parts.append(f"## {title}")
        parts.append(value if isinstance(value, str) else table_markdown(value))
        parts.append("")
    return "\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=Path("data/dataset"))
    p.add_argument("--out", type=Path, default=None, help="optional markdown output file")
    args = p.parse_args()
    if not (args.dataset / "data.yaml").exists():
        raise SystemExit(f"Not a built dataset: {args.dataset} (run scripts/build_dataset.py first)")

    images, boxes = read_boxes(args.dataset)
    report = to_markdown(summarise(images, boxes))
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n")
        print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
