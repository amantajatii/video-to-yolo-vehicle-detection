"""Quantify manual review: compare pre-labels with reviewed labels (both YOLO txt).

Boxes are matched per image with greedy class-agnostic IoU matching, then classified as
  unchanged      matched, same class, IoU >= --adjust-iou
  adjusted       matched, same class, box moved/resized (IoU < --adjust-iou)
  class_changed  matched, different class
  deleted        pre-label box without a reviewed match   (teacher false positive)
  added          reviewed box without a pre-label match    (teacher false negative)

Usage:
    python scripts/label_diff.py --before data/labels_prelabel --after data/labels_reviewed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import CLASS_NAMES


def read_boxes(path: Path) -> np.ndarray:
    """Return (N, 5): cls, x1, y1, x2, y2 in normalised coordinates."""
    rows = [list(map(float, line.split())) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return np.zeros((0, 5))
    a = np.array(rows)
    return np.column_stack([a[:, 0], a[:, 1] - a[:, 3] / 2, a[:, 2] - a[:, 4] / 2,
                            a[:, 1] + a[:, 3] / 2, a[:, 2] + a[:, 4] / 2])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 4], b[None, :, 4]) - np.maximum(a[:, None, 2], b[None, :, 2]), 0, None)
    inter = ix * iy
    area = lambda x: (x[:, 3] - x[:, 1]) * (x[:, 4] - x[:, 2])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-12)


def diff_image(before: np.ndarray, after: np.ndarray, match_iou: float, adjust_iou: float) -> list[dict]:
    events: list[dict] = []
    used_b, used_a = set(), set()
    if len(before) and len(after):
        ious = iou_matrix(before, after)
        for flat in np.argsort(-ious, axis=None):
            i, j = np.unravel_index(flat, ious.shape)
            if ious[i, j] < match_iou:
                break
            if i in used_b or j in used_a:
                continue
            used_b.add(i)
            used_a.add(j)
            cb, ca = int(before[i, 0]), int(after[j, 0])
            kind = "class_changed" if cb != ca else ("unchanged" if ious[i, j] >= adjust_iou else "adjusted")
            events.append({"event": kind, "before": CLASS_NAMES[cb], "after": CLASS_NAMES[ca]})
    events += [{"event": "deleted", "before": CLASS_NAMES[int(before[i, 0])], "after": ""}
               for i in range(len(before)) if i not in used_b]
    events += [{"event": "added", "before": "", "after": CLASS_NAMES[int(after[j, 0])]}
               for j in range(len(after)) if j not in used_a]
    return events


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--before", type=Path, default=Path("data/labels_prelabel"))
    p.add_argument("--after", type=Path, default=Path("data/labels_reviewed"))
    p.add_argument("--assignment", type=Path, default=Path("data/split_assignment.csv"))
    p.add_argument("--match-iou", type=float, default=0.3)
    p.add_argument("--adjust-iou", type=float, default=0.9)
    p.add_argument("--out", type=Path, default=Path("reports/label_review_diff.csv"))
    args = p.parse_args()

    split_of = pd.read_csv(args.assignment).set_index("file").split.to_dict()
    events = []
    for after_path in sorted(args.after.glob("*.txt")):
        before_path = args.before / after_path.name
        before = read_boxes(before_path) if before_path.exists() else np.zeros((0, 5))
        for e in diff_image(before, read_boxes(after_path), args.match_iou, args.adjust_iou):
            events.append({"file": after_path.name, "split": split_of.get(after_path.stem + ".jpg", "?"), **e})
    df = pd.DataFrame(events)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    n_before = int((df.before != "").sum())
    print(f"Pre-label boxes: {n_before}   reviewed boxes: {int((df.after != '').sum())}\n")
    summary = df.event.value_counts()
    print(pd.DataFrame({"boxes": summary, "pct_of_prelabel": (summary / n_before * 100).round(1)}))
    print("\nPer split:")
    print(pd.crosstab(df.split, df.event))
    print("\nClass changes (before -> after):")
    print(pd.crosstab(df[df.event == "class_changed"].before, df[df.event == "class_changed"].after))
    print("\nDeleted by class:", df[df.event == "deleted"].before.value_counts().to_dict())
    print("Added by class:  ", df[df.event == "added"].after.value_counts().to_dict())
    print(f"\nDetails -> {args.out}")


if __name__ == "__main__":
    main()
