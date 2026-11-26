"""Compare candidate teacher models (e.g. YOLO11m vs YOLO11x) for auto-labelling.

There is no ground truth yet, so the comparison measures agreement between the two
models on a stratified sample of extracted frames:
  - box counts per project class
  - boxes matched by both (IoU >= --iou, same class), class disagreements,
    and boxes found by only one model
  - mean inference time per image
It also saves side-by-side visualisations of the frames where the models disagree most.

Usage:
    python scripts/compare_teachers.py --frames data/frames --models yolo11m.pt yolo11x.pt
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from common import CLASS_NAMES, COCO_TO_PROJECT

COLORS = {0: (60, 200, 60), 1: (0, 140, 255), 2: (230, 60, 60)}


@dataclass(frozen=True)
class Box:
    cls: int
    xyxy: tuple[float, float, float, float]
    conf: float


def predict(model: YOLO, image: np.ndarray, conf: float, device: str) -> tuple[list[Box], float]:
    start = time.perf_counter()
    result = model.predict(image, conf=conf, classes=list(COCO_TO_PROJECT), verbose=False, device=device)[0]
    elapsed = time.perf_counter() - start
    boxes = [Box(COCO_TO_PROJECT[int(c)], tuple(xy), float(p))
             for c, xy, p in zip(result.boxes.cls.tolist(), result.boxes.xyxy.tolist(), result.boxes.conf.tolist())]
    return boxes, elapsed


def iou(a: tuple, b: tuple) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(a: list[Box], b: list[Box], thr: float) -> dict[str, int]:
    """Greedy IoU matching (class-agnostic), then split into same-class / class-mismatch."""
    pairs = sorted(((iou(x.xyxy, y.xyxy), i, j) for i, x in enumerate(a) for j, y in enumerate(b)), reverse=True)
    used_a, used_b, same, diff = set(), set(), 0, 0
    for score, i, j in pairs:
        if score < thr or i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        same, diff = (same + 1, diff) if a[i].cls == b[j].cls else (same, diff + 1)
    return {"matched_same_class": same, "matched_class_diff": diff,
            "only_a": len(a) - len(used_a), "only_b": len(b) - len(used_b)}


def draw(image: np.ndarray, boxes: list[Box], title: str) -> np.ndarray:
    canvas = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLORS[box.cls], 2)
        cv2.putText(canvas, f"{CLASS_NAMES[box.cls]} {box.conf:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[box.cls], 1)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(canvas, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return canvas


def sample_frames(frames_dir: Path, per_video: int, seed: int) -> list[Path]:
    files = sorted(frames_dir.glob("*.jpg"))
    by_video = pd.Series(files).groupby([f.name.split("_")[0] for f in files])
    return sorted(p for _, g in by_video for p in g.sample(min(per_video, len(g)), random_state=seed))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=Path, default=Path("data/frames"))
    p.add_argument("--models", nargs=2, default=["yolo11m.pt", "yolo11x.pt"])
    p.add_argument("--per-video", type=int, default=10)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("reports/teacher_comparison"))
    p.add_argument("--n-visual", type=int, default=6)
    return p


def main() -> None:
    args = build_parser().parse_args()
    files = sample_frames(args.frames, args.per_video, args.seed)
    if not files:
        raise SystemExit(f"No frames in {args.frames}")
    name_a, name_b = (Path(m).stem for m in args.models)
    model_a, model_b = (YOLO(m) for m in args.models)
    for model in (model_a, model_b):  # warm-up so timing excludes first-call overhead
        predict(model, cv2.imread(str(files[0])), args.conf, args.device)

    rows, cache = [], {}
    for path in files:
        image = cv2.imread(str(path))
        boxes_a, t_a = predict(model_a, image, args.conf, args.device)
        boxes_b, t_b = predict(model_b, image, args.conf, args.device)
        cache[path.name] = (image, boxes_a, boxes_b)
        row = {"file": path.name, f"t_{name_a}_ms": t_a * 1000, f"t_{name_b}_ms": t_b * 1000}
        for cid, cname in CLASS_NAMES.items():
            row[f"{cname}_{name_a}"] = sum(b.cls == cid for b in boxes_a)
            row[f"{cname}_{name_b}"] = sum(b.cls == cid for b in boxes_b)
        rows.append({**row, **match(boxes_a, boxes_b, args.iou)})

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "comparison.csv", index=False)
    print_summary(df, name_a, name_b, len(files))
    save_visuals(df, cache, name_a, name_b, args.out, args.n_visual)


def print_summary(df: pd.DataFrame, name_a: str, name_b: str, n: int) -> None:
    print(f"Frames compared: {n}\n")
    print("Box counts per class:")
    for cname in CLASS_NAMES.values():
        print(f"  {cname:6s} {name_a}={df[f'{cname}_{name_a}'].sum():5d}   {name_b}={df[f'{cname}_{name_b}'].sum():5d}")
    total = df[["matched_same_class", "matched_class_diff", "only_a", "only_b"]].sum()
    print(f"\nAgreement: same-class match={total.matched_same_class}  class-diff={total.matched_class_diff}  "
          f"only {name_a}={total.only_a}  only {name_b}={total.only_b}")
    print(f"Mean inference: {name_a}={df[f't_{name_a}_ms'].mean():.1f} ms  "
          f"{name_b}={df[f't_{name_b}_ms'].mean():.1f} ms")


def save_visuals(df: pd.DataFrame, cache: dict, name_a: str, name_b: str, out: Path, n: int) -> None:
    disagreement = df.matched_class_diff + df.only_a + df.only_b
    for fname in df.assign(d=disagreement).nlargest(n, "d").file:
        image, boxes_a, boxes_b = cache[fname]
        pair = np.hstack([draw(image, boxes_a, f"{name_a} | {fname}"), draw(image, boxes_b, name_b)])
        scale = 1600 / pair.shape[1]
        cv2.imwrite(str(out / f"diff_{fname}"), cv2.resize(pair, None, fx=scale, fy=scale))
    print(f"Saved {n} disagreement visualisations to {out}")


if __name__ == "__main__":
    main()
