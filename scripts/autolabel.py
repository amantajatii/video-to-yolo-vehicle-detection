"""Pre-label extracted frames with a pretrained COCO teacher model (YOLO format).

Two inference passes are merged per image:
  1. full image at --imgsz (catches large / near vehicles)
  2. tiled (SAHI-style) inference: overlapping --tile crops, each predicted at its own
     resolution, so small distant vehicles become relatively larger. Boxes that touch an
     internal tile border are discarded (they are usually partial objects that the
     neighbouring overlapping tile or the full-image pass sees whole).
Merging uses class-agnostic greedy suppression by IoU or IoS (intersection over the
smaller box), keeping the most confident box.

COCO classes are mapped to the project classes (car->mobil, truck->truk, bus->bus).
These pseudo-labels are a STARTING POINT and must be reviewed: COCO's class
boundaries differ from this project's guideline (e.g. pickups and large vans
are `mobil` here). See scripts/reclassify_clip.py for the follow-up correction step.

Outputs:
  <out>/<frame>.txt             YOLO labels: "class cx cy w h" (normalised), empty file if no box
  <predictions-csv>             one row per box incl. confidence and source (full/tile)

Usage:
    python scripts/autolabel.py --frames data/frames --out data/labels_auto --model yolo11x.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from common import CLASS_NAMES, COCO_TO_PROJECT

EDGE_MARGIN_PX = 3  # a tile box within this distance of an internal tile edge is treated as cut


def predict_boxes(model: YOLO, image: np.ndarray, imgsz: int, args: argparse.Namespace) -> np.ndarray:
    """Return an (N, 6) array: x1, y1, x2, y2, conf, coco_class."""
    result = model.predict(image, conf=args.conf, iou=args.iou, imgsz=imgsz, agnostic_nms=True,
                           classes=list(COCO_TO_PROJECT), verbose=False, device=args.device)[0]
    boxes = result.boxes
    if len(boxes) == 0:
        return np.zeros((0, 6))
    return np.column_stack([boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()])


def tile_origins(length: int, tile: int, overlap: float) -> list[int]:
    if length <= tile:
        return [0]
    stride = int(tile * (1 - overlap))
    origins = list(range(0, length - tile, stride))
    return origins + [length - tile]


def predict_tiled(model: YOLO, image: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    h, w = image.shape[:2]
    results = []
    for y0 in tile_origins(h, args.tile, args.tile_overlap):
        for x0 in tile_origins(w, args.tile, args.tile_overlap):
            crop = image[y0:y0 + args.tile, x0:x0 + args.tile]
            boxes = predict_boxes(model, crop, args.tile_imgsz, args)
            if len(boxes) == 0:
                continue
            keep = ~touches_internal_edge(boxes, x0, y0, crop.shape[1], crop.shape[0], w, h)
            shifted = boxes[keep].copy()
            shifted[:, [0, 2]] += x0
            shifted[:, [1, 3]] += y0
            results.append(shifted)
    return np.vstack(results) if results else np.zeros((0, 6))


def touches_internal_edge(boxes: np.ndarray, x0: int, y0: int, tw: int, th: int, w: int, h: int) -> np.ndarray:
    m = EDGE_MARGIN_PX
    left = (boxes[:, 0] <= m) & (x0 > 0)
    top = (boxes[:, 1] <= m) & (y0 > 0)
    right = (boxes[:, 2] >= tw - m) & (x0 + tw < w)
    bottom = (boxes[:, 3] >= th - m) & (y0 + th < h)
    return left | top | right | bottom


def merge_boxes(boxes: np.ndarray, iou_thr: float, ios_thr: float) -> np.ndarray:
    """Greedy class-agnostic suppression by IoU or intersection-over-smaller-box."""
    order = np.argsort(-boxes[:, 4])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    for i in order:
        if not keep:
            keep.append(i)
            continue
        k = np.array(keep)
        ix = np.clip(np.minimum(boxes[k, 2], boxes[i, 2]) - np.maximum(boxes[k, 0], boxes[i, 0]), 0, None)
        iy = np.clip(np.minimum(boxes[k, 3], boxes[i, 3]) - np.maximum(boxes[k, 1], boxes[i, 1]), 0, None)
        inter = ix * iy
        iou = inter / (areas[k] + areas[i] - inter)
        ios = inter / np.minimum(areas[k], areas[i])
        if not ((iou > iou_thr) | (ios > ios_thr)).any():
            keep.append(i)
    return boxes[keep]


def label_image(model: YOLO, path: Path, args: argparse.Namespace) -> list[dict]:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    h, w = image.shape[:2]
    full = predict_boxes(model, image, args.imgsz, args)
    tiled = predict_tiled(model, image, args) if args.tile > 0 else np.zeros((0, 6))
    source = np.array(["full"] * len(full) + ["tile"] * len(tiled))
    stacked = np.vstack([full, tiled])
    if len(stacked) == 0:
        return []
    stacked = np.column_stack([stacked, np.arange(len(stacked))])
    merged = merge_boxes(stacked, args.merge_iou, args.merge_ios)

    rows = []
    for x1, y1, x2, y2, conf, coco_id, src_idx in merged:
        if min(x2 - x1, y2 - y1) < args.min_box_px:
            continue
        cls = COCO_TO_PROJECT[int(coco_id)]
        rows.append({
            "file": path.name, "cls": cls, "class_name": CLASS_NAMES[cls], "conf": round(float(conf), 4),
            "source": source[int(src_idx)],
            "x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1),
            "cx": (x1 + x2) / 2 / w, "cy": (y1 + y2) / 2 / h, "w": (x2 - x1) / w, "h": (y2 - y1) / h,
        })
    return rows


def write_yolo_label(rows: list[dict], label_path: Path) -> None:
    lines = [f"{r['cls']} {r['cx']:.6f} {r['cy']:.6f} {r['w']:.6f} {r['h']:.6f}" for r in rows]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=Path, default=Path("data/frames"))
    p.add_argument("--out", type=Path, default=Path("data/labels_auto"))
    p.add_argument("--predictions-csv", type=Path, default=Path("data/autolabel_predictions.csv"))
    p.add_argument("--model", type=str, default="yolo11x.pt")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--iou", type=float, default=0.6, help="NMS IoU inside each pass (class-agnostic)")
    p.add_argument("--imgsz", type=int, default=1280, help="full-image pass size")
    p.add_argument("--tile", type=int, default=640, help="tile size in px; 0 disables tiling")
    p.add_argument("--tile-overlap", type=float, default=0.25)
    p.add_argument("--tile-imgsz", type=int, default=1280, help="inference size per tile (>tile = upscaling)")
    p.add_argument("--merge-iou", type=float, default=0.5)
    p.add_argument("--merge-ios", type=float, default=0.8)
    p.add_argument("--min-box-px", type=float, default=8.0, help="drop boxes with a side below this")
    p.add_argument("--device", type=str, default="mps")
    return p


def main() -> None:
    args = build_parser().parse_args()
    frames = sorted(args.frames.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"No frames found in {args.frames}")
    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)

    all_rows: list[dict] = []
    for i, path in enumerate(frames, 1):
        rows = label_image(model, path, args)
        write_yolo_label(rows, args.out / f"{path.stem}.txt")
        all_rows += rows
        if i % 50 == 0 or i == len(frames):
            print(f"  labelled {i}/{len(frames)}", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.predictions_csv, index=False)
    print(f"\nImages: {len(frames)}  boxes: {len(df)}  -> {args.out}")
    print(f"Images without boxes: {len(frames) - df.file.nunique()}")
    print("Boxes by source:", df.source.value_counts().to_dict())
    summary = df.groupby("class_name").agg(boxes=("cls", "size"), images=("file", "nunique"),
                                           mean_conf=("conf", "mean"), low_conf_lt_0_5=("conf", lambda c: (c < 0.5).sum()))
    print(summary.round(3))


if __name__ == "__main__":
    main()
