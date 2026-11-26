"""Model-assisted label QA: find likely label errors in train/val with a trained detector.

The trained model is run on train/val images and compared with the current labels.
Disagreements become "suspects" for a targeted second review:
  missing_label   confident prediction (conf >= --conf-missing) with no matching label
  spurious_label  label (side >= --min-side-px) with no prediction above --conf-low nearby
  class_mismatch  matched box (IoU >= --iou) whose predicted class differs, conf >= --conf-class

The TEST split is deliberately excluded: correcting test labels with the model under
evaluation would leak the model's behaviour into the ground truth.

Outputs (under --out):
  review/<frame>.jpg + <frame>.json   images to review in X-AnyLabeling (copies of current JSON)
  hints/<frame>.jpg                   same image with labels (green) and suspects drawn (red)
  suspects.csv                        one row per suspect, images ranked by number of suspects

Usage:
    python scripts/find_label_suspects.py --weights runs/yolo11n_vehicle/weights/best.pt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from common import CLASS_NAMES
from label_diff import iou_matrix, read_boxes

SPLITS = ("train", "val")
GREEN, RED, ORANGE = (0, 200, 0), (0, 0, 255), (0, 140, 255)


def predict(model: YOLO, image: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Return (N, 6): cls, x1, y1, x2, y2 (normalised), conf."""
    # Class-agnostic NMS: the model sometimes predicts both `bus` and `truk` on one bus; without it
    # the second prediction would be reported as a (false) missing label.
    r = model.predict(image, conf=args.conf_low, imgsz=args.imgsz, device=args.device,
                      agnostic_nms=True, verbose=False)[0]
    if len(r.boxes) == 0:
        return np.zeros((0, 6))
    return np.column_stack([r.boxes.cls.cpu().numpy(), r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy()])


def find_suspects(gt: np.ndarray, pred: np.ndarray, w: int, h: int, args: argparse.Namespace) -> list[dict]:
    suspects, used_gt, used_pred = [], set(), set()
    ious = iou_matrix(gt, pred) if len(gt) and len(pred) else np.zeros((len(gt), len(pred)))
    for flat in np.argsort(-ious, axis=None):
        i, j = np.unravel_index(flat, ious.shape)
        if ious[i, j] < args.iou:
            break
        if i in used_gt or j in used_pred:
            continue
        used_gt.add(i)
        used_pred.add(j)
        if gt[i, 0] != pred[j, 0] and pred[j, 5] >= args.conf_class:
            suspects.append(row("class_mismatch", gt[i], pred[j, 0], pred[j, 5]))
    for j in range(len(pred)):
        side = min((pred[j, 3] - pred[j, 1]) * w, (pred[j, 4] - pred[j, 2]) * h)
        if j not in used_pred and pred[j, 5] >= args.conf_missing and side >= args.min_side_px:
            suspects.append(row("missing_label", pred[j, :5], pred[j, 0], pred[j, 5], gt_cls=None))
    near = iou_matrix(gt, pred).max(axis=1) if len(gt) and len(pred) else np.zeros(len(gt))
    for i in range(len(gt)):
        side = min((gt[i, 3] - gt[i, 1]) * w, (gt[i, 4] - gt[i, 2]) * h)
        if i not in used_gt and near[i] < args.iou_spurious and side >= args.min_side_px:
            suspects.append(row("spurious_label", gt[i], None, None))
    return suspects


def row(kind: str, box: np.ndarray, pred_cls, conf, gt_cls="from_box") -> dict:
    gt_name = CLASS_NAMES[int(box[0])] if gt_cls == "from_box" else ""
    return {"type": kind, "gt_class": gt_name,
            "pred_class": CLASS_NAMES[int(pred_cls)] if pred_cls is not None else "",
            "conf": round(float(conf), 3) if conf is not None else None,
            "x1": box[1], "y1": box[2], "x2": box[3], "y2": box[4]}


def draw_hint(image: np.ndarray, gt: np.ndarray, suspects: list[dict]) -> np.ndarray:
    canvas = image.copy()
    h, w = canvas.shape[:2]
    for b in gt:
        cv2.rectangle(canvas, (int(b[1] * w), int(b[2] * h)), (int(b[3] * w), int(b[4] * h)), GREEN, 1)
    for s in suspects:
        color = ORANGE if s["type"] == "class_mismatch" else RED
        p1, p2 = (int(s["x1"] * w) - 3, int(s["y1"] * h) - 3), (int(s["x2"] * w) + 3, int(s["y2"] * h) + 3)
        cv2.rectangle(canvas, p1, p2, color, 3)
        text = {"missing_label": f"MISSING? {s['pred_class']} {s['conf']}",
                "spurious_label": f"WRONG BOX? {s['gt_class']}",
                "class_mismatch": f"CLASS? {s['gt_class']}->{s['pred_class']} {s['conf']}"}[s["type"]]
        cv2.putText(canvas, text, (p1[0], max(14, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(canvas, text, (p1[0], max(14, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    return canvas


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--assignment", type=Path, default=Path("data/split_assignment.csv"))
    p.add_argument("--frames", type=Path, default=Path("data/frames"))
    p.add_argument("--labels", type=Path, default=Path("data/labels_reviewed"))
    p.add_argument("--out", type=Path, default=Path("data/review_round2"))
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--conf-low", type=float, default=0.1, help="predictions below are ignored entirely")
    p.add_argument("--conf-missing", type=float, default=0.5)
    p.add_argument("--conf-class", type=float, default=0.4)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--iou-spurious", type=float, default=0.3)
    p.add_argument("--min-side-px", type=float, default=15.0, help="ignore tiny boxes (too ambiguous)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    files = pd.read_csv(args.assignment).query("split in @SPLITS")[["file", "split"]]
    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "review").mkdir(parents=True)
    (args.out / "hints").mkdir()
    model = YOLO(str(args.weights))

    rows = []
    for f in files.itertuples():
        image = cv2.imread(str(args.frames / f.file))
        gt = read_boxes(args.labels / f"{Path(f.file).stem}.txt")
        suspects = find_suspects(gt, predict(model, image, args), image.shape[1], image.shape[0], args)
        if not suspects:
            continue
        stem = Path(f.file).stem
        shutil.copy2(args.frames / f.file, args.out / "review" / f.file)
        shutil.copy2(args.frames / f"{stem}.json", args.out / "review" / f"{stem}.json")
        cv2.imwrite(str(args.out / "hints" / f.file), draw_hint(image, gt, suspects))
        rows += [{"file": f.file, "split": f.split, **s} for s in suspects]

    df = pd.DataFrame(rows)
    df["n_suspects"] = df.groupby("file").file.transform("size")
    df.sort_values(["n_suspects", "file"], ascending=[False, True]).to_csv(args.out / "suspects.csv", index=False)
    print(f"Images checked: {len(files)}   images with suspects: {df.file.nunique()}   suspects: {len(df)}")
    print(pd.crosstab(df.type, df.split, margins=True))
    print(f"\nReview folder: {args.out / 'review'}   hints: {args.out / 'hints'}")


if __name__ == "__main__":
    main()
