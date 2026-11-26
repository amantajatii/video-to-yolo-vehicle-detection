"""Evaluate a fine-tuned YOLO vehicle detector and COCO-pretrained baselines.

Requires pycocotools for the COCO comparison. Use --baseline with no values
to skip the baselines. The dataset is read-only.

Usage:
    python scripts/evaluate.py --weights runs/train/weights/best.pt --device cpu
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, img2label_paths

from common import CLASS_NAMES, COCO_TO_PROJECT
from label_diff import iou_matrix

SIZES = ("small", "medium", "large")
AREA_LIMITS = (32**2, 96**2)
PREDICT_CONF = 0.001
MAX_DET = 300
BACKGROUND = "background"
BOX_COLUMNS = ("x1_px", "y1_px", "x2_px", "y2_px")
ERROR_COLUMNS = ("file", "type", "gt_class", "pred_class", "conf", *BOX_COLUMNS, "size_bucket")
COLORS = {"GT": (0, 200, 0), "TP": (255, 70, 0), "FP": (0, 0, 255), "class_confusion": (0, 140, 255)}


@dataclass(frozen=True)
class Frame:
    path: Path
    width: int
    height: int
    gt: np.ndarray  # class, x1, y1, x2, y2 in original image pixels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data/dataset/data.yaml"))
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--baseline", nargs="*", default=["yolo11n.pt"])
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", default="mps")
    p.add_argument("--conf-analysis", type=float, default=0.25)
    p.add_argument("--iou-match", type=float, default=0.5)
    p.add_argument("--n-examples", type=int, default=12)
    p.add_argument("--out", type=Path, default=Path("reports/evaluation"))
    args = p.parse_args()
    if not 0 <= args.conf_analysis <= 1 or not 0 < args.iou_match <= 1:
        p.error("--conf-analysis must be in [0, 1], and --iou-match in (0, 1]")
    if args.imgsz <= 0 or args.n_examples < 0:
        p.error("--imgsz must be positive and --n-examples nonnegative")
    for path in (args.weights, args.data):
        if not path.is_file():
            p.error(f"File not found: {path}")
    return args


def split_paths(data_path: Path, split: str) -> list[Path]:
    data = check_det_dataset(str(data_path.resolve()), autodownload=False, split=split)
    if data["names"] != CLASS_NAMES:
        raise ValueError(f"Expected class mapping {CLASS_NAMES} in {data_path}")
    sources = data[split] if isinstance(data[split], list) else [data[split]]
    paths = []
    for source in map(Path, sources):
        if source.is_dir():
            paths.extend(p for p in source.rglob("*.*") if p.is_file())
        else:
            paths.extend(source.parent / line[2:] if line.startswith("./") else Path(line)
                         for line in source.read_text().strip().splitlines() if line)
    images = sorted(p.resolve() for p in paths if p.suffix[1:].lower() in IMG_FORMATS)
    if not images or len(images) != len(set(images)):
        raise ValueError(f"Split {split!r} must contain images with unique paths: {data_path}")
    return images


def read_frame(path: Path, label: Path) -> Frame:
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        with path.open("rb") as image_file:
            image_file.seek(-2, 2)
            if image_file.read() != b"\xff\xd9":
                raise ValueError(f"Missing JPEG end marker in {path}; Ultralytics would rewrite this image")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    height, width = image.shape[:2]
    boxes = []
    for line_no, line in enumerate(label.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            cls, cx, cy, w, h = map(float, line.split())
            if cls not in CLASS_NAMES or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError
        except ValueError as exc:
            raise ValueError(f"Invalid YOLO box at {label}:{line_no}: {line!r}") from exc
        boxes.append((cls, (cx - w / 2) * width, (cy - h / 2) * height,
                      (cx + w / 2) * width, (cy + h / 2) * height))
    return Frame(path, width, height, np.asarray(boxes, dtype=float).reshape(-1, 5))


def official_metrics(model: YOLO, args: argparse.Namespace, out: Path) -> dict:
    def memory_only_cache(_prefix: str, _path: Path, cache: dict, version: str) -> None:
        cache["version"] = version

    # Preserve Ultralytics' in-memory cache metadata while keeping labels read-only.
    with patch("ultralytics.data.dataset.save_dataset_cache_file", memory_only_cache):
        result = model.val(data=str(args.data.resolve()), split=args.split, imgsz=args.imgsz,
                           device=args.device, plots=True, project=str(out), name="ultralytics_val",
                           exist_ok=True, cache=False)
    keys = ("precision", "recall", "mAP50", "mAP50-95")
    by_id = {int(cls): dict(zip(keys, map(float, result.class_result(i))))
             for i, cls in enumerate(result.ap_class_index)}
    return {"overall": dict(zip(keys, map(float, result.mean_results()))),
            "per_class": {name: by_id.get(cls, dict.fromkeys(keys)) for cls, name in CLASS_NAMES.items()}}


def predict(model: YOLO, frames: list[Frame], args: argparse.Namespace,
            class_map: dict[int, int], conf: float = PREDICT_CONF) -> tuple[list[np.ndarray], float]:
    paths = [str(frame.path) for frame in frames]
    options = dict(conf=conf, max_det=MAX_DET, imgsz=args.imgsz, device=args.device,
                   classes=list(class_map), verbose=False, save=False)
    model.predict(paths[0], **options)
    predictions, timings = [], []
    # A list of paths is loaded into one batch by Ultralytics. Use one image per call for timing.
    for path in paths:
        result = model.predict(path, **options)[0]
        boxes = result.boxes.cpu().numpy()
        classes = np.array([class_map[int(cls)] for cls in boxes.cls])
        predictions.append(np.column_stack((classes, boxes.xyxy, boxes.conf)))
        timings.append(result.speed["inference"])
    return predictions, float(np.mean(timings))


def box_area(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    return float((x2 - x1) * (y2 - y1))


def size_bucket(box: np.ndarray) -> str:
    return SIZES[int(np.searchsorted(AREA_LIMITS, box_area(box), side="right"))]


def coco_boxes(images: list[np.ndarray]) -> list[dict]:
    rows = []
    for image_id, boxes in enumerate(images, 1):
        for cls, x1, y1, x2, y2, *score in boxes.tolist():
            bbox = [x1, y1, x2 - x1, y2 - y1]
            row = {"image_id": image_id, "category_id": int(cls), "bbox": bbox}
            if score:
                row["score"] = score[0]
            else:
                row.update(id=len(rows) + 1, area=bbox[2] * bbox[3], iscrowd=0)
            rows.append(row)
    return rows


def coco_ap(frames: list[Frame], predictions: list[np.ndarray]) -> dict[str, float | None]:
    gt = COCO()
    gt.dataset = {"images": [{"id": i, "width": f.width, "height": f.height}
                             for i, f in enumerate(frames, 1)],
                  "categories": [{"id": cls, "name": name} for cls, name in CLASS_NAMES.items()],
                  "annotations": coco_boxes([f.gt for f in frames])}
    detections = coco_boxes(predictions)
    with redirect_stdout(StringIO()):
        gt.createIndex()
        if detections:
            dt = gt.loadRes(detections)
        else:
            dt = COCO()
            dt.dataset = {**gt.dataset, "annotations": []}
            dt.createIndex()
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.maxDets = [1, 10, MAX_DET]
        # COCOeval includes both endpoints. Make adjacent buckets disjoint for AP and counts.
        evaluator.params.areaRng[1][1] = float(np.nextafter(float(AREA_LIMITS[0]), -np.inf))
        evaluator.params.areaRng[2][1] = float(np.nextafter(float(AREA_LIMITS[1]), -np.inf))
        evaluator.evaluate()
        evaluator.accumulate()

    def mean_ap(values: np.ndarray) -> float | None:
        valid = values[values >= 0]
        return float(valid.mean()) if valid.size else None

    # Precision axes are IoU, recall, category, area range and max detections.
    precision = evaluator.eval["precision"][..., -1]
    at_50 = np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.50))[0]
    at_75 = np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.75))[0]
    row = {"AP": mean_ap(precision[..., 0]), "AP50": mean_ap(precision[at_50, ..., 0]),
           "AP75": mean_ap(precision[at_75, ..., 0])}
    row.update({f"AP_{size}": mean_ap(precision[..., i]) for i, size in enumerate(SIZES, 1)})
    for i, cls in enumerate(evaluator.params.catIds):
        name = CLASS_NAMES[cls]
        row[f"AP_{name}"] = mean_ap(precision[:, :, i, 0])
        row[f"AP50_{name}"] = mean_ap(precision[at_50, :, i, 0])
    return row


def match(gt: np.ndarray, pred: np.ndarray, threshold: float) -> list[tuple[str, int | None, int | None]]:
    """Match one-to-one by descending IoU regardless of class, then classify each event."""
    events = []
    used_gt, used_pred = set(), set()
    ious = iou_matrix(gt, pred)
    for flat in np.argsort(-ious, axis=None, kind="stable"):
        i, j = np.unravel_index(flat, ious.shape)
        if ious[i, j] + 1e-12 < threshold:
            break
        if i in used_gt or j in used_pred:
            continue
        used_gt.add(i)
        used_pred.add(j)
        events.append(("TP" if gt[i, 0] == pred[j, 0] else "class_confusion", i, j))
    events.extend(("FN", i, None) for i in range(len(gt)) if i not in used_gt)
    events.extend(("FP", None, j) for j in range(len(pred)) if j not in used_pred)
    return events


def render_example(frame: Frame, overlays: list[tuple[str, str, np.ndarray]], path: Path) -> None:
    image = cv2.imread(str(frame.path))
    if image is None:
        raise ValueError(f"Cannot read image: {frame.path}")
    for kind, label, box in overlays:
        color = COLORS[kind]
        x1, y1, x2, y2 = np.rint(box).astype(int)
        inset = int(kind == "GT") * 2
        cv2.rectangle(image, (x1 - inset, y1 - inset), (x2 + inset, y2 + inset), color, 2)
        y = min(frame.height - 2, y2 + 15) if kind == "GT" else max(13, y1 - 4)
        origin = (max(0, x1), y)
        cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write error example: {path}")


def frame_errors(frame: Frame, pred: np.ndarray, threshold: float, matrix: pd.DataFrame,
                 fn_sizes: Counter) -> tuple[list[dict], list[tuple[str, str, np.ndarray]]]:
    rows = []
    overlays = [("GT", f"GT {CLASS_NAMES[int(box[0])]}", box[1:5]) for box in frame.gt]
    for kind, i, j in match(frame.gt, pred, threshold):
        gt_name = CLASS_NAMES[int(frame.gt[i, 0])] if i is not None else BACKGROUND
        pred_name = CLASS_NAMES[int(pred[j, 0])] if j is not None else BACKGROUND
        matrix.loc[gt_name, pred_name] += 1
        conf = float(pred[j, 5]) if j is not None else None
        if j is not None:
            overlays.append((kind, f"{kind} {pred_name} {conf:.2f}", pred[j, 1:5]))
        if kind != "TP":
            box = frame.gt[i, 1:5] if i is not None else pred[j, 1:5]
            size = size_bucket(box)
            rows.append({"file": frame.path.name, "type": kind, "gt_class": gt_name,
                         "pred_class": pred_name, "conf": conf, **dict(zip(BOX_COLUMNS, box)),
                         "size_bucket": size})
            if kind == "FN":
                fn_sizes[size] += 1
    return rows, overlays


def error_analysis(frames: list[Frame], predictions: list[np.ndarray],
                   args: argparse.Namespace, out: Path) -> dict:
    labels = [*CLASS_NAMES.values(), BACKGROUND]
    matrix = pd.DataFrame(0, index=labels, columns=labels)
    errors, examples = [], []
    fn_sizes: Counter = Counter(dict.fromkeys(SIZES, 0))
    for frame, pred in zip(frames, predictions):
        rows, overlays = frame_errors(frame, pred[pred[:, 5] >= args.conf_analysis], args.iou_match, matrix, fn_sizes)
        errors.extend(rows)
        if rows:
            examples.append((len(rows), frame, overlays))
    pd.DataFrame(errors, columns=ERROR_COLUMNS).to_csv(out / "errors.csv", index=False)
    matrix.to_csv(out / "confusion_matrix.csv", index_label="gt_class")

    example_dir = out / "error_examples"
    example_dir.mkdir(exist_ok=True)
    previous = set(example_dir.glob("rank-*.jpg"))
    for rank, (_, frame, overlays) in enumerate(sorted(examples, key=lambda x: -x[0])[:args.n_examples], 1):
        path = example_dir / f"rank-{rank:03d}-{frame.path.stem}.jpg"
        render_example(frame, overlays, path)
        previous.discard(path)
    for path in previous:
        path.unlink()

    scores = {}
    for name in CLASS_NAMES.values():
        tp = int(matrix.loc[name, name])
        gt_count, pred_count = int(matrix.loc[name].sum()), int(matrix[name].sum())
        scores[name] = {"precision": tp / pred_count if pred_count else None,
                        "recall": tp / gt_count if gt_count else None,
                        "TP": tp, "FP": pred_count - tp, "FN": gt_count - tp}
    counts = Counter(row["type"] for row in errors)
    return {"confidence": args.conf_analysis, "iou_match": args.iou_match, "per_class": scores,
            "counts": {"TP": sum(row["TP"] for row in scores.values()),
                       **{kind: counts[kind] for kind in ("FP", "FN", "class_confusion")}},
            "FN_by_size": dict(fn_sizes)}


def markdown_table(table: pd.DataFrame) -> str:
    rows = [table.columns, ["---"] * len(table.columns)]
    rows.extend([f"{value:.4f}" if isinstance(value, float) and np.isfinite(value)
                 else "—" if pd.isna(value) else str(value) for value in row]
                for row in table.itertuples(index=False, name=None))
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def write_summary(out: Path, args: argparse.Namespace, comparison: pd.DataFrame, metrics: dict) -> None:
    table = markdown_table(comparison)
    sizes = pd.DataFrame([metrics["GT_by_size"], metrics["error_analysis"]["FN_by_size"]], index=["GT", "FN"])
    scores = pd.DataFrame.from_dict(metrics["error_analysis"]["per_class"], orient="index")
    (out / "eval_summary.md").write_text(
        f"# Detector evaluation\n\nSplit `{args.split}` · {metrics['images']} images · {args.imgsz} px · {args.device}\n\n"
        f"{table}\n\nCOCO AP uses confidence {PREDICT_CONF}, a cap of {MAX_DET} detections per image, "
        "and original-image box areas. Small is below 32² px², medium is 32² to below 96² px², "
        "and large is 96² px² or above. — marks unavailable metrics.\n\n"
        f"Fine-tuned mean inference time was {metrics['inference_ms_per_image']:.2f} ms/image "
        "with one image per call, excluding preprocessing, postprocessing and warm-up.\n\n"
        f"## Error analysis\n\nConfidence ≥ {args.conf_analysis}, IoU ≥ {args.iou_match}. "
        "Matched wrong-class predictions count as one FP for the predicted class and one FN for the GT class. "
        "The size table counts unmatched GT only for FN. In errors.csv, class confusions use the GT box.\n\n"
        f"{markdown_table(sizes.reset_index(names='Count'))}\n\n"
        f"{markdown_table(scores.reset_index(names='Class'))}\n", encoding="utf-8")
    print(f"\n{table}\n\nFine-tuned inference: {metrics['inference_ms_per_image']:.2f} ms/image")
    print(f"Reports and plots -> {out}")


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths = split_paths(args.data, args.split)
    frames = [read_frame(path, Path(label)) for path, label in zip(paths, img2label_paths(list(map(str, paths))))]
    gt_sizes = Counter(size_bucket(box[1:]) for frame in frames for box in frame.gt)
    gt_counts = {size: gt_sizes[size] for size in SIZES}

    fine_tuned = YOLO(str(args.weights))
    if fine_tuned.names != CLASS_NAMES:
        raise ValueError(f"Fine-tuned model class mapping must be {CLASS_NAMES}: {args.weights}")
    metrics = official_metrics(fine_tuned, args, out)
    project_classes = {cls: cls for cls in CLASS_NAMES}
    predictions, speed = predict(fine_tuned, frames, args, project_classes)
    comparison = [{"model": args.weights.stem, "kind": "fine-tuned", **coco_ap(frames, predictions)}]
    for weights in args.baseline:
        baseline_predictions, _ = predict(YOLO(weights), frames, args, COCO_TO_PROJECT)
        comparison.append({"model": Path(weights).stem, "kind": "COCO", **coco_ap(frames, baseline_predictions)})
    table = pd.DataFrame(comparison).assign(**{f"GT_{size}": count for size, count in gt_counts.items()})
    table.to_csv(out / "comparison.csv", index=False)
    if args.conf_analysis < PREDICT_CONF:
        predictions, _ = predict(fine_tuned, frames, args, project_classes, args.conf_analysis)
    metrics.update(split=args.split, images=len(frames), GT_by_size=gt_counts, inference_ms_per_image=speed,
                   error_analysis=error_analysis(frames, predictions, args, out))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_summary(out, args, table, metrics)


if __name__ == "__main__":
    main()
