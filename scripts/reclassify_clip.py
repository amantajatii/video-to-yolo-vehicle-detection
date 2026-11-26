"""Second-stage class correction of teacher boxes with zero-shot CLIP.

The COCO teacher follows COCO's class boundaries (pickups -> truck, large vans -> truck
or bus), which conflict with this project's VISUAL guideline (pickup/van/SUV = mobil).
Each `truk`/`bus` box is cropped (with some context) and scored by CLIP against
fine-grained text prompts, which are grouped into the project classes:

    mobil : sedan / hatchback / SUV / pickup truck / van / minivan
    truk  : box truck / semi-trailer truck / dump truck / car carrier / tanker
    bus   : city bus / coach / minibus

A box is relabelled only when CLIP's probability for another group is >= --min-prob,
the crop is large enough (--min-crop-px) for CLIP to be meaningful, AND the transition
is in --allowed (default: truk->mobil only, validated visually). Other disagreements
(e.g. anything involving bus) are only FLAGGED for manual review. Everything else keeps
the teacher label. Output is still a PRE-label that must be reviewed.

Outputs:
  <out>/<frame>.txt        YOLO labels after correction
  <out-csv>                predictions with teacher_cls, clip_top, clip_group, clip_prob, changed
  <viz-dir>/changed_*.jpg  grids of changed crops for visual verification

Usage:
    python scripts/reclassify_clip.py --predictions data/autolabel_predictions.csv \
        --frames data/frames --out data/labels_prelabel --out-csv data/prelabel_predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image

from common import CLASS_NAMES

NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}
SUBCLASSES: dict[str, list[str]] = {
    "mobil": ["sedan car", "hatchback car", "SUV", "pickup truck", "delivery van", "passenger van",
              "minivan", "station wagon"],
    "truk": ["box truck", "semi-trailer truck", "dump truck", "car carrier truck", "tanker truck",
             "large cargo truck"],
    # No "minibus" prompt: CLIP matched van-based minibuses (Ducato/Transit), which are `mobil` here.
    "bus": ["city bus", "coach bus"],
}
TEMPLATES = ["a photo of a {}.", "a traffic camera photo of a {}.", "a low resolution photo of a {}."]
REVIEW_CLASSES = ("truk", "bus")


def build_text_features(model, tokenizer, device: str) -> tuple[torch.Tensor, list[tuple[str, str]]]:
    labels, feats = [], []
    with torch.no_grad():
        for group, names in SUBCLASSES.items():
            for name in names:
                tokens = tokenizer([t.format(name) for t in TEMPLATES]).to(device)
                emb = model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                mean = emb.mean(dim=0)
                feats.append(mean / mean.norm())
                labels.append((group, name))
    return torch.stack(feats), labels


def crop_box(image: np.ndarray, row: pd.Series, pad: float) -> np.ndarray:
    h, w = image.shape[:2]
    bw, bh = row.x2 - row.x1, row.y2 - row.y1
    x1, y1 = int(max(0, row.x1 - pad * bw)), int(max(0, row.y1 - pad * bh))
    x2, y2 = int(min(w, row.x2 + pad * bw)), int(min(h, row.y2 + pad * bh))
    return image[y1:y2, x1:x2]


def classify_crops(crops: list[np.ndarray], model, preprocess, text_feats: torch.Tensor,
                   device: str) -> np.ndarray:
    batch = torch.stack([preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops])
    with torch.no_grad():
        img = model.encode_image(batch.to(device))
        img = img / img.norm(dim=-1, keepdim=True)
        probs = (100.0 * img @ text_feats.T).softmax(dim=-1)
    return probs.cpu().numpy()


def reclassify(df: pd.DataFrame, frames_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    model, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrained)
    model = model.to(args.device).eval()
    text_feats, labels = build_text_features(model, open_clip.get_tokenizer(args.clip_model), args.device)
    groups = np.array([g for g, _ in labels])

    out = df.assign(teacher_cls=df.cls, clip_top="", clip_group="", clip_prob=np.nan, changed=False, flag=False)
    side = np.minimum(out.x2 - out.x1, out.y2 - out.y1)
    todo = out[out.class_name.isin(REVIEW_CLASSES) & (side >= args.min_crop_px)]
    for fname, rows in todo.groupby("file"):
        image = cv2.imread(str(frames_dir / fname))
        probs = classify_crops([crop_box(image, r, args.pad) for _, r in rows.iterrows()],
                               model, preprocess, text_feats, args.device)
        for idx, p in zip(rows.index, probs):
            group_probs = {g: float(p[groups == g].sum()) for g in SUBCLASSES}
            best_group = max(group_probs, key=group_probs.get)
            out.loc[idx, ["clip_top", "clip_group", "clip_prob"]] = [labels[int(p.argmax())][1], best_group,
                                                                     group_probs[best_group]]
            current = out.loc[idx, "class_name"]
            if best_group == current or group_probs[best_group] < args.min_prob:
                continue
            if f"{current}:{best_group}" in args.allowed:
                out.loc[idx, ["cls", "class_name", "changed"]] = [NAME_TO_ID[best_group], best_group, True]
            else:
                out.loc[idx, "flag"] = True  # CLIP disagrees, transition not trusted -> manual review
    return out


def write_labels(df: pd.DataFrame, frames_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame in sorted(frames_dir.glob("*.jpg")):
        rows = df[df.file == frame.name]
        lines = [f"{r.cls} {r.cx:.6f} {r.cy:.6f} {r.w:.6f} {r.h:.6f}" for r in rows.itertuples()]
        (out_dir / f"{frame.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def save_change_grids(df: pd.DataFrame, frames_dir: Path, viz_dir: Path, per_grid: int = 24) -> None:
    changed = df[df.changed].sort_values("clip_prob", ascending=False)
    viz_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    for r in changed.itertuples():
        crop = cv2.resize(crop_box(cv2.imread(str(frames_dir / r.file)), pd.Series(r._asdict()), 0.1), (200, 160))
        text = f"{CLASS_NAMES[r.teacher_cls]}->{r.class_name} {r.clip_prob:.2f}"
        cv2.rectangle(crop, (0, 0), (200, 34), (0, 0, 0), -1)
        cv2.putText(crop, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(crop, r.clip_top[:26], (3, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        tiles.append(crop)
    for g in range(0, len(tiles), per_grid):
        chunk = tiles[g:g + per_grid] + [np.zeros_like(tiles[0])] * (-len(tiles[g:g + per_grid]) % 6)
        grid = np.vstack([np.hstack(chunk[i:i + 6]) for i in range(0, len(chunk), 6)])
        cv2.imwrite(str(viz_dir / f"changed_{g // per_grid:02d}.jpg"), grid)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, default=Path("data/autolabel_predictions.csv"))
    p.add_argument("--frames", type=Path, default=Path("data/frames"))
    p.add_argument("--out", type=Path, default=Path("data/labels_prelabel"))
    p.add_argument("--out-csv", type=Path, default=Path("data/prelabel_predictions.csv"))
    p.add_argument("--viz-dir", type=Path, default=Path("reports/clip_reclass"))
    p.add_argument("--clip-model", type=str, default="ViT-B-16")
    p.add_argument("--clip-pretrained", type=str, default="laion2b_s34b_b88k")
    p.add_argument("--min-prob", type=float, default=0.6, help="min CLIP group probability to relabel")
    p.add_argument("--allowed", nargs="+", default=["truk:mobil"],
                   help="trusted teacher:clip transitions; other disagreements are only flagged. "
                        "Bus transitions were unreliable in validation (rear-view minibus -> truk, yellow van -> bus)")
    p.add_argument("--min-crop-px", type=float, default=24.0, help="skip boxes with a smaller side")
    p.add_argument("--pad", type=float, default=0.1, help="context padding around the box")
    p.add_argument("--device", type=str, default="mps")
    return p


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.predictions)
    result = reclassify(df, args.frames, args)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_csv, index=False)
    write_labels(result, args.frames, args.out)
    save_change_grids(result, args.frames, args.viz_dir)

    reviewed = result[result.clip_group != ""]
    print(f"Boxes checked by CLIP: {len(reviewed)}  changed: {int(result.changed.sum())}  "
          f"flagged for review: {int(result.flag.sum())}")
    print(pd.crosstab(result.teacher_cls.map(CLASS_NAMES), result.class_name, margins=True,
                      rownames=["teacher"], colnames=["after CLIP"]))
    print(f"\nLabels -> {args.out}   CSV -> {args.out_csv}   grids -> {args.viz_dir}")


if __name__ == "__main__":
    main()
