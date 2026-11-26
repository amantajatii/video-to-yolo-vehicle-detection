"""Extract a diverse, class-aware set of frames from the raw videos.

Pipeline per video:
  1. Candidates   : decode frames at --cand-fps.
  2. Quality      : drop blurry (per-video Laplacian percentile) and too dark/bright frames.
  3. Class scan   : run a pretrained COCO detector to count mobil/truk/bus per candidate.
                    These detections are used ONLY for frame selection, not as labels.
  4. Selection    : fill the per-video quota in priority order
                    bus -> truk -> background -> spread-in-time (farthest-point), with dedup rules
                    (minimum time gap + perceptual-hash distance).
  5. Output       : write JPGs + frames_manifest.csv (one row per selected frame).

Usage:
    python scripts/extract_frames.py --videos datasets --out data/frames \
        --quotas Video_1=90,Video_2=140,Video_3=170
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

from common import CLASS_NAMES, COCO_TO_PROJECT, video_code

QUALITY_THUMB_WIDTH = 640
BRIGHTNESS_RANGE = (40.0, 220.0)
BACKGROUND_MAX_VEHICLES = 1


@dataclass(frozen=True)
class Candidate:
    video: str
    frame_idx: int
    timestamp_s: float
    blur: float
    brightness: float
    phash: str
    n_mobil: int = 0
    n_truk: int = 0
    n_bus: int = 0
    reason: str = ""

    @property
    def n_vehicles(self) -> int:
        return self.n_mobil + self.n_truk + self.n_bus


@dataclass(frozen=True)
class SelectionConfig:
    quota: int
    bus_share: float
    truk_share: float
    background_share: float
    min_gap_s: float
    min_gap_rare_s: float
    min_phash_dist: int


# --------------------------------------------------------------------------- #
# Pass 1: candidate scoring
# --------------------------------------------------------------------------- #
def quality_metrics(frame: np.ndarray) -> tuple[float, float, str]:
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (QUALITY_THUMB_WIDTH, int(h * QUALITY_THUMB_WIDTH / w)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    phash = str(imagehash.phash(Image.fromarray(gray)))
    return blur, brightness, phash


def count_classes(model: YOLO, frame: np.ndarray, args: argparse.Namespace) -> dict[str, int]:
    result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, agnostic_nms=True,
                           classes=list(COCO_TO_PROJECT), verbose=False, device=args.device)[0]
    counts = {name: 0 for name in CLASS_NAMES.values()}
    for coco_id in result.boxes.cls.tolist():
        counts[CLASS_NAMES[COCO_TO_PROJECT[int(coco_id)]]] += 1
    return counts


def scan_video(video_path: Path, model: YOLO, args: argparse.Namespace) -> list[Candidate]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / args.cand_fps))
    candidates: list[Candidate] = []
    idx = 0
    try:
        while cap.grab():
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    blur, brightness, phash = quality_metrics(frame)
                    counts = count_classes(model, frame, args)
                    candidates.append(Candidate(
                        video=video_path.name, frame_idx=idx, timestamp_s=round(idx / fps, 3),
                        blur=blur, brightness=brightness, phash=phash,
                        n_mobil=counts["mobil"], n_truk=counts["truk"], n_bus=counts["bus"],
                    ))
            idx += 1
    finally:
        cap.release()
    return candidates


def filter_quality(candidates: list[Candidate], blur_percentile: float) -> list[Candidate]:
    if not candidates:
        return []
    blur_threshold = float(np.percentile([c.blur for c in candidates], blur_percentile))
    lo, hi = BRIGHTNESS_RANGE
    return [c for c in candidates if c.blur >= blur_threshold and lo <= c.brightness <= hi]


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def is_distinct(cand: Candidate, chosen: list[Candidate], min_gap_s: float, min_phash: int) -> bool:
    cand_hash = imagehash.hex_to_hash(cand.phash)
    for other in chosen:
        if abs(cand.timestamp_s - other.timestamp_s) < min_gap_s:
            return False
        if cand_hash - imagehash.hex_to_hash(other.phash) < min_phash:
            return False
    return True


def pick_greedy(pool: list[Candidate], chosen: list[Candidate], limit: int,
                gap_s: float, min_phash: int, reason: str) -> list[Candidate]:
    picked: list[Candidate] = []
    for cand in pool:
        if len(picked) >= limit:
            break
        if is_distinct(cand, chosen + picked, gap_s, min_phash):
            picked.append(replace(cand, reason=reason))
    return picked


def pick_spread(pool: list[Candidate], chosen: list[Candidate], limit: int,
                gap_s: float, min_phash: int) -> list[Candidate]:
    """Farthest-point sampling in time: repeatedly take the frame farthest from all chosen
    frames (ties -> sharper), so the remaining quota is spread evenly over the timeline."""
    picked: list[Candidate] = []
    remaining = list(pool)
    while len(picked) < limit and remaining:
        times = np.array([c.timestamp_s for c in chosen + picked]) if (chosen or picked) else None
        def distance(c: Candidate) -> float:
            return float(np.abs(times - c.timestamp_s).min()) if times is not None else 0.0
        best = max(remaining, key=lambda c: (distance(c), c.blur))
        remaining = [c for c in remaining if c is not best]
        if is_distinct(best, chosen + picked, gap_s, min_phash):
            picked.append(replace(best, reason="uniform"))
    return picked


def select_frames(candidates: list[Candidate], cfg: SelectionConfig) -> list[Candidate]:
    used = lambda: {c.frame_idx for c in chosen}  # noqa: E731
    chosen: list[Candidate] = []

    bus_pool = sorted((c for c in candidates if c.n_bus > 0), key=lambda c: (-c.n_bus, -c.blur))
    chosen += pick_greedy(bus_pool, chosen, round(cfg.quota * cfg.bus_share),
                          cfg.min_gap_rare_s, cfg.min_phash_dist, "bus")

    truk_pool = sorted((c for c in candidates if c.n_truk > 0 and c.frame_idx not in used()),
                       key=lambda c: (-c.n_truk, -c.blur))
    chosen += pick_greedy(truk_pool, chosen, round(cfg.quota * cfg.truk_share),
                          cfg.min_gap_rare_s, cfg.min_phash_dist, "truk")

    bg_pool = sorted((c for c in candidates if c.n_vehicles <= BACKGROUND_MAX_VEHICLES
                      and c.frame_idx not in used()), key=lambda c: -c.blur)
    chosen += pick_greedy(bg_pool, chosen, round(cfg.quota * cfg.background_share),
                          cfg.min_gap_s, cfg.min_phash_dist, "background")

    rest = [c for c in candidates if c.frame_idx not in used()]
    chosen += pick_spread(rest, chosen, cfg.quota - len(chosen), cfg.min_gap_s, cfg.min_phash_dist)
    return sorted(chosen, key=lambda c: c.frame_idx)


# --------------------------------------------------------------------------- #
# Pass 2: write selected frames
# --------------------------------------------------------------------------- #
def frame_filename(video_path: Path, cand: Candidate) -> str:
    return f"{video_code(video_path)}_t{round(cand.timestamp_s * 1000):07d}.jpg"


def write_frames(video_path: Path, selected: list[Candidate], out_dir: Path, jpg_quality: int) -> list[dict]:
    wanted = {c.frame_idx: c for c in selected}
    last = max(wanted)
    cap = cv2.VideoCapture(str(video_path))
    rows: list[dict] = []
    idx = 0
    try:
        while idx <= last and cap.grab():
            if idx in wanted:
                ok, frame = cap.retrieve()
                if not ok:
                    raise RuntimeError(f"Failed to decode frame {idx} of {video_path.name}")
                name = frame_filename(video_path, wanted[idx])
                cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
                rows.append({"file": name, **asdict(wanted[idx])})
            idx += 1
    finally:
        cap.release()
    if len(rows) != len(selected):
        raise RuntimeError(f"Wrote {len(rows)}/{len(selected)} frames for {video_path.name}")
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_quotas(text: str) -> dict[str, int]:
    quotas = {}
    for item in text.split(","):
        stem, value = item.split("=")
        quotas[stem.strip()] = int(value)
    return quotas


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", type=Path, default=Path("datasets"))
    p.add_argument("--out", type=Path, default=Path("data/frames"))
    p.add_argument("--quotas", type=str, default="Video_1=90,Video_2=140,Video_3=170")
    p.add_argument("--model", type=str, default="yolo11x.pt",
                   help="COCO model used for selection only (11m over-predicted bus on vans)")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--cand-fps", type=float, default=4.0)
    p.add_argument("--blur-percentile", type=float, default=10.0)
    p.add_argument("--bus-share", type=float, default=0.40)
    p.add_argument("--truk-share", type=float, default=0.25)
    p.add_argument("--background-share", type=float, default=0.05)
    p.add_argument("--min-gap", type=float, default=1.0, help="seconds between frames (common)")
    p.add_argument("--min-gap-rare", type=float, default=0.5, help="seconds between bus/truk frames")
    p.add_argument("--min-phash", type=int, default=2,
                   help="min pHash Hamming distance; low because static cameras keep pHash nearly constant")
    p.add_argument("--jpg-quality", type=int, default=95)
    return p


def process_video(video: Path, quota: int, model: YOLO, args: argparse.Namespace) -> list[dict]:
    candidates = scan_video(video, model, args)
    good = filter_quality(candidates, args.blur_percentile)
    cfg = SelectionConfig(quota=quota, bus_share=args.bus_share, truk_share=args.truk_share,
                          background_share=args.background_share, min_gap_s=args.min_gap,
                          min_gap_rare_s=args.min_gap_rare, min_phash_dist=args.min_phash)
    selected = select_frames(good, cfg)
    print(f"{video.name}: candidates={len(candidates)} after_quality={len(good)} "
          f"selected={len(selected)}/{quota}")
    return write_frames(video, selected, args.out, args.jpg_quality)


def main() -> None:
    args = build_parser().parse_args()
    quotas = parse_quotas(args.quotas)
    videos = sorted(args.videos.glob("*.mp4"))
    missing = set(quotas) - {v.stem for v in videos}
    if missing:
        raise SystemExit(f"Videos not found for quotas: {sorted(missing)}")

    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    rows: list[dict] = []
    for video in videos:
        if video.stem in quotas:
            rows += process_video(video, quotas[video.stem], model, args)

    manifest = pd.DataFrame(rows)
    manifest_path = args.out.parent / "frames_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nTotal frames: {len(manifest)} -> {args.out}  (manifest: {manifest_path})")
    print(manifest.groupby(["video", "reason"]).size().unstack(fill_value=0))
    has = lambda col: (manifest[col] > 0).groupby(manifest["video"]).sum()  # noqa: E731
    print("\nFrames containing class (per selection detector):")
    print(pd.DataFrame({"mobil": has("n_mobil"), "truk": has("n_truk"), "bus": has("n_bus")}))


if __name__ == "__main__":
    main()
