"""Build a contact sheet (grid of evenly spaced frames) for each raw video.

Used to inspect scene content before deciding frame-sampling quotas.

Usage:
    python scripts/inspect_videos.py --videos datasets --out reports/contact_sheets
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

GRID_COLS = 5
GRID_ROWS = 4
THUMB_WIDTH = 384


def read_frame_at(cap: cv2.VideoCapture, index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    return frame if ok else None


def make_thumb(frame: np.ndarray, label: str) -> np.ndarray:
    h, w = frame.shape[:2]
    thumb = cv2.resize(frame, (THUMB_WIDTH, int(h * THUMB_WIDTH / w)))
    cv2.rectangle(thumb, (0, 0), (110, 24), (0, 0, 0), -1)
    cv2.putText(thumb, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return thumb


def build_sheet(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = GRID_COLS * GRID_ROWS
        indices = np.linspace(0, total - 1, n, dtype=int)
        thumbs = []
        for idx in indices:
            frame = read_frame_at(cap, int(idx))
            if frame is None:
                continue
            thumbs.append(make_thumb(frame, f"t={idx / fps:6.1f}s"))
    finally:
        cap.release()

    if not thumbs:
        raise RuntimeError(f"No frames decoded from {video_path}")
    blank = np.zeros_like(thumbs[0])
    thumbs += [blank] * (n - len(thumbs))
    rows = [np.hstack(thumbs[r * GRID_COLS:(r + 1) * GRID_COLS]) for r in range(GRID_ROWS)]
    return np.vstack(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=Path("datasets"))
    parser.add_argument("--out", type=Path, default=Path("reports/contact_sheets"))
    args = parser.parse_args()

    videos = sorted(args.videos.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No .mp4 files found in {args.videos}")
    args.out.mkdir(parents=True, exist_ok=True)
    for video in videos:
        sheet = build_sheet(video)
        out_path = args.out / f"{video.stem}.jpg"
        cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"{video.name} -> {out_path}")


if __name__ == "__main__":
    main()
