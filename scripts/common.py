"""Shared constants for the dataset pipeline."""

from __future__ import annotations

import re
from pathlib import Path

# Project classes (Indonesian names, as specified by the assignment).
CLASS_NAMES: dict[int, str] = {0: "mobil", 1: "truk", 2: "bus"}

# COCO class id -> project class id. Only used when reading pretrained COCO output.
COCO_TO_PROJECT: dict[int, int] = {2: 0, 7: 1, 5: 2}  # car, truck, bus


def video_code(video_path: Path) -> str:
    """Return a short id like 'v1' from a file name such as 'Video_1.mp4'."""
    match = re.search(r"(\d+)", video_path.stem)
    if match is None:
        raise ValueError(f"Video name must contain a number: {video_path.name}")
    return f"v{int(match.group(1))}"
