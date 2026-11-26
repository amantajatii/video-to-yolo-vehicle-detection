"""Fine-tune a YOLO detector on the vehicle dataset from a YAML config.

All training hyper-parameters live in the config file (default: configs/train.yaml) so the
experiment is fully described by one versioned file. Ultralytics additionally writes the
resolved arguments to <run>/args.yaml. This script also records wall-clock time and the
software/hardware environment in <run>/train_meta.json for the report.

Usage:
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --config configs/train.yaml --override epochs=1 device=cpu   # smoke test
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO


def parse_overrides(items: list[str]) -> dict:
    overrides = {}
    for item in items:
        key, _, value = item.partition("=")
        if not key or not _:
            raise ValueError(f"Override must be key=value, got {item!r}")
        overrides[key] = yaml.safe_load(value)
    return overrides


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    p.add_argument("--override", nargs="*", default=[], help="key=value pairs that override the config")
    args = p.parse_args()

    cfg = {**yaml.safe_load(args.config.read_text()), **parse_overrides(args.override)}
    if not Path(cfg["data"]).exists():
        raise SystemExit(f"Dataset yaml not found: {cfg['data']} (run scripts/build_dataset.py first)")

    # Ultralytics nests relative project dirs under runs/detect/; an absolute path keeps runs/<name>.
    cfg["project"] = str(Path(cfg.get("project", "runs")).resolve())
    model = YOLO(cfg.pop("model"))
    start = time.time()
    results = model.train(**cfg)
    elapsed = time.time() - start

    run_dir = Path(results.save_dir)
    shutil.copy2(args.config, run_dir / "train_config.yaml")
    meta = {"config": str(args.config), "overrides": args.override, "train_seconds": round(elapsed, 1),
            "best_weights": str(run_dir / "weights" / "best.pt"), "environment": environment()}
    (run_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nTraining finished in {elapsed / 60:.1f} min -> {run_dir}")


if __name__ == "__main__":
    main()
