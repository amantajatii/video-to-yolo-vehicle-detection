"""Compare several training runs on the SAME validation and test sets (seed / label-version study).

Each run is given as `<version>:<run_name>` (a folder under --runs-dir with weights/best.pt).
Every run is evaluated with the same data.yaml, so differences come from the runs, not from the
evaluation labels. The table reports val and test mAP plus per-class AP50 for truk and bus, and a
mean ± std per version.

Model selection must use the val columns only; test columns are for reporting.

Usage:
    python scripts/compare_runs.py --data data/dataset/data.yaml \
        --runs v1:yolo11n_vehicle v1:yolo11n_vehicle_v1_s1 v2:yolo11n_vehicle_v2 v2:yolo11n_vehicle_v2_s1 \
        --out reports/seed_study.csv
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO

SPLITS = ("val", "test")


def best_epoch(run_dir: Path) -> tuple[int, int]:
    res = pd.read_csv(run_dir / "results.csv")
    res.columns = res.columns.str.strip()
    return int(res.loc[res["metrics/mAP50-95(B)"].idxmax(), "epoch"]), int(res.epoch.iloc[-1])


def evaluate(weights: Path, data: Path, split: str, args: argparse.Namespace, scratch: Path) -> dict:
    r = YOLO(str(weights)).val(data=str(data.resolve()), split=split, imgsz=args.imgsz, device=args.device,
                               verbose=False, plots=False, project=str(scratch), name="val", exist_ok=True)
    per_class = {r.names[int(c)]: r.class_result(i)[2] for i, c in enumerate(r.ap_class_index)}
    return {f"{split}_mAP50": r.box.map50, f"{split}_mAP5095": r.box.map,
            f"{split}_truk50": per_class.get("truk", 0.0), f"{split}_bus50": per_class.get("bus", 0.0)}


def evaluate_run(spec: str, args: argparse.Namespace, scratch: Path) -> dict:
    version, _, name = spec.partition(":")
    if not name:
        raise SystemExit(f"Run must be <version>:<run_name>, got {spec!r}")
    run_dir = args.runs_dir / name
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        raise SystemExit(f"Missing weights: {weights}")
    seed = yaml.safe_load((run_dir / "args.yaml").read_text()).get("seed")
    best, last = best_epoch(run_dir)
    row = {"version": version, "seed": seed, "run": name, "best_epoch": best, "epochs": last}
    for split in SPLITS:
        row.update(evaluate(weights, args.data, split, args, scratch))
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="<version>:<run_name> entries")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--data", type=Path, default=Path("data/dataset/data.yaml"),
                   help="one data.yaml used for every run (same val/test labels)")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--out", type=Path, default=Path("reports/seed_study.csv"))
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        df = pd.DataFrame([evaluate_run(spec, args, Path(tmp)) for spec in args.runs]).round(4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    metrics = [c for c in df.columns if c.startswith(SPLITS)]
    print("\nmean ± std per version (select on val only)")
    print(df.groupby("version")[metrics].agg(["mean", "std"]).round(3).T.to_string())
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
