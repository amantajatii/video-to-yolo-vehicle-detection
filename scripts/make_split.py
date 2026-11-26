"""Assign frames to train/val/test using contiguous TIME BLOCKS (not random frames).

Adjacent video frames are near-duplicates, so a random per-frame split leaks
almost-identical images into the test set. Here each video's timeline is cut into
--blocks equal-duration blocks; whole blocks go to one split. Frames closer than
--buffer seconds to a boundary between two DIFFERENT splits are dropped.

The block->split assignment is chosen by a seeded random search that minimises the
deviation from the target ratios, both for image counts and for truk/bus box counts
(taken from the auto-labels, i.e. before review), under the constraint that every
video contributes at least one block to every split.

Outputs:
  data/split_assignment.csv   file, video, timestamp_s, block, split (dropped -> "drop")
  data/review_order.txt       suggested review order: test -> val -> train (bus/truk/low-conf first)

Usage:
    python scripts/make_split.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def load_frames(manifest: Path, predictions: Path) -> pd.DataFrame:
    frames = pd.read_csv(manifest)[["file", "video", "timestamp_s"]]
    preds = pd.read_csv(predictions)
    counts = preds.pivot_table(index="file", columns="class_name", values="cls", aggfunc="size", fill_value=0)
    min_conf = preds.groupby("file").conf.min().rename("min_conf")
    flag_col = preds.flag.astype(int) if "flag" in preds else pd.Series(0, index=preds.index)
    flags = flag_col.groupby(preds.file).sum().rename("n_flags")
    df = (frames.merge(counts, on="file", how="left").merge(min_conf, on="file", how="left")
          .merge(flags, on="file", how="left"))
    df["n_flags"] = df["n_flags"].fillna(0).astype(int)
    for col in ("mobil", "truk", "bus"):
        df[col] = df[col].fillna(0).astype(int) if col in df else 0
    return df


def add_blocks(df: pd.DataFrame, n_blocks: int) -> pd.DataFrame:
    def block_of(t: pd.Series) -> np.ndarray:
        edges = np.linspace(t.min(), t.max() + 1e-6, n_blocks + 1)
        return np.searchsorted(edges, t, side="right") - 1
    return df.assign(block=df.groupby("video").timestamp_s.transform(block_of))


def score(df: pd.DataFrame, assign: dict, targets: np.ndarray) -> float:
    split = df.set_index(["video", "block"]).index.map(assign)
    total = 0.0
    for col in ("n_img", "truk", "bus"):
        values = df["one"] if col == "n_img" else df[col]
        per_split = np.array([values[split == s].sum() for s in SPLITS], dtype=float)
        if per_split.sum() > 0:
            weight = 2.0 if col == "bus" else 1.0
            total += weight * np.abs(per_split / per_split.sum() - targets).sum()
    return total


def random_assignment(rng: np.random.Generator, videos: list[str], n_blocks: int,
                      targets: np.ndarray) -> dict:
    assign = {}
    for video in videos:
        labels = rng.choice(len(SPLITS), size=n_blocks, p=targets)
        forced = rng.choice(n_blocks, size=len(SPLITS), replace=False)  # every split gets >=1 block
        labels[forced] = np.arange(len(SPLITS))
        assign.update({(video, b): SPLITS[labels[b]] for b in range(n_blocks)})
    return assign


def search_assignment(df: pd.DataFrame, n_blocks: int, targets: np.ndarray,
                      trials: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    videos = sorted(df.video.unique())
    best, best_score = None, np.inf
    for _ in range(trials):
        candidate = random_assignment(rng, videos, n_blocks, targets)
        s = score(df, candidate, targets)
        if s < best_score:
            best, best_score = candidate, s
    print(f"Best assignment score: {best_score:.3f} (lower is better)")
    return best


def drop_boundary_frames(df: pd.DataFrame, buffer_s: float) -> pd.DataFrame:
    """Mark frames within buffer_s of a neighbouring frame that belongs to another split."""
    out = df.sort_values(["video", "timestamp_s"]).copy()
    drop = np.zeros(len(out), dtype=bool)
    for _, g in out.groupby("video"):
        idx, t, s = g.index.to_numpy(), g.timestamp_s.to_numpy(), g.split.to_numpy()
        for i in range(len(g)):
            near = np.abs(t - t[i]) < buffer_s
            if (s[near] != s[i]).any():
                drop[out.index.get_indexer([idx[i]])[0]] = True
    out.loc[drop, "split"] = "drop"
    return out


def write_review_order(df: pd.DataFrame, path: Path) -> None:
    kept = df[df.split != "drop"].copy()
    kept["priority"] = np.select([(kept.n_flags > 0) | (kept.bus > 0), kept.truk > 0], [0, 1], default=2)
    lines = ["# Urutan review label (X-AnyLabeling). Cek semua gambar di bagian TEST dan VAL.",
             "# Dalam tiap bagian: flag CLIP / bus -> truk -> sisanya (confidence terendah dulu).",
             "# flag = CLIP tidak setuju dengan kelas kotak ini tapi tidak diubah otomatis; cek kelasnya.", ""]
    for split in ("test", "val", "train"):
        part = kept[kept.split == split].sort_values(["priority", "min_conf"])
        lines.append(f"## {split.upper()} ({len(part)} gambar)")
        lines += [f"{r.file}  bus={r.bus} truk={r.truk} flag={r.n_flags} min_conf={r.min_conf:.2f}"
                  for r in part.itertuples()]
        lines.append("")
    path.write_text("\n".join(lines))


def print_summary(df: pd.DataFrame) -> None:
    kept = df[df.split != "drop"]
    print(f"Dropped at split boundaries: {(df.split == 'drop').sum()}  kept: {len(kept)}")
    summary = kept.groupby("split").agg(images=("file", "size"), mobil=("mobil", "sum"),
                                        truk=("truk", "sum"), bus=("bus", "sum"),
                                        imgs_with_bus=("bus", lambda b: (b > 0).sum()))
    summary["img_ratio"] = (summary.images / summary.images.sum()).round(3)
    print(summary.reindex(list(SPLITS)))
    print("\nImages per video x split:")
    print(pd.crosstab(kept.video, kept.split)[list(SPLITS)])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=Path("data/frames_manifest.csv"))
    p.add_argument("--predictions", type=Path, default=Path("data/autolabel_predictions.csv"))
    p.add_argument("--out", type=Path, default=Path("data/split_assignment.csv"))
    p.add_argument("--review-order", type=Path, default=Path("data/review_order.txt"))
    p.add_argument("--blocks", type=int, default=10, help="time blocks per video")
    p.add_argument("--ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15])
    p.add_argument("--buffer", type=float, default=1.5, help="seconds dropped around split boundaries")
    p.add_argument("--trials", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reuse-assignment", action="store_true",
                   help="keep the split stored in --out; only refresh stats and review order")
    return p


def main() -> None:
    args = build_parser().parse_args()
    targets = np.array(args.ratios) / sum(args.ratios)
    df = add_blocks(load_frames(args.manifest, args.predictions), args.blocks).assign(one=1)
    if args.reuse_assignment:
        # Keep the locked split; only refresh label statistics and the review order.
        locked = pd.read_csv(args.out)[["file", "split"]]
        df = df.merge(locked, on="file", how="left", validate="one_to_one")
        if df.split.isna().any():
            raise SystemExit(f"Frames missing from locked assignment: {df[df.split.isna()].file.tolist()[:5]}")
    else:
        assign = search_assignment(df, args.blocks, targets, args.trials, args.seed)
        df["split"] = df.set_index(["video", "block"]).index.map(assign)
        df = drop_boundary_frames(df, args.buffer)
    df.drop(columns="one").to_csv(args.out, index=False)
    write_review_order(df, args.review_order)
    print_summary(df)
    print(f"\nSaved: {args.out}, {args.review_order}")


if __name__ == "__main__":
    main()
