#!/usr/bin/env bash
# Seed / label-version study: train the same config on two label versions with several seeds.
#
#   v1 = round-1 labels  (data/dataset_v1, submitted dataset)
#   v2 = rounds 1-3 labels (data/dataset, after model-assisted label QA)
#
# Images, split and test labels are identical in both datasets; only train/val labels differ.
# Seed 42 runs are the default runs (yolo11n_vehicle, yolo11n_vehicle_v2); this script adds more seeds.
# Each run takes ~40-90 min on an Apple M1, so the script keeps the machine awake on macOS.
#
# Usage (from the repo root):
#   bash scripts/run_seed_study.sh 1 2        # seeds to add
# Then compare all runs on the same val/test labels with scripts/compare_runs.py (see README §5).
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
SEEDS=("${@:-1 2}")
KEEP_AWAKE=()
command -v caffeinate >/dev/null 2>&1 && KEEP_AWAKE=(caffeinate -i)

for seed in ${SEEDS[@]}; do
  for version in v1 v2; do
    if [[ "$version" == "v1" ]]; then data="data/dataset_v1/data.yaml"; else data="data/dataset/data.yaml"; fi
    name="yolo11n_vehicle_${version}_s${seed}"
    echo "=== $(date +%H:%M:%S) start ${name} (${data})"
    "${KEEP_AWAKE[@]}" "$PYTHON" scripts/train.py --override "name=${name}" "seed=${seed}" "data=${data}" \
      > "runs/train_${name}.log" 2>&1
    echo "=== $(date +%H:%M:%S) done ${name}"
  done
done
