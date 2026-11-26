# Vehicle Detection Dataset and YOLO Model (CVOps Take-Home Test)

Training-ready YOLO object-detection dataset built from three raw traffic videos, plus a fine-tuned
YOLO11n detector and its evaluation. Target classes (Indonesian names, as specified):

| id | class   | covers |
|----|---------|--------|
| 0  | `mobil` | passenger cars: sedan, hatchback, SUV, MPV/minivan, **pickup**, **van / panel van** |
| 1  | `truk`  | goods vehicles with a cargo area separate from the cab: box, container, dump, car-carrier, light cab-over trucks |
| 2  | `bus`   | bus-bodied passenger vehicles: city bus, coach, bus-bodied minibus |

> The dataset and trained weights are not in this repository (submitted separately).
> The full write-up of results and limitations is in [`REPORT.md`](REPORT.md).

---

## 1. Setup (Python 3.10)

```bash
uv venv --python 3.10 .venv            # or: python3.10 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

Place the raw videos in `datasets/` as `Video_1.mp4`, `Video_2.mp4`, `Video_3.mp4`
(the provided `Video_2(1).mp4` was renamed to `Video_3.mp4`; it is a different video).
Pretrained weights (`yolo11n/m/x.pt`, CLIP) are downloaded automatically on first use.
All scripts run from the repo root; `--device mps` is the default (use `cpu` or `0` for CUDA).

## 2. Repository layout

```
configs/train.yaml          all training hyper-parameters (single source of truth)
metadata/                   frames_manifest.csv (why each frame was picked), split_assignment.csv (locked split)
scripts/
  common.py                 class names + COCO→project class mapping
  inspect_videos.py         contact sheets for a first look at each video
  extract_frames.py         quality-filtered, class-aware frame sampling
  compare_teachers.py       YOLO11m vs YOLO11x as auto-label teacher
  autolabel.py              pre-labels: full-image + tiled (SAHI-style) inference
  reclassify_clip.py        zero-shot CLIP correction of teacher classes (truk→mobil)
  make_split.py             time-block train/val/test split + review order
  json_to_yolo.py           X-AnyLabeling JSON → YOLO txt after manual review
  label_diff.py             statistics of the manual review (pre-label vs reviewed)
  find_label_suspects.py    model-assisted label QA: flags likely label errors in train/val
  build_dataset.py          assemble + validate the final dataset and data.yaml
  dataset_stats.py          dataset statistics used in REPORT.md §1 (counts, size buckets, box sizes)
  train.py                  fine-tune YOLO from configs/train.yaml
  run_seed_study.sh         extra training runs per label version and seed (seed study)
  evaluate.py               test-set metrics, COCO baseline, AP by size, error analysis
  compare_runs.py           compare several runs on the same val/test labels (mean ± std per version)
reports/                    figures, evaluation summaries and seed_study.csv used in REPORT.md
```

## 3. Dataset preparation methodology

```
3 videos ─► frame sampling ─► pre-label (teacher + tiling) ─► CLIP class fix ─► time-block split
        ─► manual review (X-AnyLabeling) ─► YOLO labels ─► dataset build + validation
```

1. **Inspect**: `python scripts/inspect_videos.py` builds a 20-frame contact sheet per video.
   V1: static highway camera, mostly cars. V2: hand-held, ~4 camera positions, some exposure changes.
   V3: static urban camera, dense traffic, many vans, the main source of trucks and buses.
2. **Frame sampling**: `python scripts/extract_frames.py` (≈19 min on an M1):
   candidates at 4 fps → drop the 10 % blurriest frames per video (Laplacian variance) and
   frames outside brightness 40–220 → count classes per candidate with YOLO11x (selection only,
   *not* labels) → fill per-video quotas (V1 90, V2 140, V3 170) in priority order
   bus → truk → background → spread-in-time (farthest-point sampling), enforcing a minimum time gap
   (1.0 s; 0.5 s for bus/truk) and a perceptual-hash distance. Result: 397 frames, every choice and its
   reason logged in `metadata/frames_manifest.csv`. Class-aware sampling raised the number of frames
   containing a bus from ~41 to ~73.
3. **Teacher choice**: `python scripts/compare_teachers.py` compared YOLO11m and YOLO11x on 30 frames;
   YOLO11x found more small vehicles and made fewer van→bus / pickup→truck errors → chosen as teacher.
4. **Pre-labelling**: `python scripts/autolabel.py`: YOLO11x at 1280 px on the full image plus
   640-px tiles (25 % overlap) upscaled to 1280 px, merged with class-agnostic IoU/IoS suppression;
   conf ≥ 0.3, boxes with a side < 8 px dropped. COCO `car/truck/bus` are mapped to `mobil/truk/bus`.
5. **Class correction**: `python scripts/reclassify_clip.py`: every `truk`/`bus` crop is scored by
   zero-shot CLIP (ViT-B-16) against fine-grained prompts. Only the validated transition
   `truk → mobil` (pickups, vans) is applied automatically (205 boxes); disagreements involving
   `bus` are only flagged for review, because they proved unreliable.
6. **Split**: the split was computed from the first (non-tiled) pre-label statistics:
   `python scripts/autolabel.py --tile 0 --min-box-px 12 --out data/labels_auto_v1 --predictions-csv data/autolabel_predictions_v1.csv`
   then `python scripts/make_split.py --predictions data/autolabel_predictions_v1.csv` (see §4; this
   reproduces `metadata/split_assignment.csv` exactly).
7. **Manual review**: every image (train, val and test) was reviewed once in X-AnyLabeling following
   the guideline below; the test split was then reviewed a second time image by image (all 54 marked
   *Passed*). Then `python scripts/json_to_yolo.py` and
   `python scripts/label_diff.py` (review statistics in `REPORT.md`).
8. **Model-assisted label QA (rounds 2–3)**: after a first model was trained,
   `python scripts/find_label_suspects.py --weights runs/yolo11n_vehicle/weights/best.pt` compared its
   predictions with the labels on train/val only (test is excluded to avoid leaking the evaluated
   model into the ground truth). It writes a review folder plus hint images marking likely missing
   labels, wrong boxes and class mismatches. 175 images were re-reviewed and 126 boxes (≈2.2 %) changed.
   The effect on test mAP50–95 changed direction between seeds. We selected v1 on validation.
9. **Build**: `python scripts/build_dataset.py` copies images/labels per split, validates every
   label line and writes `data.yaml`. Both label versions use the same images and split:
   `python scripts/build_dataset.py --labels data/review_backup/labels_reviewed_round1 --out data/dataset_v1` (submitted) and
   `python scripts/build_dataset.py --labels data/labels_reviewed --out data/dataset` (rounds 1–3).

**Annotation guideline (visual, not functional):** a class is decided from what the camera sees.
Label every vehicle that is recognisable without guessing and whose box side is ≥ ~8 px; label the
visible part of vehicles that are ≥ ~30 % visible (truncated / occluded); include equipment mounted on a
vehicle; do not label motorcycles, bicycles, pedestrians or towed trailers. The rationale (for example, why pickups and vans are `mobil`) is in `REPORT.md` §2.

## 4. Dataset structure

```
dataset/
  data.yaml              # no `path:` key → splits resolve relative to this file (portable)
  dataset_stats.csv      # per-image class counts and split
  images/{train,val,test}/<video>_t<ms>.jpg   # e.g. v2_t0136533.jpg = Video_2 at 136.533 s
  labels/{train,val,test}/<same stem>.txt     # YOLO: "cls cx cy w h" (normalised)
```

```yaml
train: images/train
val: images/val
test: images/test
nc: 3
names: {0: mobil, 1: truk, 2: bus}
```

**Split by time blocks, not random frames.** Neighbouring frames are near-duplicates, so a random
split would leak almost identical images into test. Each video is cut into 10 equal time blocks; whole
blocks go to one split. The block→split assignment is a seeded random search (5 000 trials, seed 42)
matching 70/15/15 for images, truk boxes and bus boxes (bus weighted ×2), with every video in every split.
Frames within 1.5 s of a boundary between different splits are dropped (17). The assignment was
locked before the manual review and is stored in `metadata/split_assignment.csv`.

| split | images | mobil | truk | bus |
|-------|-------:|------:|-----:|----:|
| train | 275 | 4,558 | 213 | 52 |
| val   |  51 |   899 |  61 |  9 |
| test  |  54 |   882 |  54 | 10 |
| **total** | **380** | **6,339** | **328** | **71** |

`python scripts/dataset_stats.py --dataset data/dataset_v1` reproduces these counts and the rest of
the dataset statistics in `REPORT.md` §1, including images per video, boxes per video and class, COCO size buckets and median box sizes.

## 5. Training configuration

`python scripts/train.py --config configs/train.yaml --override data=data/dataset_v1/data.yaml`. Every value lives in
[`configs/train.yaml`](configs/train.yaml); the run folder also stores Ultralytics' resolved `args.yaml`,
`train_config.yaml` and `train_meta.json` (time, software, hardware).

| parameter | value | why |
|-----------|-------|-----|
| model | `yolo11n.pt` (COCO-pretrained) | lightweight, trains on a laptop GPU/CPU |
| imgsz | 960 | many vehicles are small; at 640 a 12-px car becomes ~4 px |
| epochs / patience | 100 / 25 | early stopping on validation fitness |
| batch | 8 | fits Apple-M1 memory at 960 px |
| optimizer / lr0 / lrf | AdamW / 0.001 / 0.01 | set explicitly instead of `auto` |
| weight_decay / warmup | 0.0005 / 3 epochs | Ultralytics defaults |
| augmentation | mosaic 1.0 (off for last 10 epochs), fliplr 0.5, HSV, scale 0.5, translate 0.1 | Ultralytics defaults |
| seed / deterministic | 42 / true | reproducibility |
| device | `mps` (Apple M1) | |

To compare label versions or measure run-to-run variance, the same config is reused with overrides, e.g.
`python scripts/train.py --override name=yolo11n_vehicle_v2_s1 seed=1 data=data/dataset/data.yaml`.
`bash scripts/run_seed_study.sh 1 2` runs these overrides for both label versions and the given seeds.
The study in `REPORT.md` §5 used seeds 42 and 1 per version; seed 2 was stopped to stay within the deadline.

## 6. Evaluation procedure

```bash
# From the repo root, place both submission ZIPs in data/submission/ first.
unzip -q data/submission/dataset_vehicle_yolo.zip -d data/submission/
unzip -q data/submission/model_yolo11n_vehicle.zip -d data/submission/
python scripts/evaluate.py --data data/submission/dataset/data.yaml \
  --weights data/submission/model/best.pt --baseline yolo11n.pt --device cpu \
  --out data/submission/evaluation
```

On the test split only (never used for training or model selection) it reports:
1. Ultralytics metrics: precision, recall, mAP50, mAP50-95 overall and per class, PR/F1 curves and
   confusion matrix (`ultralytics_val/`);
2. a COCO-style comparison of the fine-tuned model with the same architecture before fine-tuning
   (YOLO11n COCO, classes mapped car→mobil, truck→truk, bus→bus), including AP by object size
   (small < 32², medium < 96², large ≥ 96² px²);
3. error analysis at conf 0.25 / IoU 0.5: per-class precision/recall, FP / FN / class confusions
   (`errors.csv`, `confusion_matrix.csv`) and the images with most errors drawn (`error_examples/`);
4. inference speed (ms/image).

Seed / label-version study (`reports/seed_study.csv`, `REPORT.md` §5): every run is scored on the same
val and test labels, and the version is chosen on the val columns only.

```bash
python scripts/compare_runs.py --data data/dataset/data.yaml \
  --runs v1:yolo11n_vehicle v1:yolo11n_vehicle_v1_s1 v2:yolo11n_vehicle_v2 v2:yolo11n_vehicle_v2_s1 \
  --out reports/seed_study.csv
```

Results and their analysis: see [`REPORT.md`](REPORT.md).
