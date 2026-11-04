# Measure It — Segmentation and Metric Measurement

xis.ai AI / Computer Vision technical assessment (v2.0).  
Branch: **`dev-test01`**

Photograph a USB flash drive (or similar), segment it, and report **width and height in millimetres** using a coplanar credit-card reference.

## Why this model

**Torchvision DeepLabV3-MobileNetV3** — semantic segmentation with a pretrained backbone, fine-tuned on **our own** masks.  
Not Ultralytics YOLO. Not Roboflow. Chosen for a clear segmentation API, lighter CPU/Colab footprint than Mask R-CNN, and straightforward IoU / precision / recall evaluation.

## Hardware (fill after first real run)

- OS:
- CPU:
- GPU: none on the assessment laptop (use Colab for training if needed)
- RAM:
- Camera model / locked resolution:
- Weights: models/checkpoints/best.pt  
- Weights SHA-256: BFFEE3EF71257031F6189F79968A853F3CB5289F16251FF57BA5EBD909FCF867

## Dataset summary (fill after capture)

- Object: USB flash drive (see `dataset/OBJECT.md`)
- Reference: credit card (calliper sizes in `measurement/data/reference.json`)
- Image count:
- Split: 70 / 20 / 10 via `dataset/splits/{train,val,test}.txt` (committed before training)
- Labelling tool: CVAT / Label Studio / …

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline (run in order)

### 1. Calibrate (≥20 checkerboard images)

1. Print a checkerboard (≥7×9 inner corners), tape flat, measure one square with a calliper.
2. Update `calibration/board_config.json` (`square_size_mm`, corner counts).
3. Lock phone focus / zoom / resolution. Capture ≥20 board photos → `calibration/images/`.
4. Run:

```bash
python calibration/calibrate.py
python calibration/undistort.py --input path/to/test.jpg --output outputs/undistorted_check.jpg
```

You are ready when straight edges stay straight after undistortion.

### 2. Dataset (≥70 labelled images)

1. Capture object + reference photos with the **same locked camera** → `dataset/images/raw/`.
2. Undistort:

```bash
python scripts/undistort_dataset.py
```

3. Label masks (0=bg, 1=object, 2=reference) into `dataset/labels/` (see `dataset/labels/README.md`).
4. Freeze splits **before training** and commit them:

```bash
python scripts/make_splits.py
git add dataset/splits && git commit -m "Freeze train/val/test split lists before training."
```

### 3. Train + evaluate

```bash
python -m models.train
python -m models.evaluate --split val
python -m models.evaluate --split test
```

Test split is report-only — never use it for tuning or checkpoint pick.

### 4. One command: image → millimetres

```bash
python -m inference.measure --image path/to/new_photo.jpg --out outputs/overlay.png
```

Prints JSON with `width_mm`, `height_mm`, `confidence`, or refuses with `object_not_found` / `reference_not_found` / low-confidence / tilted-plane reasons.

**Pixels → mm** lives in `inference/geometry.py` (`mm_per_pixel_from_reference`, `measure_object_mm`). Valid only when object and reference are coplanar and the image is undistorted with the locked calibration.

### 5. Accuracy vs calliper

1. Measure ≥10 instances with a calliper; fill `measurement/data/ground_truth.csv` (from the template).
2. Run:

```bash
python measurement/evaluate_accuracy.py
python measurement/undistortion_proof.py
```

## Documents

| Path | Role |
|------|------|
| `docs/Plan.md` | Order of work, skips, decision record |
| `docs/Objectives.md` | MO-1 / MO-2 measurable targets |
| `docs/Definition_of_Done.md` | Checklist with evidence |

## Repo layout

```
calibration/   images, calibrate.py, camera_params.npz
dataset/       images, labels, splits, OBJECT.md
models/        DeepLabV3 train/eval (weights linked + SHA-256)
inference/     image-to-millimetres command
measurement/   accuracy + undistortion proof
docs/          Plan, Objectives, Definition of Done
```

## Submission checklist

- [ ] Public GitHub repo, branch `dev-test01`, progressive commits
- [ ] PR `dev-test01` → `main`
- [ ] Three docs + README
- [ ] Weights + SHA-256
- [ ] Loom ≤5 min (K1–K4)
