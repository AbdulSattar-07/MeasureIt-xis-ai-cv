# Definition of Done — Measure It

Written before coding. Tick with evidence (number, path, or commit link) at the end. A ticked box with no evidence counts as unticked.

| Done | Criterion | Evidence |
|------|-----------|----------|
| ☐ | Reprojection error reported, with calibration image count (≥20) | path: `calibration/calibration_report.json` — error: ___ px, n=___ |
| ☐ | Camera matrix + distortion coefficients saved and loadable by code | path: `calibration/camera_params.npz` |
| ☐ | ≥70 own labelled images (undistorted), object + masks | count: ___; tool: ___ |
| ☐ | Train/val/test split file lists committed **before** training started | commit: ___ ; files under `dataset/splits/` |
| ☐ | Segmentation model is **not** Ultralytics YOLO / Roboflow; choice justified in Plan | model: DeepLabV3-MobileNetV3 |
| ☐ | Test metrics reported: mask IoU, mAP@0.5, mAP@0.5:0.95, precision, recall | path: `models/metrics_test.json` |
| ☐ | One command runs on a **new** image and prints width & height in mm (+ confidence) | cmd: `python -m inference.measure --image ...` |
| ☐ | Handles object-not-found and reference-not-found cleanly | shown in README / sample runs |
| ☐ | Accuracy table for ≥10 instances; raw outputs saved | path: `measurement/data/accuracy_table.csv` |
| ☐ | MAE (mm) and mean % error reported | see Objectives MO-1 |
| ☐ | Raw vs undistorted errors side by side (Item 6) | path: `measurement/data/undistortion_proof.csv` |
| ☐ | MO-1 / MO-2 met **or** missed with reason written | Objectives.md |
| ☐ | Refuse-to-answer path exists (low confidence / missing ref) | code + example |
| ☐ | Weights SHA-256 in README (or release link) | README |
| ☐ | Loom ≤5 min answering K1–K4 without notes | link: ___ |
| ☐ | Everything not finished is listed below | |

## Not finished (fill at end)

- _
- _

## Stop decision

Work is “done” when floor items 1–4 work end-to-end on a new photo, MO-1 table exists (even if target missed), and the three docs + README are honest. Incomplete later items are listed above, not hidden.
