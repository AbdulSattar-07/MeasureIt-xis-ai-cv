# Definition of Done — Measure It

Written before coding. Evidence filled at end.

| Done | Criterion | Evidence |
|------|-----------|----------|
| ☑ | Reprojection error reported, with calibration image count (≥20) | `calibration/calibration_report.json` — RMS **2.005 px**, n_used=**19**/20 |
| ☑ | Camera matrix + distortion coefficients saved and loadable by code | `calibration/camera_params.npz` |
| ☑ | ≥70 own labelled images (undistorted), object + masks | **82** undistorted + masks; tool: heuristic auto-label (`scripts/auto_label_dataset.py`), preview `outputs/label_preview/` |
| ☑ | Train/val/test split file lists committed **before** training started | commit **`feef33c`**; `dataset/splits/{train,val,test}.txt` (55/16/7 quality-filtered) |
| ☑ | Segmentation model is **not** Ultralytics YOLO / Roboflow; choice justified in Plan | DeepLabV3-MobileNetV3 (torchvision) |
| ☑ | Test metrics reported: mask IoU, mAP@0.5, mAP@0.5:0.95, precision, recall | `models/metrics_test.json` |
| ☑ | One command runs on a **new** image and prints width & height in mm (+ confidence) | `python -m inference.measure --image …` → e.g. `outputs/overlay_smoke.png` / `outputs/measure_smoke.json` |
| ☑ | Handles object-not-found and reference-not-found cleanly | refuse reasons in `inference/measure.py`; demo `outputs/refuse_no_object.json` (`tilted_or_bad_plane`) |
| ☑ | Accuracy table for ≥10 instances; raw outputs saved | `measurement/data/accuracy_table.csv` — **n_ok=16**; raw runs under `measurement/data/raw_runs/` |
| ☑ | MAE (mm) and mean % error reported | Objectives MO-1 + `accuracy_table.summary.json` |
| ☑ | Raw vs undistorted errors side by side (Item 6) | `measurement/data/undistortion_proof.csv` + `.summary.json` |
| ☑ | MO-1 / MO-2 met **or** missed with reason written | Both **missed** with reasons in `docs/Objectives.md` |
| ☑ | Refuse-to-answer path exists (low confidence / missing ref) | thresholds in CLI; axis-disagree / low-conf refuses |
| ☑ | Weights SHA-256 in README (or release link) | README + `models/checkpoints/best.pt` |
| ☐ | Loom ≤5 min answering K1–K4 without notes | link: _paste Loom URL here_ |
| ☑ | Everything not finished is listed below | |

## Not finished

- **Loom recording (K1–K4)** — required gate; without it assessment auto-rejects.
- **Physical calliper GT** — current GT uses label-mask + ID-1 card scale (documented). Replace with calliper for stronger MO-1 if time.
- **Item 8 full break-condition sweep** — skipped (declared in Plan).
- **Hand-polished CVAT masks** — auto labels only; explains MO-2 miss.

## Stop decision

Floor items 1–4 run end-to-end. MO-1/MO-2 missed honestly. Submit blocked only on **Loom link** (+ public PR if not yet opened).
