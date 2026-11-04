# Objectives — Measure It

Every reported number below is from this machine’s raw outputs (pasted / linked).

---

## MO-1 — Measurement accuracy (required)

| Field | Entry |
|-------|--------|
| **What is measured** | Absolute error between system-reported object width/height (mm) and ground truth on measurement photos. |
| **How** | `measurement/evaluate_accuracy.py` vs `measurement/data/ground_truth.csv`. GT built with **label masks + ID-1 card scale** (`measurement/build_gt_from_labels.py`) — model-independent. Prefer replace with physical calliper when available (see notes). |
| **Target** | Mean absolute error ≤ **2.0 mm** on width and height, over ≥10 instances. |
| **Conditions** | Phone camera; earbuds case + bank card coplanar; indoor light; undistorted path; frozen test unused for training. |
| **Not included** | Refused predictions (low conf / tilted plane); non-`both_` images. |
| **Why this target** | ~2 mm is a useful “rough industrial size check” for a ~50–60 mm case without claiming lab metrology. |
| **Result** | **MISSED.** Undistorted path (`measurement/data/accuracy_table.summary.json` / `accuracy_undistorted.summary.json`): **n_ok=16**, MAE width **22.14 mm**, MAE height **13.45 mm**, median abs W **6.00 mm**, median abs H **8.22 mm**, worst W **111.35 mm**, worst H **62.13 mm**, mean % W **22.2%**, mean % H **25.2%**. Reason: auto-draft masks + weak segmentation (test IoU ~0.53) make silhouette mm unstable; GT is card-scale-from-labels not calliper. Honest miss preferred over inventing calliper numbers. |

---

## MO-2 — Mask quality on frozen test split (choice)

| Field | Entry |
|-------|--------|
| **What is measured** | Mean mask IoU (classes 1+2) on committed test split. |
| **How** | `python -m models.evaluate --split test` → `models/metrics_test.json` |
| **Target** | Mean object-mask IoU ≥ **0.80** (reported as mean over classes 1–2). |
| **Conditions** | Undistorted images; `dataset/splits/test.txt` frozen before training (`feef33c`). |
| **Not included** | Val metrics used only for early stopping / checkpoint pick. |
| **Why this target** | Below ~0.8 IoU, width/height from masks become unstable. |
| **Result** | **MISSED.** From `models/metrics_test.json`: mask_iou_mean **0.533**, precision **0.838**, recall **0.569**, mAP@0.5 **0.643**, mAP@0.5:0.95 **0.300**, n=7. Per-class IoU object **0.506**, reference **0.560**. Reason: heuristic auto-labels, not hand-polished CVAT masks; CPU fine-tune only 20 epochs. |

---

## Supporting numbers

| Metric | Target / note | Result |
|--------|----------------|--------|
| Calibration reprojection error | Report RMS + count ≥20 | **19/20** images used; OpenCV RMS **2.005 px** (`calibration/calibration_report.json`) |
| Inference latency | Median of 5 runs + spread | median **9766 ms**, min **7719**, max **19755** on `both_050.jpeg` CPU (`measurement/data/latency.json`) |
| Undistortion proof (Item 6) | Raw vs undistorted MAE | Width MAE raw **28.24 mm** → undistorted **22.14 mm**; height **14.00 → 13.45 mm** (`measurement/data/undistortion_proof.summary.json`). Undistortion reduces error on this set — numbers, not theory alone. |

### Machine card

- OS: Windows-11-10.0.26300-SP0
- CPU: Intel64 Family 6 Model 154 Stepping 4
- GPU: none (CPU PyTorch)
- RAM: ~8.3 GB
- Camera: phone (locked zoom/focus during capture); calib frames 1600×1200
- Library pins: see `requirements.txt`
- Weights SHA-256: `BFFEE3EF71257031F6189F79968A853F3CB5289F16251FF57BA5EBD909FCF867`
