# Objectives — Measure It

Hardware / environment will be filled with real values after first run (CPU, RAM, OS, camera model, image resolution). Every reported number must be pasted from this machine’s raw output.

---

## MO-1 — Measurement accuracy (required)

| Field | Entry |
|-------|--------|
| **What is measured** | Absolute error between system-reported object width/height (mm) and calliper ground truth on held-out measurement photos. |
| **How** | Script `measurement/evaluate_accuracy.py` compares system CLI output to a hand-entered calliper CSV for ≥10 instances. Reports per-instance table, mean, median, worst-case absolute error (mm) and mean percentage error. |
| **Target** | Mean absolute error ≤ **2.0 mm** on both width and height, over ≥10 instances (same photos every run). |
| **Conditions** | Phone camera with locked focus/zoom/resolution; object and credit-card reference coplanar and fully visible; indoor light; tilt within ~15° of face-on; undistorted images; test/measurement images never used for training. |
| **Not included** | Photos where the reference is cut off, mask confidence below refuse threshold, or the card and object are clearly on different planes. |
| **Why this target** | A USB drive is ~60–70 mm long; 2 mm ≈ 3% — useful for a demo industrial “is this roughly the right size” check without claiming metrology-lab precision on a phone. |
| **Result** | _TBD — paste raw table + mean/median/worst_ |

---

## MO-2 — Mask quality on frozen test split (choice)

| Field | Entry |
|-------|--------|
| **What is measured** | Mean mask IoU of the object class on the committed test split. |
| **How** | `models/evaluate.py` runs once on the test list after training; test images never used for tuning or checkpoint pick. |
| **Target** | Mean object-mask IoU ≥ **0.80** on the test split. |
| **Conditions** | Same camera lock as calibration; undistorted images; fixed `dataset/splits/test.txt`. |
| **Not included** | Validation metrics used for early stopping (val may guide training; test is report-only). |
| **Why this target** | Below ~0.8 IoU, silhouette width/height in pixels becomes unstable enough that MO-1 is unlikely even with perfect scale. |
| **Result** | _TBD — paste IoU, mAP@0.5, mAP@0.5:0.95, precision, recall_ |

---

## Supporting numbers (not separate MOs)

| Metric | Target / note | Result |
|--------|----------------|--------|
| Calibration reprojection error | Report RMS (pixels) and image count (≥20) | _TBD_ |
| Inference latency (MO-2 related timing) | Median of 5 runs on one image; report spread | _TBD_ |
| Undistortion proof (Item 6) | MAE raw vs MAE undistorted, same 10 instances | _TBD_ |

### Machine card (fill before reporting)

- OS:
- CPU:
- GPU: none / Colab —
- RAM:
- Camera model / resolution:
- Library pins: see `requirements.txt`
