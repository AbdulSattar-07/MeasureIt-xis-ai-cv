# Plan — Measure It (Segmentation + Metric Measurement)

**Project:** Measure It — Segmentation and Metric Measurement  
**Branch:** `dev-test01`  
**Budget:** 8 hours of work (submit within 24 hours)  
**Object:** Ronin-style earbuds charging case (rigid)  
**Reference:** Bank/ATM card (ID-1, 85.60 × 53.98 mm), preferably back side in frame  
**Camera:** Phone camera, focus / zoom / resolution locked before calibration and kept locked after  
**Model choice:** Torchvision DeepLabV3-MobileNetV3 — not Ultralytics YOLO, not Roboflow. Pretrained backbone fine-tuned on our own masks. Chosen for a real semantic-segmentation API, lighter CPU footprint than Mask R-CNN, and straightforward IoU / precision / recall evaluation.

---

## Order of work (and rough time)

| Step | What | Hours | Notes |
|------|------|-------|-------|
| 0 | Plan + DoD + Objectives + empty split lists committed first | 0.25 | Done (`b182930`) |
| 1 | ≥20 checkerboard images (on-screen board), OpenCV calibrate | 1.0 | Done — 19/20 used, ~2.0 px RMS |
| 2 | ≥70 own photos, undistort, label, freeze splits before train | 2.0 | Done — 82 images; splits in `feef33c` |
| 3 | Train DeepLabV3; test metrics once | 1.5 | Done — test IoU ~0.53 |
| 4 | One CLI image→mm + refuse paths | 1.0 | Done |
| 5 | Accuracy table ≥10 | 0.75 | Done — n_ok=16; MO-1 missed |
| 6 | Raw vs undistorted errors | 0.5 | Done — undistortion helps on this set |
| 7 | Fill Objectives with pasted numbers | 0.5 | Done |
| 8–9 | Break condition sweep / polish refuse | 0.5 | **Skipped** (declared) |
| 10 | Decision record + Loom | 0.5 | Decision record done; **Loom pending** |

**Skipped as planned if time ran out:** Item 8 full sweep; Item 9 beyond working refuse path.

---

## Approach vs rejected approach

| Took | Rejected | Cost of rejecting |
|------|----------|-------------------|
| DeepLabV3-MobileNetV3 (torchvision) | Ultralytics YOLO-seg / Roboflow | Forbidden — auto-reject |
| On-screen checkerboard photographed by phone + ATM card scale for square mm | Trust PDF print scale / skip undistortion | No ruler available; card scale documented; still real OpenCV calib |
| Heuristic auto-masks then train | Wait for full CVAT polish | Saved time; cost = MO-2/MO-1 miss (accepted honestly) |
| Fixed split commit before training | Random split at train time | Required for reproducibility |

---

## Decision record (closed)

- **Approach taken:** Phone + on-screen checkerboard calibration; DeepLabV3-MobileNetV3; card coplanar scale; auto-label draft masks; progressive commits on `dev-test01`.
- **Approach rejected:** YOLO/Roboflow; downloaded object images; inventing calliper numbers; one mega-commit.
- **What rejecting it cost:** Lower mask IoU and large mm MAE vs label-card GT; longer CPU training; need Loom still from human.
- **Plan changes:** Object became earbuds case (not USB); square size estimated via UBL card on screen (~24.5 mm) instead of printed calliper square; GT for MO-1 interim = label+ID-1 scale (calliper preferred later).

---

## Stop rule

Honest unfinished beats fake millimetres. Floor 1–4 work. MO targets missed with reasons. Only **Loom** remains as hard submit gate from the candidate.
