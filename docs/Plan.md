# Plan — Measure It (Segmentation + Metric Measurement)

**Project:** Measure It — Segmentation and Metric Measurement  
**Branch:** `dev-test01`  
**Budget:** 8 hours of work (submit within 24 hours)  
**Object (default):** USB flash drive (rigid, easy to photograph many times, ~60–70 mm long)  
**Reference:** Credit card / bank card in the same plane (measured with calliper; nominal 85.60 × 53.98 mm — we still measure ourselves)  
**Camera:** Phone camera, focus / zoom / resolution locked before calibration and kept locked for every image after  
**Model choice (pre-decided):** Torchvision DeepLabV3-MobileNetV3 — not Ultralytics YOLO, not Roboflow. Pretrained backbone fine-tuned on our own masks. Chosen because it is a real semantic-segmentation architecture, lighter than Mask R-CNN for CPU/Colab, and easy to evaluate with IoU / precision / recall.

---

## Order of work (and rough time)

| Step | What | Hours | Notes |
|------|------|-------|-------|
| 0 | This Plan + Definition of Done + Objectives draft + empty/fixed split file lists committed | 0.25 | **Must be first commit — before any code** |
| 1 | Print / tape checkerboard, measure square with calliper, capture ≥20 calibration images, run OpenCV calibrate, save params + reprojection error | 1.0 | Floor item 1 |
| 2 | Capture ≥70 object photos (varied pose/distance/light), undistort all, label segmentation masks (object + reference), freeze 70/20/10 split file lists | 2.0 | Floor item 2 |
| 3 | Train DeepLabV3 on train/val only; evaluate once on test: mask IoU, mAP@0.5, mAP@0.5:0.95, precision, recall | 1.5 | Floor item 3 (training wait does not count) |
| 4 | One CLI: image → undistort → segment → find reference → width/height mm + confidence; handle object/ref not found | 1.0 | Floor item 4 |
| 5 | Calliper ground truth on ≥10 instances; per-instance table, MAE (mm), mean % error | 0.75 | Item 5 |
| 6 | Same 10 on raw vs undistorted; side-by-side errors + explanation from numbers | 0.5 | Item 6 |
| 7 | Hit or miss MO-1 / MO-2; fill Objectives with pasted raw outputs | 0.5 | Item 7 |
| 8–9 | One break condition + refuse-to-answer threshold (if time) | 0.5 | Items 8–9 |
| 10 | Decision record at end of this Plan; Loom ≤5 min (K1–K4) | 0.5 | Item 10 + recording |

**Already decided to skip if time runs out:** Item 8 full sweep and Item 9 polished thresholding after a working refuse path. Floor (1–4) is non-negotiable.

---

## Approach vs rejected approach

| Took | Rejected | Cost of rejecting |
|------|----------|-------------------|
| DeepLabV3-MobileNetV3 (torchvision), fine-tuned on our masks | Ultralytics YOLO-seg / Roboflow train | Forbidden by brief; would be auto-reject |
| Phone + locked focus/zoom + OpenCV checkerboard | Assume pinhole with no undistortion | Distortion would bias mm conversion; Item 6 would fail |
| Same-plane rigid reference (card) for px→mm | Assume fixed camera height / known focal length only | Fragile if distance changes; harder to prove |
| Fixed split file lists committed before training | Random split at train time | Violates reproducibility / auto-reject risk |

---

## Decision record (fill at end)

- **Approach taken:** _TBD after work_
- **Approach rejected:** _TBD_
- **What rejecting it cost:** _TBD_
- **Plan changes:** _Note any deviation and why_

---

## Stop rule

An honest unfinished submission beats a rushed wrong millimetre. If floor items 1–4 are not solid, stop and document what was not reached in Definition of Done — do not invent accuracy numbers.
