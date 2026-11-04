# Loom ≤5 min — what to say (do NOT read notes on camera; memorize)

## Timing
1. **~1 min:** New phone photo of earbuds+card → run  
   `python -m inference.measure --image NEW.jpg --out outputs/loom.png`  
   Then show calliper/ruler on the real object.
2. **~1.5 min:** Walk code on that image: undistort → model → reference → mm.
3. **~2.5 min:** Answer K1–K4 out loud.

## K1 — Which file/line turns pixels into mm?
- File: `inference/geometry.py`
- Functions: `mm_per_pixel_from_reference`, `measure_object_mm`
- Valid only if: object and reference **same plane**, card size correct, image **undistorted**, both fully visible.

## K2 — Where is undistortion? How much did skipping it change error?
- Applied in `inference/measure.py` via `calibration.undistort.undistort_bgr` (unless `--skip-undistort`).
- Our numbers: width MAE raw **28.24 mm** → undistorted **22.14 mm**; height **14.00 → 13.45 mm** (`measurement/data/undistortion_proof.summary.json`).

## K3 — One thing you would reverse, and cost?
- Example: using **auto heuristic labels** instead of careful CVAT masks.
- Cost: test IoU stuck ~0.53; MO-1 mm errors large; time saved ~hours of labelling.

## K4 — One thing you did not understand, and what you did?
- Example: DeepLab aux head channel count on MobileNet / BatchNorm batch=1 crash.
- What you did: fixed aux `in_channels`, set `drop_last=True`, re-ran training; checked torchvision source + error message.
