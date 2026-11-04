# Labels

Use CVAT, Label Studio, or similar. Export **per-image PNG masks** (preferred) or COCO and convert.

## Classes (pixel values in mask PNG)

| Value | Class |
|------:|-------|
| 0 | background |
| 1 | object (USB drive / measured item) |
| 2 | reference (credit card) |

## Naming

For each undistorted image `dataset/images/undistorted/img_001.png` save:

`dataset/labels/img_001.png` (same stem, single-channel mask)

## Rules

- Mask the full visible silhouette of object and reference.
- Object and reference must appear in every **measurement** photo; training photos may omit reference only if you still train object well — for this assessment keep **both** in most images so the inference path is realistic.
- Never put test-split images into training.
