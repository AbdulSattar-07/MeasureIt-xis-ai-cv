"""Draft segmentation masks for earbuds (class 1) + white card (class 2).

Heuristic colour/geometry labelling so the pipeline can proceed.
Review/fix important images before trusting metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def largest_components(mask: np.ndarray, keep: int = 2) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return np.zeros_like(mask)
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
    areas.sort(key=lambda t: t[1], reverse=True)
    out = np.zeros_like(mask)
    for i, _ in areas[:keep]:
        out[labels == i] = 1
    return out


def label_one(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # White / light card
    light = cv2.inRange(hsv, (0, 0, 150), (180, 70, 255))
    light = cv2.morphologyEx(light, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    light = cv2.morphologyEx(light, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
    light = largest_components(light, keep=1)

    # Dark earbuds / case (avoid pure black background noise with area filter)
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 70))
    # remove very bright fabric noise
    dark = cv2.bitwise_and(dark, cv2.bitwise_not(light * 255))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    dark = largest_components(dark, keep=2)

    # Prefer dark blob that is compact (case-like), not huge fabric patches
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
    object_mask = np.zeros((h, w), np.uint8)
    candidates = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if area < 0.002 * h * w or area > 0.35 * h * w:
            continue
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > 4.5:
            continue
        candidates.append((area, i))
    candidates.sort(reverse=True)
    for _, i in candidates[:2]:
        object_mask[labels == i] = 1

    # If card empty, try gray card fallback
    if light.sum() < 0.005 * h * w:
        gray_card = cv2.inRange(gray, 160, 245)
        gray_card = cv2.morphologyEx(gray_card, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
        light = largest_components(gray_card, keep=1)

    mask = np.zeros((h, w), np.uint8)
    mask[object_mask > 0] = 1
    mask[light > 0] = 2
    # object wins overlaps slightly for case-on-card rare cases: keep card where only light
    return mask


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=root / "dataset" / "images" / "raw")
    parser.add_argument("--out", type=Path, default=root / "dataset" / "labels")
    parser.add_argument("--preview", type=Path, default=root / "outputs" / "label_preview")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.preview.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    ok = weak = 0
    for path in sorted(args.images.iterdir()):
        if path.suffix.lower() not in exts:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        mask = label_one(img)
        out_path = args.out / f"{path.stem}.png"
        cv2.imwrite(str(out_path), mask)

        obj = int((mask == 1).sum())
        ref = int((mask == 2).sum())
        status = "ok" if obj > 500 and ref > 500 else "weak"
        if status == "ok":
            ok += 1
        else:
            weak += 1

        # preview overlay
        vis = img.copy()
        vis[mask == 1] = (0.5 * vis[mask == 1] + np.array([0, 200, 0]) * 0.5).astype(np.uint8)
        vis[mask == 2] = (0.5 * vis[mask == 2] + np.array([0, 160, 255]) * 0.5).astype(np.uint8)
        cv2.putText(vis, f"{status} obj={obj} ref={ref}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(str(args.preview / f"{path.stem}.jpg"), vis)
        print(f"{path.name}: {status} obj_px={obj} ref_px={ref}")

    print(f"done ok={ok} weak={weak} labels->{args.out} preview->{args.preview}")


if __name__ == "__main__":
    main()
