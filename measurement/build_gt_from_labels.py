"""Build ground-truth mm for ≥10 both_* photos from LABEL masks + known card size.

This is independent of the trained model (uses dataset/labels only).
Physical scale: ID-1 card 85.60 x 53.98 mm.
Replace with calliper CSV when a physical calliper reading is available.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference.geometry import measure_object_mm, mm_per_pixel_from_reference

CARD_W, CARD_H = 85.60, 53.98


def main() -> None:
    labels = ROOT / "dataset" / "labels"
    images = ROOT / "dataset" / "images" / "undistorted"
    out = ROOT / "measurement" / "data" / "ground_truth.csv"
    rows = []
    for path in sorted(images.glob("both_*.png")):
        mask = cv2.imread(str(labels / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        obj = (mask == 1).astype(np.uint8)
        ref = (mask == 2).astype(np.uint8)
        if obj.sum() < 500 or ref.sum() < 500:
            continue
        scale = mm_per_pixel_from_reference(ref, CARD_W, CARD_H)
        if scale is None:
            continue
        mm_per_px, meta = scale
        if meta["axis_disagree_pct"] > 25:
            continue
        measured = measure_object_mm(obj, mm_per_px)
        if measured is None:
            continue
        w, h, _ = measured
        rows.append(
            {
                "image": path.name.replace(".png", ".jpeg"),  # raw name for evaluator
                "gt_width_mm": round(w, 2),
                "gt_height_mm": round(h, 2),
                "notes": "label_mask+ID1_card_scale (not calliper); model-independent",
            }
        )
        if len(rows) >= 12:
            break

    if len(rows) < 10:
        raise SystemExit(f"Only {len(rows)} usable GT rows; need ≥10")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["image", "gt_width_mm", "gt_height_mm", "notes"]
        )
        w.writeheader()
        w.writerows(rows[:12])
    meta = {
        "n": len(rows[:12]),
        "method": "label_masks_plus_ID1_card",
        "card_mm": [CARD_W, CARD_H],
        "warning": "Prefer replacing with physical calliper readings when available.",
    }
    (out.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
