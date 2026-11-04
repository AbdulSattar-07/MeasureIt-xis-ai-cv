"""Pixel silhouette → millimetres using a coplanar reference of known size.

K1 (Loom): this module is where pixels become millimetres.
The conversion is valid only when the reference and object lie in the same
plane (approximately face-on), the reference size is measured correctly, and
the image is undistorted with the locked calibration.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def load_reference(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def largest_contour(mask: np.ndarray):
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def oriented_bbox_wh(mask: np.ndarray) -> tuple[float, float, tuple] | None:
    contour = largest_contour(mask)
    if contour is None or cv2.contourArea(contour) < 10:
        return None
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect
    # minAreaRect width/height ordering is arbitrary; return larger as length-like
    width_px = float(max(w, h))
    height_px = float(min(w, h))
    return width_px, height_px, ((cx, cy), (w, h), angle)


def mm_per_pixel_from_reference(
    ref_mask: np.ndarray,
    ref_width_mm: float,
    ref_height_mm: float,
) -> tuple[float, dict] | None:
    dims = oriented_bbox_wh(ref_mask)
    if dims is None:
        return None
    width_px, height_px, rect = dims
    # Use the long side of the card against the known long side in mm
    long_mm = max(ref_width_mm, ref_height_mm)
    short_mm = min(ref_width_mm, ref_height_mm)
    scale_long = long_mm / max(width_px, 1e-6)
    scale_short = short_mm / max(height_px, 1e-6)
    # Average the two axes; large disagreement means tilt / bad mask
    scale = 0.5 * (scale_long + scale_short)
    axis_disagree_pct = abs(scale_long - scale_short) / max(scale, 1e-6) * 100.0
    meta = {
        "ref_width_px": width_px,
        "ref_height_px": height_px,
        "mm_per_px_long": scale_long,
        "mm_per_px_short": scale_short,
        "mm_per_px": scale,
        "axis_disagree_pct": axis_disagree_pct,
        "ref_rect": {
            "center": [float(rect[0][0]), float(rect[0][1])],
            "size": [float(rect[1][0]), float(rect[1][1])],
            "angle": float(rect[2]),
        },
    }
    return scale, meta


def measure_object_mm(
    object_mask: np.ndarray,
    mm_per_px: float,
) -> tuple[float, float, dict] | None:
    dims = oriented_bbox_wh(object_mask)
    if dims is None:
        return None
    width_px, height_px, rect = dims
    width_mm = width_px * mm_per_px
    height_mm = height_px * mm_per_px
    meta = {
        "object_width_px": width_px,
        "object_height_px": height_px,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "object_rect": {
            "center": [float(rect[0][0]), float(rect[0][1])],
            "size": [float(rect[1][0]), float(rect[1][1])],
            "angle": float(rect[2]),
        },
    }
    return width_mm, height_mm, meta
