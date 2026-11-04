"""Estimate checkerboard square size (mm) using an ATM/ID-1 card as the ruler.

Take ONE photo where:
  - the on-screen checkerboard is visible, AND
  - your ATM card (peeth OK) lies against the screen near the board
    so both are clear.

Usage:
  python scripts/estimate_square_mm_with_card.py --image path/to/photo.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CARD_LONG_MM = 85.60  # ID-1 / ATM card long side


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--corners-x", type=int, default=9)
    parser.add_argument("--corners-y", type=int, default=6)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "calibration" / "board_config.json",
    )
    parser.add_argument("--update-config", action="store_true")
    args = parser.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"Cannot read {args.image}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray, (args.corners_x, args.corners_y), None
    )
    if not found:
        raise SystemExit(
            "Checkerboard corners not found. Retake: full board visible, less glare."
        )
    # average adjacent-corner distance in pixels (= one square)
    pts = corners.reshape(-1, 2)
    # reshape to grid
    grid = pts.reshape(args.corners_y, args.corners_x, 2)
    dists = []
    for y in range(args.corners_y):
        for x in range(args.corners_x - 1):
            dists.append(np.linalg.norm(grid[y, x + 1] - grid[y, x]))
    for y in range(args.corners_y - 1):
        for x in range(args.corners_x):
            dists.append(np.linalg.norm(grid[y + 1, x] - grid[y, x]))
    square_px = float(np.median(dists))

    # Detect card as large light rectangle; use long side in pixels
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    light = cv2.inRange(hsv, (0, 0, 140), (180, 90, 255))
    light = cv2.morphologyEx(light, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    contours, _ = cv2.findContours(light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise SystemExit("Card not found. Place ATM card clearly next to the board.")
    card = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(card)
    (cw, ch) = rect[1]
    card_long_px = float(max(cw, ch))
    if card_long_px < 10:
        raise SystemExit("Card detection too small / failed.")

    square_mm = CARD_LONG_MM * (square_px / card_long_px)
    print(f"square_px={square_px:.2f}")
    print(f"card_long_px={card_long_px:.2f} (assumed {CARD_LONG_MM} mm)")
    print(f"estimated square_size_mm={square_mm:.2f}")

    if args.update_config:
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        cfg["square_size_mm"] = round(square_mm, 2)
        cfg["notes"] = (
            f"Estimated via ATM card ({CARD_LONG_MM} mm) in {args.image.name}; "
            "not a printer PDF scale."
        )
        args.config.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"updated {args.config}")


if __name__ == "__main__":
    main()
