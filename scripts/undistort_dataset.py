"""Undistort every raw dataset image into dataset/images/undistorted/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibration.undistort import load_camera_params, undistort_bgr  # noqa: E402
import cv2  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        type=Path,
        default=ROOT / "calibration" / "camera_params.npz",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "dataset" / "images" / "raw",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dataset" / "images" / "undistorted",
    )
    args = parser.parse_args()

    camera_matrix, dist_coeffs = load_camera_params(args.params)
    args.output.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    count = 0
    for path in sorted(args.input.iterdir()):
        if path.suffix.lower() not in exts:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        out = undistort_bgr(img, camera_matrix, dist_coeffs)
        # Normalise extension to .png for consistent labelling
        dest = args.output / f"{path.stem}.png"
        cv2.imwrite(str(dest), out)
        count += 1
        print(dest.name)
    print(f"undistorted {count} images -> {args.output}")


if __name__ == "__main__":
    main()
