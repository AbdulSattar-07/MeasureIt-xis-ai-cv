"""Load saved camera params and undistort an image or a folder."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def load_camera_params(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["camera_matrix"], data["dist_coeffs"]


def undistort_bgr(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    new_k, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    return cv2.undistort(image_bgr, camera_matrix, dist_coeffs, None, new_k)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Undistort image(s)")
    parser.add_argument("--params", type=Path, default=root / "camera_params.npz")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.params.exists():
        raise SystemExit(f"Missing calibration file: {args.params}")

    camera_matrix, dist_coeffs = load_camera_params(args.params)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    if args.input.is_file():
        img = cv2.imread(str(args.input))
        if img is None:
            raise SystemExit(f"Cannot read {args.input}")
        out = undistort_bgr(img, camera_matrix, dist_coeffs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output), out)
        print(f"wrote {args.output}")
        return

    args.output.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.input.iterdir()):
        if path.suffix.lower() not in exts:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        out = undistort_bgr(img, camera_matrix, dist_coeffs)
        dest = args.output / path.name
        cv2.imwrite(str(dest), out)
        print(f"wrote {dest}")


if __name__ == "__main__":
    main()
