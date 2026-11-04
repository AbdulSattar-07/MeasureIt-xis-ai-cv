"""Calibrate camera from checkerboard images using OpenCV.

Saves camera matrix, distortion coefficients, and a reprojection-error report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_board_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def collect_object_image_points(
    image_paths: list[Path],
    corners_x: int,
    corners_y: int,
    square_mm: float,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int] | None, list[str]]:
    objp = np.zeros((corners_y * corners_x, 3), np.float32)
    objp[:, :2] = np.mgrid[0:corners_x, 0:corners_y].T.reshape(-1, 2)
    objp *= square_mm

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    used: list[str] = []

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FILTER_QUADS
    )

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"skip unreadable: {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])
        h = gray.shape[0]
        # Prefer screen region (drop laptop keyboard), also try full frame
        rois = [gray[: int(h * 0.82)], gray, cv2.equalizeHist(gray[: int(h * 0.82)])]
        found = False
        corners = None
        for roi in rois:
            found, corners = cv2.findChessboardCorners(
                roi, (corners_x, corners_y), flags
            )
            if found:
                break
            if hasattr(cv2, "findChessboardCornersSB"):
                found, corners = cv2.findChessboardCornersSB(
                    roi, (corners_x, corners_y)
                )
                if found:
                    break
        if not found:
            print(f"skip no corners: {path.name}")
            continue
        # cornerSubPix needs same-sized source as detection; use full gray when possible
        src = gray
        if corners is not None and corners[:, 0, 1].max() < h * 0.82:
            src = gray
        corners = cv2.cornerSubPix(src, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        used.append(path.name)
        print(f"ok: {path.name}")

    return objpoints, imgpoints, image_size, used


def mean_reprojection_error(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    rvecs,
    tvecs,
    camera_matrix,
    dist_coeffs,
) -> float:
    total_error = 0.0
    total_points = 0
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        err = cv2.norm(imgpoints[i], projected, cv2.NORM_L2)
        n = len(objpoints[i])
        total_error += err * err
        total_points += n
    return float(np.sqrt(total_error / max(total_points, 1)))


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="OpenCV checkerboard calibration")
    parser.add_argument(
        "--images",
        type=Path,
        default=root / "images",
        help="Folder of calibration images",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "board_config.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "camera_params.npz",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "calibration_report.json",
    )
    args = parser.parse_args()

    cfg = load_board_config(args.config)
    corners_x = int(cfg["inner_corners_x"])
    corners_y = int(cfg["inner_corners_y"])
    square_mm = float(cfg["square_size_mm"])

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths = sorted(
        p for p in args.images.iterdir() if p.suffix.lower() in exts
    )
    if len(paths) < 20:
        print(
            f"WARNING: found {len(paths)} images; assessment asks for 20 or more."
        )
    if not paths:
        raise SystemExit(f"No images in {args.images}")

    objpoints, imgpoints, image_size, used = collect_object_image_points(
        paths, corners_x, corners_y, square_mm
    )
    if len(objpoints) < 3:
        raise SystemExit("Need at least 3 successful checkerboard detections.")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    mean_err = mean_reprojection_error(
        objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs
    )

    np.savez(
        args.out,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_width=image_size[0],
        image_height=image_size[1],
        square_size_mm=square_mm,
        inner_corners_x=corners_x,
        inner_corners_y=corners_y,
    )

    report = {
        "num_images_on_disk": len(paths),
        "num_images_used": len(used),
        "images_used": used,
        "image_size_wh": list(image_size),
        "opencv_rms": float(rms),
        "mean_reprojection_error_px": mean_err,
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        "params_file": str(args.out.name),
        "board": cfg,
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Calibration report ===")
    print(f"images used: {len(used)} / {len(paths)}")
    print(f"OpenCV RMS: {rms:.4f} px")
    print(f"mean reprojection error: {mean_err:.4f} px")
    print(f"saved params: {args.out}")
    print(f"saved report: {args.report}")


if __name__ == "__main__":
    main()
