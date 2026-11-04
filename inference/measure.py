"""One command: image → undistort → segment → reference scale → width/height mm.

Usage:
  python -m inference.measure --image path/to/photo.jpg --out out/overlay.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration.undistort import load_camera_params, undistort_bgr
from inference.geometry import (
    load_reference,
    measure_object_mm,
    mm_per_pixel_from_reference,
)
from models.train import build_model


OBJECT_CLASS = 1
REFERENCE_CLASS = 2


def preprocess(image_bgr: np.ndarray, image_size: int) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0)


def mask_confidence(logits: torch.Tensor, class_id: int, mask_hw: np.ndarray) -> float:
    """Mean softmax probability of class_id inside the predicted mask region."""
    probs = torch.softmax(logits, dim=1)[0, class_id].cpu().numpy()
    region = mask_hw > 0
    if region.sum() == 0:
        return 0.0
    # Resize probs to mask size if needed
    if probs.shape != mask_hw.shape:
        probs = cv2.resize(probs, (mask_hw.shape[1], mask_hw.shape[0]))
    return float(probs[region].mean())


def draw_overlay(
    image_bgr: np.ndarray,
    object_mask: np.ndarray,
    ref_mask: np.ndarray,
    text_lines: list[str],
) -> np.ndarray:
    out = image_bgr.copy()
    color_obj = np.zeros_like(out)
    color_ref = np.zeros_like(out)
    color_obj[:, :] = (0, 200, 0)
    color_ref[:, :] = (0, 160, 255)
    obj = object_mask.astype(bool)
    ref = ref_mask.astype(bool)
    out[obj] = (0.55 * out[obj] + 0.45 * color_obj[obj]).astype(np.uint8)
    out[ref] = (0.55 * out[ref] + 0.45 * color_ref[ref]).astype(np.uint8)
    y = 28
    for line in text_lines:
        cv2.putText(
            out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA
        )
        y += 28
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure object width/height in mm")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--params",
        type=Path,
        default=ROOT / "calibration" / "camera_params.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models" / "checkpoints" / "best.pt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "models" / "config.yaml",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "measurement" / "data" / "reference.json",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "overlay.png")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--skip-undistort", action="store_true")
    parser.add_argument("--min-object-conf", type=float, default=0.55)
    parser.add_argument("--min-ref-conf", type=float, default=0.55)
    parser.add_argument("--max-axis-disagree-pct", type=float, default=12.0)
    parser.add_argument("--min-mask-area", type=int, default=200)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    image_size = int(cfg["image_size"])
    ref_info = load_reference(args.reference)

    raw = cv2.imread(str(args.image))
    if raw is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    t0 = time.perf_counter()
    if args.skip_undistort:
        image = raw
        undistorted = False
    else:
        if not args.params.exists():
            raise SystemExit(f"Missing camera params: {args.params}")
        K, dist = load_camera_params(args.params)
        image = undistort_bgr(raw, K, dist)
        undistorted = True

    if not args.checkpoint.exists():
        raise SystemExit(
            f"Missing checkpoint: {args.checkpoint}. Train first with models/train.py"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(int(cfg["num_classes"]), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tensor = preprocess(image, image_size).to(device)
    with torch.no_grad():
        logits = model(tensor)["out"]
        pred_small = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)

    h, w = image.shape[:2]
    pred = cv2.resize(pred_small, (w, h), interpolation=cv2.INTER_NEAREST)
    object_mask = (pred == OBJECT_CLASS).astype(np.uint8)
    ref_mask = (pred == REFERENCE_CLASS).astype(np.uint8)

    obj_area = int(object_mask.sum())
    ref_area = int(ref_mask.sum())
    obj_conf = mask_confidence(logits, OBJECT_CLASS, object_mask)
    ref_conf = mask_confidence(logits, REFERENCE_CLASS, ref_mask)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    result = {
        "image": str(args.image),
        "undistorted": undistorted,
        "elapsed_ms": elapsed_ms,
        "object_area_px": obj_area,
        "reference_area_px": ref_area,
        "object_confidence": obj_conf,
        "reference_confidence": ref_conf,
        "status": "ok",
        "width_mm": None,
        "height_mm": None,
        "confidence": None,
        "refuse_reason": None,
    }

    def refuse(reason: str, overlay_lines: list[str]) -> None:
        result["status"] = "refused"
        result["refuse_reason"] = reason
        args.out.parent.mkdir(parents=True, exist_ok=True)
        overlay = draw_overlay(image, object_mask, ref_mask, overlay_lines)
        cv2.imwrite(str(args.out), overlay)
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        raise SystemExit(2)

    if obj_area < args.min_mask_area:
        refuse(
            "object_not_found",
            ["REFUSED: object not found", f"area={obj_area}px"],
        )
    if ref_area < args.min_mask_area:
        refuse(
            "reference_not_found",
            ["REFUSED: reference not found", f"area={ref_area}px"],
        )
    if obj_conf < args.min_object_conf:
        refuse(
            "low_object_confidence",
            [f"REFUSED: low object conf {obj_conf:.2f}", f"threshold={args.min_object_conf}"],
        )
    if ref_conf < args.min_ref_conf:
        refuse(
            "low_reference_confidence",
            [f"REFUSED: low ref conf {ref_conf:.2f}", f"threshold={args.min_ref_conf}"],
        )

    scale_pack = mm_per_pixel_from_reference(
        ref_mask,
        float(ref_info["width_mm"]),
        float(ref_info["height_mm"]),
    )
    if scale_pack is None:
        refuse("reference_geometry_failed", ["REFUSED: bad reference geometry"])

    mm_per_px, ref_meta = scale_pack
    if ref_meta["axis_disagree_pct"] > args.max_axis_disagree_pct:
        refuse(
            "tilted_or_bad_plane",
            [
                "REFUSED: reference axes disagree",
                f"disagree={ref_meta['axis_disagree_pct']:.1f}%",
            ],
        )

    measured = measure_object_mm(object_mask, mm_per_px)
    if measured is None:
        refuse("object_geometry_failed", ["REFUSED: bad object geometry"])

    width_mm, height_mm, obj_meta = measured
    conf = float(min(obj_conf, ref_conf))
    result.update(
        {
            "width_mm": round(width_mm, 2),
            "height_mm": round(height_mm, 2),
            "confidence": round(conf, 4),
            "mm_per_px": mm_per_px,
            "reference_meta": ref_meta,
            "object_meta": obj_meta,
        }
    )

    lines = [
        f"W={width_mm:.1f} mm  H={height_mm:.1f} mm",
        f"conf={conf:.2f}  undistort={undistorted}",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    overlay = draw_overlay(image, object_mask, ref_mask, lines)
    cv2.imwrite(str(args.out), overlay)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"overlay -> {args.out}")


if __name__ == "__main__":
    main()
