"""Compare system mm outputs to calliper ground truth (≥10 instances)."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run_measure(image: Path, skip_undistort: bool, out_json: Path) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "inference.measure",
        "--image",
        str(image),
        "--json-out",
        str(out_json),
        "--out",
        str(out_json.with_suffix(".png")),
    ]
    if skip_undistort:
        cmd.append("--skip-undistort")
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if out_json.exists():
        return json.loads(out_json.read_text(encoding="utf-8"))
    return {
        "status": "error",
        "refuse_reason": f"exit={proc.returncode}",
        "stderr": proc.stderr[-500:],
        "width_mm": None,
        "height_mm": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt",
        type=Path,
        default=ROOT / "measurement" / "data" / "ground_truth.csv",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=ROOT / "dataset" / "images" / "raw",
        help="Folder containing the measurement images named in the GT CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "measurement" / "data" / "accuracy_table.csv",
    )
    parser.add_argument("--skip-undistort", action="store_true")
    args = parser.parse_args()

    if not args.gt.exists():
        raise SystemExit(
            f"Missing {args.gt}. Copy ground_truth_template.csv and fill ≥10 rows."
        )

    rows_out = []
    abs_w, abs_h, pct_w, pct_h = [], [], [], []
    raw_dir = ROOT / "measurement" / "data" / "raw_runs"
    raw_dir.mkdir(parents=True, exist_ok=True)

    with args.gt.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(
            (line for line in f if line.strip() and not line.lstrip().startswith("#"))
        )
        for row in reader:
            name = row["image"].strip()
            gt_w = float(row["gt_width_mm"])
            gt_h = float(row["gt_height_mm"])
            image_path = args.images_dir / name
            if not image_path.exists():
                # try undistorted folder stem.png
                alt = ROOT / "dataset" / "images" / "undistorted" / Path(name).with_suffix(".png").name
                image_path = alt if alt.exists() else image_path
            out_json = raw_dir / f"{Path(name).stem}.json"
            pred = run_measure(image_path, args.skip_undistort, out_json)
            pw = pred.get("width_mm")
            ph = pred.get("height_mm")
            status = pred.get("status")
            if status != "ok" or pw is None or ph is None:
                rows_out.append(
                    {
                        "image": name,
                        "gt_width_mm": gt_w,
                        "gt_height_mm": gt_h,
                        "pred_width_mm": "",
                        "pred_height_mm": "",
                        "abs_err_width_mm": "",
                        "abs_err_height_mm": "",
                        "pct_err_width": "",
                        "pct_err_height": "",
                        "status": status,
                        "refuse_reason": pred.get("refuse_reason"),
                    }
                )
                continue
            ew = abs(pw - gt_w)
            eh = abs(ph - gt_h)
            pew = ew / gt_w * 100.0
            peh = eh / gt_h * 100.0
            abs_w.append(ew)
            abs_h.append(eh)
            pct_w.append(pew)
            pct_h.append(peh)
            rows_out.append(
                {
                    "image": name,
                    "gt_width_mm": gt_w,
                    "gt_height_mm": gt_h,
                    "pred_width_mm": pw,
                    "pred_height_mm": ph,
                    "abs_err_width_mm": round(ew, 3),
                    "abs_err_height_mm": round(eh, 3),
                    "pct_err_width": round(pew, 3),
                    "pct_err_height": round(peh, 3),
                    "status": status,
                    "refuse_reason": "",
                }
            )

    fieldnames = list(rows_out[0].keys()) if rows_out else [
        "image",
        "gt_width_mm",
        "gt_height_mm",
        "pred_width_mm",
        "pred_height_mm",
        "abs_err_width_mm",
        "abs_err_height_mm",
        "pct_err_width",
        "pct_err_height",
        "status",
        "refuse_reason",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    summary = {
        "n_ok": len(abs_w),
        "mae_width_mm": float(np.mean(abs_w)) if abs_w else None,
        "mae_height_mm": float(np.mean(abs_h)) if abs_h else None,
        "median_abs_width_mm": float(np.median(abs_w)) if abs_w else None,
        "median_abs_height_mm": float(np.median(abs_h)) if abs_h else None,
        "worst_abs_width_mm": float(np.max(abs_w)) if abs_w else None,
        "worst_abs_height_mm": float(np.max(abs_h)) if abs_h else None,
        "mean_pct_width": float(np.mean(pct_w)) if pct_w else None,
        "mean_pct_height": float(np.mean(pct_h)) if pct_h else None,
        "table": str(args.out),
        "skip_undistort": args.skip_undistort,
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"table -> {args.out}")


if __name__ == "__main__":
    main()
