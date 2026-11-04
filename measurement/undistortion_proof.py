"""Item 6: measure the same instances on raw vs undistorted; compare errors."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(gt: Path, skip: bool, out: Path) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "measurement" / "evaluate_accuracy.py"),
        "--gt",
        str(gt),
        "--out",
        str(out),
    ]
    if skip:
        cmd.append("--skip-undistort")
    subprocess.run(cmd, cwd=str(ROOT), check=False)
    summary = out.with_suffix(".summary.json")
    if not summary.exists():
        raise SystemExit(f"Missing summary {summary}")
    return json.loads(summary.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt",
        type=Path,
        default=ROOT / "measurement" / "data" / "ground_truth.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "measurement" / "data" / "undistortion_proof.csv",
    )
    args = parser.parse_args()

    raw_table = ROOT / "measurement" / "data" / "accuracy_raw.csv"
    und_table = ROOT / "measurement" / "data" / "accuracy_undistorted.csv"
    raw = run(args.gt, skip=True, out=raw_table)
    und = run(args.gt, skip=False, out=und_table)

    # Merge per-image abs errors if both tables exist
    def load_errs(path: Path) -> dict[str, tuple[float | None, float | None]]:
        data = {}
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    ew = float(row["abs_err_width_mm"]) if row["abs_err_width_mm"] else None
                    eh = float(row["abs_err_height_mm"]) if row["abs_err_height_mm"] else None
                except ValueError:
                    ew = eh = None
                data[row["image"]] = (ew, eh)
        return data

    raw_errs = load_errs(raw_table)
    und_errs = load_errs(und_table)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "raw_abs_err_width_mm",
                "raw_abs_err_height_mm",
                "undistorted_abs_err_width_mm",
                "undistorted_abs_err_height_mm",
            ],
        )
        writer.writeheader()
        for name in sorted(set(raw_errs) | set(und_errs)):
            rw, rh = raw_errs.get(name, (None, None))
            uw, uh = und_errs.get(name, (None, None))
            writer.writerow(
                {
                    "image": name,
                    "raw_abs_err_width_mm": rw,
                    "raw_abs_err_height_mm": rh,
                    "undistorted_abs_err_width_mm": uw,
                    "undistorted_abs_err_height_mm": uh,
                }
            )

    proof = {
        "raw_mae_width_mm": raw.get("mae_width_mm"),
        "raw_mae_height_mm": raw.get("mae_height_mm"),
        "undistorted_mae_width_mm": und.get("mae_width_mm"),
        "undistorted_mae_height_mm": und.get("mae_height_mm"),
        "table": str(args.out),
        "explanation_template": (
            "Undistortion changed width MAE from "
            f"{raw.get('mae_width_mm')} mm to {und.get('mae_width_mm')} mm "
            "and height MAE from "
            f"{raw.get('mae_height_mm')} mm to {und.get('mae_height_mm')} mm "
            "on the same instances. Interpret using these numbers in Objectives.md."
        ),
    }
    proof_path = args.out.with_suffix(".summary.json")
    proof_path.write_text(json.dumps(proof, indent=2), encoding="utf-8")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
