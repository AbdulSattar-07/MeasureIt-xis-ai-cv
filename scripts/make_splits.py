"""Create fixed 70/20/10 train/val/test file lists from labelled undistorted images.

Run ONCE after labelling, then commit the split files BEFORE training.
Re-running after training has started is forbidden for the test list.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images",
        type=Path,
        default=root / "dataset" / "images" / "undistorted",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=root / "dataset" / "labels",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "dataset" / "splits",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.20)
    args = parser.parse_args()

    exts = {".png", ".jpg", ".jpeg"}
    names = []
    for p in sorted(args.images.iterdir()):
        if p.suffix.lower() not in exts:
            continue
        label = args.labels / f"{p.stem}.png"
        if not label.exists():
            print(f"skip (no label): {p.name}")
            continue
        names.append(p.name)

    if len(names) < 70:
        print(f"WARNING: {len(names)} labelled images; assessment asks for 70+.")

    rng = random.Random(args.seed)
    rng.shuffle(names)
    n = len(names)
    n_train = int(round(n * args.train))
    n_val = int(round(n * args.val))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    train = names[:n_train]
    val = names[n_train : n_train + n_val]
    test = names[n_train + n_val :]

    args.out.mkdir(parents=True, exist_ok=True)
    header = (
        "# Fixed split — filenames relative to dataset/images/undistorted/\n"
        "# COMMITTED BEFORE TRAINING. Do not edit the test list after training starts.\n"
    )
    (args.out / "train.txt").write_text(
        header + "\n".join(train) + ("\n" if train else ""), encoding="utf-8"
    )
    (args.out / "val.txt").write_text(
        header + "\n".join(val) + ("\n" if val else ""), encoding="utf-8"
    )
    (args.out / "test.txt").write_text(
        header
        + "# Never use for training, tuning, or checkpoint pick.\n"
        + "\n".join(test)
        + ("\n" if test else ""),
        encoding="utf-8",
    )
    print(f"total={n} train={len(train)} val={len(val)} test={len(test)} seed={args.seed}")
    print(f"wrote lists under {args.out}")
    print("Commit these split files BEFORE starting training.")


if __name__ == "__main__":
    main()
