"""Evaluate a checkpoint on a frozen split (val or test). Test only once for reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from models.metrics import evaluate_batch
from models.seg_dataset import SegDataset
from models.train import build_model

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "models" / "config.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ckpt_path = args.checkpoint or (ROOT / cfg["best_checkpoint"])
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

    split_key = "test_list" if args.split == "test" else "val_list"
    ds = SegDataset(
        ROOT / cfg[split_key],
        ROOT / cfg["images_dir"],
        ROOT / cfg["labels_dir"],
        image_size=int(cfg["image_size"]),
    )
    loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(int(cfg["num_classes"]), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds, all_gts = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            out = model(images)["out"]
            all_preds.append(out.argmax(1).cpu().numpy())
            all_gts.append(batch["mask"].numpy())

    metrics = evaluate_batch(
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_gts, axis=0),
        class_ids=(1, 2),
    )
    metrics["split"] = args.split
    metrics["checkpoint"] = str(ckpt_path)
    metrics["num_images"] = len(ds)

    out_path = args.out or (
        ROOT / (cfg["metrics_test"] if args.split == "test" else cfg["metrics_val"])
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
