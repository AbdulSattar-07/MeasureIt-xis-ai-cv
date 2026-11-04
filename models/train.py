"""Train DeepLabV3-MobileNetV3 on our own masks (not YOLO / not Roboflow)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)

from models.metrics import evaluate_batch
from models.seg_dataset import SegDataset


ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = (
        DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    )
    # Keep aux head so train/eval checkpoints always match
    model = deeplabv3_mobilenet_v3_large(weights=weights, aux_loss=True)
    # Replace classifier head for our class count
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        # MobileNet aux FCN head uses fewer channels than ResNet (often 10, not 256)
        aux_in = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(aux_in, num_classes, kernel_size=1)
    return model


@torch.no_grad()
def run_eval(model, loader, device) -> dict:
    model.eval()
    all_preds = []
    all_gts = []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].numpy()
        out = model(images)["out"]
        pred = out.argmax(1).cpu().numpy()
        all_preds.append(pred)
        all_gts.append(masks)
    preds = np.concatenate(all_preds, axis=0)
    gts = np.concatenate(all_gts, axis=0)
    return evaluate_batch(preds, gts, class_ids=(1, 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "models" / "config.yaml",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train_ds = SegDataset(
        ROOT / cfg["train_list"],
        ROOT / cfg["images_dir"],
        ROOT / cfg["labels_dir"],
        image_size=int(cfg["image_size"]),
    )
    val_ds = SegDataset(
        ROOT / cfg["val_list"],
        ROOT / cfg["images_dir"],
        ROOT / cfg["labels_dir"],
        image_size=int(cfg["image_size"]),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        drop_last=True,  # BatchNorm in DeepLab needs batch>1
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
    )

    model = build_model(int(cfg["num_classes"]), bool(cfg["pretrained"])).to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = ROOT / cfg["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ROOT / cfg["best_checkpoint"]
    best_iou = -1.0
    patience = int(cfg["early_stop_patience"])
    stale = 0
    history = []

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        losses = []
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optim.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs["out"], masks)
            if "aux" in outputs:
                loss = loss + 0.4 * criterion(outputs["aux"], masks)
            loss.backward()
            optim.step()
            losses.append(float(loss.item()))

        val_metrics = run_eval(model, val_loader, device)
        mean_loss = float(np.mean(losses)) if losses else 0.0
        row = {"epoch": epoch, "train_loss": mean_loss, **val_metrics}
        history.append(row)
        print(
            f"epoch {epoch}: loss={mean_loss:.4f} "
            f"val_iou={val_metrics['mask_iou_mean']:.4f} "
            f"mAP50={val_metrics['mAP@0.5']:.4f}"
        )

        if val_metrics["mask_iou_mean"] > best_iou:
            best_iou = val_metrics["mask_iou_mean"]
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "val_metrics": val_metrics,
                    "epoch": epoch,
                },
                best_path,
            )
            print(f"  saved best -> {best_path}")
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop at epoch {epoch}")
                break

    (ROOT / cfg["metrics_val"]).write_text(
        json.dumps({"best_iou": best_iou, "history": history}, indent=2),
        encoding="utf-8",
    )
    print(f"best val IoU={best_iou:.4f}")
    print("Do NOT touch test split until evaluate.py --split test")


if __name__ == "__main__":
    main()
