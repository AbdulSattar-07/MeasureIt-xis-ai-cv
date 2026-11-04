"""Segmentation dataset from fixed split file lists + PNG masks."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def read_split_list(path: Path) -> list[str]:
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line)
    return names


class SegDataset(Dataset):
    def __init__(
        self,
        split_file: Path,
        images_dir: Path,
        labels_dir: Path,
        image_size: int = 512,
    ) -> None:
        self.names = read_split_list(split_file)
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = image_size
        if not self.names:
            raise FileNotFoundError(
                f"Split list empty: {split_file}. Populate and commit before training."
            )

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> dict:
        name = self.names[idx]
        img_path = self.images_dir / name
        if not img_path.exists():
            # allow stem-only entries
            stem = Path(name).stem
            candidates = list(self.images_dir.glob(f"{stem}.*"))
            if not candidates:
                raise FileNotFoundError(img_path)
            img_path = candidates[0]
        label_path = self.labels_dir / f"{img_path.stem}.png"
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Missing image/mask for {name}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )
        mask = cv2.resize(
            mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST
        )

        image_t = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        # ImageNet normalisation expected by torchvision pretrained weights
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_t = (image_t - mean) / std
        mask_t = torch.from_numpy(mask.astype(np.int64))
        return {"image": image_t, "mask": mask_t, "name": img_path.name}
