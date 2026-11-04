"""Segmentation metrics: IoU, precision, recall, and simple mask mAP proxies."""

from __future__ import annotations

import numpy as np


def _binarize(mask: np.ndarray, class_id: int) -> np.ndarray:
    return (mask == class_id).astype(np.uint8)


def binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)


def binary_precision_recall(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, 1 - gt).sum()
    fn = np.logical_and(1 - pred, gt).sum()
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    return precision, recall


def average_precision_from_iou(iou: float, thresholds: np.ndarray) -> float:
    """Single-detection AP proxy: 1 at each threshold the mask IoU clears."""
    return float((iou >= thresholds).mean()) if len(thresholds) else 0.0


def evaluate_batch(
    preds: np.ndarray,
    gts: np.ndarray,
    class_ids: tuple[int, ...] = (1, 2),
) -> dict:
    """preds/gts: HxW int masks or NxHxW."""
    if preds.ndim == 2:
        preds = preds[None, ...]
        gts = gts[None, ...]

    thresholds_05 = np.array([0.5])
    thresholds_coco = np.arange(0.5, 1.0, 0.05)

    per_class: dict[str, dict] = {}
    ious_all = []
    precs = []
    recs = []
    ap50s = []
    ap5095s = []

    for cid in class_ids:
        class_ious = []
        class_p = []
        class_r = []
        class_ap50 = []
        class_ap = []
        for i in range(len(preds)):
            p = _binarize(preds[i], cid)
            g = _binarize(gts[i], cid)
            iou = binary_iou(p, g)
            precision, recall = binary_precision_recall(p, g)
            class_ious.append(iou)
            class_p.append(precision)
            class_r.append(recall)
            class_ap50.append(average_precision_from_iou(iou, thresholds_05))
            class_ap.append(average_precision_from_iou(iou, thresholds_coco))
        per_class[str(cid)] = {
            "iou": float(np.mean(class_ious)),
            "precision": float(np.mean(class_p)),
            "recall": float(np.mean(class_r)),
            "mAP@0.5": float(np.mean(class_ap50)),
            "mAP@0.5:0.95": float(np.mean(class_ap)),
        }
        ious_all.extend(class_ious)
        precs.extend(class_p)
        recs.extend(class_r)
        ap50s.extend(class_ap50)
        ap5095s.extend(class_ap)

    return {
        "mask_iou_mean": float(np.mean(ious_all)) if ious_all else 0.0,
        "precision": float(np.mean(precs)) if precs else 0.0,
        "recall": float(np.mean(recs)) if recs else 0.0,
        "mAP@0.5": float(np.mean(ap50s)) if ap50s else 0.0,
        "mAP@0.5:0.95": float(np.mean(ap5095s)) if ap5095s else 0.0,
        "per_class": per_class,
    }
