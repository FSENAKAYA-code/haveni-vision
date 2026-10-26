"""Foreground segmentation and manual-mask evaluation helpers."""

import cv2
import numpy as np


def grabcut_foreground(
    bgr: np.ndarray, bbox: tuple[int, int, int, int], *, iterations: int = 5
) -> np.ndarray:
    """Run OpenCV GrabCut using a user-provided cat bounding box.

    The box should tightly contain the cat while leaving some visible background
    around it. Returns a boolean mask; the result must be visually reviewed.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("image must be an HxWx3 uint8 BGR array")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    x, y, width, height = bbox
    image_height, image_width = bgr.shape[:2]
    if (
        x < 0 or y < 0 or width < 2 or height < 2
        or x + width > image_width or y + height > image_height
    ):
        raise ValueError("bbox must have positive size and fit inside the image")

    labels = np.zeros((image_height, image_width), dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        bgr, labels, (x, y, width, height), background_model, foreground_model,
        iterations, cv2.GC_INIT_WITH_RECT,
    )
    result = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
    if not result.any():
        raise RuntimeError("GrabCut returned an empty foreground mask")
    return result


def dice_iou(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Compute binary Dice and IoU against a manually prepared reference mask."""
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference masks must have the same shape")
    pred = prediction.astype(bool)
    ref = reference.astype(bool)
    intersection = int(np.count_nonzero(pred & ref))
    union = int(np.count_nonzero(pred | ref))
    denominator = int(pred.sum() + ref.sum())
    if union == 0:
        return {"dice": 1.0, "iou": 1.0}
    return {
        "dice": float(2 * intersection / denominator),
        "iou": float(intersection / union),
    }
