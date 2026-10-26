"""Heuristic coat-colour classification in HSV colour space."""

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

DEFAULT_THRESHOLDS = {
    "white_max_saturation": 75,
    "white_min_value": 155,
    "ginger_min_saturation": 45,
    "ginger_min_value": 45,
    "ginger_hue_min": 0,
    "ginger_hue_max": 32,
}


def classify_pixels(
    rgb: np.ndarray,
    foreground: np.ndarray,
    thresholds: Mapping[str, int] | None = None,
) -> np.ndarray:
    """Return labels: 0 background, 1 white, 2 ginger, 3 other/uncertain.

    Thresholds are visual heuristics, not calibrated probabilities or a learned
    classifier. White is checked first so pale cream fur is not double-counted.
    """
    _validate_rgb_mask(rgb, foreground)
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    in_foreground = foreground.astype(bool)

    white = (
        in_foreground
        & (saturation <= limits["white_max_saturation"])
        & (value >= limits["white_min_value"])
    )
    ginger_hue = (hue >= limits["ginger_hue_min"]) & (
        hue <= limits["ginger_hue_max"]
    )
    ginger = (
        in_foreground
        & ~white
        & ginger_hue
        & (saturation >= limits["ginger_min_saturation"])
        & (value >= limits["ginger_min_value"])
    )

    labels = np.zeros(foreground.shape, dtype=np.uint8)
    labels[in_foreground] = 3
    labels[white] = 1
    labels[ginger] = 2
    return labels


def summarize_labels(labels: np.ndarray, region: np.ndarray | None = None) -> dict:
    """Summarize class proportions among foreground pixels in an optional region."""
    if labels.ndim != 2:
        raise ValueError("labels must be a two-dimensional array")
    selected = labels > 0
    if region is not None:
        if region.shape != labels.shape:
            raise ValueError("region and labels must have the same shape")
        selected &= region.astype(bool)
    count = int(selected.sum())
    if count == 0:
        return {"foreground_pixels": 0, "white_pct": None, "ginger_pct": None,
                "other_pct": None, "classified_pct": None}

    white_pct = float(np.count_nonzero((labels == 1) & selected) / count * 100)
    ginger_pct = float(np.count_nonzero((labels == 2) & selected) / count * 100)
    other_pct = float(np.count_nonzero((labels == 3) & selected) / count * 100)
    return {
        "foreground_pixels": count,
        "white_pct": white_pct,
        "ginger_pct": ginger_pct,
        "other_pct": other_pct,
        "classified_pct": white_pct + ginger_pct,
    }


def _validate_rgb_mask(rgb: np.ndarray, foreground: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be an HxWx3 uint8 array")
    if foreground.shape != rgb.shape[:2]:
        raise ValueError("foreground mask dimensions must match the image")
    if not np.any(foreground):
        raise ValueError("foreground mask is empty")


def normalize_polygon(points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    """Validate polygon vertices expressed as normalized x/y coordinates."""
    if len(points) < 3:
        raise ValueError("a region polygon must have at least three points")
    normalized = []
    for point in points:
        if len(point) != 2:
            raise ValueError("each polygon point must contain x and y")
        x, y = float(point[0]), float(point[1])
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("normalized polygon coordinates must be between 0 and 1")
        normalized.append((x, y))
    return normalized


def polygon_mask(
    shape: tuple[int, int], points: Sequence[Sequence[float]]
) -> np.ndarray:
    """Rasterize a normalized image polygon into a boolean mask."""
    height, width = shape
    if height < 1 or width < 1:
        raise ValueError("image dimensions must be positive")
    polygon = normalize_polygon(points)
    pixels = np.array(
        [[round(x * (width - 1)), round(y * (height - 1))] for x, y in polygon],
        dtype=np.int32,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pixels], 1)
    return mask.astype(bool)
