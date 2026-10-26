"""Image analysis orchestration and visual artifact generation."""

import json
from pathlib import Path

import cv2
import numpy as np

from haveni_vision.colors import classify_pixels, polygon_mask, summarize_labels
from haveni_vision.segmentation import grabcut_foreground

REGION_NAMES = {"chest", "belly", "paws", "back"}


def load_region_polygons(path: str | Path) -> dict[str, list[list[float]]]:
    """Read optional user-drawn normalized polygons from a JSON sidecar."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    polygons = document.get("regions")
    if not isinstance(polygons, dict):
        raise TypeError("region JSON must contain an object named 'regions'")
    unknown = set(polygons) - REGION_NAMES
    if unknown:
        raise ValueError(f"unknown region names: {', '.join(sorted(unknown))}")
    return polygons


def analyze_image(
    image_path: str | Path,
    bbox: tuple[int, int, int, int],
    *,
    region_polygons: dict[str, list[list[float]]] | None = None,
    iterations: int = 5,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Return JSON-ready measurements, binary foreground mask, and overlay."""
    source = Path(image_path)
    bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not read image: {source}")
    foreground = grabcut_foreground(bgr, bbox, iterations=iterations)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    labels = classify_pixels(rgb, foreground)
    report = {
        "image": source.name,
        "image_width": int(bgr.shape[1]),
        "image_height": int(bgr.shape[0]),
        "segmentation": "opencv_grabcut_with_user_bbox",
        "color_method": "fixed_hsv_heuristics",
        "global_coat_estimates": summarize_labels(labels),
        "regions": {},
        "limitations": [
            "Segmentation and colour labels are heuristic and require visual review.",
            "Anatomical regions are measured only when supplied as manual polygons.",
            "Percentages are not calibrated probabilities or identity evidence.",
        ],
    }
    for name, points in (region_polygons or {}).items():
        if name not in REGION_NAMES:
            raise ValueError(f"unknown region name: {name}")
        region = polygon_mask(labels.shape, points)
        report["regions"][name] = summarize_labels(labels, region)

    overlay = bgr.copy()
    tint = np.zeros_like(bgr)
    tint[:] = (141, 165, 155)  # sage, BGR
    overlay[foreground] = cv2.addWeighted(
        bgr[foreground], 0.56, tint[foreground], 0.44, 0
    )
    contours, _ = cv2.findContours(
        foreground.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (74, 93, 47), 2)
    return report, foreground.astype(np.uint8) * 255, overlay


def save_outputs(
    report: dict,
    mask: np.ndarray,
    overlay: np.ndarray,
    output_dir: str | Path,
) -> None:
    """Save the report, mask, and overlay in a local output directory."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not cv2.imwrite(str(destination / "foreground-mask.png"), mask):
        raise OSError("could not save foreground-mask.png")
    if not cv2.imwrite(str(destination / "overlay.png"), overlay):
        raise OSError("could not save overlay.png")
