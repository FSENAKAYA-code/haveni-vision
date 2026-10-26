import numpy as np
import pytest

from haveni_vision.colors import classify_pixels, polygon_mask, summarize_labels
from haveni_vision.segmentation import dice_iou, grabcut_foreground


def test_color_summary_counts_white_ginger_and_other_foreground():
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb[:3] = (245, 245, 240)
    rgb[3:7] = (190, 105, 45)
    rgb[7:] = (30, 30, 30)
    foreground = np.ones((10, 10), dtype=bool)
    labels = classify_pixels(rgb, foreground)
    summary = summarize_labels(labels)
    assert summary["white_pct"] == pytest.approx(30.0)
    assert summary["ginger_pct"] == pytest.approx(40.0)
    assert summary["other_pct"] == pytest.approx(30.0)
    assert summary["classified_pct"] == pytest.approx(70.0)


def test_region_polygon_is_normalized_and_rasterized():
    region = polygon_mask((10, 10), [[0, 0], [0.49, 0], [0.49, 1], [0, 1]])
    assert region.shape == (10, 10)
    assert region[:, :5].all()
    assert not region[:, 7:].any()


def test_dice_and_iou_match_manual_binary_masks():
    predicted = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    reference = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    assert dice_iou(predicted, reference) == {"dice": 0.5, "iou": 1 / 3}


def test_grabcut_rejects_invalid_bbox():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="bbox"):
        grabcut_foreground(image, (-1, 0, 10, 10))


def test_color_classifier_rejects_empty_foreground():
    rgb = np.zeros((3, 3, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="empty"):
        classify_pixels(rgb, np.zeros((3, 3), dtype=bool))
