# HaveniVision

Local-first computer-vision prototype for estimating the white and ginger coat
colour proportions in a cat photo. The first baseline uses OpenCV GrabCut with a
user-drawn bounding box and fixed HSV thresholds. It exports a foreground mask, a
sage-colour overlay, and JSON measurements. No model has been trained and no
accuracy or coat measurement result is claimed.

## Current scope and limits

- Segmentation requires a rectangle around the cat and must be visually checked.
- White/ginger labels use simple colour thresholds. Lighting, shadows, camera white
  balance, and cream or dark fur can change the estimates.
- Chest, belly, paws, and back percentages are reported only for polygons supplied by
  the user in a JSON sidecar. The tool does not infer anatomy automatically.
- The “other” share is unclassified by these two colour heuristics; it is not a
  calibrated uncertainty score. No identity claim is made from photo similarity.
- There are no Haveni photos, annotations, trained weights, or benchmark results in
  this repository. Keep personal photos and masks local; `.gitignore` excludes them.

## Setup

Requires Python 3.12 and `uv`.

```bash
uv sync
uv run pytest
uv run ruff check .
```

## Use on a local photo

Pass a rectangle `(x, y, width, height)` that encloses the cat and leaves some
background visible around it:

```bash
uv run python scripts/analyze_cat.py /path/to/haveni-photo.jpg \
  --bbox 80 40 900 1100 \
  --regions configs/regions.example.json \
  --out reports/generated/haveni
```

Output files are `foreground-mask.png`, `overlay.png`, and `report.json`. Coordinates
in the example region file are normalized to the full image and are illustrative;
draw polygons that match the actual cat pose before interpreting region-level values.
Outputs are local and ignored by Git.

## Data source and licensing

The initial prototype accepts user-provided local photos and does not download a
dataset. For a reproducible segmentation evaluation, the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
provides pet images and pixel-level trimap annotations. Oxford states that it is
available under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) and
that image copyright remains with the original owners. Obtain it from the official
source, review the license and attribution obligations, and do not commit its images
or annotations here. Its trimaps can evaluate foreground segmentation, but they do
not label anatomical colour regions; those require separate manual annotations.

Reference: O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. V. Jawahar, “Cats and Dogs,”
CVPR 2012.

## Evaluation plan

Prepare a small, manually checked set with foreground masks and pose-specific region
polygons. Compare GrabCut against Oxford trimap references using Dice and IoU; evaluate
coat-colour estimates only on manually reviewed colour labels. Keep photos from the
same capture session together when splitting evaluation data. Report measured results
only after running the experiment, including sample count, conditions, and known
limitations.
