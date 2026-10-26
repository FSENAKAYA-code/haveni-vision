"""Run local coat-colour analysis on one image."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from haveni_vision.analysis import analyze_image, load_region_polygons, save_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Local cat photo; not uploaded anywhere")
    parser.add_argument(
        "--bbox", type=int, nargs=4, required=True, metavar=("X", "Y", "W", "H"),
        help="Rectangle around the cat in source-image pixels",
    )
    parser.add_argument("--regions", type=Path, help="JSON file with manual region polygons")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("reports/generated/haveni"))
    args = parser.parse_args()
    polygons = load_region_polygons(args.regions) if args.regions else None
    report, mask, overlay = analyze_image(
        args.image, tuple(args.bbox), region_polygons=polygons,
        iterations=args.iterations,
    )
    save_outputs(report, mask, overlay, args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
