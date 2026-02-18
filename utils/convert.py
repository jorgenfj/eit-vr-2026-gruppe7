#!/usr/bin/env python3
import argparse
from pathlib import Path

from PIL import Image


def png_to_jpg(in_path: Path, out_path: Path, quality: int = 95, background=(255, 255, 255)) -> None:
    """Convert a PNG to JPG, handling alpha by compositing onto a background."""
    with Image.open(in_path) as im:
        # Handle alpha/transparency
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, background + (255,))
            im = Image.alpha_composite(bg, im).convert("RGB")
        else:
            im = im.convert("RGB")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path, format="JPEG", quality=quality, optimize=True)


def main():
    parser = argparse.ArgumentParser(description="Convert PNGs to JPG and keep every Nth image (sorted by name).")
    parser.add_argument("--input", "-i", required=True, help="Input folder containing PNG images.")
    parser.add_argument("--output", "-o", required=True, help="Output folder for JPG images.")
    parser.add_argument("--keep-every", "-k", type=int, default=100, help="Keep one of every K images (default: 5).")
    parser.add_argument("--offset", type=int, default=0, help="Start offset within the sequence (default: 0).")
    parser.add_argument("--quality", "-q", type=int, default=95, help="JPEG quality 1-100 (default: 95).")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)

    if not in_dir.is_dir():
        raise SystemExit(f"Input path is not a directory: {in_dir}")

    pngs = sorted(in_dir.glob("*.png"))
    if not pngs:
        raise SystemExit(f"No .png files found in: {in_dir}")

    k = args.keep_every
    if k <= 0:
        raise SystemExit("--keep-every must be >= 1")

    offset = args.offset % k
    selected = [p for idx, p in enumerate(pngs) if (idx - offset) % k == 0]

    print(f"Found {len(pngs)} PNGs. Keeping {len(selected)} (every {k}th, offset={offset}).")

    for p in selected:
        out_path = out_dir / (p.stem + ".jpg")
        png_to_jpg(p, out_path, quality=args.quality)

    print(f"Done. Wrote JPGs to: {out_dir}")


if __name__ == "__main__":
    main()
