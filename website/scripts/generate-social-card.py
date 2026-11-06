#!/usr/bin/env python3
"""Generate website/static/img/social-card.png (1200x630) from the site banner.

The banner is a wide 1145x196 strip. Link-preview crawlers (Open Graph, X's
summary_large_image) crop to 1.91:1 or 2:1, which would cut a full-bleed strip
at the sides. This centers the wordmark on a 1200x630 canvas of the banner's
own background, inside a 1000x500 safe area, so both crops keep every letter.

Deterministic: same banner in, same bytes out.

    python3 website/scripts/generate-social-card.py [--check]

--check exits 1 if the committed PNG differs from a fresh render.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageChops

STATIC_IMG = Path(__file__).resolve().parent.parent / "static" / "img"
BANNER = STATIC_IMG / "hermes-agent-banner.png"
OUTPUT = STATIC_IMG / "social-card.png"

SIZE = (1200, 630)
SAFE_WIDTH = 1000  # wordmark fits inside a centered 1000x500 safe area


def render() -> Image.Image:
    banner = Image.open(BANNER).convert("RGB")
    background = banner.getpixel((0, 0))

    # Crop to the wordmark's own bounds so padding is even on every side.
    bounds = ImageChops.difference(
        banner, Image.new("RGB", banner.size, background)
    ).getbbox()
    wordmark = banner.crop(bounds)

    scale = SAFE_WIDTH / wordmark.width
    wordmark = wordmark.resize(
        (SAFE_WIDTH, round(wordmark.height * scale)), Image.Resampling.LANCZOS
    )

    card = Image.new("RGB", SIZE, background)
    card.paste(
        wordmark, ((SIZE[0] - wordmark.width) // 2, (SIZE[1] - wordmark.height) // 2)
    )
    return card


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "--check", action="store_true", help="fail if the committed PNG is stale"
    )
    args = parser.parse_args()

    card = render()
    if args.check:
        current = Image.open(OUTPUT).convert("RGB") if OUTPUT.exists() else None
        if (
            current is None
            or current.size != card.size
            or ImageChops.difference(current, card).getbbox()
        ):
            print(
                f"{OUTPUT} is stale; run python3 website/scripts/generate-social-card.py"
            )
            return 1
        return 0

    card.save(OUTPUT, optimize=True)
    print(f"wrote {OUTPUT} {card.size} {OUTPUT.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
