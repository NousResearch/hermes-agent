#!/usr/bin/env python3
"""Check the social-card tags Docusaurus actually emits.

    python3 website/scripts/check_social_card.py [BUILD_DIR]
    python3 website/scripts/check_social_card.py --html page.html [--html ...]

Given a build dir (default website/build), checks the docs homepage and the
plugin catalog-submission page. Given --html, checks those saved pages (for
example the live site). A page passes when og:image and twitter:image are the
same absolute https URL ending in /img/social-card.png, twitter:card is
summary_large_image exactly once, and (build dir only) the image file is a
1200x630 PNG under 1 MB.
"""

from __future__ import annotations

import argparse
import struct
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

PAGES = ("index.html", "developer-guide/plugins/catalog-submission/index.html")
IMAGE_SUFFIX = "/img/social-card.png"  # locale builds emit /docs/<locale>/img/...
MAX_BYTES = 1_000_000


class Meta(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, str]] = []

    def handle_starttag(self, tag, attrs):
        if tag != "meta":
            return
        a = dict(attrs)
        key = a.get("property") or a.get("name")
        if key in ("og:image", "twitter:image", "twitter:card"):
            self.tags.append((key, a.get("content") or ""))


def check_html(label: str, html: str) -> list[str]:
    parser = Meta()
    parser.feed(html)
    errors = []
    values = {
        k: [v for kk, v in parser.tags if kk == k]
        for k in ("og:image", "twitter:image", "twitter:card")
    }
    if values["twitter:card"] != ["summary_large_image"]:
        errors.append(
            f"{label}: twitter:card is {values['twitter:card']}, want ['summary_large_image']"
        )
    images = {k: values[k] for k in ("og:image", "twitter:image")}
    for key, found in images.items():
        if len(found) != 1:
            errors.append(f"{label}: {key} appears {len(found)} times, want 1")
        elif not is_card_url(found[0]):
            errors.append(
                f"{label}: {key} is {found[0]!r}, want absolute https URL ending {IMAGE_SUFFIX}"
            )
    og, tw = images["og:image"], images["twitter:image"]
    if len(og) == 1 and len(tw) == 1 and og != tw:
        errors.append(f"{label}: og:image {og[0]!r} != twitter:image {tw[0]!r}")
    return errors


def is_card_url(url: str) -> bool:
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        return False
    return parts.scheme == "https" and bool(host) and parts.path.endswith(IMAGE_SUFFIX)


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[:24]
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    return struct.unpack(">II", data[16:24])


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "build_dir",
        nargs="?",
        default=str(Path(__file__).resolve().parent.parent / "build"),
    )
    parser.add_argument(
        "--html",
        action="append",
        default=[],
        help="saved page to check instead of a build dir",
    )
    args = parser.parse_args()

    errors: list[str] = []
    if args.html:
        for name in args.html:
            errors += check_html(name, Path(name).read_text(encoding="utf-8"))
    else:
        build = Path(args.build_dir)
        for page in PAGES:
            errors += check_html(page, (build / page).read_text(encoding="utf-8"))
        image = build / "img" / "social-card.png"
        if not image.exists():
            errors.append(f"{image} missing from build")
        else:
            if png_size(image) != (1200, 630):
                errors.append(f"{image} is {png_size(image)}, want (1200, 630)")
            if image.stat().st_size >= MAX_BYTES:
                errors.append(
                    f"{image} is {image.stat().st_size} bytes, want < {MAX_BYTES}"
                )

    for error in errors:
        print(error)
    print("social card: FAIL" if errors else "social card: ok")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
