#!/usr/bin/env python3
"""Print the analytics bias-trap checklist for result review."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_CHECKLIST = Path(__file__).resolve().parents[1] / "yallaplay-wiki" / "reference" / "analytics_rules.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=DEFAULT_CHECKLIST, help="checklist markdown path")
    parser.add_argument("--section", default="Bias-trap checklist", help="heading text to print")
    return parser.parse_args()


def heading_level(line: str) -> int | None:
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return None
    return len(stripped) - len(stripped.lstrip("#"))


def extract_section(text: str, section: str) -> str:
    lines = text.splitlines()
    start = None
    level = None
    needle = section.lower()
    for index, line in enumerate(lines):
        current_level = heading_level(line)
        if current_level is None:
            continue
        title = line.lstrip("#").strip().lower()
        if needle in title:
            start = index
            level = current_level
            break
    if start is None or level is None:
        return text.strip()
    end = len(lines)
    for index in range(start + 1, len(lines)):
        current_level = heading_level(lines[index])
        if current_level is not None and current_level <= level:
            end = index
            break
    return "\n".join(lines[start:end]).strip()


def main() -> int:
    args = parse_args()
    if not args.file.exists():
        raise SystemExit(f"Checklist not found: {args.file}")
    print(extract_section(args.file.read_text(encoding="utf-8"), args.section))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
