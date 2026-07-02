#!/usr/bin/env python3
"""Create a new yallaplay-wiki markdown page from a safe template."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path


DEFAULT_WIKI_ROOT = Path(__file__).resolve().parents[1] / "yallaplay-wiki"
KIND_DIRS = {
    "finding": "operations/incidents",
    "definition": "reference",
    "methodology": "reference",
    "query": "reference/queries",
    "schema": "reference/warehouse",
    "runbook": "operations",
    "tool": "tools",
}


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "untitled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=sorted(KIND_DIRS), help="page template kind")
    parser.add_argument("title", help="page title")
    parser.add_argument("--wiki-root", type=Path, default=DEFAULT_WIKI_ROOT)
    parser.add_argument("--domain", help="frontmatter domain override")
    parser.add_argument("--slug", help="filename slug override")
    parser.add_argument("--dir", help="directory under wiki root override")
    parser.add_argument("--force", action="store_true", help="overwrite an existing file")
    parser.add_argument("--print", action="store_true", help="print the template instead of writing")
    return parser.parse_args()


def current_git_user() -> str:
    try:
        result = subprocess.run(
            ["git", "config", "user.name"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return ""
    return result.stdout.strip()


def template(kind: str, title: str, domain: str) -> str:
    today = date.today().isoformat()
    author = current_git_user()
    common = f"""---
title: {title}
domain: {domain}
status: draft
source_refs: []
see_also: []
built: {today}
---

# {title}
"""
    if kind == "finding":
        return common + """
## Summary

- 

## Evidence

- 

## Caveats

- 

## Future Use

- 
"""
    if kind == "definition":
        return common + """
## Definition


## Why It Matters


## Related Terms

- 
"""
    if kind == "methodology":
        return common + """
## Use When


## Procedure

1. 

## Bias Traps

- 

## Output Pattern

- 
"""
    if kind == "query":
        return common + """
## Purpose


## Parameters

- 

## SQL

```sql
-- Add reusable SQL here.
```

## Caveats

- 
"""
    if kind == "schema":
        return common + """
## Object


## Columns / Fields

- 

## Known Gotchas

- 

## Example Queries

- 
"""
    if kind == "runbook":
        return common + """
## Trigger


## Steps

1. 

## Escalation

- 

## Rollback / Safety

- 
"""
    if kind == "tool":
        return common + """
## What It Is For


## Access / Auth


## Data It Holds


## Safe Usage

- 
"""
    raise ValueError(kind)


def main() -> int:
    args = parse_args()
    root = args.wiki_root.resolve()
    if not root.exists():
        print(f"wiki root not found: {root}", file=sys.stderr)
        return 2

    domain = args.domain or args.kind
    content = template(args.kind, args.title, domain)
    if args.print:
        print(content, end="")
        return 0

    rel_dir = Path(args.dir or KIND_DIRS[args.kind])
    if rel_dir.is_absolute() or ".." in rel_dir.parts:
        print("--dir must stay inside the wiki root", file=sys.stderr)
        return 2

    slug = slugify(args.slug or args.title)
    path = root / rel_dir / f"{slug}.md"
    if path.exists() and not args.force:
        print(f"refusing to overwrite existing file: {path.relative_to(root)}", file=sys.stderr)
        return 1

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(path.relative_to(root).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
