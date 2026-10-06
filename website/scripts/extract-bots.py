#!/usr/bin/env python3
"""Publish the reviewed Bot Marketplace without weakening its canonical schema.

The backend catalog models are the only validator. Any malformed entry or removal
row aborts extraction before existing website feeds are touched.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hermes_cli.bot_catalog import (  # noqa: E402
    BotCatalogError,
    BotCatalogEntry,
    RemovedBot,
    _tree_catalog,
)

DEFAULT_CATALOG_DIR = REPO_ROOT / "bot-catalog"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "website" / "static" / "api"
DEFAULT_WORKFLOW_ROOT = REPO_ROOT / "optional-skills" / "bots"
TIERS = ("official", "community")
SAMPLE_PREVIEW_LIMIT = 4000
_SLUG_RE = re.compile(r"[^a-z0-9._-]+")


def maintainer_slug(maintainer: str) -> str:
    """Match Plugin Catalog author slugs while excluding reserved dot routes."""
    slug = _SLUG_RE.sub("-", maintainer.strip().lower()).strip("-")
    return "unknown" if not slug or slug in {".", ".."} else slug


def author_page_path(slug: str) -> str:
    return f"/bots/by/{quote(slug, safe='')}"


def _sample_title(content: str, fallback: str) -> str:
    for line in content.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            return match.group(1)[:120]
    return fallback.replace("-", " ").replace("_", " ").title()


def _workflow_package(workflow_root: Path, name: str) -> dict | None:
    root = workflow_root / name
    if not (root / "SKILL.md").is_file():
        return None
    samples: list[dict] = []
    candidates = [(root / "references" / "sample-input.md", "input")]
    templates = root / "templates"
    if templates.is_dir():
        candidates.extend((path, "template") for path in sorted(templates.iterdir()) if path.is_file())
    example_tasks: list[str] = []
    for path, kind in candidates:
        try:
            content = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            continue
        if not content:
            continue
        preview = content[:SAMPLE_PREVIEW_LIMIT]
        samples.append(
            {
                "kind": kind,
                "title": _sample_title(content, path.stem),
                "path": path.relative_to(root).as_posix(),
                "preview": preview,
                "truncated": len(content) > len(preview),
            }
        )
        if kind == "input":
            match = re.search(
                r"(?im)^(?:\*\*)?Expected task:(?:\*\*)?\s*(.+(?:\n(?!\s*#|\s*$).+)*)",
                content,
            )
            if match:
                task = " ".join(line.strip() for line in match.group(1).splitlines()).strip()
                if task:
                    example_tasks.append(task[:500])
    if not samples:
        return None
    return {"skill": f"official/bots/{name}", "exampleTasks": example_tasks, "samples": samples}


def _page_entry(entry: BotCatalogEntry, workflow_root: Path) -> tuple[dict, dict]:
    canonical = entry.model_dump(mode="json")
    slug = maintainer_slug(entry.maintainer)
    page = {
        **canonical,
        "maintainerSlug": slug,
        "authorPath": author_page_path(slug),
    }
    workflow_package = _workflow_package(workflow_root, entry.name)
    if workflow_package is not None:
        page["workflowPackage"] = workflow_package
        if workflow_package["exampleTasks"]:
            page["starterExample"] = workflow_package["exampleTasks"][0]
    if "starterExample" not in page:
        page["starterExample"] = entry.summary
    return page, canonical


def _validated_catalog(catalog_dir: Path) -> tuple[list[BotCatalogEntry], list[RemovedBot]]:
    # This shared loader owns byte/count limits, defaults, closed-field checks,
    # duplicate detection, skill/toolset review, and removal-list validation.
    return _tree_catalog(catalog_dir)


def load_catalog(
    catalog_dir: Path, workflow_root: Path = DEFAULT_WORKFLOW_ROOT
) -> tuple[list[dict], list[dict]]:
    entries, _ = _validated_catalog(catalog_dir)
    normalized = [_page_entry(entry, workflow_root) for entry in entries]
    pages = sorted((page for page, _ in normalized), key=lambda item: item["name"])
    canonical = sorted((feed for _, feed in normalized), key=lambda item: item["name"])
    return pages, canonical


def load_catalog_entries(catalog_dir: Path) -> list[dict]:
    return load_catalog(catalog_dir)[0]


def load_removed(catalog_dir: Path) -> list[dict]:
    _, removed = _validated_catalog(catalog_dir)
    return [item.model_dump(mode="json") for item in removed]


def main(
    catalog_dir: Path = DEFAULT_CATALOG_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    workflow_root: Path = DEFAULT_WORKFLOW_ROOT,
) -> int:
    entries, removed_models = _validated_catalog(catalog_dir)
    normalized = [_page_entry(entry, workflow_root) for entry in entries]
    pages = [page for page, _ in normalized]
    canonical = [feed for _, feed in normalized]
    removed = [item.model_dump(mode="json") for item in removed_models]
    blocked = {entry["name"] for entry in removed}
    pages = sorted((entry for entry in pages if entry["name"] not in blocked), key=lambda item: item["name"])
    canonical = sorted((entry for entry in canonical if entry["name"] not in blocked), key=lambda item: item["name"])

    by_tier = Counter(entry["tier"] for entry in pages)
    by_category = Counter(entry["category"] for entry in pages)
    generated_at = datetime.now(timezone.utc).isoformat()
    documents = {
        "bots.json": pages,
        "bots-meta.json": {
            "generatedAt": generated_at,
            "total": len(pages),
            "byTier": {tier: by_tier.get(tier, 0) for tier in TIERS},
            "byCategory": dict(sorted(by_category.items())),
            "removedCount": len(removed),
        },
        "bot-catalog.json": {
            "generated_at": generated_at,
            "entries": canonical,
            "removed": removed,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        (output_dir / name).write_text(
            json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    print(f"Extracted {len(pages)} bot catalog entries to {output_dir / 'bots.json'}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, default=DEFAULT_CATALOG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workflow-root", type=Path, default=DEFAULT_WORKFLOW_ROOT)
    args = parser.parse_args()
    try:
        raise SystemExit(main(args.catalog_dir, args.output_dir, args.workflow_root))
    except BotCatalogError as exc:
        print(f"[extract-bots] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
