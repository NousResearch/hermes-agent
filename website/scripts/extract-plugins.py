#!/usr/bin/env python3
"""Extract plugin-catalog entries into website/static/api/plugins.json.

Feeds the Plugin Catalog page at /docs/plugins (website/src/pages/plugins/).

Data source: ``plugin-catalog/*.yaml`` at the repo root — one YAML file per
catalog entry (see plugin-catalog/README.md for the entry schema), plus
``plugin-catalog/removed.yaml`` listing plugins pulled from the catalog.
No network, no crawling: the catalog is human-merged data in the checkout.

Graceful degradation: when ``plugin-catalog/`` does not exist yet (the
catalog PR may not have merged), we emit an EMPTY catalog list and zeroed
meta counts and exit 0 so the docs build stays green. The page renders a
"catalog is just getting started" state.

Outputs (both under website/static/api/, CDN-served at /docs/api/):

- ``plugins.json``       — list of catalog entries for the page
- ``plugins-meta.json``  — counts by tier + generatedAt + removedCount
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_DIR = REPO_ROOT / "plugin-catalog"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "website" / "static" / "api"

CATALOG_TIERS = ("official", "community")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _log(msg: str) -> None:
    print(f"[extract-plugins] {msg}", file=sys.stderr)


def _str_list(value) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(x) for x in value if x]
    return []


def _normalize_capabilities(raw) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    return {
        "providesTools": _str_list(raw.get("provides_tools")),
        "providesHooks": _str_list(raw.get("provides_hooks")),
        "providesMiddleware": _str_list(raw.get("provides_middleware")),
        "requiresEnv": _str_list(raw.get("requires_env")),
    }


def load_catalog_entries(catalog_dir: Path) -> list[dict]:
    """Parse all ``*.yaml`` files (except removed.yaml) into page entries.

    Entries missing any of name/repo/sha are skipped with a stderr log —
    a malformed community entry must never break the docs deploy.
    """
    entries: list[dict] = []
    if not catalog_dir.is_dir():
        return entries

    for path in sorted(catalog_dir.glob("*.yaml")):
        if path.name == "removed.yaml":
            continue
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (yaml.YAMLError, OSError) as e:
            _log(f"skipping {path.name}: unreadable YAML ({e})")
            continue
        if not isinstance(raw, dict):
            _log(f"skipping {path.name}: not a mapping")
            continue

        name = str(raw.get("name") or "").strip()
        repo = str(raw.get("repo") or "").strip()
        sha = str(raw.get("sha") or "").strip().lower()
        missing = [
            field
            for field, value in (("name", name), ("repo", repo), ("sha", sha))
            if not value
        ]
        if missing:
            _log(f"skipping {path.name}: missing required field(s) {', '.join(missing)}")
            continue
        if not SHA_RE.match(sha):
            _log(f"skipping {path.name} ({name}): sha is not a 40-hex commit pin")
            continue

        tier = str(raw.get("tier") or "community").strip().lower()
        if tier not in CATALOG_TIERS:
            _log(f"{path.name} ({name}): unknown tier {tier!r}, treating as community")
            tier = "community"

        entries.append({
            "name": name,
            "description": str(raw.get("description") or "").strip(),
            "repo": repo,
            "sha": sha,
            "shaShort": sha[:7],
            "tier": tier,
            "maintainer": str(raw.get("maintainer") or "").strip(),
            "requiresHermes": str(raw.get("requires_hermes") or "").strip(),
            "platforms": _str_list(raw.get("platforms")),
            "capabilities": _normalize_capabilities(raw.get("capabilities")),
            "docsUrl": str(raw.get("docs_url") or "").strip(),
            "installCommand": f"hermes plugins install {name}",
        })

    entries.sort(key=lambda e: (0 if e["tier"] == "official" else 1, e["name"]))
    return entries


def count_removed(catalog_dir: Path) -> int:
    """Number of entries in plugin-catalog/removed.yaml (``removed:`` list)."""
    removed_path = catalog_dir / "removed.yaml"
    if not removed_path.is_file():
        return 0
    try:
        raw = yaml.safe_load(removed_path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError) as e:
        _log(f"could not read removed.yaml: {e}")
        return 0
    if not isinstance(raw, dict):
        return 0
    removed = raw.get("removed")
    return len(removed) if isinstance(removed, list) else 0


def main(catalog_dir: Path = DEFAULT_CATALOG_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> int:
    if not catalog_dir.is_dir():
        _log(
            f"plugin-catalog directory not found at {catalog_dir}; "
            "emitting empty catalog (this is expected until the catalog lands)"
        )

    entries = load_catalog_entries(catalog_dir)
    removed_count = count_removed(catalog_dir)

    by_tier = Counter(e["tier"] for e in entries)
    meta = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "total": len(entries),
        "byTier": {tier: by_tier.get(tier, 0) for tier in CATALOG_TIERS},
        "removedCount": removed_count,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "plugins.json", "w", encoding="utf-8") as f:
        json.dump(entries, f, separators=(",", ":"), ensure_ascii=False)
    with open(output_dir / "plugins-meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, separators=(",", ":"), ensure_ascii=False)

    print(
        f"Extracted {len(entries)} plugin catalog entries "
        f"({meta['byTier']['official']} official, {meta['byTier']['community']} community, "
        f"{removed_count} removed) to {output_dir / 'plugins.json'}"
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, default=DEFAULT_CATALOG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    sys.exit(main(catalog_dir=args.catalog_dir, output_dir=args.output_dir))
