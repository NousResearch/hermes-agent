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

- ``plugins.json``        — list of catalog entries for the page (camelCase)
- ``plugins-meta.json``   — counts by tier + generatedAt + removedCount
- ``plugin-catalog.json`` — ``{"entries": [raw YAML mappings], "removed": [...]}`` in the loader's own
  schema; installed Hermes clients fetch this for live catalog refresh
  (``hermes_cli.plugin_catalog.LIVE_CATALOG_URL``) so new entries and removals reach them without
  updating. Emitting it here means the docs deploy IS the publish step — no second pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_DIR = REPO_ROOT / "plugin-catalog"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "website" / "static" / "api"
# Written by fetch-plugin-stars.py (at most one GitHub probe per day); absent → no ranking data.
DEFAULT_STARS_FILE = DEFAULT_OUTPUT_DIR / "plugin-stars.json"
_GITHUB_REPO_RE = re.compile(r"^https://github\.com/([^/\s]+)/([^/\s#?]+?)(?:\.git)?/?$")

CATALOG_TIERS = ("official", "community")
CATALOG_CATEGORIES = ("desktop", "memory", "platform", "web", "tools", "voice", "automation", "models", "general")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
# Keep in sync with scripts/validate_plugin_catalog.py (cosmetic fields attached to the pin).
VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,31}$")
IMAGE_HOSTS = ("raw.githubusercontent.com", "github.com")
IMAGE_HOST_SUFFIX = ".githubusercontent.com"
MAX_SCREENSHOTS = 6
# Forges whose raw-file URL scheme the site knows; ``readme: true`` on any other host is ignored.
_FORGE_REPO_RE = re.compile(r"^https://(github\.com|gitlab\.com)/([^/\s]+)/([^/\s#?]+?)(?:\.git)?/?$")
_SLUG_RE = re.compile(r"[^a-z0-9._-]+")


def _log(msg: str) -> None:
    print(f"[extract-plugins] {msg}", file=sys.stderr)


def _is_allowed_image_url(url: str) -> bool:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    return parts.scheme == "https" and bool(host) and (host in IMAGE_HOSTS or host.endswith(IMAGE_HOST_SUFFIX))


def _cosmetic(value, accept, file_name: str, entry: str, key: str) -> str:
    """A cosmetic field is dropped, never fatal: the site must not lose an entry a reviewer merged."""
    text = str(value or "").strip()
    if text and not accept(text):
        _log(f"{file_name} ({entry}): dropping invalid {key} {text!r}")
        return ""
    return text


def maintainer_slug(maintainer: str) -> str:
    """URL segment for the author page (/docs/plugins/by/<slug>); shared by every entry of one maintainer."""
    return _SLUG_RE.sub("-", maintainer.strip().lower()).strip("-") or "unknown"


def readme_url(repo: str, sha: str, subdir: str) -> str:
    """Raw README.md URL at the pinned commit for GitHub/GitLab repos; "" when the forge is unknown."""
    m = _FORGE_REPO_RE.match(repo)
    if not m:
        return ""
    host, owner, name = m.groups()
    path = f"{subdir.strip('/')}/README.md" if subdir.strip("/") else "README.md"
    if host == "github.com":
        return f"https://raw.githubusercontent.com/{owner}/{name}/{sha}/{path}"
    return f"https://gitlab.com/{owner}/{name}/-/raw/{sha}/{path}"


def _screenshots(raw, file_name: str, entry: str) -> list[str]:
    shots = [s for s in _str_list(raw) if _is_allowed_image_url(s)]
    if len(shots) != len(_str_list(raw)):
        _log(f"{file_name} ({entry}): dropping screenshots off GitHub hosts")
    return shots[:MAX_SCREENSHOTS]


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


def load_stars(stars_file: Path) -> dict[str, int]:
    """``{"owner/repo": stars}`` from fetch-plugin-stars.py's cache; empty when missing/unreadable."""
    try:
        data = json.loads(stars_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    raw = data.get("stars") if isinstance(data, dict) else None
    return {str(k): int(v) for k, v in raw.items() if isinstance(v, (int, float))} if isinstance(raw, dict) else {}


def _repo_stars(repo: str, stars: dict[str, int]) -> int | None:
    m = _GITHUB_REPO_RE.match(repo)
    return stars.get(f"{m.group(1)}/{m.group(2)}") if m else None


def load_git_dates(catalog_dir: Path) -> dict[str, dict[str, str]]:
    """``{"<file>.yaml": {"addedAt": iso, "updatedAt": iso}}`` from the catalog's git history.

    One ``git log`` over the directory, newest first: the first commit seen touching a file is
    its last update, the last one is when it entered the catalog. Renames (``R``) carry the
    old path's history onto the new name so a renamed entry keeps its original addedAt.
    Committer dates, not author dates: rebase-merged PRs are stamped when they LAND on main,
    which is when the entry actually became installable.

    Empty when git is unavailable or the checkout is shallow — a depth-1 clone would report
    every entry as added in the tip commit, which is worse than no dates (deploy-site.yml
    checks out with ``fetch-depth: 0`` for this reason).
    """
    if not catalog_dir.is_dir():
        return {}
    try:
        shallow = subprocess.run(
            ["git", "rev-parse", "--is-shallow-repository"],
            cwd=catalog_dir, capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip()
        if shallow == "true":
            _log("shallow checkout: skipping addedAt/updatedAt (need fetch-depth: 0)")
            return {}
        log = subprocess.run(
            ["git", "log", "--format=%x00%cI", "--name-status", "--relative", "-M", "--", "."],
            cwd=catalog_dir, capture_output=True, text=True, check=True, timeout=120,
        ).stdout
    except (OSError, subprocess.SubprocessError) as e:
        _log(f"git history unavailable, skipping addedAt/updatedAt ({e})")
        return {}

    dates: dict[str, dict[str, str]] = {}
    alias: dict[str, str] = {}  # old file name → current name, for renamed entries

    def touch(name: str, when: str) -> None:
        if "/" in name or not name.endswith(".yaml"):
            return
        while name in alias:
            name = alias[name]
        rec = dates.setdefault(name, {"addedAt": when, "updatedAt": when})
        rec["addedAt"] = when  # newest-first walk: the last write wins = oldest commit

    when = ""
    for line in log.splitlines():
        if line.startswith("\x00"):
            when = line[1:].strip()
            continue
        parts = line.split("\t")
        if len(parts) < 2 or not when:
            continue
        if parts[0].startswith("R") and len(parts) == 3:
            old, new = parts[1], parts[2]
            touch(new, when)
            alias[old] = new
            continue
        touch(parts[-1], when)
    return dates


def load_catalog_entries(catalog_dir: Path, stars: dict[str, int] | None = None,
                         dates: dict[str, dict[str, str]] | None = None) -> list[dict]:
    """Parse all ``*.yaml`` files (except removed.yaml) into page entries.

    Entries missing any of name/repo/sha are skipped with a stderr log —
    a malformed community entry must never break the docs deploy.
    ``dates`` is ``load_git_dates()`` output keyed by file name; absent → null addedAt/updatedAt.
    """
    entries: list[dict] = []
    stars = stars or {}
    dates = dates or {}
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
        category = str(raw.get("category") or "desktop").strip().lower()
        if category not in CATALOG_CATEGORIES:
            _log(f"{path.name} ({name}): unknown category {category!r}, treating as general")
            category = "general"

        subdir = str(raw.get("subdir") or "").strip()
        entries.append({
            "name": name,
            "description": str(raw.get("description") or "").strip(),
            "repo": repo,
            "sha": sha,
            "shaShort": sha[:7],
            "tier": tier,
            "category": category,
            "maintainer": str(raw.get("maintainer") or "").strip(),
            "subdir": subdir,
            "requiresHermes": str(raw.get("requires_hermes") or "").strip(),
            "platforms": _str_list(raw.get("platforms")),
            "capabilities": _normalize_capabilities(raw.get("capabilities")),
            "docsUrl": str(raw.get("docs_url") or "").strip(),
            "version": _cosmetic(raw.get("version"), VERSION_RE.match, path.name, name, "version"),
            "image": _cosmetic(raw.get("image"), _is_allowed_image_url, path.name, name, "image"),
            "screenshots": _screenshots(raw.get("screenshots"), path.name, name),
            # README renders by default from the pinned commit; `readme: false` opts an entry out.
            "readme": raw.get("readme") is not False and bool(readme_url(repo, sha, subdir)),
            "readmeUrl": readme_url(repo, sha, subdir) if raw.get("readme") is not False else "",
            "maintainerSlug": maintainer_slug(str(raw.get("maintainer") or "")),
            "installCommand": f"hermes plugins install {name}",
            "stars": _repo_stars(repo, stars),
            "addedAt": dates.get(path.name, {}).get("addedAt"),
            "updatedAt": dates.get(path.name, {}).get("updatedAt"),
        })

    # Most-starred first (unknown = 0), then name so the order is stable; tier does not rank.
    entries.sort(key=lambda e: (-(e["stars"] or 0), e["name"]))
    return entries


def _stars_fetched_at(stars_file: Path) -> str | None:
    try:
        data = json.loads(stars_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return str(data.get("fetched_at")) if isinstance(data, dict) and data.get("fetched_at") else None


def load_removed(catalog_dir: Path) -> list[dict]:
    """``removed:`` list from plugin-catalog/removed.yaml (mappings only)."""
    removed_path = catalog_dir / "removed.yaml"
    if not removed_path.is_file():
        return []
    try:
        raw = yaml.safe_load(removed_path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError) as e:
        _log(f"could not read removed.yaml: {e}")
        return []
    removed = raw.get("removed") if isinstance(raw, dict) else None
    return [r for r in removed if isinstance(r, dict)] if isinstance(removed, list) else []


def count_removed(catalog_dir: Path) -> int:
    return len(load_removed(catalog_dir))


def load_raw_entries(catalog_dir: Path) -> list[dict]:
    """Raw entry mappings (loader schema, snake_case) for ``plugin-catalog.json``; the client re-validates."""
    entries: list[dict] = []
    if not catalog_dir.is_dir():
        return entries
    for path in sorted(catalog_dir.glob("*.yaml")):
        if path.name == "removed.yaml":
            continue
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (yaml.YAMLError, OSError):
            continue
        if isinstance(raw, dict) and raw.get("name") and raw.get("repo") and raw.get("sha"):
            entries.append(raw)
    return entries


def main(catalog_dir: Path = DEFAULT_CATALOG_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR,
         stars_file: Path | None = None) -> int:
    if not catalog_dir.is_dir():
        _log(
            f"plugin-catalog directory not found at {catalog_dir}; "
            "emitting empty catalog (this is expected until the catalog lands)"
        )

    stars_path = stars_file if stars_file is not None else output_dir / "plugin-stars.json"
    stars = load_stars(stars_path)
    entries = load_catalog_entries(catalog_dir, stars, load_git_dates(catalog_dir))
    removed_count = count_removed(catalog_dir)

    by_tier = Counter(e["tier"] for e in entries)
    by_category = Counter(e["category"] for e in entries)
    meta = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "total": len(entries),
        "byTier": {tier: by_tier.get(tier, 0) for tier in CATALOG_TIERS},
        "byCategory": {c: by_category.get(c, 0) for c in CATALOG_CATEGORIES if by_category.get(c)},
        "starsFetchedAt": _stars_fetched_at(stars_path),
        "removedCount": removed_count,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "plugins.json", "w", encoding="utf-8") as f:
        json.dump(entries, f, separators=(",", ":"), ensure_ascii=False)
    with open(output_dir / "plugins-meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, separators=(",", ":"), ensure_ascii=False)
    with open(output_dir / "plugin-catalog.json", "w", encoding="utf-8") as f:
        json.dump({"generated_at": meta["generatedAt"], "entries": load_raw_entries(catalog_dir),
                   "removed": load_removed(catalog_dir)}, f, separators=(",", ":"), ensure_ascii=False)

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
    parser.add_argument("--stars-file", type=Path, default=None,
                        help="plugin-stars.json from fetch-plugin-stars.py (default: <output-dir>/plugin-stars.json)")
    args = parser.parse_args()
    sys.exit(main(catalog_dir=args.catalog_dir, output_dir=args.output_dir, stars_file=args.stars_file))
