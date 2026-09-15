#!/usr/bin/env python3
"""Refresh GitHub star counts for plugin-catalog repos, at most once a day.

Writes ``website/static/api/plugin-stars.json``::

    {"fetched_at": "<ISO-8601 UTC>", "stars": {"owner/repo": 123, ...}}

``extract-plugins.py`` merges these into ``plugins.json`` so the catalog page can rank
entries by stars.

Rate-limit discipline (the whole point of this file): the docs site deploys many times a
day and GitHub's per-installation API budget is shared with every other workflow. So this
script NEVER calls the GitHub API unless the cache is stale:

1. Download the live site's current ``plugin-stars.json`` (one CDN GET, not the API).
2. If its ``fetched_at`` is younger than ``--max-age-hours`` (default 24), write it back to
   disk unchanged and exit. Zero GitHub calls.
3. Otherwise probe ``GET /repos/{owner}/{repo}`` once per unique repo. A 403/429 or any
   network error keeps the previous count for that repo rather than dropping it.

Local ``npm run build`` without network or token degrades to whatever is on disk / an
empty map; the page then simply ranks alphabetically.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_DIR = REPO_ROOT / "plugin-catalog"
DEFAULT_OUTPUT = REPO_ROOT / "website" / "static" / "api" / "plugin-stars.json"
LIVE_URL = "https://hermes-agent.nousresearch.com/docs/api/plugin-stars.json"
_GITHUB_REPO_RE = re.compile(r"^https://github\.com/([^/\s]+)/([^/\s#?]+?)(?:\.git)?/?$")


def _log(msg: str) -> None:
    print(f"[fetch-plugin-stars] {msg}", file=sys.stderr)


def github_slug(repo_url: str) -> str | None:
    """``owner/repo`` for a github.com URL, else None (non-GitHub hosts are never probed)."""
    m = _GITHUB_REPO_RE.match(repo_url.strip())
    return f"{m.group(1)}/{m.group(2)}" if m else None


def catalog_slugs(catalog_dir: Path) -> list[str]:
    slugs: set[str] = set()
    for path in sorted(catalog_dir.glob("*.yaml")):
        if path.name == "removed.yaml":
            continue
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (yaml.YAMLError, OSError):
            continue
        slug = github_slug(str((raw or {}).get("repo") or "")) if isinstance(raw, dict) else None
        if slug:
            slugs.add(slug)
    return sorted(slugs)


def _http_json(url: str, headers: dict[str, str], timeout: float = 15.0):
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-agent-docs", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_previous(output: Path, live_url: str | None) -> dict:
    """Newest of {live site copy, on-disk copy}; ``{}`` when neither exists."""
    candidates: list[dict] = []
    if live_url:
        try:
            data = _http_json(live_url, {})
            if isinstance(data, dict) and isinstance(data.get("stars"), dict):
                candidates.append(data)
        except (urllib.error.URLError, OSError, ValueError) as e:
            _log(f"live cache unavailable ({e}); continuing without it")
    if output.is_file():
        try:
            data = json.loads(output.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("stars"), dict):
                candidates.append(data)
        except (OSError, ValueError):
            pass
    return max(candidates, key=lambda d: str(d.get("fetched_at") or ""), default={})


def is_fresh(previous: dict, max_age: timedelta, now: datetime) -> bool:
    try:
        fetched = datetime.fromisoformat(str(previous.get("fetched_at")))
    except (TypeError, ValueError):
        return False
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=timezone.utc)
    return now - fetched < max_age


def probe_stars(slugs: list[str], previous: dict[str, int], token: str | None) -> dict[str, int]:
    """One ``GET /repos/{slug}`` each; on any failure keep the previous count (never regress to 0)."""
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    stars: dict[str, int] = {}
    rate_limited = False
    for slug in slugs:
        if rate_limited:
            if slug in previous:
                stars[slug] = previous[slug]
            continue
        try:
            data = _http_json(f"https://api.github.com/repos/{slug}", headers)
            stars[slug] = int(data.get("stargazers_count") or 0)
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                _log(f"rate limited at {slug} (HTTP {e.code}); keeping previous counts for the rest")
                rate_limited = True
            else:
                _log(f"{slug}: HTTP {e.code}; keeping previous count")
            if slug in previous:
                stars[slug] = previous[slug]
        except (urllib.error.URLError, OSError, ValueError) as e:
            _log(f"{slug}: {e}; keeping previous count")
            if slug in previous:
                stars[slug] = previous[slug]
    return stars


def main(catalog_dir: Path = DEFAULT_CATALOG_DIR, output: Path = DEFAULT_OUTPUT,
         max_age_hours: float = 24.0, force: bool = False, live_url: str | None = LIVE_URL,
         token: str | None = None) -> int:
    now = datetime.now(timezone.utc)
    previous = load_previous(output, live_url)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not force and is_fresh(previous, timedelta(hours=max_age_hours), now):
        output.write_text(json.dumps(previous, separators=(",", ":")), encoding="utf-8")
        print(f"Reused plugin stars from {previous.get('fetched_at')} "
              f"({len(previous.get('stars', {}))} repos, no GitHub calls)")
        return 0

    slugs = catalog_slugs(catalog_dir)
    prev_stars = {k: int(v) for k, v in (previous.get("stars") or {}).items() if isinstance(v, (int, float))}
    stars = probe_stars(slugs, prev_stars, token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"))
    fetched_at = now.isoformat() if stars else str(previous.get("fetched_at") or "")
    output.write_text(json.dumps({"fetched_at": fetched_at, "stars": stars}, separators=(",", ":")),
                      encoding="utf-8")
    print(f"Probed {len(slugs)} repos, wrote {len(stars)} star counts to {output}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, default=DEFAULT_CATALOG_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-age-hours", type=float, default=24.0)
    parser.add_argument("--force", action="store_true", help="probe GitHub even if the cache is fresh")
    parser.add_argument("--no-live", action="store_true", help="do not consult the live site's cache")
    args = parser.parse_args()
    sys.exit(main(catalog_dir=args.catalog_dir, output=args.output, max_age_hours=args.max_age_hours,
                  force=args.force, live_url=None if args.no_live else LIVE_URL))
