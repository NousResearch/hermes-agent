#!/usr/bin/env python3
"""Refresh GitHub star counts for plugin-catalog repos, at most once a day.

Writes ``website/static/api/plugin-stars.json``::

    {"fetched_at": "<ISO-8601 UTC>", "stars": {"owner/repo": 123, ...}}

``extract-plugins.py`` merges these into ``plugins.json`` so the catalog page can rank
entries by stars.

Rate-limit discipline follows the skills-index rule: GitHub is only ever consulted from the
scheduled ``skills-index.yml`` run (twice daily, ``--probe``), which uploads the result as an
artifact. Docs deploys run with ``--reuse-only`` and NEVER touch the API: they take the
scheduled artifact when given one, else the live site's copy (one CDN GET), else whatever is
on disk, else an empty map (the page then ranks alphabetically).

The probe starts with one GraphQL request for every catalog repo (aliased ``repository``
fields), then uses REST only for unresolved repos. Any failure (rate limit, network, bad
token) keeps the previous counts instead of regressing them to zero.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_DIR = REPO_ROOT / "plugin-catalog"
DEFAULT_OUTPUT = REPO_ROOT / "website" / "static" / "api" / "plugin-stars.json"
LIVE_URL = "https://hermes-agent.nousresearch.com/docs/api/plugin-stars.json"
_GITHUB_REPO_RE = re.compile(r"^https://github\.com/([^/\s]+)/([^/\s#?]+?)(?:\.git)?/?$")


def _log(msg: str) -> None:
    print(f"[fetch-plugin-stars] {msg}", file=sys.stderr)


def _warning(msg: str) -> None:
    print(f"::warning::{msg}", file=sys.stderr)


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


_GRAPHQL_URL = "https://api.github.com/graphql"


def _graphql(query: str, token: str) -> dict:
    req = urllib.request.Request(
        _GRAPHQL_URL, data=json.dumps({"query": query}).encode("utf-8"), method="POST",
        headers={"User-Agent": "hermes-agent-docs", "Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def stars_query(slugs: list[str]) -> str:
    fields = "\n".join(
        f'r{i}: repository(owner: {json.dumps(slug.split("/", 1)[0])}, name: {json.dumps(slug.split("/", 1)[1])})'
        " { stargazerCount }"
        for i, slug in enumerate(slugs))
    return "query {\n" + fields + "\n}"


def probe_stars(
    slugs: list[str], previous: dict[str, int], token: str | None,
) -> tuple[dict[str, int], bool]:
    """Fetch every slug, preserving cached counts for repos that remain unresolved."""
    if not slugs:
        return {}, True
    if not token:
        _log("no GITHUB_TOKEN; keeping previous counts without probing")
        return {s: previous[s] for s in slugs if s in previous}, False

    fresh: dict[str, int] = {}
    try:
        payload = _graphql(stars_query(slugs), token)
    except (urllib.error.URLError, OSError, ValueError) as e:
        _log(f"GraphQL probe failed ({e}); keeping previous counts")
        return {slug: previous[slug] for slug in slugs if slug in previous}, False
    else:
        if not isinstance(payload, dict):
            _log("GraphQL probe returned a malformed payload; keeping previous counts")
            return {slug: previous[slug] for slug in slugs if slug in previous}, False
        errors = payload.get("errors")
        if isinstance(errors, list):
            for err in errors:
                _log(f"GraphQL: {err.get('message') if isinstance(err, dict) else err}")
        elif errors is not None:
            _log("GraphQL probe returned malformed errors")
            return {slug: previous[slug] for slug in slugs if slug in previous}, False
        data = payload.get("data")
        if not isinstance(data, dict):
            _log("GraphQL probe returned no repository data; keeping previous counts")
            return {slug: previous[slug] for slug in slugs if slug in previous}, False
        for i, slug in enumerate(slugs):
            alias = f"r{i}"
            if alias not in data:
                _log(f"GraphQL probe omitted {alias}; keeping previous counts")
                return {slug: previous[slug] for slug in slugs if slug in previous}, False
            node = data[alias]
            if node is None:
                continue
            count = node.get("stargazerCount") if isinstance(node, dict) else None
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                _log(f"GraphQL probe returned a malformed star count for {slug}; keeping previous counts")
                return {slug: previous[slug] for slug in slugs if slug in previous}, False
            fresh[slug] = count

    headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}"}
    for slug in (slug for slug in slugs if slug not in fresh):
        try:
            payload = _http_json(f"https://api.github.com/repos/{slug}", headers)
            if not isinstance(payload, dict):
                _log(f"REST fallback returned a malformed payload for {slug}")
                continue
            count = payload.get("stargazers_count")
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                fresh[slug] = count
            else:
                _log(f"REST fallback returned no star count for {slug}")
        except urllib.error.HTTPError as e:
            _log(f"REST fallback failed for {slug} ({e})")
            if e.code in (403, 429):
                _log("REST fallback rate limited; keeping previous counts for remaining repos")
                break
        except (urllib.error.URLError, OSError, ValueError) as e:
            _log(f"REST fallback failed for {slug} ({e})")

    stars = {slug: previous[slug] for slug in slugs if slug in previous}
    stars.update(fresh)
    return stars, len(fresh) == len(slugs)


def main(catalog_dir: Path = DEFAULT_CATALOG_DIR, output: Path = DEFAULT_OUTPUT,
         probe: bool = False, live_url: str | None = LIVE_URL, token: str | None = None) -> int:
    previous = load_previous(output, live_url)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not probe:
        output.write_text(json.dumps(previous or {"fetched_at": None, "stars": {}},
                                     separators=(",", ":")), encoding="utf-8")
        print(f"Reused plugin stars from {previous.get('fetched_at')} "
              f"({len(previous.get('stars', {}))} repos, no GitHub calls)")
        return 0

    slugs = catalog_slugs(catalog_dir)
    prev_stars = {k: int(v) for k, v in (previous.get("stars") or {}).items() if isinstance(v, (int, float))}
    stars, probe_succeeded = probe_stars(
        slugs, prev_stars, token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"))
    fetched_at = (datetime.now(timezone.utc).isoformat() if probe_succeeded
                  else previous.get("fetched_at"))
    if not probe_succeeded:
        missing = [slug for slug in slugs if slug not in stars]
        if missing:
            _warning("Plugin star probe incomplete; missing cached counts for: " + ", ".join(missing))
    output.write_text(json.dumps({"fetched_at": fetched_at, "stars": stars}, separators=(",", ":")),
                      encoding="utf-8")
    print(f"Probed {len(slugs)} repos, wrote {len(stars)} star counts to {output}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, default=DEFAULT_CATALOG_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--probe", action="store_true",
                        help="call GitHub (one GraphQL request); only the scheduled skills-index run does this")
    parser.add_argument("--no-live", action="store_true", help="do not consult the live site's cache")
    args = parser.parse_args()
    sys.exit(main(catalog_dir=args.catalog_dir, output=args.output, probe=args.probe,
                  live_url=None if args.no_live else LIVE_URL))
