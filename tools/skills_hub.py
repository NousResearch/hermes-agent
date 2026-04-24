#!/usr/bin/env python3
"""
Skills Hub — Source adapters and hub state management for the Hermes Skills Hub.

This is a library module (not an agent tool). It provides:
  - GitHubAuth: Shared GitHub API authentication (PAT, gh CLI, GitHub App)
  - SkillSource ABC: Interface for all skill registry adapters
  - OptionalSkillSource: Official optional skills shipped with the repo (not activated by default)
  - GitHubSource: Fetch skills from any GitHub repo via the Contents API
  - HubLockFile: Track provenance of installed hub skills
  - Hub state directory management (quarantine, audit log, taps, index cache)

Used by hermes_cli/skills_hub.py for CLI commands and the /skills slash command.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from hermes_constants import get_hermes_home
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse, urlunparse

import httpx
import yaml

from tools.skills_guard import (
    ScanResult, content_hash, TRUSTED_REPOS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERMES_HOME = get_hermes_home()
SKILLS_DIR = HERMES_HOME / "skills"
HUB_DIR = SKILLS_DIR / ".hub"
LOCK_FILE = HUB_DIR / "lock.json"
QUARANTINE_DIR = HUB_DIR / "quarantine"
AUDIT_LOG = HUB_DIR / "audit.log"
TAPS_FILE = HUB_DIR / "taps.json"
INDEX_CACHE_DIR = HUB_DIR / "index-cache"

# Cache duration for remote index fetches
INDEX_CACHE_TTL = 3600  # 1 hour


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SkillMeta:
    """Minimal metadata returned by search results."""
    name: str
    description: str
    source: str           # "official", "github", "clawhub", "claude-marketplace", "lobehub"
    identifier: str       # source-specific ID (e.g. "openai/skills/skill-creator")
    trust_level: str      # "builtin" | "trusted" | "community"
    repo: Optional[str] = None
    path: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillBundle:
    """A downloaded skill ready for quarantine/scanning/installation."""
    name: str
    files: Dict[str, Union[str, bytes]]   # relative_path -> file content
    source: str
    identifier: str
    trust_level: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _normalize_bundle_path(path_value: str, *, field_name: str, allow_nested: bool) -> str:
    """Normalize and validate bundle-controlled paths before touching disk."""
    if not isinstance(path_value, str):
        raise ValueError(f"Unsafe {field_name}: expected a string")

    raw = path_value.strip()
    if not raw:
        raise ValueError(f"Unsafe {field_name}: empty path")

    normalized = raw.replace("\\", "/")
    path = PurePosixPath(normalized)
    parts = [part for part in path.parts if part not in ("", ".")]

    if normalized.startswith("/") or path.is_absolute():
        raise ValueError(f"Unsafe {field_name}: {path_value}")
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe {field_name}: {path_value}")
    if re.fullmatch(r"[A-Za-z]:", parts[0]):
        raise ValueError(f"Unsafe {field_name}: {path_value}")
    if not allow_nested and len(parts) != 1:
        raise ValueError(f"Unsafe {field_name}: {path_value}")

    return "/".join(parts)


def _validate_skill_name(name: str) -> str:
    return _normalize_bundle_path(name, field_name="skill name", allow_nested=False)


def _validate_category_name(category: str) -> str:
    return _normalize_bundle_path(category, field_name="category", allow_nested=False)


def _validate_bundle_rel_path(rel_path: str) -> str:
    return _normalize_bundle_path(rel_path, field_name="bundle file path", allow_nested=True)


# ---------------------------------------------------------------------------
# GitHub Authentication
# ---------------------------------------------------------------------------

class GitHubAuth:
    """
    GitHub API authentication. Tries methods in priority order:
      1. GITHUB_TOKEN / GH_TOKEN env var (PAT — the default)
      2. `gh auth token` subprocess (if gh CLI is installed)
      3. GitHub App JWT + installation token (if app credentials configured)
      4. Unauthenticated (60 req/hr, public repos only)
    """

    def __init__(self):
        self._cached_token: Optional[str] = None
        self._cached_method: Optional[str] = None
        self._app_token_expiry: float = 0

    def get_headers(self) -> Dict[str, str]:
        """Return authorization headers for GitHub API requests."""
        token = self._resolve_token()
        headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            headers["Authorization"] = f"token {token}"
        return headers

    def is_authenticated(self) -> bool:
        return self._resolve_token() is not None

    def auth_method(self) -> str:
        """Return which auth method is active: 'pat', 'gh-cli', 'github-app', or 'anonymous'."""
        self._resolve_token()
        return self._cached_method or "anonymous"

    def _resolve_token(self) -> Optional[str]:
        # Return cached token if still valid
        if self._cached_token:
            if self._cached_method != "github-app" or time.time() < self._app_token_expiry:
                return self._cached_token

        # 1. Environment variable
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            self._cached_token = token
            self._cached_method = "pat"
            return token

        # 2. gh CLI
        token = self._try_gh_cli()
        if token:
            self._cached_token = token
            self._cached_method = "gh-cli"
            return token

        # 3. GitHub App
        token = self._try_github_app()
        if token:
            self._cached_token = token
            self._cached_method = "github-app"
            self._app_token_expiry = time.time() + 3500  # ~58 min (tokens last 1 hour)
            return token

        self._cached_method = "anonymous"
        return None

    def _try_gh_cli(self) -> Optional[str]:
        """Try to get a token from the gh CLI."""
        try:
            result = subprocess.run(
                ["gh", "auth", "token"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.debug("gh CLI token lookup failed: %s", e)
        return None

    def _try_github_app(self) -> Optional[str]:
        """Try GitHub App JWT authentication if credentials are configured."""
        app_id = os.environ.get("GITHUB_APP_ID")
        key_path = os.environ.get("GITHUB_APP_PRIVATE_KEY_PATH")
        installation_id = os.environ.get("GITHUB_APP_INSTALLATION_ID")

        if not all([app_id, key_path, installation_id]):
            return None

        try:
            import jwt  # PyJWT
        except ImportError:
            logger.debug("PyJWT not installed, skipping GitHub App auth")
            return None

        try:
            key_file = Path(key_path)
            if not key_file.exists():
                return None
            private_key = key_file.read_text()

            now = int(time.time())
            payload = {
                "iat": now - 60,
                "exp": now + (10 * 60),
                "iss": app_id,
            }
            encoded_jwt = jwt.encode(payload, private_key, algorithm="RS256")

            resp = httpx.post(
                f"https://api.github.com/app/installations/{installation_id}/access_tokens",
                headers={
                    "Authorization": f"Bearer {encoded_jwt}",
                    "Accept": "application/vnd.github.v3+json",
                },
                timeout=10,
            )
            if resp.status_code == 201:
                return resp.json().get("token")
        except Exception as e:
            logger.debug(f"GitHub App auth failed: {e}")

        return None


# ---------------------------------------------------------------------------
# Source adapter interface
# ---------------------------------------------------------------------------

class SkillSource(ABC):
    """Abstract base for all skill registry adapters."""

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        """Search for skills matching a query string."""
        ...

    @abstractmethod
    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        """Download a skill bundle by identifier."""
        ...

    @abstractmethod
    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        """Fetch metadata for a skill without downloading all files."""
        ...

    @abstractmethod
    def source_id(self) -> str:
        """Unique identifier for this source (e.g. 'github', 'clawhub')."""
        ...

    def trust_level_for(self, identifier: str) -> str:
        """Determine trust level for a skill from this source."""
        return "community"


# ---------------------------------------------------------------------------
# GitHub source adapter
# ---------------------------------------------------------------------------

class GitHubSource(SkillSource):
    """Fetch skills from GitHub repos via the Contents API."""

    DEFAULT_TAPS = [
        {"repo": "openai/skills", "path": "skills/"},
        {"repo": "anthropics/skills", "path": "skills/"},
        {"repo": "VoltAgent/awesome-agent-skills", "path": "skills/"},
        {"repo": "garrytan/gstack", "path": ""},
        {"repo": "MiniMax-AI/cli", "path": "skill/"},
    ]

    def __init__(self, auth: GitHubAuth, extra_taps: Optional[List[Dict]] = None):
        self.auth = auth
        self.taps = list(self.DEFAULT_TAPS)
        if extra_taps:
            self.taps.extend(extra_taps)
        # Per-instance cache: repo -> (default_branch, tree_entries)
        # Survives within a single search/install flow, avoiding redundant API calls.
        self._tree_cache: Dict[str, Tuple[str, List[dict]]] = {}
        # Set when GitHub returns 403 with rate limit exhausted
        self._rate_limited: bool = False

    def source_id(self) -> str:
        return "github"

    @property
    def is_rate_limited(self) -> bool:
        """Whether GitHub API rate limit was hit during operations."""
        return self._rate_limited

    def trust_level_for(self, identifier: str) -> str:
        # identifier format: "owner/repo/path/to/skill"
        parts = identifier.split("/", 2)
        if len(parts) >= 2:
            repo = f"{parts[0]}/{parts[1]}"
            if repo in TRUSTED_REPOS:
                return "trusted"
        return "community"

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        """Search all taps for skills matching the query."""
        results: List[SkillMeta] = []
        query_lower = query.lower()

        for tap in self.taps:
            try:
                skills = self._list_skills_in_repo(tap["repo"], tap.get("path", ""))
                for skill in skills:
                    searchable = f"{skill.name} {skill.description} {' '.join(skill.tags)}".lower()
                    if query_lower in searchable:
                        results.append(skill)
            except Exception as e:
                logger.debug(f"Failed to search {tap['repo']}: {e}")
                continue

        # Deduplicate by name, preferring higher trust levels
        _trust_rank = {"builtin": 2, "trusted": 1, "community": 0}
        seen = {}
        for r in results:
            if r.name not in seen:
                seen[r.name] = r
            elif _trust_rank.get(r.trust_level, 0) > _trust_rank.get(seen[r.name].trust_level, 0):
                seen[r.name] = r
        results = list(seen.values())

        return results[:limit]

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        """
        Download a skill from GitHub.
        identifier format: "owner/repo/path/to/skill-dir"
        """
        parts = identifier.split("/", 2)
        if len(parts) < 3:
            return None

        repo = f"{parts[0]}/{parts[1]}"
        skill_path = parts[2]

        files = self._download_directory(repo, skill_path)
        if not files or "SKILL.md" not in files:
            return None

        skill_name = skill_path.rstrip("/").split("/")[-1]
        trust = self.trust_level_for(identifier)

        return SkillBundle(
            name=skill_name,
            files=files,
            source="github",
            identifier=identifier,
            trust_level=trust,
        )

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        """Fetch just the SKILL.md metadata for preview."""
        parts = identifier.split("/", 2)
        if len(parts) < 3:
            return None

        repo = f"{parts[0]}/{parts[1]}"
        skill_path = parts[2].rstrip("/")
        skill_md_path = f"{skill_path}/SKILL.md"

        content = self._fetch_file_content(repo, skill_md_path)
        if not content:
            return None

        fm = self._parse_frontmatter_quick(content)
        skill_name = fm.get("name", skill_path.split("/")[-1])
        description = fm.get("description", "")

        tags = []
        metadata = fm.get("metadata", {})
        if isinstance(metadata, dict):
            hermes_meta = metadata.get("hermes", {})
            if isinstance(hermes_meta, dict):
                tags = hermes_meta.get("tags", [])
        if not tags:
            raw_tags = fm.get("tags", [])
            tags = raw_tags if isinstance(raw_tags, list) else []

        return SkillMeta(
            name=skill_name,
            description=str(description),
            source="github",
            identifier=identifier,
            trust_level=self.trust_level_for(identifier),
            repo=repo,
            path=skill_path,
            tags=[str(t) for t in tags],
        )

    # -- Internal helpers --

    def _list_skills_in_repo(self, repo: str, path: str) -> List[SkillMeta]:
        """List skill directories in a GitHub repo path, using cached index."""
        cache_key = f"{repo}_{path}".replace("/", "_").replace(" ", "_")
        cached = self._read_cache(cache_key)
        if cached is not None:
            return [SkillMeta(**s) for s in cached]

        url = f"https://api.github.com/repos/{repo}/contents/{path.rstrip('/')}"
        try:
            resp = httpx.get(url, headers=self.auth.get_headers(), timeout=15, follow_redirects=True)
            if resp.status_code != 200:
                return []
        except httpx.HTTPError:
            return []

        entries = resp.json()
        if not isinstance(entries, list):
            return []

        skills: List[SkillMeta] = []
        for entry in entries:
            if entry.get("type") != "dir":
                continue

            dir_name = entry["name"]
            if dir_name.startswith((".", "_")):
                continue

            prefix = path.rstrip("/")
            skill_identifier = f"{repo}/{prefix}/{dir_name}" if prefix else f"{repo}/{dir_name}"
            meta = self.inspect(skill_identifier)
            if meta:
                skills.append(meta)

        # Cache the results
        self._write_cache(cache_key, [self._meta_to_dict(s) for s in skills])
        return skills

    # -- Repo tree cache (avoids redundant API calls) --

    def _get_repo_tree(self, repo: str) -> Optional[Tuple[str, List[dict]]]:
        """Get cached or fresh repo tree.

        Returns ``(default_branch, tree_entries)`` or ``None``.
        A single install can call ``_download_directory_via_tree`` and
        ``_find_skill_in_repo_tree`` multiple times for the same repo — this
        cache eliminates the redundant ``GET /repos/{repo}`` +
        ``GET /repos/{repo}/git/trees/{branch}`` round-trips (previously up to
        6 duplicated pairs per install, consuming ~12 of the 60/hr
        unauthenticated rate limit for nothing).
        """
        if repo in self._tree_cache:
            return self._tree_cache[repo]

        headers = self.auth.get_headers()

        # Resolve default branch
        try:
            resp = httpx.get(
                f"https://api.github.com/repos/{repo}",
                headers=headers, timeout=15, follow_redirects=True,
            )
            if resp.status_code != 200:
                self._check_rate_limit_response(resp)
                return None
            default_branch = resp.json().get("default_branch", "main")
        except (httpx.HTTPError, ValueError):
            return None

        # Fetch recursive tree
        try:
            resp = httpx.get(
                f"https://api.github.com/repos/{repo}/git/trees/{default_branch}",
                params={"recursive": "1"},
                headers=headers, timeout=30, follow_redirects=True,
            )
            if resp.status_code != 200:
                self._check_rate_limit_response(resp)
                return None
            tree_data = resp.json()
            if tree_data.get("truncated"):
                logger.debug("Git tree truncated for %s, cannot cache", repo)
                return None
        except (httpx.HTTPError, ValueError):
            return None

        entries = tree_data.get("tree", [])
        self._tree_cache[repo] = (default_branch, entries)
        return (default_branch, entries)

    def _check_rate_limit_response(self, resp: "httpx.Response") -> None:
        """Flag the instance as rate-limited when GitHub returns 403 + exhausted quota."""
        if resp.status_code == 403:
            remaining = resp.headers.get("X-RateLimit-Remaining", "")
            if remaining == "0":
                self._rate_limited = True
                logger.warning(
                    "GitHub API rate limit exhausted (unauthenticated: 60 req/hr). "
                    "Set GITHUB_TOKEN or install the gh CLI to raise the limit to 5,000/hr."
                )

    def _download_directory(self, repo: str, path: str) -> Dict[str, str]:
        """Recursively download all text files from a GitHub directory.

        Uses the Git Trees API first (single call for the entire tree) to
        avoid per-directory rate limiting that causes silent subdirectory
        loss.  Falls back to the recursive Contents API when the tree
        endpoint is unavailable or the response is truncated.
        """
        files = self._download_directory_via_tree(repo, path)
        if files is not None:
            return files
        logger.debug("Tree API unavailable for %s/%s, falling back to Contents API", repo, path)
        return self._download_directory_recursive(repo, path)

    def _download_directory_via_tree(self, repo: str, path: str) -> Optional[Dict[str, str]]:
        """Download an entire directory using the Git Trees API (single request).

        Returns:
            dict of files if the path exists and has content,
            empty dict ``{}`` if the tree is cached but the path doesn't exist
            (prevents unnecessary Contents API fallback),
            ``None`` if the tree couldn't be fetched (triggers Contents API fallback).
        """
        path = path.rstrip("/")

        cached = self._get_repo_tree(repo)
        if cached is None:
            return None
        _default_branch, tree_entries = cached

        # Check if ANY entry lives under the target path
        prefix = f"{path}/"
        has_entries = any(
            item.get("path", "").startswith(prefix) for item in tree_entries
        )
        if not has_entries:
            # Path definitively doesn't exist in the repo — return empty
            # instead of None to skip the Contents API fallback.
            return {}

        # Filter to blobs under our target path and fetch content
        files: Dict[str, str] = {}
        for item in tree_entries:
            if item.get("type") != "blob":
                continue
            item_path = item.get("path", "")
            if not item_path.startswith(prefix):
                continue
            rel_path = item_path[len(prefix):]
            content = self._fetch_file_content(repo, item_path)
            if content is not None:
                files[rel_path] = content
            else:
                logger.debug("Skipped file (fetch failed): %s/%s", repo, item_path)

        return files if files else None

    def _download_directory_recursive(self, repo: str, path: str) -> Dict[str, str]:
        """Recursively download via Contents API (fallback)."""
        url = f"https://api.github.com/repos/{repo}/contents/{path.rstrip('/')}"
        try:
            resp = httpx.get(url, headers=self.auth.get_headers(), timeout=15, follow_redirects=True)
            if resp.status_code != 200:
                logger.debug("Contents API returned %d for %s/%s", resp.status_code, repo, path)
                return {}
        except httpx.HTTPError:
            return {}

        entries = resp.json()
        if not isinstance(entries, list):
            return {}

        files: Dict[str, str] = {}
        for entry in entries:
            name = entry.get("name", "")
            entry_type = entry.get("type", "")

            if entry_type == "file":
                content = self._fetch_file_content(repo, entry.get("path", ""))
                if content is not None:
                    rel_path = name
                    files[rel_path] = content
            elif entry_type == "dir":
                sub_files = self._download_directory_recursive(repo, entry.get("path", ""))
                if not sub_files:
                    logger.debug("Empty or failed subdirectory: %s/%s", repo, entry.get("path", ""))
                for sub_name, sub_content in sub_files.items():
                    files[f"{name}/{sub_name}"] = sub_content

        return files

    def _find_skill_in_repo_tree(self, repo: str, skill_name: str) -> Optional[str]:
        """Use the GitHub Trees API to find a skill directory anywhere in the repo.

        Returns the full identifier (``repo/path/to/skill``) or ``None``.
        This is a single API call regardless of repo depth, so it efficiently
        handles deeply nested directory structures like
        ``cli-tool/components/skills/development/<skill>/SKILL.md``.
        """
        cached = self._get_repo_tree(repo)
        if cached is None:
            return None
        _default_branch, tree_entries = cached

        # Look for SKILL.md files inside directories named <skill_name>
        skill_md_suffix = f"/{skill_name}/SKILL.md"
        for entry in tree_entries:
            if entry.get("type") != "blob":
                continue
            path = entry.get("path", "")
            if path.endswith(skill_md_suffix) or path == f"{skill_name}/SKILL.md":
                # Strip /SKILL.md to get the skill directory path
                skill_dir = path[: -len("/SKILL.md")]
                return f"{repo}/{skill_dir}"

        return None

    def _fetch_file_content(self, repo: str, path: str) -> Optional[str]:
        """Fetch a single file's content from GitHub."""
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        try:
            resp = httpx.get(
                url,
                headers={**self.auth.get_headers(), "Accept": "application/vnd.github.v3.raw"},
                timeout=15, follow_redirects=True,
            )
            if resp.status_code == 200:
                return resp.text
            self._check_rate_limit_response(resp)
        except httpx.HTTPError as e:
            logger.debug("GitHub contents API fetch failed: %s", e)
        return None

    def _read_cache(self, key: str) -> Optional[list]:
        """Read cached index if not expired."""
        cache_file = INDEX_CACHE_DIR / f"{key}.json"
        if not cache_file.exists():
            return None
        try:
            stat = cache_file.stat()
            if time.time() - stat.st_mtime > INDEX_CACHE_TTL:
                return None
            return json.loads(cache_file.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, key: str, data: list) -> None:
        """Write index data to cache."""
        INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = INDEX_CACHE_DIR / f"{key}.json"
        try:
            cache_file.write_text(json.dumps(data, ensure_ascii=False))
        except OSError as e:
            logger.debug("Could not write cache: %s", e)

    @staticmethod
    def _meta_to_dict(meta: SkillMeta) -> dict:
        return {
            "name": meta.name,
            "description": meta.description,
            "source": meta.source,
            "identifier": meta.identifier,
            "trust_level": meta.trust_level,
            "repo": meta.repo,
            "path": meta.path,
            "tags": meta.tags,
        }

    @staticmethod
    def _parse_frontmatter_quick(content: str) -> dict:
        """Parse YAML frontmatter from SKILL.md content."""
        if not content.startswith("---"):
            return {}
        match = re.search(r'\n---\s*\n', content[3:])
        if not match:
            return {}
        yaml_text = content[3:match.start() + 3]
        try:
            parsed = yaml.safe_load(yaml_text)
            return parsed if isinstance(parsed, dict) else {}
        except yaml.YAMLError:
            return {}


# ---------------------------------------------------------------------------
# Well-known Agent Skills endpoint source adapter
# ---------------------------------------------------------------------------

class WellKnownSkillSource(SkillSource):
    """Read skills from a domain exposing /.well-known/skills/index.json."""

    BASE_PATH = "/.well-known/skills"

    def source_id(self) -> str:
        return "well-known"

    def trust_level_for(self, identifier: str) -> str:
        return "community"

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        index_url = self._query_to_index_url(query)
        if not index_url:
            return []

        parsed = self._parse_index(index_url)
        if not parsed:
            return []

        results: List[SkillMeta] = []
        for entry in parsed["skills"][:limit]:
            name = entry.get("name")
            if not isinstance(name, str) or not name:
                continue
            description = entry.get("description", "")
            files = entry.get("files", ["SKILL.md"])
            results.append(SkillMeta(
                name=name,
                description=str(description),
                source="well-known",
                identifier=self._wrap_identifier(parsed["base_url"], name),
                trust_level="community",
                path=name,
                extra={
                    "index_url": parsed["index_url"],
                    "base_url": parsed["base_url"],
                    "files": files if isinstance(files, list) else ["SKILL.md"],
                },
            ))
        return results

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        parsed = self._parse_identifier(identifier)
        if not parsed:
            return None

        entry = self._index_entry(parsed["index_url"], parsed["skill_name"])
        if not entry:
            return None

        skill_md = self._fetch_text(f"{parsed['skill_url']}/SKILL.md")
        if skill_md is None:
            return None

        fm = GitHubSource._parse_frontmatter_quick(skill_md)
        description = str(fm.get("description") or entry.get("description") or "")
        name = str(fm.get("name") or parsed["skill_name"])
        return SkillMeta(
            name=name,
            description=description,
            source="well-known",
            identifier=self._wrap_identifier(parsed["base_url"], parsed["skill_name"]),
            trust_level="community",
            path=parsed["skill_name"],
            extra={
                "index_url": parsed["index_url"],
                "base_url": parsed["base_url"],
                "files": entry.get("files", ["SKILL.md"]),
                "endpoint": parsed["skill_url"],
            },
        )

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        parsed = self._parse_identifier(identifier)
        if not parsed:
            return None

        try:
            skill_name = _validate_skill_name(parsed["skill_name"])
        except ValueError:
            logger.warning("Well-known skill identifier contained unsafe skill name: %s", identifier)
            return None

        entry = self._index_entry(parsed["index_url"], parsed["skill_name"])
        if not entry:
            return None

        files = entry.get("files", ["SKILL.md"])
        if not isinstance(files, list) or not files:
            files = ["SKILL.md"]

        downloaded: Dict[str, str] = {}
        for rel_path in files:
            if not isinstance(rel_path, str) or not rel_path:
                continue
            try:
                safe_rel_path = _validate_bundle_rel_path(rel_path)
            except ValueError:
                logger.warning(
                    "Well-known skill %s advertised unsafe file path: %r",
                    identifier,
                    rel_path,
                )
                return None
            text = self._fetch_text(f"{parsed['skill_url']}/{safe_rel_path}")
            if text is None:
                return None
            downloaded[safe_rel_path] = text

        if "SKILL.md" not in downloaded:
            return None

        return SkillBundle(
            name=skill_name,
            files=downloaded,
            source="well-known",
            identifier=self._wrap_identifier(parsed["base_url"], skill_name),
            trust_level="community",
            metadata={
                "index_url": parsed["index_url"],
                "base_url": parsed["base_url"],
                "endpoint": parsed["skill_url"],
                "files": files,
            },
        )

    def _query_to_index_url(self, query: str) -> Optional[str]:
        query = query.strip()
        if not query.startswith(("http://", "https://")):
            return None
        if query.endswith("/index.json"):
            return query
        if f"{self.BASE_PATH}/" in query:
            base_url = query.split(f"{self.BASE_PATH}/", 1)[0] + self.BASE_PATH
            return f"{base_url}/index.json"
        return query.rstrip("/") + f"{self.BASE_PATH}/index.json"

    def _parse_identifier(self, identifier: str) -> Optional[dict]:
        raw = identifier[len("well-known:"):] if identifier.startswith("well-known:") else identifier
        if not raw.startswith(("http://", "https://")):
            return None

        parsed_url = urlparse(raw)
        clean_url = urlunparse(parsed_url._replace(fragment=""))
        fragment = parsed_url.fragment

        if clean_url.endswith("/index.json"):
            if not fragment:
                return None
            base_url = clean_url[:-len("/index.json")]
            skill_name = fragment
            skill_url = f"{base_url}/{skill_name}"
            return {
                "index_url": clean_url,
                "base_url": base_url,
                "skill_name": skill_name,
                "skill_url": skill_url,
            }

        if clean_url.endswith("/SKILL.md"):
            skill_url = clean_url[:-len("/SKILL.md")]
        else:
            skill_url = clean_url.rstrip("/")

        if f"{self.BASE_PATH}/" not in skill_url:
            return None

        base_url, skill_name = skill_url.rsplit("/", 1)
        return {
            "index_url": f"{base_url}/index.json",
            "base_url": base_url,
            "skill_name": skill_name,
            "skill_url": skill_url,
        }

    def _parse_index(self, index_url: str) -> Optional[dict]:
        cache_key = f"well_known_index_{hashlib.md5(index_url.encode()).hexdigest()}"
        cached = _read_index_cache(cache_key)
        if isinstance(cached, dict) and isinstance(cached.get("skills"), list):
            return cached

        try:
            resp = httpx.get(index_url, timeout=20, follow_redirects=True)
            if resp.status_code != 200:
                return None
            data = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            return None

        skills = data.get("skills", []) if isinstance(data, dict) else []
        if not isinstance(skills, list):
            return None

        parsed = {
            "index_url": index_url,
            "base_url": index_url[:-len("/index.json")],
            "skills": skills,
        }
        _write_index_cache(cache_key, parsed)
        return parsed

    def _index_entry(self, index_url: str, skill_name: str) -> Optional[dict]:
        parsed = self._parse_index(index_url)
        if not parsed:
            return None
        for entry in parsed["skills"]:
            if isinstance(entry, dict) and entry.get("name") == skill_name:
                return entry
        return None

    @staticmethod
    def _fetch_text(url: str) -> Optional[str]:
        try:
            resp = httpx.get(url, timeout=20, follow_redirects=True)
            if resp.status_code == 200:
                return resp.text
        except httpx.HTTPError:
            return None
        return None

    @staticmethod
    def _wrap_identifier(base_url: str, skill_name: str) -> str:
        return f"well-known:{base_url.rstrip('/')}/{skill_name}"


# ---------------------------------------------------------------------------
# skills.sh source adapter
# ---------------------------------------------------------------------------

class SkillsShSource(SkillSource):
    """Discover skills via skills.sh and fetch content from the underlying GitHub repo."""

    BASE_URL = "https://skills.sh"
    SEARCH_URL = f"{BASE_URL}/api/search"
    _SKILL_LINK_RE = re.compile(r'href=["\']/(?P<id>(?!agents/|_next/|api/)[^"\'/]+/[^"\'/]+/[^"\'/]+)["\']')
    _INSTALL_CMD_RE = re.compile(
        r'npx\s+skills\s+add\s+(?P<repo>https?://github\.com/[^\s<]+|[^\s<]+)'
        r'(?:\s+--skill\s+(?P<skill>[^\s<]+))?',
        re.IGNORECASE,
    )
    _PAGE_H1_RE = re.compile(r'<h1[^>]*>(?P<title>.*?)</h1>', re.IGNORECASE | re.DOTALL)
    _PROSE_H1_RE = re.compile(
        r'<div[^>]*class=["\'][^"\']*prose[^"\']*["\'][^>]*>.*?<h1[^>]*>(?P<title>.*?)</h1>',
        re.IGNORECASE | re.DOTALL,
    )
    _PROSE_P_RE = re.compile(
        r'<div[^>]*class=["\'][^"\']*prose[^"\']*["\'][^>]*>.*?<p[^>]*>(?P<body>.*?)</p>',
        re.IGNORECASE | re.DOTALL,
    )
    _WEEKLY_INSTALLS_RE = re.compile(r'Weekly Installs.*?children\\":\\"(?P<count>[0-9.,Kk]+)\\"', re.DOTALL)

    def __init__(self, auth: GitHubAuth):
        self.auth = auth
        self.github = GitHubSource(auth=auth)

    def source_id(self) -> str:
        return "skills-sh"

    def trust_level_for(self, identifier: str) -> str:
        return self.github.trust_level_for(self._normalize_identifier(identifier))

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        if not query.strip():
            return self._featured_skills(limit)

        cache_key = f"skills_sh_search_{hashlib.md5(f'{query}|{limit}'.encode()).hexdigest()}"
        cached = _read_index_cache(cache_key)
        if cached is not None:
            return [SkillMeta(**item) for item in cached][:limit]

        try:
            resp = httpx.get(
                self.SEARCH_URL,
                params={"q": query, "limit": limit},
                timeout=20,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            return []

        items = data.get("skills", []) if isinstance(data, dict) else []
        if not isinstance(items, list):
            return []

        results: List[SkillMeta] = []
        for item in items[:limit]:
            meta = self._meta_from_search_item(item)
            if meta:
                results.append(meta)

        _write_index_cache(cache_key, [_skill_meta_to_dict(item) for item in results])
        return results

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        canonical = self._normalize_identifier(identifier)
        detail = self._fetch_detail_page(canonical)
        for candidate in self._candidate_identifiers(canonical):
            bundle = self.github.fetch(candidate)
            if bundle:
                bundle.source = "skills.sh"
                bundle.identifier = self._wrap_identifier(canonical)
                bundle.metadata.update(self._detail_to_metadata(canonical, detail))
                return bundle

        resolved = self._discover_identifier(canonical, detail=detail)
        if resolved:
            bundle = self.github.fetch(resolved)
            if bundle:
                bundle.source = "skills.sh"
                bundle.identifier = self._wrap_identifier(canonical)
                bundle.metadata.update(self._detail_to_metadata(canonical, detail))
                return bundle
        return None

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        canonical = self._normalize_identifier(identifier)
        detail = self._fetch_detail_page(canonical)
        meta = self._resolve_github_meta(canonical, detail=detail)
        if meta:
            return self._finalize_inspect_meta(meta, canonical, detail)
        return None

    def _featured_skills(self, limit: int) -> List[SkillMeta]:
        cache_key = "skills_sh_featured"
        cached = _read_index_cache(cache_key)
        if cached is not None:
            return [SkillMeta(**item) for item in cached][:limit]

        try:
            resp = httpx.get(self.BASE_URL, timeout=20)
            if resp.status_code != 200:
                return []
        except httpx.HTTPError:
            return []

        seen: set[str] = set()
        results: List[SkillMeta] = []
        for match in self._SKILL_LINK_RE.finditer(resp.text):
            canonical = match.group("id")
            if canonical in seen:
                continue
            seen.add(canonical)
            parts = canonical.split("/", 2)
            if len(parts) < 3:
                continue
            repo = f"{parts[0]}/{parts[1]}"
            skill_path = parts[2]
            results.append(SkillMeta(
                name=skill_path.split("/")[-1],
                description=f"Featured on skills.sh from {repo}",
                source="skills.sh",
                identifier=self._wrap_identifier(canonical),
                trust_level=self.github.trust_level_for(canonical),
                repo=repo,
                path=skill_path,
            ))
            if len(results) >= limit:
                break

        _write_index_cache(cache_key, [_skill_meta_to_dict(item) for item in results])
        return results

    def _meta_from_search_item(self, item: dict) -> Optional[SkillMeta]:
        if not isinstance(item, dict):
            return None

        canonical = item.get("id")
        repo = item.get("source")
        skill_path = item.get("skillId")
        if not isinstance(canonical, str) or canonical.count("/") < 2:
            if not (isinstance(repo, str) and isinstance(skill_path, str)):
                return None
            canonical = f"{repo}/{skill_path}"

        parts = canonical.split("/", 2)
        if len(parts) < 3:
            return None

        repo = f"{parts[0]}/{parts[1]}"
        skill_path = parts[2]
        installs = item.get("installs")
        installs_label = f" · {int(installs):,} installs" if isinstance(installs, int) else ""

        return SkillMeta(
            name=str(item.get("name") or skill_path.split("/")[-1]),
            description=f"Indexed by skills.sh from {repo}{installs_label}",
            source="skills.sh",
            identifier=self._wrap_identifier(canonical),
            trust_level=self.github.trust_level_for(canonical),
            repo=repo,
            path=skill_path,
            extra={
                "installs": installs,
                "detail_url": f"{self.BASE_URL}/{canonical}",
                "repo_url": f"https://github.com/{repo}",
            },
        )

    def _fetch_detail_page(self, identifier: str) -> Optional[dict]:
        cache_key = f"skills_sh_detail_{hashlib.md5(identifier.encode()).hexdigest()}"
        cached = _read_index_cache(cache_key)
        if isinstance(cached, dict):
            return cached

        try:
            resp = httpx.get(f"{self.BASE_URL}/{identifier}", timeout=20)
            if resp.status_code != 200:
                return None
        except httpx.HTTPError:
            return None

        detail = self._parse_detail_page(identifier, resp.text)
        if detail:
            _write_index_cache(cache_key, detail)
        return detail

    def _parse_detail_page(self, identifier: str, html: str) -> Optional[dict]:
        parts = identifier.split("/", 2)
        if len(parts) < 3:
            return None

        default_repo = f"{parts[0]}/{parts[1]}"
        skill_token = parts[2]
        repo = default_repo
        install_skill = skill_token

        install_command = None
        install_match = self._INSTALL_CMD_RE.search(html)
        if install_match:
            install_command = install_match.group(0).strip()
            repo_value = (install_match.group("repo") or "").strip()
            install_skill = (install_match.group("skill") or install_skill).strip()
            repo = self._extract_repo_slug(repo_value) or repo

        page_title = self._extract_first_match(self._PAGE_H1_RE, html)
        body_title = self._extract_first_match(self._PROSE_H1_RE, html)
        body_summary = self._extract_first_match(self._PROSE_P_RE, html)
        weekly_installs = self._extract_weekly_installs(html)
        security_audits = self._extract_security_audits(html, identifier)

        return {
            "repo": repo,
            "install_skill": install_skill,
            "page_title": page_title,
            "body_title": body_title,
            "body_summary": body_summary,
            "weekly_installs": weekly_installs,
            "install_command": install_command,
            "repo_url": f"https://github.com/{repo}",
            "detail_url": f"{self.BASE_URL}/{identifier}",
            "security_audits": security_audits,
        }

    def _discover_identifier(self, identifier: str, detail: Optional[dict] = None) -> Optional[str]:
        parts = identifier.split("/", 2)
        if len(parts) < 3:
            return None

        default_repo = f"{parts[0]}/{parts[1]}"
        repo = detail.get("repo", default_repo) if isinstance(detail, dict) else default_repo
        skill_token=parts[2].split("/")[-1]
        tokens=[skill_token]
        if isinstance(detail, dict):
            tokens.extend([
                detail.get("install_skill", ""),
                detail.get("page_title", ""),
                detail.get("body_title", ""),
            ])

        # Standard skill paths
        base_paths = ["skills/", ".agents/skills/", ".claude/skills/"]

        for base_path in base_paths:
            try:
                skills = self.github._list_skills_in_repo(repo, base_path)
            except Exception:
                continue
            for meta in skills:
                if self._matches_skill_tokens(meta, tokens):
                    return meta.identifier

        # Prefer a single recursive tree lookup before brute-forcing every
        # top-level directory. This avoids large request bursts on categorized
        # repos like borghei/claude-skills.
        tree_result = self.github._find_skill_in_repo_tree(repo, skill_token)
        if tree_result:
            return tree_result

        # Fallback: scan repo root for directories that might contain skills
        try:
            root_url = f"https://api.github.com/repos/{repo}/contents/"
            resp = httpx.get(root_url, headers=self.github.auth.get_headers(),
                             timeout=15, follow_redirects=True)
            if resp.status_code == 200:
                entries = resp.json()
                if isinstance(entries, list):
                    for entry in entries:
                        if entry.get("type") != "dir":
                            continue
                        dir_name = entry["name"]
                        if dir_name.startswith((".", "_")):
                            continue
                        if dir_name in ("skills", ".agents", ".claude"):
                            continue  # already tried
                        # Try direct: repo/dir/skill_token
                        direct_id = f"{repo}/{dir_name}/{skill_token}"
                        meta = self.github.inspect(direct_id)
                        if meta:
                            return meta.identifier
                        # Try listing skills in this directory
                        try:
                            skills = self.github._list_skills_in_repo(repo, dir_name + "/")
                        except Exception:
                            continue
                        for meta in skills:
                            if self._matches_skill_tokens(meta, tokens):
                                return meta.identifier
        except Exception:
            pass

        return None

    def _resolve_github_meta(self, identifier: str, detail: Optional[dict] = None) -> Optional[SkillMeta]:
        for candidate in self._candidate_identifiers(identifier):
            meta = self.github.inspect(candidate)
            if meta:
                return meta

        resolved = self._discover_identifier(identifier, detail=detail)
        if resolved:
            return self.github.inspect(resolved)
        return None

    def _finalize_inspect_meta(self, meta: SkillMeta, canonical: str, detail: Optional[dict]) -> SkillMeta:
        meta.source = "skills.sh"
        meta.identifier = self._wrap_identifier(canonical)
        meta.trust_level = self.trust_level_for(canonical)
        merged_extra = dict(meta.extra)
        merged_extra.update(self._detail_to_metadata(canonical, detail))
        meta.extra = merged_extra

        if isinstance(detail, dict):
            body_summary = detail.get("body_summary")
            weekly_installs = detail.get("weekly_installs")
            if body_summary:
                meta.description = body_summary
            elif meta.description and weekly_installs:
                meta.description = f"{meta.description} · {weekly_installs} weekly installs on skills.sh"
        return meta

    @classmethod
    def _matches_skill_tokens(cls, meta: SkillMeta, skill_tokens: List[str]) -> bool:
        candidates = set()
        candidates.update(cls._token_variants(meta.name))
        candidates.update(cls._token_variants(meta.path))
        candidates.update(cls._token_variants(meta.identifier.split("/", 2)[-1] if meta.identifier else None))

        for token in skill_tokens:
            variants = cls._token_variants(token)
            if variants & candidates:
                return True
        return False

    @staticmethod
    def _token_variants(value: Optional[str]) -> set[str]:
        if not value:
            return set()

        plain = SkillsShSource._strip_html(str(value)).strip().strip("/").lower()
        if not plain:
            return set()

        base = plain.split("/")[-1]
        sanitized = re.sub(r'[^a-z0-9/_-]+', '-', plain).strip('-')
        sanitized_base = sanitized.split("/")[-1] if sanitized else ""
        slash_tail = plain.split("/")[-1]
        slash_tail_clean = slash_tail.lstrip('@')
        slash_tail_clean = slash_tail_clean.split('/')[-1]

        variants = {
            plain,
            plain.replace("_", "-"),
            plain.replace("/", "-"),
            base,
            base.replace("_", "-"),
            base.replace("/", "-"),
            sanitized,
            sanitized.replace("/", "-") if sanitized else "",
            sanitized_base,
            slash_tail_clean,
            slash_tail_clean.replace("_", "-"),
        }
        return {v for v in variants if v}

    @staticmethod
    def _extract_repo_slug(repo_value: str) -> Optional[str]:
        repo_value = repo_value.strip()
        if repo_value.startswith("https://github.com/"):
            repo_value = repo_value[len("https://github.com/"):]
        repo_value = repo_value.strip("/")
        parts = repo_value.split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return None

    @staticmethod
    def _extract_first_match(pattern: re.Pattern, text: str) -> Optional[str]:
        match = pattern.search(text)
        if not match:
            return None
        value = next((group for group in match.groups() if group), None)
        if value is None:
            return None
        return SkillsShSource._strip_html(value).strip() or None

    def _detail_to_metadata(self, canonical: str, detail: Optional[dict]) -> Dict[str, Any]:
        parts = canonical.split("/", 2)
        repo = f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else ""
        metadata = {
            "detail_url": f"{self.BASE_URL}/{canonical}",
        }
        if repo:
            metadata["repo_url"] = f"https://github.com/{repo}"
        if isinstance(detail, dict):
            for key in ("weekly_installs", "install_command", "repo_url", "detail_url", "security_audits"):
                value = detail.get(key)
                if value:
                    metadata[key] = value
        return metadata

    @staticmethod
    def _extract_weekly_installs(html: str) -> Optional[str]:
        match = SkillsShSource._WEEKLY_INSTALLS_RE.search(html)
        if not match:
            return None
        return match.group("count")

    @staticmethod
    def _extract_security_audits(html: str, identifier: str) -> Dict[str, str]:
        audits: Dict[str, str] = {}
        for audit in ("agent-trust-hub", "socket", "snyk"):
            idx = html.find(f"/security/{audit}")
            if idx == -1:
                continue
            window = html[idx:idx + 500]
            match = re.search(r'(Pass|Warn|Fail)', window, re.IGNORECASE)
            if match:
                audits[audit] = match.group(1).title()
        return audits

    @staticmethod
    def _strip_html(value: str) -> str:
        return re.sub(r'<[^>]+>', '', value)

    @staticmethod
    def _normalize_identifier(identifier: str) -> str:
        prefix_aliases = (
            "skills-sh/",
            "skills.sh/",
            "skils-sh/",
            "skils.sh/",
        )
        for prefix in prefix_aliases:
            if identifier.startswith(prefix):
                return identifier[len(prefix):]
        return identifier

    @staticmethod
    def _candidate_identifiers(identifier: str) -> List[str]:
        parts = identifier.split("/", 2)
        if len(parts) < 3:
            return [identifier]

        repo = f"{parts[0]}/{parts[1]}"
        skill_path = parts[2].lstrip("/")
        candidates = [
            f"{repo}/{skill_path}",
            f"{repo}/skills/{skill_path}",
            f"{repo}/.agents/skills/{skill_path}",
            f"{repo}/.claude/skills/{skill_path}",
        ]

        seen = set()
        deduped: List[str] = []
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                deduped.append(candidate)
        return deduped

    @staticmethod
    def _wrap_identifier(identifier: str) -> str:
        return f"skills-sh/{identifier}"


# ---------------------------------------------------------------------------
# ClawHub source adapter
# ---------------------------------------------------------------------------

class ClawHubSource(SkillSource):
    """
    Fetch skills from ClawHub (clawhub.ai) via their HTTP API.
    All skills are treated as community trust — ClawHavoc incident showed
    their vetting is insufficient (341 malicious skills found Feb 2026).
    """

    BASE_URL = "https://clawhub.ai/api/v1"

    def source_id(self) -> str:
        return "clawhub"

    def trust_level_for(self, identifier: str) -> str:
        return "community"

    @staticmethod
    def _normalize_tags(tags: Any) -> List[str]:
        if isinstance(tags, list):
            return [str(t) for t in tags]
        if isinstance(tags, dict):
            return [str(k) for k in tags if str(k) != "latest"]
        return []

    @staticmethod
    def _coerce_skill_payload(data: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(data, dict):
            return None
        nested = data.get("skill")
        if isinstance(nested, dict):
            merged = dict(nested)
            latest_version = data.get("latestVersion")
            if latest_version is not None and "latestVersion" not in merged:
                merged["latestVersion"] = latest_version
            return merged
        return data

    @staticmethod
    def _query_terms(query: str) -> List[str]:
        return [term for term in re.split(r"[^a-z0-9]+", query.lower()) if term]

    @classmethod
    def _search_score(cls, query: str, meta: SkillMeta) -> int:
        query_norm = query.strip().lower()
        if not query_norm:
            return 1

        identifier = (meta.identifier or "").lower()
        name = (meta.name or "").lower()
        description = (meta.description or "").lower()
        normalized_identifier = " ".join(cls._query_terms(identifier))
        normalized_name = " ".join(cls._query_terms(name))
        query_terms = cls._query_terms(query_norm)
        identifier_terms = cls._query_terms(identifier)
        name_terms = cls._query_terms(name)
        score = 0

        if query_norm == identifier:
            score += 140
        if query_norm == name:
            score += 130
        if normalized_identifier == query_norm:
            score += 125
        if normalized_name == query_norm:
            score += 120
        if normalized_identifier.startswith(query_norm):
            score += 95
        if normalized_name.startswith(query_norm):
            score += 90
        if query_terms and identifier_terms[: len(query_terms)] == query_terms:
            score += 70
        if query_terms and name_terms[: len(query_terms)] == query_terms:
            score += 65
        if query_norm in identifier:
            score += 40
        if query_norm in name:
            score += 35
        if query_norm in description:
            score += 10

        for term in query_terms:
            if term in identifier_terms:
                score += 15
            if term in name_terms:
                score += 12
            if term in description:
                score += 3

        return score

    @staticmethod
    def _dedupe_results(results: List[SkillMeta]) -> List[SkillMeta]:
        seen: set[str] = set()
        deduped: List[SkillMeta] = []
        for result in results:
            key = (result.identifier or result.name).lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(result)
        return deduped

    def _exact_slug_meta(self, query: str) -> Optional[SkillMeta]:
        slug = query.strip().split("/")[-1]
        query_terms = self._query_terms(query)
        candidates: List[str] = []

        if slug and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", slug):
            candidates.append(slug)

        if query_terms:
            base_slug = "-".join(query_terms)
            if len(query_terms) >= 2:
                candidates.extend([
                    f"{base_slug}-agent",
                    f"{base_slug}-skill",
                    f"{base_slug}-tool",
                    f"{base_slug}-assistant",
                    f"{base_slug}-playbook",
                    base_slug,
                ])
            else:
                candidates.append(base_slug)

        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            meta = self.inspect(candidate)
            if meta:
                return meta

        return None

    def _finalize_search_results(self, query: str, results: List[SkillMeta], limit: int) -> List[SkillMeta]:
        query_norm = query.strip()
        if not query_norm:
            return self._dedupe_results(results)[:limit]

        filtered = [meta for meta in results if self._search_score(query_norm, meta) > 0]
        filtered.sort(
            key=lambda meta: (
                -self._search_score(query_norm, meta),
                meta.name.lower(),
                meta.identifier.lower(),
            )
        )
        filtered = self._dedupe_results(filtered)

        exact = self._exact_slug_meta(query_norm)
        if exact:
            filtered = [meta for meta in filtered if self._search_score(query_norm, meta) >= 20]
            filtered = self._dedupe_results([exact] + filtered)

        if filtered:
            return filtered[:limit]

        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", query_norm):
            return []

        return self._dedupe_results(results)[:limit]

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        query = query.strip()

        if query:
            query_terms = self._query_terms(query)
            if len(query_terms) >= 2:
                direct = self._exact_slug_meta(query)
                if direct:
                    return [direct]

            results = self._search_catalog(query, limit=limit)
            if results:
                return results

        # Empty query or catalog fallback failure: use the lightweight listing API.
        cache_key = f"clawhub_search_listing_v1_{hashlib.md5(query.encode()).hexdigest()}_{limit}"
        cached = _read_index_cache(cache_key)
        if cached is not None:
            return self._finalize_search_results(
                query,
                [SkillMeta(**s) for s in cached],
                limit,
            )

        try:
            resp = httpx.get(
                f"{self.BASE_URL}/skills",
                params={"search": query, "limit": limit},
                timeout=15,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            return []

        skills_data = data.get("items", data) if isinstance(data, dict) else data
        if not isinstance(skills_data, list):
            return []

        results = []
        for item in skills_data[:limit]:
            slug = item.get("slug")
            if not slug:
                continue
            display_name = item.get("displayName") or item.get("name") or slug
            summary = item.get("summary") or item.get("description") or ""
            tags = self._normalize_tags(item.get("tags", []))
            results.append(SkillMeta(
                name=display_name,
                description=summary,
                source="clawhub",
                identifier=slug,
                trust_level="community",
                tags=tags,
            ))

        final_results = self._finalize_search_results(query, results, limit)
        _write_index_cache(cache_key, [_skill_meta_to_dict(s) for s in final_results])
        return final_results

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        slug = identifier.split("/")[-1]

        skill_data = self._get_json(f"{self.BASE_URL}/skills/{slug}")
        if not isinstance(skill_data, dict):
            return None

        latest_version = self._resolve_latest_version(slug, skill_data)
        if not latest_version:
            logger.warning("ClawHub fetch failed for %s: could not resolve latest version", slug)
            return None

        # Primary method: download the skill as a ZIP bundle from /download
        files = self._download_zip(slug, latest_version)

        # Fallback: try the version metadata endpoint for inline/raw content
        if "SKILL.md" not in files:
            version_data = self._get_json(f"{self.BASE_URL}/skills/{slug}/versions/{latest_version}")
            if isinstance(version_data, dict):
                # Files may be nested under version_data["version"]["files"]
                files = self._extract_files(version_data) or files
                if "SKILL.md" not in files:
                    nested = version_data.get("version", {})
                    if isinstance(nested, dict):
                        files = self._extract_files(nested) or files

        if "SKILL.md" not in files:
            logger.warning(
                "ClawHub fetch for %s resolved version %s but could not retrieve file content",
                slug,
                latest_version,
            )
            return None

        return SkillBundle(
            name=slug,
            files=files,
            source="clawhub",
            identifier=slug,
            trust_level="community",
        )

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        slug = identifier.split("/")[-1]
        data = self._coerce_skill_payload(self._get_json(f"{self.BASE_URL}/skills/{slug}"))
        if not isinstance(data, dict):
            return None

        tags = self._normalize_tags(data.get("tags", []))

        return SkillMeta(
            name=data.get("displayName") or data.get("name") or data.get("slug") or slug,
            description=data.get("summary") or data.get("description") or "",
            source="clawhub",
            identifier=data.get("slug") or slug,
            trust_level="community",
            tags=tags,
        )

    def _search_catalog(self, query: str, limit: int = 10) -> List[SkillMeta]:
        cache_key = f"clawhub_search_catalog_v1_{hashlib.md5(f'{query}|{limit}'.encode()).hexdigest()}"
        cached = _read_index_cache(cache_key)
        if cached is not None:
            return [SkillMeta(**s) for s in cached][:limit]

        catalog = self._load_catalog_index()
        if not catalog:
            return []

        results = self._finalize_search_results(query, catalog, limit)
        _write_index_cache(cache_key, [_skill_meta_to_dict(s) for s in results])
        return results

    def _load_catalog_index(self) -> List[SkillMeta]:
        cache_key = "clawhub_catalog_v1"
        cached = _read_index_cache(cache_key)
        if cached is not None:
            return [SkillMeta(**s) for s in cached]

        cursor: Optional[str] = None
        results: List[SkillMeta] = []
        seen: set[str] = set()
        max_pages = 50

        for _ in range(max_pages):
            params: Dict[str, Any] = {"limit": 200}
            if cursor:
                params["cursor"] = cursor

            try:
                resp = httpx.get(f"{self.BASE_URL}/skills", params=params, timeout=30)
                if resp.status_code != 200:
                    break
                data = resp.json()
            except (httpx.HTTPError, json.JSONDecodeError):
                break

            items = data.get("items", []) if isinstance(data, dict) else []
            if not isinstance(items, list) or not items:
                break

            for item in items:
                slug = item.get("slug")
                if not isinstance(slug, str) or not slug or slug in seen:
                    continue
                seen.add(slug)
                display_name = item.get("displayName") or item.get("name") or slug
                summary = item.get("summary") or item.get("description") or ""
                tags = self._normalize_tags(item.get("tags", []))
                results.append(SkillMeta(
                    name=display_name,
                    description=summary,
                    source="clawhub",
                    identifier=slug,
                    trust_level="community",
                    tags=tags,
                ))

            cursor = data.get("nextCursor") if isinstance(data, dict) else None
            if not isinstance(cursor, str) or not cursor:
                break

        _write_index_cache(cache_key, [_skill_meta_to_dict(s) for s in results])
        return results

    def _get_json(self, url: str, timeout: int = 20) -> Optional[Any]:
        try:
            resp = httpx.get(url, timeout=timeout)
            if resp.status_code != 200:
                return None
            return resp.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            return None

    def _resolve_latest_version(self, slug: str, skill_data: Dict[str, Any]) -> Optional[str]:
        latest = skill_data.get("latestVersion")
        if isinstance(latest, dict):
            version = latest.get("version")
            if isinstance(version, str) and version:
                return version

        tags = skill_data.get("tags")
        if isinstance(tags, dict):
            latest_tag = tags.get("latest")
            if isinstance(latest_tag, str) and latest_tag:
                return latest_tag

        versions_data = self._get_json(f"{self.BASE_URL}/skills/{slug}/versions")
        if isinstance(versions_data, list) and versions_data:
            first = versions_data[0]
            if isinstance(first, dict):
                version = first.get("version")
                if isinstance(version, str) and version:
                    return version
        return None

    def _extract_files(self, version_data: Dict[str, Any]) -> Dict[str, str]:
        files: Dict[str, str] = {}
        file_list = version_data.get("files")

        if isinstance(file_list, dict):
            return {k: v for k, v in file_list.items() if isinstance(v, str)}

        if not isinstance(file_list, list):
            return files

        for file_meta in file_list:
            if not isinstance(file_meta, dict):
                continue

            fname = file_meta.get("path") or file_meta.get("name")
            if not fname or not isinstance(fname, str):
                continue

            inline_content = file_meta.get("content")
            if isinstance(inline_content, str):
                files[fname] = inline_content
                continue

            raw_url = file_meta.get("rawUrl") or file_meta.get("downloadUrl") or file_meta.get("url")
            if isinstance(raw_url, str) and raw_url.startswith("http"):
                content = self._fetch_text(raw_url)
                if content is not None:
                    files[fname] = content

        return files

    def _download_zip(self, slug: str, version: str) -> Dict[str, str]:
        """Download skill as a ZIP bundle from the /download endpoint and extract text files."""
        import io
        import zipfile

        files: Dict[str, str] = {}
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = httpx.get(
                    f"{self.BASE_URL}/download",
                    params={"slug": slug, "version": version},
                    timeout=30,
                    follow_redirects=True,
                )
                if resp.status_code == 429:
                    try:
                        retry_after = int(resp.headers.get("retry-after", "5"))
                    except (ValueError, TypeError):
                        retry_after = 5
                    retry_after = min(retry_after, 15)  # Cap wait time
                    logger.debug(
                        "ClawHub download rate-limited for %s, retrying in %ds (attempt %d/%d)",
                        slug, retry_after, attempt + 1, max_retries,
                    )
                    time.sleep(retry_after)
                    continue
                if resp.status_code != 200:
                    logger.debug("ClawHub ZIP download for %s v%s returned %s", slug, version, resp.status_code)
                    return files

                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        try:
                            name = _validate_bundle_rel_path(info.filename)
                        except ValueError:
                            logger.debug("Skipping unsafe ZIP member path: %s", info.filename)
                            continue
                        # Only extract text-sized files (skip large binaries)
                        if info.file_size > 500_000:
                            logger.debug("Skipping large file in ZIP: %s (%d bytes)", name, info.file_size)
                            continue
                        try:
                            raw = zf.read(info.filename)
                            files[name] = raw.decode("utf-8")
                        except (UnicodeDecodeError, KeyError):
                            logger.debug("Skipping non-text file in ZIP: %s", name)
                            continue

                return files

            except zipfile.BadZipFile:
                logger.warning("ClawHub returned invalid ZIP for %s v%s", slug, version)
                return files
            except httpx.HTTPError as exc:
                logger.debug("ClawHub ZIP download failed for %s v%s: %s", slug, version, exc)
                return files

        logger.debug("ClawHub ZIP download exhausted retries for %s v%s", slug, version)
        return files

    def _fetch_text(self, url: str) -> Optional[str]:
        try:
            resp = httpx.get(url, timeout=20)
            if resp.status_code == 200:
                return resp.text
        except httpx.HTTPError:
            return None
        return None


# ---------------------------------------------------------------------------
# Claude Code marketplace source adapter
# ---------------------------------------------------------------------------

class ClaudeMarketplaceSource(SkillSource):
    """
    Discover skills from Claude Code marketplace repos.
    Marketplace repos contain .claude-plugin/marketplace.json with plugin listings.
    """

    KNOWN_MARKETPLACES = [
        "anthropics/skills",
        "aiskillstore/marketplace",
    ]

    def __init__(self, auth: GitHubAuth):
        self.auth = auth

    def source_id(self) -> str:
        return "claude-marketplace"

    def trust_level_for(self, identifier: str) -> str:
        parts = identifier.split("/", 2)
        if len(parts) >= 2:
            repo = f"{parts[0]}/{parts[1]}"
            if repo in TRUSTED_REPOS:
                return "trusted"
        return "community"

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        results: List[SkillMeta] = []
        query_lower = query.lower()

        for marketplace_repo in self.KNOWN_MARKETPLACES:
            plugins = self._fetch_marketplace_index(marketplace_repo)
            for plugin in plugins:
                searchable = f"{plugin.get('name', '')} {plugin.get('description', '')}".lower()
                if query_lower in searchable:
                    source_path = plugin.get("source", "")
                    if source_path.startswith("./"):
                        identifier = f"{marketplace_repo}/{source_path[2:]}"
                    elif "/" in source_path:
                        identifier = source_path
                    else:
                        identifier = f"{marketplace_repo}/{source_path}"

                    results.append(SkillMeta(
                        name=plugin.get("name", ""),
                        description=plugin.get("description", ""),
                        source="claude-marketplace",
                        identifier=identifier,
                        trust_level=self.trust_level_for(identifier),
                        repo=marketplace_repo,
                    ))

        return results[:limit]

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        # Delegate to GitHub Contents API since marketplace skills live in GitHub repos
        gh = GitHubSource(auth=self.auth)
        bundle = gh.fetch(identifier)
        if bundle:
            bundle.source = "claude-marketplace"
        return bundle

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        gh = GitHubSource(auth=self.auth)
        meta = gh.inspect(identifier)
        if meta:
            meta.source = "claude-marketplace"
            meta.trust_level = self.trust_level_for(identifier)
        return meta

    def _fetch_marketplace_index(self, repo: str) -> List[dict]:
        """Fetch and parse .claude-plugin/marketplace.json from a repo."""
        cache_key = f"claude_marketplace_{repo.replace('/', '_')}"
        cached = _read_index_cache(cache_key)
        if cached is not None:
            return cached

        url = f"https://api.github.com/repos/{repo}/contents/.claude-plugin/marketplace.json"
        try:
            resp = httpx.get(
                url,
                headers={**self.auth.get_headers(), "Accept": "application/vnd.github.v3.raw"},
                timeout=15,
            )
            if resp.status_code != 200:
                return []
            data = json.loads(resp.text)
        except (httpx.HTTPError, json.JSONDecodeError):
            return []

        plugins = data.get("plugins", [])
        _write_index_cache(cache_key, plugins)
        return plugins


# ---------------------------------------------------------------------------
# LobeHub source adapter
# ---------------------------------------------------------------------------

class LobeHubSource(SkillSource):
    """
    Fetch skills from LobeHub's agent marketplace (14,500+ agents).
    LobeHub agents are system prompt templates — we convert them to SKILL.md on fetch.
    Data lives in GitHub: lobehub/lobe-chat-agents.
    """

    INDEX_URL = "https://chat-agents.lobehub.com/index.json"

    def source_id(self) -> str:
        return "lobehub"

    def trust_level_for(self, identifier: str) -> str:
        return "community"

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        index = self._fetch_index()
        if not index:
            return []

        query_lower = query.lower()
        results: List[SkillMeta] = []

        agents = index.get("agents", index) if isinstance(index, dict) else index
        if not isinstance(agents, list):
            return []

        for agent in agents:
            meta = agent.get("meta", agent)
            title = meta.get("title", agent.get("identifier", ""))
            desc = meta.get("description", "")
            tags = meta.get("tags", [])

            searchable = f"{title} {desc} {' '.join(tags) if isinstance(tags, list) else ''}".lower()
            if query_lower in searchable:
                identifier = agent.get("identifier", title.lower().replace(" ", "-"))
                results.append(SkillMeta(
                    name=identifier,
                    description=desc[:200],
                    source="lobehub",
                    identifier=f"lobehub/{identifier}",
                    trust_level="community",
                    tags=tags if isinstance(tags, list) else [],
                ))

            if len(results) >= limit:
                break

        return results

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        # Strip "lobehub/" prefix if present
        agent_id = identifier.split("/", 1)[-1] if identifier.startswith("lobehub/") else identifier

        agent_data = self._fetch_agent(agent_id)
        if not agent_data:
            return None

        skill_md = self._convert_to_skill_md(agent_data)
        return SkillBundle(
            name=agent_id,
            files={"SKILL.md": skill_md},
            source="lobehub",
            identifier=f"lobehub/{agent_id}",
            trust_level="community",
        )

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        agent_id = identifier.split("/", 1)[-1] if identifier.startswith("lobehub/") else identifier
        index = self._fetch_index()
        if not index:
            return None

        agents = index.get("agents", index) if isinstance(index, dict) else index
        if not isinstance(agents, list):
            return None

        for agent in agents:
            if agent.get("identifier") == agent_id:
                meta = agent.get("meta", agent)
                return SkillMeta(
                    name=agent_id,
                    description=meta.get("description", ""),
                    source="lobehub",
                    identifier=f"lobehub/{agent_id}",
                    trust_level="community",
                    tags=meta.get("tags", []) if isinstance(meta.get("tags"), list) else [],
                )
        return None

    def _fetch_index(self) -> Optional[Any]:
        """Fetch the LobeHub agent index (cached for 1 hour)."""
        cache_key = "lobehub_index"
        cached = _read_index_cache(cache_key)
        if cached is not None:
            return cached

        try:
            resp = httpx.get(self.INDEX_URL, timeout=30)
            if resp.status_code != 200:
                return None
            data = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            return None

        _write_index_cache(cache_key, data)
        return data

    def _fetch_agent(self, agent_id: str) -> Optional[dict]:
        """Fetch a single agent's JSON file."""
        url = f"https://chat-agents.lobehub.com/{agent_id}.json"
        try:
            resp = httpx.get(url, timeout=15)
            if resp.status_code == 200:
                return resp.json()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            logger.debug("LobeHub agent fetch failed: %s", e)
        return None

    @staticmethod
    def _convert_to_skill_md(agent_data: dict) -> str:
        """Convert a LobeHub agent JSON into SKILL.md format."""
        meta = agent_data.get("meta", agent_data)
        identifier = agent_data.get("identifier", "lobehub-agent")
        title = meta.get("title", identifier)
        description = meta.get("description", "")
        tags = meta.get("tags", [])
        system_role = agent_data.get("config", {}).get("systemRole", "")

        tag_list = tags if isinstance(tags, list) else []
        fm_lines = [
            "---",
            f"name: {identifier}",
            f"description: {description[:500]}",
            "metadata:",
            "  hermes:",
            f"    tags: [{', '.join(str(t) for t in tag_list)}]",
            "  lobehub:",
            "    source: lobehub",
            "---",
        ]

        body_lines = [
            f"# {title}",
            "",
            description,
            "",
            "## Instructions",
            "",
            system_role if system_role else "(No system role defined)",
        ]

        return "\n".join(fm_lines) + "\n\n" + "\n".join(body_lines) + "\n"


# ---------------------------------------------------------------------------
# Official optional skills source adapter
# ---------------------------------------------------------------------------

class OptionalSkillSource(SkillSource):
    """
    Fetch skills from the optional-skills/ directory shipped with the repo.

    These skills are official (maintained by Nous Research) but not activated
    by default — they don't appear in the system prompt and aren't copied to
    ~/.hermes/skills/ during setup.  They are discoverable via the Skills Hub
    (search / install / inspect) and labelled "official" with "builtin" trust.
    """

    def __init__(self):
        from hermes_constants import get_optional_skills_dir

        self._optional_dir = get_optional_skills_dir(
            Path(__file__).parent.parent / "optional-skills"
        )

    def source_id(self) -> str:
        return "official"

    def trust_level_for(self, identifier: str) -> str:
        return "builtin"

    # -- search -----------------------------------------------------------

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        results: List[SkillMeta] = []
        query_lower = query.lower()

        for meta in self._scan_all():
            searchable = f"{meta.name} {meta.description} {' '.join(meta.tags)}".lower()
            if query_lower in searchable:
                results.append(meta)
            if len(results) >= limit:
                break

        return results

    # -- fetch ------------------------------------------------------------

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        # identifier format: "official/category/skill" or "official/skill"
        rel = identifier.split("/", 1)[-1] if identifier.startswith("official/") else identifier
        skill_dir = self._optional_dir / rel

        # Guard against path traversal (e.g. "official/../../etc")
        try:
            resolved = skill_dir.resolve()
            if not str(resolved).startswith(str(self._optional_dir.resolve())):
                return None
        except (OSError, ValueError):
            return None

        if not resolved.is_dir():
            # Try searching by skill name only (last segment)
            skill_name = rel.rsplit("/", 1)[-1]
            skill_dir = self._find_skill_dir(skill_name)
            if not skill_dir:
                return None
        else:
            skill_dir = resolved

        files: Dict[str, Union[str, bytes]] = {}
        for f in skill_dir.rglob("*"):
            if (
                f.is_file()
                and not f.name.startswith(".")
                and "__pycache__" not in f.parts
                and f.suffix != ".pyc"
            ):
                rel_path = str(f.relative_to(skill_dir))
                try:
                    files[rel_path] = f.read_bytes()
                except OSError:
                    continue

        if not files:
            return None

        # Determine category from directory structure
        name = skill_dir.name

        return SkillBundle(
            name=name,
            files=files,
            source="official",
            identifier=f"official/{skill_dir.relative_to(self._optional_dir)}",
            trust_level="builtin",
        )

    # -- inspect ----------------------------------------------------------

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        rel = identifier.split("/", 1)[-1] if identifier.startswith("official/") else identifier
        skill_name = rel.rsplit("/", 1)[-1]

        for meta in self._scan_all():
            if meta.name == skill_name:
                return meta
        return None

    # -- internal helpers -------------------------------------------------

    def _find_skill_dir(self, name: str) -> Optional[Path]:
        """Find a skill directory by name anywhere in optional-skills/."""
        if not self._optional_dir.is_dir():
            return None
        for skill_md in self._optional_dir.rglob("SKILL.md"):
            if skill_md.parent.name == name:
                return skill_md.parent
        return None

    def _scan_all(self) -> List[SkillMeta]:
        """Enumerate all optional skills with metadata."""
        if not self._optional_dir.is_dir():
            return []

        results: List[SkillMeta] = []
        for skill_md in sorted(self._optional_dir.rglob("SKILL.md")):
            parent = skill_md.parent
            rel_parts = parent.relative_to(self._optional_dir).parts
            if any(part.startswith(".") for part in rel_parts):
                continue

            try:
                content = skill_md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            fm = self._parse_frontmatter(content)
            name = fm.get("name", parent.name)
            desc = fm.get("description", "")
            tags = []
            meta_block = fm.get("metadata", {})
            if isinstance(meta_block, dict):
                hermes_meta = meta_block.get("hermes", {})
                if isinstance(hermes_meta, dict):
                    tags = hermes_meta.get("tags", [])

            rel_path = str(parent.relative_to(self._optional_dir))

            results.append(SkillMeta(
                name=name,
                description=desc[:200],
                source="official",
                identifier=f"official/{rel_path}",
                trust_level="builtin",
                path=rel_path,
                tags=tags if isinstance(tags, list) else [],
            ))

        return results

    @staticmethod
    def _parse_frontmatter(content: str) -> dict:
        """Parse YAML frontmatter from SKILL.md content."""
        if not content.startswith("---"):
            return {}
        match = re.search(r'\n---\s*\n', content[3:])
        if not match:
            return {}
        yaml_text = content[3:match.start() + 3]
        try:
            parsed = yaml.safe_load(yaml_text)
            return parsed if isinstance(parsed, dict) else {}
        except yaml.YAMLError:
            return {}


# ---------------------------------------------------------------------------
# Shared cache helpers (used by multiple adapters)
# ---------------------------------------------------------------------------

def _read_index_cache(key: str) -> Optional[Any]:
    """Read cached data if not expired."""
    cache_file = INDEX_CACHE_DIR / f"{key}.json"
    if not cache_file.exists():
        return None
    try:
        stat = cache_file.stat()
        if time.time() - stat.st_mtime > INDEX_CACHE_TTL:
            return None
        return json.loads(cache_file.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_index_cache(key: str, data: Any) -> None:
    """Write data to cache."""
    INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure .ignore exists so ripgrep (and tools respecting .ignore) skip
    # this directory.  Cache files contain unvetted community content that
    # could include adversarial text (prompt injection via catalog entries).
    ignore_file = HUB_DIR / ".ignore"
    if not ignore_file.exists():
        try:
            ignore_file.write_text("# Exclude hub internals from search tools\n*\n")
        except OSError:
            pass
    cache_file = INDEX_CACHE_DIR / f"{key}.json"
    try:
        cache_file.write_text(json.dumps(data, ensure_ascii=False, default=str))
    except OSError as e:
        logger.debug("Could not write cache: %s", e)


def _skill_meta_to_dict(meta: SkillMeta) -> dict:
    """Convert a SkillMeta to a dict for caching."""
    return {
        "name": meta.name,
        "description": meta.description,
        "source": meta.source,
        "identifier": meta.identifier,
        "trust_level": meta.trust_level,
        "repo": meta.repo,
        "path": meta.path,
        "tags": meta.tags,
        "extra": meta.extra,
    }


# ---------------------------------------------------------------------------
# Lock file management
# ---------------------------------------------------------------------------

class HubLockFile:
    """Manages skills/.hub/lock.json — tracks provenance of installed hub skills."""

    def __init__(self, path: Path = LOCK_FILE):
        self.path = path

    def load(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "installed": {}}
        try:
            return json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return {"version": 1, "installed": {}}

    def save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    def record_install(
        self,
        name: str,
        source: str,
        identifier: str,
        trust_level: str,
        scan_verdict: str,
        skill_hash: str,
        install_path: str,
        files: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        data = self.load()
        data["installed"][name] = {
            "source": source,
            "identifier": identifier,
            "trust_level": trust_level,
            "scan_verdict": scan_verdict,
            "content_hash": skill_hash,
            "install_path": install_path,
            "files": files,
            "metadata": metadata or {},
            "installed_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save(data)

    def record_uninstall(self, name: str) -> None:
        data = self.load()
        data["installed"].pop(name, None)
        self.save(data)

    def get_installed(self, name: str) -> Optional[dict]:
        data = self.load()
        return data["installed"].get(name)

    def list_installed(self) -> List[dict]:
        data = self.load()
        result = []
        for name, entry in data["installed"].items():
            result.append({"name": name, **entry})
        return result


# ---------------------------------------------------------------------------
# Taps management
# ---------------------------------------------------------------------------

class TapsManager:
    """Manages the taps.json file — custom GitHub repo sources."""

    def __init__(self, path: Path = TAPS_FILE):
        self.path = path

    def load(self) -> List[dict]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text())
            return data.get("taps", [])
        except (json.JSONDecodeError, OSError):
            return []

    def save(self, taps: List[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"taps": taps}, indent=2) + "\n")

    def add(self, repo: str, path: str = "skills/") -> bool:
        """Add a tap. Returns False if already exists."""
        taps = self.load()
        if any(t["repo"] == repo for t in taps):
            return False
        taps.append({"repo": repo, "path": path})
        self.save(taps)
        return True

    def remove(self, repo: str) -> bool:
        """Remove a tap by repo name. Returns False if not found."""
        taps = self.load()
        new_taps = [t for t in taps if t["repo"] != repo]
        if len(new_taps) == len(taps):
            return False
        self.save(new_taps)
        return True

    def list_taps(self) -> List[dict]:
        return self.load()


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def append_audit_log(action: str, skill_name: str, source: str,
                     trust_level: str, verdict: str, extra: str = "") -> None:
    """Append a line to the audit log."""
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    parts = [timestamp, action, skill_name, f"{source}:{trust_level}", verdict]
    if extra:
        parts.append(extra)
    line = " ".join(parts) + "\n"
    try:
        with open(AUDIT_LOG, "a") as f:
            f.write(line)
    except OSError as e:
        logger.debug("Could not write audit log: %s", e)


# ---------------------------------------------------------------------------
# Hub operations (high-level)
# ---------------------------------------------------------------------------

def ensure_hub_dirs() -> None:
    """Create the .hub directory structure if it doesn't exist."""
    HUB_DIR.mkdir(parents=True, exist_ok=True)
    QUARANTINE_DIR.mkdir(exist_ok=True)
    INDEX_CACHE_DIR.mkdir(exist_ok=True)
    if not LOCK_FILE.exists():
        LOCK_FILE.write_text('{"version": 1, "installed": {}}\n')
    if not AUDIT_LOG.exists():
        AUDIT_LOG.touch()
    if not TAPS_FILE.exists():
        TAPS_FILE.write_text('{"taps": []}\n')


def quarantine_bundle(bundle: SkillBundle) -> Path:
    """Write a skill bundle to the quarantine directory for scanning."""
    ensure_hub_dirs()
    skill_name = _validate_skill_name(bundle.name)
    validated_files: List[Tuple[str, Union[str, bytes]]] = []
    for rel_path, file_content in bundle.files.items():
        safe_rel_path = _validate_bundle_rel_path(rel_path)
        validated_files.append((safe_rel_path, file_content))

    dest = QUARANTINE_DIR / skill_name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    for rel_path, file_content in validated_files:
        file_dest = dest.joinpath(*rel_path.split("/"))
        file_dest.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(file_content, bytes):
            file_dest.write_bytes(file_content)
        else:
            file_dest.write_text(file_content, encoding="utf-8")

    return dest


def install_from_quarantine(
    quarantine_path: Path,
    skill_name: str,
    category: str,
    bundle: SkillBundle,
    scan_result: ScanResult,
) -> Path:
    """Move a scanned skill from quarantine into the skills directory."""
    safe_skill_name = _validate_skill_name(skill_name)
    safe_category = _validate_category_name(category) if category else ""
    quarantine_resolved = quarantine_path.resolve()
    quarantine_root = QUARANTINE_DIR.resolve()
    if not quarantine_resolved.is_relative_to(quarantine_root):
        raise ValueError(f"Unsafe quarantine path: {quarantine_path}")

    if safe_category:
        install_dir = SKILLS_DIR / safe_category / safe_skill_name
    else:
        install_dir = SKILLS_DIR / safe_skill_name

    if install_dir.exists():
        shutil.rmtree(install_dir)

    # Warn (but don't block) if SKILL.md is very large
    skill_md = quarantine_path / "SKILL.md"
    if skill_md.exists():
        try:
            skill_size = skill_md.stat().st_size
            if skill_size > 100_000:
                logger.warning(
                    "Skill '%s' has a large SKILL.md (%s chars). "
                    "Large skills consume significant context when loaded. "
                    "Consider asking the author to split it into smaller files.",
                    safe_skill_name,
                    f"{skill_size:,}",
                )
        except OSError:
            pass

    install_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(quarantine_path), str(install_dir))

    # Record in lock file
    lock = HubLockFile()
    lock.record_install(
        name=safe_skill_name,
        source=bundle.source,
        identifier=bundle.identifier,
        trust_level=bundle.trust_level,
        scan_verdict=scan_result.verdict,
        skill_hash=content_hash(install_dir),
        install_path=str(install_dir.relative_to(SKILLS_DIR)),
        files=list(bundle.files.keys()),
        metadata=bundle.metadata,
    )

    append_audit_log(
        "INSTALL", safe_skill_name, bundle.source,
        bundle.trust_level, scan_result.verdict,
        content_hash(install_dir),
    )

    return install_dir


def uninstall_skill(skill_name: str) -> Tuple[bool, str]:
    """Remove a hub-installed skill. Refuses to remove builtins."""
    lock = HubLockFile()
    entry = lock.get_installed(skill_name)
    if not entry:
        return False, f"'{skill_name}' is not a hub-installed skill (may be a builtin)"

    install_path = SKILLS_DIR / entry["install_path"]
    if install_path.exists():
        shutil.rmtree(install_path)

    lock.record_uninstall(skill_name)
    append_audit_log("UNINSTALL", skill_name, entry["source"], entry["trust_level"], "n/a", "user_request")

    return True, f"Uninstalled '{skill_name}' from {entry['install_path']}"


def bundle_content_hash(bundle: SkillBundle) -> str:
    """Compute a deterministic hash for an in-memory skill bundle."""
    h = hashlib.sha256()
    for rel_path in sorted(bundle.files):
        h.update(bundle.files[rel_path].encode("utf-8"))
    return f"sha256:{h.hexdigest()[:16]}"


def _source_matches(source: SkillSource, source_name: str) -> bool:
    aliases = {
        "skills.sh": "skills-sh",
    }
    normalized = aliases.get(source_name, source_name)
    return source.source_id() == normalized


def check_for_skill_updates(
    name: Optional[str] = None,
    *,
    lock: Optional[HubLockFile] = None,
    sources: Optional[List[SkillSource]] = None,
    auth: Optional[GitHubAuth] = None,
) -> List[dict]:
    """Check installed hub skills for upstream changes."""
    lock = lock or HubLockFile()
    installed = lock.list_installed()
    if name:
        installed = [entry for entry in installed if entry.get("name") == name]

    if sources is None:
        sources = create_source_router(auth=auth)

    results: List[dict] = []
    for entry in installed:
        identifier = entry.get("identifier", "")
        source_name = entry.get("source", "")
        candidate_sources = [src for src in sources if _source_matches(src, source_name)] or sources

        bundle = None
        for src in candidate_sources:
            try:
                bundle = src.fetch(identifier)
            except Exception:
                bundle = None
            if bundle:
                break

        if not bundle:
            results.append({
                "name": entry.get("name", ""),
                "identifier": identifier,
                "source": source_name,
                "status": "unavailable",
            })
            continue

        current_hash = entry.get("content_hash", "")
        latest_hash = bundle_content_hash(bundle)
        status = "up_to_date" if current_hash == latest_hash else "update_available"
        results.append({
            "name": entry.get("name", ""),
            "identifier": identifier,
            "source": source_name,
            "status": status,
            "current_hash": current_hash,
            "latest_hash": latest_hash,
            "bundle": bundle,
        })

    return results


# ---------------------------------------------------------------------------
# Hermes centralized index source
# ---------------------------------------------------------------------------

HERMES_INDEX_URL = "https://hermes-agent.nousresearch.com/docs/api/skills-index.json"
HERMES_INDEX_CACHE_FILE = INDEX_CACHE_DIR / "hermes-index.json"
HERMES_INDEX_TTL = 6 * 3600  # 6 hours


def _load_hermes_index() -> Optional[dict]:
    """Fetch the centralized skills index, with local cache.

    The index is a JSON file hosted on the docs site, rebuilt daily by CI.
    We cache it locally for HERMES_INDEX_TTL seconds to avoid repeated
    downloads within a session.
    """
    # Check local cache
    if HERMES_INDEX_CACHE_FILE.exists():
        try:
            age = time.time() - HERMES_INDEX_CACHE_FILE.stat().st_mtime
            if age < HERMES_INDEX_TTL:
                return json.loads(HERMES_INDEX_CACHE_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            pass

    # Fetch from docs site
    try:
        resp = httpx.get(HERMES_INDEX_URL, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            logger.debug("Hermes index fetch returned %d", resp.status_code)
            return _load_stale_index_cache()
        data = resp.json()
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        logger.debug("Hermes index fetch failed: %s", e)
        return _load_stale_index_cache()

    # Validate structure
    if not isinstance(data, dict) or "skills" not in data:
        return _load_stale_index_cache()

    # Cache locally
    try:
        HERMES_INDEX_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        HERMES_INDEX_CACHE_FILE.write_text(json.dumps(data))
    except OSError:
        pass

    return data


def _load_stale_index_cache() -> Optional[dict]:
    """Fall back to stale cache when the network fetch fails."""
    if HERMES_INDEX_CACHE_FILE.exists():
        try:
            return json.loads(HERMES_INDEX_CACHE_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    return None


class HermesIndexSource(SkillSource):
    """Skill source backed by the centralized Hermes Skills Index.

    The index is a JSON catalog published to the docs site and rebuilt
    daily by CI.  It contains metadata + resolved GitHub paths for every
    skill, eliminating the need for users to hit the GitHub API for
    search or path discovery.

    When the index is unavailable, all methods return empty / None so
    downstream sources take over transparently.
    """

    def __init__(self, auth: GitHubAuth):
        self._index: Optional[dict] = None
        self._loaded = False
        self.auth = auth
        # Lazily create GitHubSource for fetch — only used when actually
        # downloading files, which requires real GitHub API calls.
        self._github: Optional[GitHubSource] = None

    def _ensure_loaded(self) -> dict:
        if not self._loaded:
            self._index = _load_hermes_index()
            self._loaded = True
        return self._index or {}

    def _get_github(self) -> GitHubSource:
        if self._github is None:
            self._github = GitHubSource(auth=self.auth)
        return self._github

    def source_id(self) -> str:
        return "hermes-index"

    @property
    def is_available(self) -> bool:
        """Whether the index is loaded and has skills."""
        index = self._ensure_loaded()
        return bool(index.get("skills"))

    def trust_level_for(self, identifier: str) -> str:
        index = self._ensure_loaded()
        for skill in index.get("skills", []):
            if skill.get("identifier") == identifier:
                return skill.get("trust_level", "community")
        return "community"

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        """Search the cached index.  Zero API calls."""
        index = self._ensure_loaded()
        skills = index.get("skills", [])
        if not skills:
            return []

        if not query.strip():
            # No query — return featured/popular
            return [self._to_meta(s) for s in skills[:limit]]

        query_lower = query.lower()
        results: List[SkillMeta] = []
        for s in skills:
            searchable = f"{s.get('name', '')} {s.get('description', '')} {' '.join(s.get('tags', []))}".lower()
            if query_lower in searchable:
                results.append(self._to_meta(s))
                if len(results) >= limit:
                    break
        return results

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        """Fetch a skill using the resolved path from the index.

        If the index has a ``resolved_github_id`` for this skill, we skip
        the entire candidate/discovery chain and go directly to GitHub
        with the exact path.  This reduces install from ~31 API calls to
        just the file content downloads (~5-22 depending on skill size).
        """
        index = self._ensure_loaded()
        entry = self._find_entry(identifier, index)
        if not entry:
            return None

        # Use resolved path if available
        resolved = entry.get("resolved_github_id")
        if resolved:
            bundle = self._get_github().fetch(resolved)
            if bundle:
                bundle.source = entry.get("source", "hermes-index")
                bundle.identifier = identifier
                return bundle

        # Fall back to identifier-based fetch via repo/path
        repo = entry.get("repo", "")
        path = entry.get("path", "")
        if repo and path:
            github_id = f"{repo}/{path}"
            bundle = self._get_github().fetch(github_id)
            if bundle:
                bundle.source = entry.get("source", "hermes-index")
                bundle.identifier = identifier
                return bundle

        return None

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        """Return metadata from the index.  Zero API calls."""
        index = self._ensure_loaded()
        entry = self._find_entry(identifier, index)
        if entry:
            return self._to_meta(entry)
        return None

    def _find_entry(self, identifier: str, index: dict) -> Optional[dict]:
        """Look up a skill in the index by identifier or name."""
        skills = index.get("skills", [])

        # Exact identifier match
        for s in skills:
            if s.get("identifier") == identifier:
                return s

        # Try without source prefix (e.g. "skills-sh/" stripped)
        normalized = identifier
        for prefix in ("skills-sh/", "skills.sh/", "official/", "github/", "clawhub/"):
            if identifier.startswith(prefix):
                normalized = identifier[len(prefix):]
                break

        # Match on normalized identifier or name
        for s in skills:
            sid = s.get("identifier", "")
            # Strip prefix from stored identifier too
            stored_normalized = sid
            for prefix in ("skills-sh/", "skills.sh/", "official/", "github/", "clawhub/"):
                if sid.startswith(prefix):
                    stored_normalized = sid[len(prefix):]
                    break
            if stored_normalized == normalized:
                return s

        return None

    @staticmethod
    def _to_meta(entry: dict) -> SkillMeta:
        return SkillMeta(
            name=entry.get("name", ""),
            description=entry.get("description", ""),
            source=entry.get("source", "hermes-index"),
            identifier=entry.get("identifier", ""),
            trust_level=entry.get("trust_level", "community"),
            repo=entry.get("repo"),
            path=entry.get("path"),
            tags=entry.get("tags", []),
            extra=entry.get("extra", {}),
        )


def create_source_router(auth: Optional[GitHubAuth] = None) -> List[SkillSource]:
    """
    Create all configured source adapters.
    Returns a list of active sources for search/fetch operations.
    """
    if auth is None:
        auth = GitHubAuth()

    taps_mgr = TapsManager()
    extra_taps = taps_mgr.list_taps()

    sources: List[SkillSource] = [
        OptionalSkillSource(),        # Official optional skills (highest priority)
        HermesIndexSource(auth=auth), # Centralized index (search + resolved install paths)
        SkillsShSource(auth=auth),
        WellKnownSkillSource(),
        GitHubSource(auth=auth, extra_taps=extra_taps),
        ClawHubSource(),
        ClaudeMarketplaceSource(auth=auth),
        LobeHubSource(),
    ]

    return sources


def _search_one_source(
    src: SkillSource, query: str, limit: int
) -> Tuple[str, List[SkillMeta]]:
    """Search a single source.  Runs in a thread for parallelism."""
    try:
        return src.source_id(), src.search(query, limit=limit)
    except Exception as e:
        logger.debug("Search failed for %s: %s", src.source_id(), e)
        return src.source_id(), []


def parallel_search_sources(
    sources: List[SkillSource],
    query: str = "",
    per_source_limits: Optional[Dict[str, int]] = None,
    source_filter: str = "all",
    overall_timeout: float = 30,
    on_source_done: Optional[Any] = None,
) -> Tuple[List[SkillMeta], Dict[str, int], List[str]]:
    """Search all sources in parallel with per-source timeout.

    Returns ``(all_results, source_counts, timed_out_ids)``.

    *on_source_done* is an optional callback ``(source_id, count) -> None``
    invoked as each source completes — useful for progress indicators.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    per_source_limits = per_source_limits or {}

    active: List[SkillSource] = []
    # When the centralized index is available and the user hasn't filtered
    # to a specific source, skip external API sources (github, skills-sh,
    # clawhub, etc.) — the index already has their data.  This avoids
    # ~70 GitHub API calls per search for unauthenticated users.
    _index_available = False
    _api_source_ids = frozenset({"github", "skills-sh", "clawhub",
                                  "claude-marketplace", "lobehub", "well-known"})
    if source_filter == "all":
        for src in sources:
            if (src.source_id() == "hermes-index"
                    and getattr(src, "is_available", False)):
                _index_available = True
                break

    for src in sources:
        sid = src.source_id()
        if source_filter != "all" and sid != source_filter and sid != "official":
            continue
        # Skip external API sources when the index covers them
        if _index_available and sid in _api_source_ids:
            continue
        active.append(src)

    all_results: List[SkillMeta] = []
    source_counts: Dict[str, int] = {}
    timed_out_ids: List[str] = []

    if not active:
        return all_results, source_counts, timed_out_ids

    with ThreadPoolExecutor(max_workers=min(len(active), 8)) as pool:
        futures = {}
        for src in active:
            lim = per_source_limits.get(src.source_id(), 50)
            fut = pool.submit(_search_one_source, src, query, lim)
            futures[fut] = src.source_id()

        try:
            for fut in as_completed(futures, timeout=overall_timeout):
                try:
                    sid, results = fut.result(timeout=0)
                    source_counts[sid] = len(results)
                    all_results.extend(results)
                    if on_source_done:
                        on_source_done(sid, len(results))
                except Exception:
                    pass
        except TimeoutError:
            timed_out_ids = [
                futures[f] for f in futures if not f.done()
            ]
            if timed_out_ids:
                logger.debug(
                    "Skills browse timed out waiting for: %s",
                    ", ".join(timed_out_ids),
                )

    return all_results, source_counts, timed_out_ids


def unified_search(query: str, sources: List[SkillSource],
                   source_filter: str = "all", limit: int = 10) -> List[SkillMeta]:
    """Search all sources (in parallel) and merge results."""
    all_results, _, _ = parallel_search_sources(
        sources,
        query=query,
        source_filter=source_filter,
        overall_timeout=30,
    )

    # Deduplicate by name, preferring higher trust levels
    _TRUST_RANK = {"builtin": 2, "trusted": 1, "community": 0}
    seen: Dict[str, SkillMeta] = {}
    for r in all_results:
        if r.name not in seen:
            seen[r.name] = r
        elif _TRUST_RANK.get(r.trust_level, 0) > _TRUST_RANK.get(seen[r.name].trust_level, 0):
            seen[r.name] = r
    deduped = list(seen.values())

    return deduped[:limit]
