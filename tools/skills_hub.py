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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import yaml

from tools.skills_guard import (
    ScanResult, scan_skill, should_allow_install, content_hash, TRUSTED_REPOS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERMES_HOME = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
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


@dataclass
class SkillBundle:
    """A downloaded skill ready for quarantine/scanning/installation."""
    name: str
    files: Dict[str, str]   # relative_path -> text content
    source: str
    identifier: str
    trust_level: str


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
    ]

    def __init__(self, auth: GitHubAuth, extra_taps: Optional[List[Dict]] = None):
        self.auth = auth
        self.taps = list(self.DEFAULT_TAPS)
        if extra_taps:
            self.taps.extend(extra_taps)

    def source_id(self) -> str:
        return "github"

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
            resp = httpx.get(url, headers=self.auth.get_headers(), timeout=15)
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
            if dir_name.startswith(".") or dir_name.startswith("_"):
                continue

            skill_identifier = f"{repo}/{path.rstrip('/')}/{dir_name}"
            meta = self.inspect(skill_identifier)
            if meta:
                skills.append(meta)

        # Cache the results
        self._write_cache(cache_key, [self._meta_to_dict(s) for s in skills])
        return skills

    def _download_directory(self, repo: str, path: str) -> Dict[str, str]:
        """Recursively download all text files from a GitHub directory."""
        url = f"https://api.github.com/repos/{repo}/contents/{path.rstrip('/')}"
        try:
            resp = httpx.get(url, headers=self.auth.get_headers(), timeout=15)
            if resp.status_code != 200:
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
                sub_files = self._download_directory(repo, entry.get("path", ""))
                for sub_name, sub_content in sub_files.items():
                    files[f"{name}/{sub_name}"] = sub_content

        return files

    def _fetch_file_content(self, repo: str, path: str) -> Optional[str]:
        """Fetch a single file's content from GitHub."""
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        try:
            resp = httpx.get(
                url,
                headers={**self.auth.get_headers(), "Accept": "application/vnd.github.v3.raw"},
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.text
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
        for candidate in self._candidate_identifiers(canonical):
            bundle = self.github.fetch(candidate)
            if bundle:
                bundle.source = "skills.sh"
                bundle.identifier = self._wrap_identifier(canonical)
                return bundle

        detail = self._fetch_detail_page(canonical)
        resolved = self._discover_identifier(canonical, detail=detail)
        if resolved:
            bundle = self.github.fetch(resolved)
            if bundle:
                bundle.source = "skills.sh"
                bundle.identifier = self._wrap_identifier(canonical)
                return bundle
        return None

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        canonical = self._normalize_identifier(identifier)
        detail: Optional[dict] = None
        for candidate in self._candidate_identifiers(canonical):
            meta = self.github.inspect(candidate)
            if meta:
                detail = self._fetch_detail_page(canonical)
                return self._finalize_inspect_meta(meta, canonical, detail)

        detail = self._fetch_detail_page(canonical)
        resolved = self._discover_identifier(canonical, detail=detail)
        if resolved:
            meta = self.github.inspect(resolved)
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

        install_match = self._INSTALL_CMD_RE.search(html)
        if install_match:
            repo_value = (install_match.group("repo") or "").strip()
            install_skill = (install_match.group("skill") or install_skill).strip()
            repo = self._extract_repo_slug(repo_value) or repo

        page_title = self._extract_first_match(self._PAGE_H1_RE, html)
        body_title = self._extract_first_match(self._PROSE_H1_RE, html)
        body_summary = self._extract_first_match(self._PROSE_P_RE, html)
        weekly_installs = self._extract_weekly_installs(html)

        return {
            "repo": repo,
            "install_skill": install_skill,
            "page_title": page_title,
            "body_title": body_title,
            "body_summary": body_summary,
            "weekly_installs": weekly_installs,
        }

    def _discover_identifier(self, identifier: str, detail: Optional[dict] = None) -> Optional[str]:
        parts = identifier.split("/", 2)
        if len(parts) < 3:
            return None

        default_repo = f"{parts[0]}/{parts[1]}"
        repo = detail.get("repo", default_repo) if isinstance(detail, dict) else default_repo
        skill_token = parts[2]
        tokens = [skill_token]
        if isinstance(detail, dict):
            tokens.extend([
                detail.get("install_skill", ""),
                detail.get("page_title", ""),
                detail.get("body_title", ""),
            ])

        for base_path in ("skills/", ".agents/skills/", ".claude/skills/"):
            try:
                skills = self.github._list_skills_in_repo(repo, base_path)
            except Exception:
                continue
            for meta in skills:
                if self._matches_skill_tokens(meta, tokens):
                    return meta.identifier
        return None

    def _finalize_inspect_meta(self, meta: SkillMeta, canonical: str, detail: Optional[dict]) -> SkillMeta:
        meta.source = "skills.sh"
        meta.identifier = self._wrap_identifier(canonical)
        meta.trust_level = self.trust_level_for(canonical)

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

    @staticmethod
    def _extract_weekly_installs(html: str) -> Optional[str]:
        match = SkillsShSource._WEEKLY_INSTALLS_RE.search(html)
        if not match:
            return None
        return match.group("count")

    @staticmethod
    def _strip_html(value: str) -> str:
        return re.sub(r'<[^>]+>', '', value)

    @staticmethod
    def _normalize_identifier(identifier: str) -> str:
        if identifier.startswith("skills-sh/"):
            return identifier[len("skills-sh/"):]
        if identifier.startswith("skills.sh/"):
            return identifier[len("skills.sh/"):]
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

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        cache_key = f"clawhub_search_{hashlib.md5(query.encode()).hexdigest()}"
        cached = _read_index_cache(cache_key)
        if cached is not None:
            return [SkillMeta(**s) for s in cached][:limit]

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
            tags = item.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            results.append(SkillMeta(
                name=display_name,
                description=summary,
                source="clawhub",
                identifier=slug,
                trust_level="community",
                tags=[str(t) for t in tags],
            ))

        _write_index_cache(cache_key, [_skill_meta_to_dict(s) for s in results])
        return results

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
        data = self._get_json(f"{self.BASE_URL}/skills/{slug}")
        if not isinstance(data, dict):
            return None

        tags = data.get("tags", [])
        if not isinstance(tags, list):
            tags = []

        return SkillMeta(
            name=data.get("displayName") or data.get("name") or data.get("slug") or slug,
            description=data.get("summary") or data.get("description") or "",
            source="clawhub",
            identifier=data.get("slug") or slug,
            trust_level="community",
            tags=[str(t) for t in tags],
        )

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
                    retry_after = int(resp.headers.get("retry-after", "5"))
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
                        # Sanitize path — strip leading slashes and ..
                        name = info.filename.lstrip("/")
                        if ".." in name or name.startswith("/"):
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
    REPO = "lobehub/lobe-chat-agents"

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
            f"  lobehub:",
            f"    source: lobehub",
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
        self._optional_dir = Path(__file__).parent.parent / "optional-skills"

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

        files: Dict[str, str] = {}
        for f in skill_dir.rglob("*"):
            if f.is_file() and not f.name.startswith("."):
                rel_path = str(f.relative_to(skill_dir))
                try:
                    files[rel_path] = f.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
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

    def is_hub_installed(self, name: str) -> bool:
        data = self.load()
        return name in data["installed"]


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
    dest = QUARANTINE_DIR / bundle.name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    for rel_path, file_content in bundle.files.items():
        file_dest = dest / rel_path
        file_dest.parent.mkdir(parents=True, exist_ok=True)
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
    if category:
        install_dir = SKILLS_DIR / category / skill_name
    else:
        install_dir = SKILLS_DIR / skill_name

    if install_dir.exists():
        shutil.rmtree(install_dir)

    install_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(quarantine_path), str(install_dir))

    # Record in lock file
    lock = HubLockFile()
    lock.record_install(
        name=skill_name,
        source=bundle.source,
        identifier=bundle.identifier,
        trust_level=bundle.trust_level,
        scan_verdict=scan_result.verdict,
        skill_hash=content_hash(install_dir),
        install_path=str(install_dir.relative_to(SKILLS_DIR)),
        files=list(bundle.files.keys()),
    )

    append_audit_log(
        "INSTALL", skill_name, bundle.source,
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
        SkillsShSource(auth=auth),
        GitHubSource(auth=auth, extra_taps=extra_taps),
        ClawHubSource(),
        ClaudeMarketplaceSource(auth=auth),
        LobeHubSource(),
    ]

    return sources


def unified_search(query: str, sources: List[SkillSource],
                   source_filter: str = "all", limit: int = 10) -> List[SkillMeta]:
    """Search all sources and merge results."""
    all_results: List[SkillMeta] = []

    for src in sources:
        if source_filter != "all" and src.source_id() != source_filter:
            continue
        try:
            results = src.search(query, limit=limit)
            all_results.extend(results)
        except Exception as e:
            logger.debug(f"Search failed for {src.source_id()}: {e}")

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
