#!/usr/bin/env python3
"""
Skills Hub — hub state management and the public facade for the source adapters.

Library module (not an agent tool). Owns the hub paths, guarded HTTP, index
cache, lock file, taps and audit log. Install/uninstall/update live in
``skills_hub_install``, the index fetch/source router/search in
``skills_hub_search``, and the adapters in the other ``tools.skills_hub_*``
siblings; import each name from its defining module.

Used by hermes_cli/skills_hub.py for CLI commands and the /skills slash command.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import httpx

from hermes_constants import get_hermes_home
from tools.url_safety import is_safe_url
from tools.website_policy import check_website_access
from tools.skills_hub_models import _normalize_lock_install_path, _validate_skill_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolved per-call (not frozen at import) so the profile override is honored;
# import-time constants leaked across profiles in single-process multi-profile
# runtimes. The path names (SKILLS_DIR, ...) resolve live via __getattr__ below.

INDEX_CACHE_TTL = 3600  # 1 hour


def _path_resolver(name: str, parent: str, leaf: str):
    """Resolver for hub path ``<parent>/<leaf>``: a test-injected real module attribute
    (patch.object/monkeypatch on SKILLS_DIR etc.) wins over live resolution."""
    def resolve() -> Path:
        forced = globals().get(name)
        return Path(forced) if forced is not None else _DYNAMIC_PATH_RESOLVERS[parent]() / leaf
    resolve.__name__ = f"_{name.lower()}"
    return resolve


def _hermes_home() -> Path:
    return get_hermes_home()


_skills_dir = _path_resolver("SKILLS_DIR", "HERMES_HOME", "skills")
_hub_dir = _path_resolver("HUB_DIR", "SKILLS_DIR", ".hub")
_lock_file = _path_resolver("LOCK_FILE", "HUB_DIR", "lock.json")
_quarantine_dir = _path_resolver("QUARANTINE_DIR", "HUB_DIR", "quarantine")
_audit_log = _path_resolver("AUDIT_LOG", "HUB_DIR", "audit.log")
_taps_file = _path_resolver("TAPS_FILE", "HUB_DIR", "taps.json")
_index_cache_dir = _path_resolver("INDEX_CACHE_DIR", "HUB_DIR", "index-cache")
_DYNAMIC_PATH_RESOLVERS = {"HERMES_HOME": _hermes_home, **{
    r.__name__[1:].upper(): r
    for r in (_skills_dir, _hub_dir, _lock_file, _quarantine_dir, _audit_log, _taps_file, _index_cache_dir)
}}


def __getattr__(name: str):
    """Resolve legacy path constants dynamically (PEP 562) so they reflect the
    active profile override; a test's patch.object-set real attribute shadows it."""
    resolver = _DYNAMIC_PATH_RESOLVERS.get(name)
    if resolver is not None:
        return resolver()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Install-path safety + guarded HTTP
# ---------------------------------------------------------------------------

_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
_MAX_SKILL_FETCH_REDIRECTS = 5


def _ssrf_safe_http_get(url: str, *, timeout: int = 20) -> httpx.Response:
    """Fetch one URL with connect-time SSRF validation and no automatic redirects."""
    from tools.url_safety import create_ssrf_safe_client

    with create_ssrf_safe_client(timeout=timeout, follow_redirects=False) as client:
        return client.get(url)


def _guarded_http_get(url: str, *, timeout: int = 20) -> Optional[httpx.Response]:
    """Fetch a URL with SSRF and redirect-target validation (each hop re-checked)."""
    from tools.url_safety import SSRFConnectionBlocked

    current_url = url

    for _ in range(_MAX_SKILL_FETCH_REDIRECTS + 1):
        if not is_safe_url(current_url):
            logger.warning("Blocked unsafe Skills Hub URL: %s", current_url)
            return None

        blocked = check_website_access(current_url)
        if blocked:
            logger.info(
                "Blocked Skills Hub fetch for %s by rule %s",
                blocked["host"],
                blocked["rule"],
            )
            return None

        try:
            resp = _ssrf_safe_http_get(current_url, timeout=timeout)
        except (SSRFConnectionBlocked, httpx.HTTPError) as exc:
            logger.debug("Skills Hub fetch failed for %s: %s", current_url, exc)
            return None

        if resp.status_code in _REDIRECT_STATUS_CODES:
            location = getattr(resp, "headers", {}).get("location")
            if not location:
                return None
            current_url = urljoin(current_url, location)
            continue

        return resp

    logger.warning("Skills Hub fetch exceeded redirect limit for %s", url)
    return None


# ---------------------------------------------------------------------------
# Shared index cache (used by every adapter)
# ---------------------------------------------------------------------------

def _read_json_if_fresh(path: Path, ttl: float) -> Optional[Any]:
    """Parsed JSON from ``path`` when it exists and is younger than ``ttl`` seconds."""
    try:
        if time.time() - path.stat().st_mtime > ttl:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_index_cache(key: str) -> Optional[Any]:
    return _read_json_if_fresh(_index_cache_dir() / f"{key}.json", INDEX_CACHE_TTL)


def _write_index_cache(key: str, data: Any) -> None:
    index_cache_dir = _index_cache_dir()
    index_cache_dir.mkdir(parents=True, exist_ok=True)
    # Cache files hold unvetted community text (possible prompt injection);
    # a .ignore keeps ripgrep and .ignore-aware tools out of the hub dir.
    ignore_file = _hub_dir() / ".ignore"
    if not ignore_file.exists():
        try:
            ignore_file.write_text("# Exclude hub internals from search tools\n*\n", encoding="utf-8")
        except OSError:
            pass
    try:
        (index_cache_dir / f"{key}.json").write_text(
            json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8"
        )
    except OSError as e:
        logger.debug("Could not write cache: %s", e)


# ---------------------------------------------------------------------------
# Hub state files: lock.json, taps.json, audit.log
# ---------------------------------------------------------------------------

class _JsonStateFile:
    """A JSON file under the hub dir with a fixed empty shape (``EMPTY``, deep-copied
    on every miss/corrupt read); ``DEFAULT_PATH`` is the hub path resolver."""

    EMPTY: dict = {}
    DEFAULT_PATH: Any = None

    def __init__(self, path: Optional[Path] = None):
        self.path = path if path is not None else type(self).DEFAULT_PATH()

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return json.loads(json.dumps(self.EMPTY))

    def _write(self, data: dict, **dumps_kwargs) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, **dumps_kwargs) + "\n", encoding="utf-8")


class HubLockFile(_JsonStateFile):
    """skills/.hub/lock.json — provenance of installed hub skills."""

    EMPTY = {"version": 1, "installed": {}}
    DEFAULT_PATH = staticmethod(_lock_file)

    def load(self) -> dict:
        return self._read()

    def save(self, data: dict) -> None:
        self._write(data, ensure_ascii=False)

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
        scan_provenance: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Validate name and install-path SHAPE at write time: a poisoned lock
        # entry is the precondition for the uninstall_skill rmtree-escape.
        safe_name = _validate_skill_name(name)
        safe_install_path = _normalize_lock_install_path(install_path, safe_name)
        data = self.load()
        now = datetime.now(timezone.utc).isoformat()
        data["installed"][safe_name] = {
            "source": source,
            "identifier": identifier,
            "trust_level": trust_level,
            "scan_verdict": scan_verdict,
            "content_hash": skill_hash,
            "install_path": safe_install_path,
            "files": files,
            "metadata": metadata or {},
            "scan_provenance": scan_provenance or {},
            "installed_at": now,
            "updated_at": now,
        }
        self.save(data)

    def record_uninstall(self, name: str) -> None:
        data = self.load()
        data["installed"].pop(name, None)
        self.save(data)

    def get_installed(self, name: str) -> Optional[dict]:
        return self.load()["installed"].get(name)

    def list_installed(self) -> List[dict]:
        return [{"name": name, **entry} for name, entry in self.load()["installed"].items()]


class TapsManager(_JsonStateFile):
    """skills/.hub/taps.json — custom GitHub repo sources."""

    EMPTY = {"taps": []}
    DEFAULT_PATH = staticmethod(_taps_file)

    def load(self) -> List[dict]:
        return self._read().get("taps", [])

    def save(self, taps: List[dict]) -> None:
        self._write({"taps": taps})

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

    list_taps = load


def append_audit_log(action: str, skill_name: str, source: str,
                     trust_level: str, verdict: str, extra: str = "") -> None:
    """Append one space-separated line to the audit log (best-effort)."""
    audit_log = _audit_log()
    audit_log.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    parts = [timestamp, action, skill_name, f"{source}:{trust_level}", verdict]
    if extra:
        parts.append(extra)
    try:
        with open(audit_log, "a", encoding="utf-8") as f:
            f.write(" ".join(parts) + "\n")
    except OSError as e:
        logger.debug("Could not write audit log: %s", e)


# ---------------------------------------------------------------------------
# Hub operations (high-level)
# ---------------------------------------------------------------------------

def ensure_hub_dirs() -> None:
    """Create the .hub directory structure if it doesn't exist."""
    _hub_dir().mkdir(parents=True, exist_ok=True)
    _quarantine_dir().mkdir(exist_ok=True)
    _index_cache_dir().mkdir(exist_ok=True)
    for path, initial in (
        (_lock_file(), '{"version": 1, "installed": {}}\n'),
        (_audit_log(), ""),
        (_taps_file(), '{"taps": []}\n'),
    ):
        if not path.exists():
            path.write_text(initial, encoding="utf-8")


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from abc import ABC  # noqa: F401,E402
from pathlib import PurePosixPath  # noqa: F401,E402
from typing import Tuple  # noqa: F401,E402
from typing import Union  # noqa: F401,E402
from abc import abstractmethod  # noqa: F401,E402
from dataclasses import dataclass  # noqa: F401,E402
from dataclasses import field  # noqa: F401,E402
import hashlib  # noqa: F401,E402
import os  # noqa: F401,E402
from urllib.parse import quote  # noqa: F401,E402
import re  # noqa: F401,E402
import shutil  # noqa: F401,E402
import subprocess  # noqa: F401,E402
from urllib.parse import unquote  # noqa: F401,E402
from urllib.parse import urlparse  # noqa: F401,E402
from urllib.parse import urlsplit  # noqa: F401,E402
from urllib.parse import urlunparse  # noqa: F401,E402
import yaml  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'BrowseShSource': ('tools.skills_hub_sources', 'BrowseShSource'),
    'ClawHubSource': ('tools.skills_hub_clawhub', 'ClawHubSource'),
    'GITHUB_TAP_PROVIDERS': ('tools.skills_hub_github', 'GITHUB_TAP_PROVIDERS'),
    'GitHubAuth': ('tools.skills_hub_github', 'GitHubAuth'),
    'GitHubSource': ('tools.skills_hub_github', 'GitHubSource'),
    'HERMES_INDEX_TTL': ('tools.skills_hub_search', 'HERMES_INDEX_TTL'),
    'HERMES_INDEX_URL': ('tools.skills_hub_search', 'HERMES_INDEX_URL'),
    'HermesIndexSource': ('tools.skills_hub_official', 'HermesIndexSource'),
    'LobeHubSource': ('tools.skills_hub_sources', 'LobeHubSource'),
    'OptionalSkillSource': ('tools.skills_hub_official', 'OptionalSkillSource'),
    'ScanResult': ('tools.skills_guard', 'ScanResult'),
    'SkillBundle': ('tools.skills_hub_models', 'SkillBundle'),
    'SkillMeta': ('tools.skills_hub_models', 'SkillMeta'),
    'SkillSource': ('tools.skills_hub_models', 'SkillSource'),
    'SkillsShSource': ('tools.skills_hub_skillssh', 'SkillsShSource'),
    'TRUSTED_REPOS': ('tools.skills_guard', 'TRUSTED_REPOS'),
    'UrlSource': ('tools.skills_hub_sources', 'UrlSource'),
    'WellKnownSkillSource': ('tools.skills_hub_sources', 'WellKnownSkillSource'),
    'bundle_content_hash': ('tools.skills_hub_install', 'bundle_content_hash'),
    'check_for_skill_updates': ('tools.skills_hub_install', 'check_for_skill_updates'),
    'content_hash': ('tools.skills_guard', 'content_hash'),
    'create_source_router': ('tools.skills_hub_search', 'create_source_router'),
    'github_provider_for': ('tools.skills_hub_github', 'github_provider_for'),
    'install_from_quarantine': ('tools.skills_hub_install', 'install_from_quarantine'),
    'is_excluded_skill_path': ('agent.skill_utils', 'is_excluded_skill_path'),
    'parallel_search_sources': ('tools.skills_hub_search', 'parallel_search_sources'),
    'quarantine_bundle': ('tools.skills_hub_install', 'quarantine_bundle'),
    'source_url_for_bundle': ('tools.skills_hub_models', 'source_url_for_bundle'),
    'unified_search': ('tools.skills_hub_search', 'unified_search'),
    'uninstall_skill': ('tools.skills_hub_install', 'uninstall_skill'),
    'windows_hide_flags': ('hermes_cli._subprocess_compat', 'windows_hide_flags'),
}

_plugin_compat_prev_getattr = __getattr__


def __getattr__(name):  # PEP 562 — chained onto the module's own __getattr__
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        return _plugin_compat_prev_getattr(name)
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
