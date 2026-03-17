"""Website access policy helpers for URL-capable tools.

This module loads a user-managed website blocklist from ~/.hermes/config.yaml
and optional shared list files. It is intentionally lightweight so web/browser
tools can enforce URL policy without pulling in the heavier CLI config stack.

Policy is cached in memory with a short TTL so config changes take effect
quickly without re-reading the file on every URL check.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_DEFAULT_WEBSITE_BLOCKLIST = {
    "enabled": False,
    "domains": [],
    "shared_files": [],
}

# Cache: parsed policy + timestamp.  Avoids re-reading config.yaml on every
# URL check (a web_crawl with 50 pages would otherwise mean 51 YAML parses).
_CACHE_TTL_SECONDS = 30.0
_cache_lock = threading.Lock()
_cached_policy: Optional[Dict[str, Any]] = None
_cached_policy_path: Optional[str] = None
_cached_policy_time: float = 0.0


def _get_hermes_home() -> Path:
    return Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))


def _get_default_config_path() -> Path:
    return _get_hermes_home() / "config.yaml"


class WebsitePolicyError(Exception):
    """Raised when a website policy file is malformed."""


def _normalize_host(host: str) -> str:
    return (host or "").strip().lower().rstrip(".")


def _normalize_rule(rule: Any) -> Optional[str]:
    if not isinstance(rule, str):
        return None
    value = rule.strip().lower()
    if not value or value.startswith("#"):
        return None
    if "://" in value:
        parsed = urlparse(value)
        value = parsed.netloc or parsed.path
    value = value.split("/", 1)[0].strip().rstrip(".")
    if value.startswith("www."):
        value = value[4:]
    return value or None


def _iter_blocklist_file_rules(path: Path) -> List[str]:
    """Load rules from a shared blocklist file.

    Missing or unreadable files log a warning and return an empty list
    rather than raising — a bad file path should not disable all web tools.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Shared blocklist file not found (skipping): %s", path)
        return []
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Failed to read shared blocklist file %s (skipping): %s", path, exc)
        return []

    rules: List[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        normalized = _normalize_rule(stripped)
        if normalized:
            rules.append(normalized)
    return rules


def _load_policy_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    config_path = config_path or _get_default_config_path()
    if not config_path.exists():
        return dict(_DEFAULT_WEBSITE_BLOCKLIST)

    try:
        import yaml
    except ImportError:
        logger.debug("PyYAML not installed — website blocklist disabled")
        return dict(_DEFAULT_WEBSITE_BLOCKLIST)

    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise WebsitePolicyError(f"Invalid config YAML at {config_path}: {exc}") from exc
    except OSError as exc:
        raise WebsitePolicyError(f"Failed to read config file {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise WebsitePolicyError("config root must be a mapping")

    security = config.get("security", {})
    if security is None:
        security = {}
    if not isinstance(security, dict):
        raise WebsitePolicyError("security must be a mapping")

    website_blocklist = security.get("website_blocklist", {})
    if website_blocklist is None:
        website_blocklist = {}
    if not isinstance(website_blocklist, dict):
        raise WebsitePolicyError("security.website_blocklist must be a mapping")

    policy = dict(_DEFAULT_WEBSITE_BLOCKLIST)
    policy.update(website_blocklist)
    return policy


def load_website_blocklist(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and return the parsed website blocklist policy.

    Results are cached for ``_CACHE_TTL_SECONDS`` to avoid re-reading
    config.yaml on every URL check.  Pass an explicit ``config_path``
    to bypass the cache (used by tests).
    """
    global _cached_policy, _cached_policy_path, _cached_policy_time

    resolved_path = str(config_path) if config_path else "__default__"
    now = time.monotonic()

    # Return cached policy if still fresh and same path
    if config_path is None:
        with _cache_lock:
            if (
                _cached_policy is not None
                and _cached_policy_path == resolved_path
                and (now - _cached_policy_time) < _CACHE_TTL_SECONDS
            ):
                return _cached_policy

    config_path = config_path or _get_default_config_path()
    policy = _load_policy_config(config_path)

    raw_domains = policy.get("domains", []) or []
    if not isinstance(raw_domains, list):
        raise WebsitePolicyError("security.website_blocklist.domains must be a list")

    raw_shared_files = policy.get("shared_files", []) or []
    if not isinstance(raw_shared_files, list):
        raise WebsitePolicyError("security.website_blocklist.shared_files must be a list")

    enabled = policy.get("enabled", True)
    if not isinstance(enabled, bool):
        raise WebsitePolicyError("security.website_blocklist.enabled must be a boolean")

    rules: List[Dict[str, str]] = []
    seen: set[Tuple[str, str]] = set()

    for raw_rule in raw_domains:
        normalized = _normalize_rule(raw_rule)
        if normalized and ("config", normalized) not in seen:
            rules.append({"pattern": normalized, "source": "config"})
            seen.add(("config", normalized))

    for shared_file in raw_shared_files:
        if not isinstance(shared_file, str) or not shared_file.strip():
            continue
        path = Path(shared_file).expanduser()
        if not path.is_absolute():
            path = (_get_hermes_home() / path).resolve()
        for normalized in _iter_blocklist_file_rules(path):
            key = (str(path), normalized)
            if key in seen:
                continue
            rules.append({"pattern": normalized, "source": str(path)})
            seen.add(key)

    result = {"enabled": enabled, "rules": rules}

    # Cache the result (only for the default path — explicit paths are tests)
    if config_path == _get_default_config_path():
        with _cache_lock:
            _cached_policy = result
            _cached_policy_path = "__default__"
            _cached_policy_time = now

    return result


def invalidate_cache() -> None:
    """Force the next ``check_website_access`` call to re-read config."""
    global _cached_policy
    with _cache_lock:
        _cached_policy = None


def _match_host_against_rule(host: str, pattern: str) -> bool:
    if not host or not pattern:
        return False
    if pattern.startswith("*."):
        return fnmatch.fnmatch(host, pattern)
    return host == pattern or host.endswith(f".{pattern}")


def _extract_host_from_urlish(url: str) -> str:
    parsed = urlparse(url)
    host = _normalize_host(parsed.hostname or parsed.netloc)
    if host:
        return host

    if "://" not in url:
        schemeless = urlparse(f"//{url}")
        host = _normalize_host(schemeless.hostname or schemeless.netloc)
        if host:
            return host

    return ""


def check_website_access(url: str, config_path: Optional[Path] = None) -> Optional[Dict[str, str]]:
    """Check whether a URL is allowed by the website blocklist policy.

    Returns ``None`` if access is allowed, or a dict with block metadata
    (``host``, ``rule``, ``source``, ``message``) if blocked.

    Never raises on policy errors — logs a warning and returns ``None``
    (fail-open) so a config typo doesn't break all web tools.  Pass
    ``config_path`` explicitly (tests) to get strict error propagation.
    """
    # Fast path: if no explicit config_path and the cached policy is disabled
    # or empty, skip all work (no YAML read, no host extraction).
    if config_path is None:
        with _cache_lock:
            if _cached_policy is not None and not _cached_policy.get("enabled"):
                return None

    host = _extract_host_from_urlish(url)
    if not host:
        return None

    try:
        policy = load_website_blocklist(config_path)
    except WebsitePolicyError as exc:
        if config_path is not None:
            raise  # Tests pass explicit paths — let errors propagate
        logger.warning("Website policy config error (failing open): %s", exc)
        return None
    except Exception as exc:
        logger.warning("Unexpected error loading website policy (failing open): %s", exc)
        return None

    if not policy.get("enabled"):
        return None

    for rule in policy.get("rules", []):
        pattern = rule.get("pattern", "")
        if _match_host_against_rule(host, pattern):
            logger.info("Blocked URL %s — matched rule '%s' from %s",
                        url, pattern, rule.get("source", "config"))
            return {
                "url": url,
                "host": host,
                "rule": pattern,
                "source": rule.get("source", "config"),
                "message": (
                    f"Blocked by website policy: '{host}' matched rule '{pattern}'"
                    f" from {rule.get('source', 'config')}"
                ),
            }
    return None
