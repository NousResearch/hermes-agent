"""Website access policy helpers for URL-capable tools.

Loads a user-managed website blocklist (``security.website_blocklist`` in ~/.hermes/config.yaml plus
optional shared list files) without the heavier CLI config stack. The parsed policy is cached with a
short TTL so config edits take effect quickly without re-parsing YAML on every URL check.
"""

from __future__ import annotations

import fnmatch
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from hermes_constants import get_hermes_home
from tools.url_safety import _normalize_hostname as _normalize_host

logger = logging.getLogger(__name__)

_DEFAULT_WEBSITE_BLOCKLIST = {"enabled": False, "domains": [], "shared_files": []}

# Without this cache a 50-URL extract would mean 51 YAML parses of config.yaml.
_CACHE_TTL_SECONDS = 30.0
_cache_lock = threading.Lock()
_cached_policy: Optional[Dict[str, Any]] = None
_cached_policy_path: Optional[str] = None
_cached_policy_time: float = 0.0


class WebsitePolicyError(Exception):
    """Raised when a website policy file is malformed."""


def _normalize_rule(rule: Any) -> Optional[str]:
    """Reduce a rule (bare host, URL, or ``host/path``) to a lowercase host; None for blanks/comments."""
    if not isinstance(rule, str) or not (value := rule.strip().lower()) or value.startswith("#"):
        return None
    if "://" in value:
        parsed = urlparse(value)
        value = parsed.netloc or parsed.path
    return value.split("/", 1)[0].strip().rstrip(".").removeprefix("www.") or None


def _iter_blocklist_file_rules(path: Path) -> List[str]:
    """Rules from a shared blocklist file; missing/unreadable files warn and yield nothing rather than
    raising — a bad file path must not disable all web tools."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Shared blocklist file not found (skipping): %s", path)
        return []
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Failed to read shared blocklist file %s (skipping): %s", path, exc)
        return []
    return [rule for rule in map(_normalize_rule, raw.splitlines()) if rule]


def _require_mapping(value: Any, label: str) -> Dict[str, Any]:
    """``None`` (empty YAML section) counts as an empty mapping; other non-dicts are errors."""
    if value is not None and not isinstance(value, dict):
        raise WebsitePolicyError(f"{label} must be a mapping")
    return value or {}


def _load_policy_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return dict(_DEFAULT_WEBSITE_BLOCKLIST)
    try:
        import yaml
    except ImportError:
        logger.debug("PyYAML not installed — website blocklist disabled")
        return dict(_DEFAULT_WEBSITE_BLOCKLIST)
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise WebsitePolicyError(f"Invalid config YAML at {config_path}: {exc}") from exc
    except OSError as exc:
        raise WebsitePolicyError(f"Failed to read config file {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise WebsitePolicyError("config root must be a mapping")
    security = _require_mapping(config.get("security", {}), "security")
    website_blocklist = _require_mapping(security.get("website_blocklist", {}), "security.website_blocklist")
    return {**_DEFAULT_WEBSITE_BLOCKLIST, **website_blocklist}


def _require_type(policy: Dict[str, Any], key: str, kind: type, default: Any) -> Any:
    """Typed policy field; ``None``/empty list values are coerced to ``[]`` for lists only."""
    value = policy.get(key, default)
    if kind is list:
        value = value or []
    if not isinstance(value, kind):
        kind_name = "boolean" if kind is bool else "list"
        raise WebsitePolicyError(f"security.website_blocklist.{key} must be a {kind_name}")
    return value


def load_website_blocklist(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Parsed website blocklist policy (``{"enabled", "rules"}``); cached for ``_CACHE_TTL_SECONDS`` for
    the default config path only — an explicit ``config_path`` (tests) bypasses and never populates it."""
    global _cached_policy, _cached_policy_path, _cached_policy_time
    default_path = get_hermes_home() / "config.yaml"
    resolved_path = str(config_path or default_path)
    now = time.monotonic()
    if config_path is None:
        with _cache_lock:
            fresh = _cached_policy_path == resolved_path and (now - _cached_policy_time) < _CACHE_TTL_SECONDS
            if _cached_policy is not None and fresh:
                return _cached_policy
    config_path = config_path or default_path
    policy = _load_policy_config(config_path)
    domains = map(_normalize_rule, _require_type(policy, "domains", list, []))
    pairs: List[Tuple[str, str]] = [(p, "config") for p in domains if p]
    shared_files = _require_type(policy, "shared_files", list, [])
    enabled = _require_type(policy, "enabled", bool, True)
    for shared_file in shared_files:
        if not isinstance(shared_file, str) or not shared_file.strip():
            continue
        path = Path(shared_file).expanduser()
        path = path if path.is_absolute() else (get_hermes_home() / path).resolve()
        pairs += [(normalized, str(path)) for normalized in _iter_blocklist_file_rules(path)]
    # dict.fromkeys dedupes (pattern, source) while keeping first-seen order.
    result = {"enabled": enabled, "rules": [{"pattern": p, "source": s} for p, s in dict.fromkeys(pairs)]}
    if config_path == default_path:  # explicit paths are tests — never cache them
        with _cache_lock:
            _cached_policy, _cached_policy_path, _cached_policy_time = result, resolved_path, now
    return result


def _match_host_against_rule(host: str, pattern: str) -> bool:
    """``*.example.com`` rules glob-match; bare hosts match exactly or as a parent domain."""
    if not host or not pattern:
        return False
    if pattern.startswith("*."):
        return fnmatch.fnmatch(host, pattern)
    return host == pattern or host.endswith(f".{pattern}")


def _extract_host_from_urlish(url: str) -> str:
    """Host of ``url``; schemeless inputs (``example.com/x``) are retried as ``//url``."""
    parsed = urlparse(url)
    host = _normalize_host(parsed.hostname or parsed.netloc)
    if not host and "://" not in url:
        parsed = urlparse(f"//{url}")
        host = _normalize_host(parsed.hostname or parsed.netloc)
    return host


def check_website_access(url: str, config_path: Optional[Path] = None) -> Optional[Dict[str, str]]:
    """``None`` if the URL is allowed by the blocklist policy, else block metadata (host/rule/source/message).

    Fails open on policy errors (warn + ``None``) so a config typo can't break all web tools — except with
    an explicit ``config_path`` (tests), where errors propagate.
    """
    # Fast path: cached policy disabled/empty → no YAML read, no host extraction.
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
            raise
        logger.warning("Website policy config error (failing open): %s", exc)
        return None
    except Exception as exc:
        logger.warning("Unexpected error loading website policy (failing open): %s", exc)
        return None
    if not policy.get("enabled"):
        return None
    for rule in policy.get("rules", []):
        pattern, source = rule.get("pattern", ""), rule.get("source", "config")
        if _match_host_against_rule(host, pattern):
            logger.info("Blocked URL %s — matched rule '%s' from %s", url, pattern, source)
            return {
                "url": url, "host": host, "rule": pattern, "source": source,
                "message": f"Blocked by website policy: '{host}' matched rule '{pattern}' from {source}",
            }
    return None


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def invalidate_cache() -> None:
    """Force the next ``check_website_access`` call to re-read config."""
    global _cached_policy
    with _cache_lock:
        _cached_policy = None
# ---- END PLUGIN-COMPAT ----
