"""Website access policy helpers for URL-capable tools.

This module loads a user-managed website blocklist from ~/.hermes/config.yaml
and optional shared list files. It is intentionally lightweight so web/browser
tools can enforce URL policy without pulling in the heavier CLI config stack.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import yaml


_DEFAULT_WEBSITE_BLOCKLIST = {
    "enabled": True,
    "domains": [],
    "shared_files": [],
}


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
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise WebsitePolicyError(f"Shared blocklist file not found: {path}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise WebsitePolicyError(f"Failed to read shared blocklist file {path}: {exc}") from exc

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

    return {"enabled": enabled, "rules": rules}


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
    host = _extract_host_from_urlish(url)
    if not host:
        return None

    policy = load_website_blocklist(config_path)
    if not policy.get("enabled"):
        return None

    for rule in policy.get("rules", []):
        pattern = rule.get("pattern", "")
        if _match_host_against_rule(host, pattern):
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
