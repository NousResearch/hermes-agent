"""OSV malware check for MCP extension packages.

Before launching an MCP server via npx/uvx, queries the OSV (Open Source
Vulnerabilities) API to check if the package has any known malware advisories
(MAL-* IDs).  Regular CVEs are ignored — only confirmed malware is blocked.

The API is free, public, and maintained by Google.  Typical latency is ~300ms.
Fail-open: network errors allow the package to proceed.

Inspired by Block/goose's extension malware check.
"""
import json
import logging
import os
import re
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional, Tuple
logger = logging.getLogger(__name__)

_OSV_ENDPOINT = os.getenv("OSV_ENDPOINT", "https://api.osv.dev/v1/query")
_TIMEOUT = 10  # seconds

# Result cache: (ecosystem, package, version) -> (expiry_timestamp, result).
# MCP reconnect ladders, stdio recycles, parked-server self-probes, and
# repeated `hermes mcp test` invocations re-run the preflight for the SAME
# package on every spawn attempt. Without a cache, a flapping server turns
# into a sustained OSV query/DNS stream — the #75485 incident logged 779K
# api.osv.dev DNS queries in 16h from revival loops. Malware advisories don't
# appear or vanish on second-to-second timescales, so a successful verdict
# (clean OR blocked) is reusable. Network failures are NOT cached: fail-open
# already covers them, and caching a failure could mask a real advisory once
# connectivity returns.
#
# The cache is also persisted to disk inside the Hermes home so that separate
# `hermes mcp test` processes (and gateway restarts) reuse a warm verdict
# instead of re-querying OSV. Expiry is stored as absolute wall-clock time so
# it survives process restarts and monotonic-clock skew.
_CACHE_TTL_S = float(os.getenv("OSV_CHECK_CACHE_TTL", "3600"))
_CACHE_MAX_ENTRIES = 256
_cache: dict = {}
_cache_lock = threading.Lock()
_disk_cache_loaded = False
_DISK_CACHE_VERSION = 1


def _disk_cache_path() -> Optional[Path]:
    """Return the path for the persistent OSV verdict cache.

    Uses ``hermes_constants.get_hermes_home()`` so the cache follows the
    active profile and is isolated across Hermes homes. The cache directory
    is created on demand. Returns ``None`` when Hermes home cannot be
    resolved, in which case only the in-process cache is used.
    """
    try:
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
    except Exception:
        return None
    try:
        cache_dir = home / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "osv_check.json"
    except Exception:
        return None


def _load_disk_cache() -> None:
    """Load persistent cache entries from disk into the in-process cache.

    Called once under ``_cache_lock`` on first use. Skips expired or
    malformed entries. Uses absolute wall-clock timestamps. Only adds
    missing keys so an in-memory overwrite (e.g. a test forcing expiry)
    is not silently reversed by the disk copy.
    """
    global _disk_cache_loaded
    if _disk_cache_loaded:
        return
    _disk_cache_loaded = True

    path = _disk_cache_path()
    if path is None:
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return

    if not isinstance(data, dict) or data.get("version") != _DISK_CACHE_VERSION:
        return

    now = time.time()
    for key_str, entry in data.get("entries", {}).items():
        if not isinstance(entry, dict):
            continue
        expiry = entry.get("expiry")
        result = entry.get("result")
        if expiry is None or expiry <= now:
            continue
        parts = key_str.split("|", 2)
        if len(parts) != 3:
            continue
        key = (parts[0], parts[1], parts[2] or None)
        if key not in _cache:
            _cache[key] = (expiry, result)


def _save_disk_cache() -> None:
    """Persist the in-process cache to disk.

    Caller must hold ``_cache_lock`` for consistency. Writes atomically to
    a sibling file then renames into place.
    """
    path = _disk_cache_path()
    if path is None:
        return

    entries: dict = {}
    for key, (expiry, result) in _cache.items():
        key_str = "|".join(str(k) if k is not None else "" for k in key)
        entries[key_str] = {"expiry": expiry, "result": result}

    data = {"version": _DISK_CACHE_VERSION, "entries": entries}

    try:
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=path.parent, prefix=path.name + ".tmp-"
        )
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp_path, path)
    except Exception as exc:
        logger.debug("Failed to save OSV disk cache to %s: %s", path, exc)


def _cache_get(key) -> Tuple[bool, Optional[str]]:
    """Return (hit, result) for a fresh cache entry."""
    with _cache_lock:
        _load_disk_cache()
        entry = _cache.get(key)
        if entry is None:
            return False, None
        expiry, result = entry
        if time.time() >= expiry:
            del _cache[key]
            return False, None
        return True, result


def _cache_put(key, result: Optional[str]) -> None:
    with _cache_lock:
        _load_disk_cache()
        if len(_cache) >= _CACHE_MAX_ENTRIES:
            now = time.time()
            for k in [k for k, (exp, _) in _cache.items() if exp <= now]:
                del _cache[k]
            if len(_cache) >= _CACHE_MAX_ENTRIES:
                _cache.clear()  # tiny working set in practice; safe reset
        _cache[key] = (time.time() + _CACHE_TTL_S, result)
        _save_disk_cache()


def check_package_for_malware(
    command: str, args: list
) -> Optional[str]:
    """Check if an MCP server package has known malware advisories.

    Inspects the *command* (e.g. ``npx``, ``uvx``) and *args* to infer the
    package name and ecosystem.  Queries the OSV API for MAL-* advisories.

    Returns:
        An error message string if malware is found, or None if clean/unknown.
        Returns None (allow) on network errors or unrecognized commands.
    """
    ecosystem = _infer_ecosystem(command)
    if not ecosystem:
        return None  # not npx/uvx — skip

    package, version = _parse_package_from_args(args, ecosystem)
    if not package:
        return None

    cache_key = (ecosystem, package, version)
    hit, cached = _cache_get(cache_key)
    if hit:
        return cached

    try:
        malware = _query_osv(package, ecosystem, version)
    except Exception as exc:
        # Fail-open: network errors, timeouts, parse failures → allow.
        # Deliberately NOT cached — see _CACHE_TTL_S comment.
        logger.debug("OSV check failed for %s/%s (allowing): %s", ecosystem, package, exc)
        return None

    if malware:
        ids = ", ".join(m["id"] for m in malware[:3])
        summaries = "; ".join(
            m.get("summary", m["id"])[:100] for m in malware[:3]
        )
        result = (
            f"BLOCKED: Package '{package}' ({ecosystem}) has known malware "
            f"advisories: {ids}. Details: {summaries}"
        )
    else:
        result = None
    _cache_put(cache_key, result)
    return result


def _infer_ecosystem(command: str) -> Optional[str]:
    """Infer package ecosystem from the command name."""
    base = os.path.basename(command).lower()
    if base in {"npx", "npx.cmd"}:
        return "npm"
    if base in {"uvx", "uvx.cmd", "pipx"}:
        return "PyPI"
    return None


def _parse_package_from_args(
    args: list, ecosystem: str
) -> Tuple[Optional[str], Optional[str]]:
    """Extract package name and optional version from command args.

    Returns (package_name, version) or (None, None) if not parseable.
    """
    if not args:
        return None, None

    # Skip flags to find the package token.
    # Honor npx's explicit install target: --package=NAME / --package NAME and
    # the -p NAME short form, which name a package distinct from the executed
    # binary. Without this the first bare positional (often the command name)
    # is mistaken for the package.
    package_token = None
    take_next = False
    for arg in args:
        if not isinstance(arg, str):
            continue
        if take_next:
            package_token = arg
            break
        if arg in ("--package", "-p"):
            take_next = True
            continue
        if arg.startswith("--package="):
            package_token = arg[len("--package="):]
            break
        if arg.startswith("-"):
            continue
        package_token = arg
        break

    if not package_token:
        return None, None

    if ecosystem == "npm":
        return _parse_npm_package(package_token)
    elif ecosystem == "PyPI":
        return _parse_pypi_package(package_token)
    return package_token, None


def _parse_npm_package(token: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse npm package: @scope/name@version or name@version."""
    if token.startswith("@"):
        # Scoped: @scope/name@version
        match = re.match(r"^(@[^/]+/[^@]+)(?:@(.+))?$", token)
        if match:
            return match.group(1), match.group(2)
        return token, None
    # Unscoped: name@version
    if "@" in token:
        parts = token.rsplit("@", 1)
        name = parts[0]
        version = parts[1] if len(parts) > 1 and parts[1] != "latest" else None
        return name, version
    return token, None


def _parse_pypi_package(token: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse PyPI package: name==version or name[extras]==version."""
    # Strip extras: name[extra1,extra2]==version
    match = re.match(r"^([a-zA-Z0-9._-]+)(?:\[[^\]]*\])?(?:==(.+))?$", token)
    if match:
        return match.group(1), match.group(2)
    return token, None


def _query_osv(
    package: str, ecosystem: str, version: Optional[str] = None
) -> list:
    """Query the OSV API for MAL-* advisories. Returns list of malware vulns."""
    payload = {"package": {"name": package, "ecosystem": ecosystem}}
    if version:
        payload["version"] = version

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _OSV_ENDPOINT,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "hermes-agent-osv-check/1.0",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        result = json.loads(resp.read())

    vulns = result.get("vulns", [])
    # Only malware advisories — ignore regular CVEs
    return [v for v in vulns if v.get("id", "").startswith("MAL-")]
