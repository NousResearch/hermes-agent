"""OSV malware check for MCP extension packages.

Before launching an MCP server via npx/uvx, queries Google's free public OSV API for
known malware advisories (MAL-* IDs). Regular CVEs are ignored — only confirmed malware
is blocked. Fail-open: network errors allow the package to proceed (~300ms typical).
Inspired by Block/goose's extension malware check.
"""
import json
import logging
import os
import re
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional, Tuple
logger = logging.getLogger(__name__)

_OSV_ENDPOINT = os.getenv("OSV_ENDPOINT", "https://api.osv.dev/v1/query")
_TIMEOUT = 10  # seconds

# Result cache: (ecosystem, package, version) -> (expiry_wallclock, result). Reconnect
# ladders, parked-server self-probes and repeated `hermes mcp test` runs re-run the preflight
# for the SAME package on every spawn; uncached, a flapping server becomes a sustained OSV/DNS
# query stream. Clean AND blocked verdicts are reusable; network failures are NOT cached
# (fail-open covers them and caching one could mask a real advisory later).
# The cache is also persisted under the Hermes home so separate processes and gateway
# restarts reuse warm verdicts; expiry is absolute wall-clock time so it survives restarts.
# Trade-off: a MAL advisory published right after a clean verdict is noticed at TTL expiry
# (<= 1h by default) rather than at next process start — lower OSV_CHECK_CACHE_TTL to tighten.
# Without a cache, a flapping server turns into a sustained OSV query/DNS stream — the #75485 incident
# logged 779K api.osv.dev DNS queries in 16h from revival loops. Malware advisories don't appear or vanish
# on second-to-second timescales, so a successful verdict (clean OR blocked) is reusable. The window is the
# same one the in-process cache already accepted; it just now spans restarts.
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

    Invoked under ``_cache_lock`` from every get/put but does real work only
    once per process (``_disk_cache_loaded`` latch); a transient ``OSError``
    leaves the latch unset so the next call retries. Skips expired or
    malformed entries. Only adds missing keys so an in-memory overwrite
    (e.g. a test forcing expiry) is not silently reversed by the disk copy.
    """
    global _disk_cache_loaded
    if _disk_cache_loaded:
        return

    path = _disk_cache_path()
    if path is None:
        _disk_cache_loaded = True
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = None
    except OSError:
        # Transient I/O (file busy, brief permission flap). Retry next call.
        return
    except Exception:
        # Malformed JSON or anything else: unrecoverable, don't spin on it.
        data = None

    _disk_cache_loaded = True
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
        # Shared atomic writer (temp file + fsync + rename); mkstemp's 0600
        # is kept on create, so verdicts never sit in a world-readable file.
        from utils import atomic_write_text

        atomic_write_text(path, json.dumps(data))
    except Exception as exc:
        logger.debug("Failed to save OSV disk cache to %s: %s", path, exc)


def _cache_get(key) -> Tuple[bool, Optional[str]]:
    """Return (hit, result) for a fresh cache entry."""
    with _cache_lock:
        _load_disk_cache()
        entry = _cache.get(key)
        if entry is not None and time.time() < entry[0]:
            return True, entry[1]
        _cache.pop(key, None)  # absent or expired
        return False, None


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


def check_package_for_malware(command: str, args: list) -> Optional[str]:
    """Check an MCP server package (inferred from ``command``/``args``) for MAL-* advisories.
    Returns a BLOCKED message, else None — also on network errors/unknown commands (fail-open)."""
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
        # Fail-open; deliberately NOT cached — see _CACHE_TTL_S comment.
        logger.debug("OSV check failed for %s/%s (allowing): %s", ecosystem, package, exc)
        return None
    result = None
    if malware:
        ids = ", ".join(m["id"] for m in malware[:3])
        summaries = "; ".join(m.get("summary", m["id"])[:100] for m in malware[:3])
        result = (f"BLOCKED: Package '{package}' ({ecosystem}) has known malware "
                  f"advisories: {ids}. Details: {summaries}")
    _cache_put(cache_key, result)
    return result


_ECOSYSTEM_BY_COMMAND = {
    "npx": "npm", "npx.cmd": "npm", "uvx": "PyPI", "uvx.cmd": "PyPI", "pipx": "PyPI"}


def _infer_ecosystem(command: str) -> Optional[str]:
    return _ECOSYSTEM_BY_COMMAND.get(os.path.basename(command).lower())


def _parse_package_from_args(args: list, ecosystem: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (package_name, version) from command args, or (None, None) if not parseable."""
    # Skip flags to find the package token. npx's explicit install target (--package=NAME /
    # --package NAME / -p NAME) names a package distinct from the executed binary.
    package_token = None
    take_next = False
    for arg in args or ():
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
    parser = _PACKAGE_PARSERS.get(ecosystem)
    return parser(package_token) if parser else (package_token, None)


def _parse_npm_package(token: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse npm package: @scope/name@version or name@version."""
    if token.startswith("@"):
        match = re.match(r"^(@[^/]+/[^@]+)(?:@(.+))?$", token)
        return (match.group(1), match.group(2)) if match else (token, None)
    if "@" in token:
        name, version = token.rsplit("@", 1)
        return name, version if version != "latest" else None
    return token, None


def _parse_pypi_package(token: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse PyPI package: name==version or name[extras]==version."""
    match = re.match(r"^([a-zA-Z0-9._-]+)(?:\[[^\]]*\])?(?:==(.+))?$", token)
    return (match.group(1), match.group(2)) if match else (token, None)


_PACKAGE_PARSERS = {"npm": _parse_npm_package, "PyPI": _parse_pypi_package}


def _query_osv(package: str, ecosystem: str, version: Optional[str] = None) -> list:
    """Query the OSV API; return only MAL-* advisories (regular CVEs ignored)."""
    payload = {"package": {"name": package, "ecosystem": ecosystem}}
    if version:
        payload["version"] = version
    req = urllib.request.Request(
        _OSV_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "hermes-agent-osv-check/1.0"},
        method="POST")
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        result = json.loads(resp.read())
    return [v for v in result.get("vulns", []) if v.get("id", "").startswith("MAL-")]
