"""
RAM Cache — Fast tool result cache for Hermes Agent.

Provides sub-millisecond caching for tool results across parent and child
agents, eliminating redundant file reads, command executions, and API calls
during multi-agent delegation workflows.

Architecture:
    - File-based cache in a fast directory (tmpfs on Linux, tempdir elsewhere)
    - Cross-process sharing via atomic file writes (os.rename)
    - Thread-safe by design (no locks needed — atomic rename is the sync primitive)
    - Zero external dependencies (stdlib only)

Cache categories:
    - file:     File contents, invalidated on mtime change
    - cmd:      Read-only terminal command output, TTL-based
    - tool:     Generic tool results (search_files, etc.), TTL-based
    - api:      External API responses, TTL-based
    - briefing: Parent-to-child context sharing, session-scoped

Configuration via config.yaml:
    cache:
      enabled: true
      root: auto            # "auto" picks tmpfs on Linux, tempdir elsewhere
      max_entries: 1000     # evict oldest when exceeded
      ttl:
        file: 3600
        cmd: 60
        tool: 300
        api: 600
        briefing: 3600

Or via environment:
    HERMES_CACHE_ENABLED=1
    HERMES_CACHE_ROOT=/dev/shm/hermes-cache
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _default_cache_root() -> Path:
    """Pick the best cache directory for this platform."""
    # Linux: /dev/shm is a RAM filesystem — fastest option
    shm = Path("/dev/shm")
    if platform.system() == "Linux" and shm.is_dir():
        return shm / "hermes-cache"
    # macOS / Windows / other: use system temp directory
    return Path(tempfile.gettempdir()) / "hermes-cache"


def _load_config() -> Dict[str, Any]:
    """Load cache config from Hermes config.yaml if available."""
    defaults = {
        "enabled": True,
        "root": "auto",
        "max_entries": 1000,
        "ttl": {
            "file": 3600,
            "cmd": 60,
            "tool": 300,
            "api": 600,
            "briefing": 3600,
        },
    }
    try:
        config_path = Path.home() / ".hermes" / "config.yaml"
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            cache_cfg = cfg.get("cache", {})
            if cache_cfg:
                defaults["enabled"] = cache_cfg.get("enabled", defaults["enabled"])
                defaults["root"] = cache_cfg.get("root", defaults["root"])
                defaults["max_entries"] = cache_cfg.get("max_entries", defaults["max_entries"])
                if "ttl" in cache_cfg:
                    defaults["ttl"].update(cache_cfg["ttl"])
    except Exception:
        pass  # graceful — use defaults
    # Environment overrides
    if os.environ.get("HERMES_CACHE_ENABLED", "").lower() in ("0", "false", "no"):
        defaults["enabled"] = False
    if os.environ.get("HERMES_CACHE_ROOT"):
        defaults["root"] = os.environ["HERMES_CACHE_ROOT"]
    return defaults


_config = _load_config()
ENABLED: bool = _config["enabled"]
CACHE_ROOT: Path = (
    Path(_config["root"]) if _config["root"] != "auto"
    else _default_cache_root()
)
MAX_ENTRIES: int = _config["max_entries"]
DEFAULT_TTLS: Dict[str, int] = _config["ttl"]

# ---------------------------------------------------------------------------
# Read-only command detection
# ---------------------------------------------------------------------------

# Commands that are safe to cache (produce same output for same input)
READ_ONLY_PREFIXES: FrozenSet[str] = frozenset({
    # Filesystem inspection
    "ls", "cat", "head", "tail", "wc", "file ", "stat ", "md5sum",
    "sha256sum", "readlink", "realpath", "basename", "dirname",
    # Search
    "grep", "rg", "find", "locate", "which", "whereis", "type ",
    # System info
    "df", "du", "free", "ps", "uptime", "whoami", "hostname", "id",
    "groups", "uname", "arch", "nproc", "lscpu", "lsmem",
    # Hardware
    "nvidia-smi", "lsblk", "lspci", "lsusb", "lshw",
    # Network (read-only)
    "ip addr", "ip link", "ip route", "ss -", "netstat",
    # Package info (NOT install/upgrade)
    "pip list", "pip show", "pip3 list", "pip3 show",
    "python --version", "python3 --version",
    "node --version", "npm list", "npm ls",
    # Service status (NOT restart/stop)
    "systemctl status", "systemctl is-active", "systemctl list-units",
    "systemctl show",
    # Logs (read-only)
    "journalctl",
    # Container inspection (NOT run/exec/rm)
    "docker ps", "docker logs", "docker inspect", "docker images",
    "docker stats", "docker top",
    # Git (read-only)
    "git log", "git status", "git diff", "git branch", "git show",
    "git tag", "git remote", "git rev-parse", "git describe",
    # Kubernetes (read-only)
    "kubectl get", "kubectl describe", "kubectl logs", "kubectl top",
    "microk8s kubectl get", "microk8s kubectl describe",
    # Text processing (pure functions)
    "sort", "uniq", "awk", "sed -n", "cut ", "tr ", "jq ",
    # Web fetch (read-only)
    "curl -s", "curl --silent",
})

# Patterns that make a command non-cacheable even if prefix matches
_DYNAMIC_PATTERNS = frozenset({
    "$(", "`",       # command substitution — output varies
    "$RANDOM",       # random values
    "$$",            # PID substitution
    "mktemp",        # creates unique files
})


def is_read_only_command(command: str) -> bool:
    """
    Determine if a shell command is read-only (safe to cache).

    Conservative: returns False for anything not explicitly whitelisted.
    Also rejects commands containing dynamic substitutions.
    """
    if not ENABLED:
        return False
    cmd = command.strip()
    # Strip sudo prefix
    if cmd.startswith("sudo "):
        cmd = cmd[5:].strip()
    # Strip env var assignments (e.g., "FOO=bar command")
    while cmd and "=" in (cmd.split()[0] if cmd.split() else ""):
        parts = cmd.split(None, 1)
        cmd = parts[1] if len(parts) > 1 else ""
    if not cmd:
        return False
    # Reject dynamic patterns
    for pattern in _DYNAMIC_PATTERNS:
        if pattern in command:
            return False
    # Check against whitelist
    for prefix in READ_ONLY_PREFIXES:
        if cmd.startswith(prefix):
            return True
    return False


# ---------------------------------------------------------------------------
# Core cache operations
# ---------------------------------------------------------------------------

def _make_key(parts: Union[Tuple, list]) -> str:
    """Hash key parts into a filesystem-safe filename."""
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_path(category: str, key_parts: Union[Tuple, list]) -> Path:
    """Return the cache file path for a given category and key."""
    return CACHE_ROOT / category / f"{_make_key(key_parts)}.json"


_hit_count = 0
_miss_count = 0


def cache_get(category: str, key_parts: Union[Tuple, list],
              ttl: Optional[int] = None) -> Optional[Any]:
    """
    Retrieve a cached value.

    Args:
        category: Cache category (file, cmd, tool, api, briefing).
        key_parts: Tuple/list of values that uniquely identify this entry.
        ttl: Time-to-live in seconds. Defaults to category default.

    Returns:
        Cached value, or None on miss/expiry.

    Never raises exceptions — returns None on any error.
    """
    global _hit_count, _miss_count
    if not ENABLED:
        return None
    if ttl is None:
        ttl = DEFAULT_TTLS.get(category, 300)
    p = _cache_path(category, key_parts)
    try:
        if not p.exists():
            _miss_count += 1
            return None
        raw = p.read_bytes()
        data = json.loads(raw)
        age = time.time() - data["ts"]
        if age > ttl:
            p.unlink(missing_ok=True)
            _miss_count += 1
            logger.debug("Cache expired: %s (age=%.0fs, ttl=%ds)", category, age, ttl)
            return None
        _hit_count += 1
        logger.debug("Cache hit: %s (age=%.0fs)", category, age)
        return data["v"]
    except Exception:
        _miss_count += 1
        return None


def cache_set(category: str, key_parts: Union[Tuple, list],
              value: Any, ttl: Optional[int] = None) -> bool:
    """
    Store a value in cache with atomic write.

    Args:
        category: Cache category.
        key_parts: Unique key tuple/list.
        value: JSON-serializable value to cache.
        ttl: Time-to-live in seconds.

    Returns:
        True on success, False on failure. Never raises.
    """
    if not ENABLED:
        return False
    if ttl is None:
        ttl = DEFAULT_TTLS.get(category, 300)
    p = _cache_path(category, key_parts)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        payload = json.dumps({"v": value, "ts": time.time(), "ttl": ttl},
                             ensure_ascii=False)
        tmp.write_text(payload, encoding="utf-8")
        os.rename(str(tmp), str(p))  # atomic on same filesystem
        # Eviction check (lightweight — only count, don't sort every time)
        _maybe_evict(category)
        return True
    except Exception as e:
        logger.debug("Cache set failed: %s — %s", category, e)
        # Clean up tmp file on failure
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def cache_invalidate(category: str, key_parts: Union[Tuple, list]) -> bool:
    """Remove a specific cache entry."""
    if not ENABLED:
        return False
    p = _cache_path(category, key_parts)
    try:
        p.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def cache_clear(category: Optional[str] = None) -> int:
    """
    Clear cache entries.

    Args:
        category: If given, clear only that category. Otherwise clear all.

    Returns:
        Number of entries removed.
    """
    count = 0
    try:
        targets = []
        if category:
            cat_dir = CACHE_ROOT / category
            if cat_dir.exists():
                targets = [cat_dir]
        elif CACHE_ROOT.exists():
            targets = [d for d in CACHE_ROOT.iterdir() if d.is_dir()]

        for cat_dir in targets:
            for f in cat_dir.glob("*.json"):
                try:
                    f.unlink()
                    count += 1
                except Exception:
                    pass
    except Exception:
        pass
    return count


def cache_stats() -> Dict[str, Any]:
    """
    Return cache statistics.

    Returns:
        Dict with categories, total_entries, total_bytes, hit_rate.
    """
    global _hit_count, _miss_count
    stats: Dict[str, Any] = {
        "categories": {},
        "total_entries": 0,
        "total_bytes": 0,
        "hits": _hit_count,
        "misses": _miss_count,
        "hit_rate": (
            f"{_hit_count / (_hit_count + _miss_count) * 100:.1f}%"
            if (_hit_count + _miss_count) > 0 else "N/A"
        ),
        "root": str(CACHE_ROOT),
        "enabled": ENABLED,
        "platform": platform.system(),
    }
    try:
        if not CACHE_ROOT.exists():
            return stats
        for d in sorted(CACHE_ROOT.iterdir()):
            if d.is_dir():
                files = list(d.glob("*.json"))
                size = sum(f.stat().st_size for f in files)
                stats["categories"][d.name] = {
                    "entries": len(files),
                    "bytes": size,
                }
                stats["total_entries"] += len(files)
                stats["total_bytes"] += size
    except Exception:
        pass
    return stats


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------

def _maybe_evict(category: str) -> None:
    """Evict oldest entries if category exceeds MAX_ENTRIES."""
    try:
        cat_dir = CACHE_ROOT / category
        files = sorted(cat_dir.glob("*.json"), key=lambda f: f.stat().st_mtime)
        excess = len(files) - MAX_ENTRIES
        if excess > 0:
            for f in files[:excess]:
                f.unlink(missing_ok=True)
            logger.debug("Evicted %d entries from %s", excess, category)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# File cache helpers (mtime-validated)
# ---------------------------------------------------------------------------

def file_cache_get(path: str, offset: int = 1, limit: int = 500) -> Optional[str]:
    """
    Get cached file content, validated against current mtime.

    The file's modification time is part of the cache key, so any
    change to the file automatically invalidates the cache entry.

    Returns:
        Cached content string, or None if not cached or file changed.
    """
    try:
        mtime = os.stat(path).st_mtime
        return cache_get("file", (path, offset, limit, mtime), ttl=3600)
    except (FileNotFoundError, OSError):
        return None


def file_cache_set(path: str, offset: int, limit: int, content: Any) -> bool:
    """Store file content with current mtime as part of the key."""
    try:
        mtime = os.stat(path).st_mtime
        return cache_set("file", (path, offset, limit, mtime), content, ttl=3600)
    except (FileNotFoundError, OSError):
        return False


# ---------------------------------------------------------------------------
# Dispatch integration
# ---------------------------------------------------------------------------

def check_cache_before_dispatch(
    function_name: str,
    function_args: Dict[str, Any],
) -> Optional[str]:
    """
    Check cache before tool dispatch. Called from model_tools.handle_function_call.

    Args:
        function_name: Tool name (read_file, terminal, search_files, etc.)
        function_args: Tool arguments dict.

    Returns:
        Cached result string if hit, None if miss.
    """
    if not ENABLED:
        return None
    try:
        if function_name == "read_file":
            path = function_args.get("path", "")
            offset = function_args.get("offset", 1)
            limit = function_args.get("limit", 500)
            cached = file_cache_get(path, offset, limit)
            if cached is not None:
                return cached if isinstance(cached, str) else json.dumps(cached)

        elif function_name == "terminal":
            cmd = function_args.get("command", "")
            cwd = function_args.get("workdir", "")
            if is_read_only_command(cmd):
                key = ("cmd", cmd, cwd)
                cached = cache_get("cmd", key, ttl=DEFAULT_TTLS["cmd"])
                if cached is not None:
                    return cached if isinstance(cached, str) else json.dumps(cached)

        elif function_name == "search_files":
            key = ("tool", function_name, json.dumps(function_args, sort_keys=True))
            cached = cache_get("tool", key, ttl=DEFAULT_TTLS["tool"])
            if cached is not None:
                return cached if isinstance(cached, str) else json.dumps(cached)

    except Exception:
        pass
    return None


def store_cache_after_dispatch(
    function_name: str,
    function_args: Dict[str, Any],
    result: str,
) -> None:
    """
    Store result in cache after tool dispatch. Called from model_tools.handle_function_call.

    Only caches successful results (no error in first 50 chars).
    """
    if not ENABLED:
        return
    try:
        if function_name == "read_file":
            if result and '"error"' not in result[:50]:
                path = function_args.get("path", "")
                offset = function_args.get("offset", 1)
                limit = function_args.get("limit", 500)
                file_cache_set(path, offset, limit, result)

        elif function_name == "terminal":
            cmd = function_args.get("command", "")
            cwd = function_args.get("workdir", "")
            if is_read_only_command(cmd) and result and '"error"' not in result[:50]:
                key = ("cmd", cmd, cwd)
                cache_set("cmd", key, result, ttl=DEFAULT_TTLS["cmd"])

        elif function_name == "search_files":
            if result and '"error"' not in result[:50]:
                key = ("tool", function_name, json.dumps(function_args, sort_keys=True))
                cache_set("tool", key, result, ttl=DEFAULT_TTLS["tool"])

    except Exception:
        pass


# ---------------------------------------------------------------------------
# Hermes tool registration
# ---------------------------------------------------------------------------

try:
    from tools.registry import registry

    def _hermes_cache_stats_tool(args: dict = None, **kwargs) -> str:
        """View RAM cache statistics — entries, size, hit rate, and categories."""
        stats = cache_stats()
        lines = [
            "## RAM Cache Statistics",
            "",
            f"**Enabled:** {stats['enabled']}",
            f"**Platform:** {stats['platform']}",
            f"**Root:** `{stats['root']}`",
            f"**Total entries:** {stats['total_entries']}",
            f"**Total size:** {stats['total_bytes'] / 1024:.1f} KB",
            f"**Hit rate:** {stats['hit_rate']} ({stats['hits']} hits, {stats['misses']} misses)",
            "",
        ]
        for cat, info in stats.get("categories", {}).items():
            ttl = DEFAULT_TTLS.get(cat, 300)
            lines.append(f"- **{cat}**: {info['entries']} entries, "
                        f"{info['bytes'] / 1024:.1f} KB (TTL: {ttl}s)")
        return "\n".join(lines)

    def _hermes_cache_clear_tool(args: dict = None, **kwargs) -> str:
        """Clear RAM cache. Optionally specify a category: file, cmd, api, tool, briefing."""
        args = args or {}
        category = args.get("category", "")
        cat = category.strip() if category else None
        count = cache_clear(cat)
        scope = f"'{cat}'" if cat else "all"
        return f"Cleared {count} entries from {scope} cache."

    registry.register(
        name="cache_stats",
        handler=_hermes_cache_stats_tool,
        schema={
            "name": "cache_stats",
            "description": (
                "View RAM cache statistics — shows entries, size, hit rate, "
                "and per-category breakdown. The cache accelerates tool calls "
                "by storing results of file reads, terminal commands, and "
                "search operations in RAM."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        toolset="cache",
    )

    registry.register(
        name="cache_clear",
        handler=_hermes_cache_clear_tool,
        schema={
            "name": "cache_clear",
            "description": (
                "Clear the RAM cache. Optionally specify a category: "
                "file, cmd, api, tool, briefing. "
                "Use when you need fresh data and suspect cached results are stale."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Category to clear. Leave empty for all.",
                        "enum": ["", "file", "cmd", "tool", "api", "briefing"],
                        "default": "",
                    },
                },
                "required": [],
            },
        },
        toolset="cache",
    )

except ImportError:
    pass  # registry not available (e.g., standalone usage)
