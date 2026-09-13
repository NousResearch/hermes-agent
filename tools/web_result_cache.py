"""Result caching for web_search / web_extract; both caches TTL-bounded (default 20 min,
``web.cache_ttl_minutes``; disable with ``web.cache_enabled: false``), only successful responses cache.
* **Search memo** — in-memory, per-process, single-flighted: concurrent identical queries share one
  paid request. Limits bucket to 10/20/50/100 so near-identical requests share an entry.
* **Extract cache** — disk-backed under ``cache/web`` (cross-process) with a JSON sidecar index:
  URL digest → (file, fetched_at, title). Hits re-run the normal truncate pipeline.
Lives here, not in tool dispatch, so hits sit *after* every safety check and skip only the vendor call.
"""

import hashlib
import json
import logging
import os
import re
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Requested limits round UP to a bucket so cache keys collide on purpose.
_LIMIT_BUCKETS = (10, 20, 50, 100)

DEFAULT_TTL_MINUTES = 20

_INDEX_FILENAME = "extract-index.json"
_INDEX_MAX_ENTRIES = 500  # oldest entries evicted past this


def _web_config() -> dict:
    try:
        from tools.web_tools import _load_web_config
        return _load_web_config()
    except Exception:  # noqa: BLE001 — config problems must never break tools
        return {}


def cache_enabled() -> bool:
    """Both caches honor ``web.cache_enabled`` (default: on)."""
    return True if (val := _web_config().get("cache_enabled")) is None else bool(val)


def ttl_seconds() -> float:
    """TTL from ``web.cache_ttl_minutes`` (default 20, clamped 1–1440)."""
    raw = _web_config().get("cache_ttl_minutes")
    try:
        minutes = float(raw) if raw is not None else DEFAULT_TTL_MINUTES
    except (TypeError, ValueError):
        minutes = DEFAULT_TTL_MINUTES
    return max(1.0, min(minutes, 1440.0)) * 60.0


def bucket_limit(limit: int) -> int:
    """Round a requested result count up to the nearest bucket."""
    return next((b for b in _LIMIT_BUCKETS if limit <= b), _LIMIT_BUCKETS[-1])


def normalize_query(query: str) -> str:
    """Case-fold and collapse whitespace so trivial variants share an entry."""
    return re.sub(r"\s+", " ", (query or "").strip().lower())


def _host_slug(url: str) -> str:
    """Filesystem-safe hostname slug for cache filenames (``"page"`` when hostless). Shared with
    tools.web_tools_truncate."""
    host = (urlparse(url).hostname or "page").replace(":", "_")
    return re.sub(r"[^A-Za-z0-9._-]", "-", host)[:60].strip("-") or "page"


def _deep_copy(response: dict) -> dict:
    """Defensive copy so callers mutating a hit never corrupt the cached entry."""
    return json.loads(json.dumps(response))


# ─── Search memo (in-memory, single-flight) ───────────────────────────────────

class SearchMemo:
    """TTL memo + single-flight coalescer for search responses. Thread-safe: the parallel tool-dispatch pool
    and subagents share this process, so identical queries genuinely race; per-key locks make the losers
    wait for (and share) the winner's response."""

    def __init__(self) -> None:
        self._store: Dict[tuple, Tuple[float, dict]] = {}  # key -> (expires_at, response)
        self._store_lock = threading.Lock()
        self._key_locks: Dict[tuple, threading.Lock] = {}

    @staticmethod
    def _key(provider: str, query: str, limit: int) -> tuple:
        return (provider, normalize_query(query), bucket_limit(limit))

    def lookup(self, provider: str, query: str, limit: int) -> Optional[dict]:
        if not cache_enabled():
            return None
        key = self._key(provider, query, limit)
        with self._store_lock:
            hit = self._store.get(key)
            if hit is None or time.monotonic() >= hit[0]:
                self._store.pop(key, None)
                return None
        logger.info("web_search cache hit: %r via %s", query, provider)
        return _deep_copy(hit[1])

    def store(self, provider: str, query: str, limit: int, response: dict) -> None:
        """Cache a SUCCESSFUL response for the bucketed key."""
        if not cache_enabled() or not isinstance(response, dict) or not response.get("success"):
            return
        key = self._key(provider, query, limit)
        with self._store_lock:
            now = time.monotonic()  # opportunistic expiry sweep bounds memory
            for k in [k for k, (exp, _) in self._store.items() if now >= exp]:
                del self._store[k]
            self._store[key] = (now + ttl_seconds(), _deep_copy(response))

    def flight_lock(self, provider: str, query: str, limit: int) -> threading.Lock:
        """Per-key lock held around lookup-miss → paid request → store."""
        key = self._key(provider, query, limit)
        with self._store_lock:
            lock = self._key_locks.get(key)
            if lock is None:
                # Bound the lock table, but never evict a HELD lock: dropping one lets a concurrent
                # identical request mint a fresh lock and issue a duplicate paid call. locked() is a
                # safe snapshot under _store_lock because holders already have their reference.
                # See #94618.
                if len(self._key_locks) > 256:
                    self._key_locks = {k: v for k, v in self._key_locks.items() if v.locked()}
                lock = self._key_locks[key] = threading.Lock()
            return lock

    def clear(self) -> None:
        """Drop all cached entries (tests; config changes)."""
        with self._store_lock:
            self._store.clear()
            self._key_locks.clear()


search_memo = SearchMemo()


def slice_search_response(response: dict, limit: int) -> dict:
    """Trim a bucketed response's result list down to the caller's limit."""
    try:
        web = response.get("data", {}).get("web")
        if isinstance(web, list) and len(web) > limit:
            out = _deep_copy(response)
            out["data"]["web"] = out["data"]["web"][:limit]
            return out
    except Exception:  # noqa: BLE001
        pass
    return response


# ─── Extract cache (disk-backed, reuses cache/web) ────────────────────────────

_index_lock = threading.Lock()


def _cache_dir() -> Optional[Path]:
    try:
        from hermes_constants import get_hermes_dir
        d = get_hermes_dir("cache/web", "web_cache")
        d.mkdir(parents=True, exist_ok=True)
        return d
    except Exception:  # noqa: BLE001
        return None


def _load_index() -> dict:
    try:
        data = json.loads((_cache_dir() / _INDEX_FILENAME).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 — missing/corrupt index == empty cache
        return {}


def _save_index(index: dict) -> None:
    if (d := _cache_dir()) is None:
        return
    path = d / _INDEX_FILENAME
    try:
        if len(index) > _INDEX_MAX_ENTRIES:
            newest = sorted(index.items(), key=lambda kv: kv[1].get("fetched_at", 0), reverse=True)
            index = dict(newest[:_INDEX_MAX_ENTRIES])
        # Per-process tmp name: CLI, gateway, cron, and subagents all write this index; a shared tmp name
        # would let concurrent writers truncate each other. os.replace is atomic: worst case is a lost insert.
        tmp = path.with_suffix(f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(index), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to save web extract cache index: %s", exc)


def _url_digest(url: str, format: Optional[str], provider: str = "") -> str:
    # format AND provider are part of the key: html != markdown, and one backend's rendering is not another's.
    raw = f"{url}\n{format or 'markdown'}\n{provider or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _entry_file_path(url: str, format: Optional[str], provider: str) -> Optional[Path]:
    """Dedicated cache file per (url, format, provider) — deliberately NOT the truncate-store file
    (keyed on URL alone), which html/markdown or two providers' copies of one URL would overwrite.

    The truncate-store file keeps its role for read_file paging; these files exist only for cache reuse and
    carry the full key in their name. See #94618.
    """
    if (d := _cache_dir()) is None:
        return None
    slug = "page"
    with suppress(Exception):
        slug = _host_slug(url)
    return d / f"{slug}-{_url_digest(url, format, provider)}.cache.md"


def _host_matches_pattern(host: str, pattern: str) -> bool:
    """Case-insensitive: exact, ``*.wildcard``, or bare-domain suffix
    (``mysite.dev`` also matches ``preview.mysite.dev``)."""
    host = host.lower().strip(".")
    pattern = (pattern or "").lower().strip().strip(".").removeprefix("*.")
    return bool(pattern) and (host == pattern or host.endswith("." + pattern))


def _is_cache_exempt_host(url: str) -> bool:
    """True when the host matches ``web.cache_exempt_hosts`` — sites the user develops over public DNS
    (staging, tunnels, previews) that must fetch live."""
    try:
        patterns = _web_config().get("cache_exempt_hosts") or []
        host = (urlparse(url).hostname or "").strip("[]")
        if not isinstance(patterns, (list, tuple)) or not host:
            return False
        return any(_host_matches_pattern(host, str(p)) for p in patterns)
    except Exception:  # noqa: BLE001 — config problems never break tools
        return False


def _is_local_dev_url(url: str) -> bool:
    """True for loopback/private/LAN URLs — never cached: they are the user's own fast-changing dev servers.
    Hostname heuristics only, no DNS: this is a freshness decision, not a security boundary (SSRF enforcement
    lives in tools/url_safety.py, which blocks these by default anyway)."""
    try:
        host = (urlparse(url).hostname or "").strip("[]").lower()
        # Unparseable → don't cache; single-label (no "." / ":") == LAN name, not public DNS.
        if not host or host == "localhost" or host.endswith((".localhost", ".local")):
            return True
        if "." not in host and ":" not in host:
            return True
        import ipaddress
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return False  # public DNS name
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_unspecified
    except Exception:  # noqa: BLE001 — on doubt, don't cache
        return True


def _cacheable(url: str) -> bool:
    """Extract-cache gate: enabled, not a local-dev host, not user-exempted."""
    return cache_enabled() and not (_is_local_dev_url(url) or _is_cache_exempt_host(url))


def extract_cache_get(url: str, format: Optional[str] = None, provider: str = "") -> Optional[dict]:
    """Return {'url','title','content'} for a fresh cached page, else None."""
    if not _cacheable(url):
        return None
    with _index_lock:
        entry = _load_index().get(_url_digest(url, format, provider))
    if not entry or (time.time() - float(entry.get("fetched_at", 0))) >= ttl_seconds():
        return None
    try:
        file_path, cache_root = Path(entry["file"]), _cache_dir()
        # The index is plain JSON on disk; never let a tampered entry read outside cache/web.
        if cache_root.resolve() not in file_path.resolve().parents:
            return None
        content = file_path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001 — evicted/pruned file == miss (or no cache dir)
        return None
    logger.info("web_extract cache hit: %s", url)
    return {"url": url, "title": entry.get("title", ""), "content": content, "error": None, "cached": True}


def extract_cache_put(
    url: str, content: str, title: str = "", format: Optional[str] = None, provider: str = ""
) -> None:
    """Store one successful extraction's full clean text for TTL reuse; pages over the truncate-store
    ceiling are not cached (serving a capped copy back as if whole would silently lose the tail)."""
    if not content or not _cacheable(url):
        return
    try:
        from tools.web_tools_truncate import MAX_STORED_TEXT_CHARS
        file_path = _entry_file_path(url, format, provider)
        if len(content) > MAX_STORED_TEXT_CHARS or file_path is None:
            return
        from tools.spill_safety import write_text_exclusive
        write_text_exclusive(file_path, content, private=False, overwrite=True)
        with _index_lock:
            index = _load_index()
            index[_url_digest(url, format, provider)] = {
                "url": url, "file": str(file_path), "title": title or "", "fetched_at": time.time(),
            }
            _save_index(index)
    except Exception as exc:  # noqa: BLE001 — cache writes are best-effort
        logger.debug("Failed to cache web extract for %s: %s", url, exc)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Any  # noqa: F401,E402
from typing import List  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
