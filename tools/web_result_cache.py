"""Result caching for web_search / web_extract.

Two caches, both TTL-bounded (default 20 minutes, ``web.cache_ttl_minutes``):

* **Search memo** — in-memory, per-process. Keyed by (provider, normalized
  query, bucketed limit). Concurrent identical queries are single-flighted:
  the first caller performs the paid request while the rest wait and share
  the response. Requested limits are bucketed up to 10/20/50/100 so
  near-identical requests (limit=5 vs limit=8) share one entry; callers get
  their requested count sliced from the bucket.

* **Extract cache** — disk-backed, cross-process. Reuses the existing
  ``cache/web`` full-text store (the same files the truncate-store footer
  points read_file at) plus a small JSON sidecar index mapping URL digest →
  (file, fetched_at, title). A repeat ``web_extract`` of the same URL within
  TTL reads the stored clean text back instead of re-scraping, then re-runs
  the normal truncate pipeline with the caller's char_limit.

Why this lives here and not in generic tool dispatch (issue #8126): a
dispatch-level memo would have to reason about middleware, approval gates,
and hooks on cache hits. Down here the cache sits *after* every safety check
(secret-in-URL, SSRF, policy) and *before* the paid vendor call — hits skip
only the network request, never a control.

Disable with ``web.cache_enabled: false``; both TTLs come from
``web.cache_ttl_minutes``. Only successful responses are ever cached.
"""

import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Requested search limits are rounded UP to one of these buckets so cache
# keys collide on purpose (idea borrowed from Apodex FrontierAgent's
# web_search num-bucketing). Callers get their requested count sliced out.
_LIMIT_BUCKETS = (10, 20, 50, 100)

DEFAULT_TTL_MINUTES = 20

# Extract-index sidecar filename inside cache/web.
_INDEX_FILENAME = "extract-index.json"

# Cap index growth; oldest entries evicted past this.
_INDEX_MAX_ENTRIES = 500


def _web_config() -> dict:
    try:
        from tools.web_tools import _load_web_config
        return _load_web_config()
    except Exception:  # noqa: BLE001 — config problems must never break tools
        return {}


def cache_enabled() -> bool:
    """Both caches honor ``web.cache_enabled`` (default: on)."""
    val = _web_config().get("cache_enabled")
    if val is None:
        return True
    return bool(val)


def ttl_seconds() -> float:
    """TTL from ``web.cache_ttl_minutes`` (default 20, clamped 1–1440)."""
    raw = _web_config().get("cache_ttl_minutes")
    try:
        minutes = float(raw) if raw is not None else DEFAULT_TTL_MINUTES
    except (TypeError, ValueError):
        minutes = DEFAULT_TTL_MINUTES
    minutes = max(1.0, min(minutes, 1440.0))
    return minutes * 60.0


def bucket_limit(limit: int) -> int:
    """Round a requested result count up to the nearest bucket."""
    for b in _LIMIT_BUCKETS:
        if limit <= b:
            return b
    return _LIMIT_BUCKETS[-1]


def normalize_query(query: str) -> str:
    """Case-fold and collapse whitespace so trivial variants share an entry."""
    return re.sub(r"\s+", " ", (query or "").strip().lower())


# ---------------------------------------------------------------------------
# Search memo (in-memory, single-flight)
# ---------------------------------------------------------------------------

class SearchMemo:
    """TTL memo + single-flight coalescer for search responses.

    Thread-safe: web tools run inside the parallel tool-dispatch thread pool
    and subagents share this process, so identical queries can genuinely race.
    Per-key locks make the losers of that race wait for (and share) the
    winner's response instead of issuing their own paid request.
    """

    def __init__(self) -> None:
        self._store: Dict[tuple, Tuple[float, dict]] = {}
        self._store_lock = threading.Lock()
        self._key_locks: Dict[tuple, threading.Lock] = {}

    def _key(self, provider: str, query: str, limit: int) -> tuple:
        return (provider, normalize_query(query), bucket_limit(limit))

    def lookup(self, provider: str, query: str, limit: int) -> Optional[dict]:
        if not cache_enabled():
            return None
        key = self._key(provider, query, limit)
        with self._store_lock:
            hit = self._store.get(key)
            if hit is None:
                return None
            expires, response = hit
            if time.monotonic() >= expires:
                del self._store[key]
                return None
        logger.info("web_search cache hit: %r via %s", query, provider)
        return json.loads(json.dumps(response))  # defensive copy

    def store(self, provider: str, query: str, limit: int, response: dict) -> None:
        """Cache a SUCCESSFUL response for the bucketed key."""
        if not cache_enabled():
            return
        if not isinstance(response, dict) or not response.get("success"):
            return
        key = self._key(provider, query, limit)
        with self._store_lock:
            # Opportunistic expiry sweep to bound memory.
            now = time.monotonic()
            for k in [k for k, (exp, _) in self._store.items() if now >= exp]:
                del self._store[k]
            self._store[key] = (now + ttl_seconds(), json.loads(json.dumps(response)))

    def flight_lock(self, provider: str, query: str, limit: int) -> threading.Lock:
        """Per-key lock for single-flight coalescing.

        Callers hold this around lookup-miss → paid request → store, so a
        concurrent identical call blocks until the winner has stored, then
        finds the entry on its own lookup.
        """
        key = self._key(provider, query, limit)
        with self._store_lock:
            lock = self._key_locks.get(key)
            if lock is None:
                # Bound the lock table alongside the store — but never evict
                # a HELD lock: dropping one lets a concurrent identical
                # request mint a fresh lock and issue a duplicate paid call
                # (review finding on #94618). locked() under _store_lock is
                # a safe snapshot because flight locks are only ever
                # acquired by callers that already hold a reference.
                if len(self._key_locks) > 256:
                    self._key_locks = {
                        k: v for k, v in self._key_locks.items() if v.locked()
                    }
                lock = threading.Lock()
                self._key_locks[key] = lock
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
            out = json.loads(json.dumps(response))
            out["data"]["web"] = out["data"]["web"][:limit]
            return out
    except Exception:  # noqa: BLE001
        pass
    return response


# ---------------------------------------------------------------------------
# Extract cache (disk-backed, reuses cache/web)
# ---------------------------------------------------------------------------

_index_lock = threading.Lock()


def _cache_dir() -> Optional[Path]:
    try:
        from hermes_constants import get_hermes_dir
        d = get_hermes_dir("cache/web", "web_cache")
        d.mkdir(parents=True, exist_ok=True)
        return d
    except Exception:  # noqa: BLE001
        return None


def _index_path() -> Optional[Path]:
    d = _cache_dir()
    return (d / _INDEX_FILENAME) if d else None


def _load_index() -> dict:
    path = _index_path()
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 — corrupt index == empty cache
        return {}


def _save_index(index: dict) -> None:
    path = _index_path()
    if path is None:
        return
    try:
        if len(index) > _INDEX_MAX_ENTRIES:
            newest = sorted(
                index.items(),
                key=lambda kv: kv[1].get("fetched_at", 0),
                reverse=True,
            )[:_INDEX_MAX_ENTRIES]
            index = dict(newest)
        # Per-process tmp name: CLI, gateway, cron, and subagent processes
        # all write this index; a shared fixed tmp filename would let two
        # concurrent writers truncate each other mid-write. os.replace is
        # atomic per writer, so the worst cross-process outcome is one
        # writer's entry winning — a lost cache insert, never a torn file.
        tmp = path.with_suffix(f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(index), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to save web extract cache index: %s", exc)


def _url_digest(url: str, format: Optional[str], provider: str = "") -> str:
    # format AND provider participate in the key: an html extract is not a
    # markdown one, and one backend's rendering of a page is not another's.
    raw = f"{url}\n{format or 'markdown'}\n{provider or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _entry_file_path(url: str, format: Optional[str], provider: str) -> Optional[Path]:
    """Dedicated cache file per (url, format, provider) entry.

    Deliberately NOT the truncate-store file from ``_store_full_text`` — that
    filename keys on URL alone, so html/markdown (or two providers') copies
    of one URL would overwrite each other (review finding on #94618). The
    truncate-store file keeps its role for read_file paging; these files
    exist only for cache reuse and carry the full key in their name.
    """
    d = _cache_dir()
    if d is None:
        return None
    try:
        from urllib.parse import urlparse
        host = (urlparse(url).hostname or "page").replace(":", "_")
        slug = re.sub(r"[^A-Za-z0-9._-]", "-", host)[:60].strip("-") or "page"
    except Exception:  # noqa: BLE001
        slug = "page"
    return d / f"{slug}-{_url_digest(url, format, provider)}.cache.md"


def extract_cache_get(
    url: str,
    format: Optional[str] = None,
    provider: str = "",
) -> Optional[dict]:
    """Return {'url','title','content'} for a fresh cached page, else None."""
    if not cache_enabled():
        return None
    with _index_lock:
        index = _load_index()
        entry = index.get(_url_digest(url, format, provider))
    if not entry:
        return None
    if (time.time() - float(entry.get("fetched_at", 0))) >= ttl_seconds():
        return None
    try:
        file_path = Path(entry["file"])
        cache_root = _cache_dir()
        # The index is plain JSON on disk; never let a tampered entry read
        # outside cache/web.
        if cache_root is None or cache_root.resolve() not in file_path.resolve().parents:
            return None
        content = file_path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001 — evicted/pruned file == miss
        return None
    logger.info("web_extract cache hit: %s", url)
    return {
        "url": url,
        "title": entry.get("title", ""),
        "content": content,
        "error": None,
        "cached": True,
    }


def extract_cache_put(
    url: str,
    content: str,
    title: str = "",
    format: Optional[str] = None,
    provider: str = "",
) -> None:
    """Store one successful extraction's full clean text for TTL reuse.

    Writes a dedicated per-(url, format, provider) cache file (see
    ``_entry_file_path``) — never the URL-keyed truncate-store file, which
    different formats/providers would overwrite. Pages larger than the
    truncate-store ceiling are not cached: serving a capped copy back as if
    whole would silently lose the tail.
    """
    if not cache_enabled() or not content:
        return
    try:
        from tools.web_tools import MAX_STORED_TEXT_CHARS
        if len(content) > MAX_STORED_TEXT_CHARS:
            return
        file_path = _entry_file_path(url, format, provider)
        if file_path is None:
            return
        from tools.spill_safety import write_text_exclusive
        write_text_exclusive(file_path, content, private=False, overwrite=True)
        with _index_lock:
            index = _load_index()
            index[_url_digest(url, format, provider)] = {
                "url": url,
                "file": str(file_path),
                "title": title or "",
                "fetched_at": time.time(),
            }
            _save_index(index)
    except Exception as exc:  # noqa: BLE001 — cache writes are best-effort
        logger.debug("Failed to cache web extract for %s: %s", url, exc)
