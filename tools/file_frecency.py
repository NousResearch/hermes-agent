"""File-picker frecency store for the `@`-file reference picker.

Tracks how often and how recently the user references each file via the `@`
picker, so frequently/recently used files rank higher in completions. Shared
by all three surfaces:

  * CLI         — hermes_cli/commands.py (_fuzzy_file_completions)
  * TUI/Desktop — tui_gateway/server.py (`complete.path` RPC ranker)

and the single record point is the universal `@`-ref resolver in
``agent/context_references.py`` (``_expand_file_reference`` /
``_expand_folder_reference``) — a hit is recorded only when a referenced path
resolves to a real file on send, so we track *actual usage*, not abandoned
popover hovers, regardless of which surface issued the reference.

Algorithm
---------
``fre``-style (camdencheek/fre) single-number exponential decay. For each
tracked path we store ``(weight, t_last)``:

  * record at time ``t``:  weight = weight * e^(-λ·(t - t_last)) + 1 ; t_last = t
  * score at time ``now``: weight * e^(-λ·(now - t_last))

with ``λ = ln2 / (half_life_days · 86400)``. This is mathematically identical
to summing ``e^(-λ·age)`` over the full visit history (validated to 1e-6 over
600 trials) but in O(1) space/time. It avoids the zoxide/z/fasd "re-visit
spike" bug, where a single access to a stale file re-weights its entire
historical rank by the newest timestamp and catapults it to the top.

Default half-life is 1 day: a file you stop touching loses half its weight
every day, so the picker tracks "what am I working on right now" aggressively.

Design notes (mirrors tools/skill_usage.py)
-------------------------------------------
  * Sidecar JSON at ``~/.hermes/.file_frecency.json``, keyed by ABSOLUTE path
    (relative paths collide across repos; absolute keys keep project A's
    history out of project B).
  * Atomic writes via tempfile + os.replace; cross-process fcntl/msvcrt lock.
  * Best-effort: every failure logs at DEBUG and returns silently. A broken or
    missing sidecar never breaks completion or message send.
  * zoxide-style aging: when total weight exceeds ``max_total``, scale all
    weights so the new total is ~90% of the cap, then drop entries below 1.
  * Lazy prune: paths whose file no longer exists are dropped when scored.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# fcntl is Unix-only; on Windows use msvcrt for file locking. Same pattern as
# tools/skill_usage.py.
msvcrt = None
try:
    import fcntl
except ImportError:  # pragma: no cover - platform-specific fallback
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass


SECONDS_PER_DAY = 86400.0

# Built-in defaults; overridden by config.yaml ``picker.frecency.*``.
_DEFAULT_ENABLED = True
_DEFAULT_HALF_LIFE_DAYS = 1.0
_DEFAULT_WEIGHT = 40.0       # alpha: frecency contribution on the 0-100 static scale
_DEFAULT_MAX_ENTRIES = 4000  # hard cap on tracked paths (primary memory bound)
_DEFAULT_MAX_TOTAL = 10000.0  # secondary cap on summed weight (rescale only)


# ---------------------------------------------------------------------------
# Config (cached for the process lifetime; mirrors file_tools._get_max_read_chars)
# ---------------------------------------------------------------------------
_config_cached: Optional[Dict[str, Any]] = None
_config_loaded = False


def _frecency_config() -> Dict[str, Any]:
    """Return the ``picker.frecency`` config block, with defaults backfilled.

    Read once from config.yaml and cached. Any failure falls back to the
    built-in defaults so the store works on a bare profile.
    """
    global _config_cached, _config_loaded
    if _config_loaded and _config_cached is not None:
        return _config_cached

    cfg: Dict[str, Any] = {
        "enabled": _DEFAULT_ENABLED,
        "half_life_days": _DEFAULT_HALF_LIFE_DAYS,
        "weight": _DEFAULT_WEIGHT,
        "max_entries": _DEFAULT_MAX_ENTRIES,
        "max_total": _DEFAULT_MAX_TOTAL,
    }
    try:
        from hermes_cli.config import load_config

        raw = load_config() or {}
        picker = raw.get("picker")
        block = picker.get("frecency") if isinstance(picker, dict) else None
        if isinstance(block, dict):
            if isinstance(block.get("enabled"), bool):
                cfg["enabled"] = block["enabled"]
            hl = block.get("half_life_days")
            if isinstance(hl, (int, float)) and hl > 0:
                cfg["half_life_days"] = float(hl)
            w = block.get("weight")
            if isinstance(w, (int, float)) and w >= 0:
                cfg["weight"] = float(w)
            me = block.get("max_entries")
            if isinstance(me, int) and me > 0:
                cfg["max_entries"] = me
            mt = block.get("max_total")
            if isinstance(mt, (int, float)) and mt > 0:
                cfg["max_total"] = float(mt)
    except Exception as e:  # noqa: BLE001 — config is optional
        logger.debug("file_frecency: config load failed, using defaults: %s", e)

    _config_cached = cfg
    _config_loaded = True
    return cfg


def is_enabled() -> bool:
    return bool(_frecency_config()["enabled"])


def half_life_days() -> float:
    return float(_frecency_config()["half_life_days"])


def weight_alpha() -> float:
    return float(_frecency_config()["weight"])


def _lambda() -> float:
    return math.log(2) / (half_life_days() * SECONDS_PER_DAY)


def reset_config_cache() -> None:
    """Drop the cached config block and the store read-through cache.

    Used by tests that mutate config.yaml or the store on disk between cases so
    a stale parse can't leak across tests.
    """
    global _config_cached, _config_loaded
    _config_cached = None
    _config_loaded = False
    _invalidate_store_cache()


# ---------------------------------------------------------------------------
# Sidecar I/O (mirrors tools/skill_usage.py)
# ---------------------------------------------------------------------------
def _store_file() -> Path:
    return get_hermes_home() / ".file_frecency.json"


@contextmanager
def _store_lock():
    """Serialize the store's read-modify-write cycles across processes."""
    lock_path = _store_file().with_suffix(".json.lock")
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        yield
        return

    if fcntl is None and msvcrt is None:
        yield
        return

    if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
        try:
            lock_path.write_text(" ", encoding="utf-8")
        except OSError:
            yield
            return

    try:
        fd = open(lock_path, "r+" if msvcrt else "a+", encoding="utf-8")
    except OSError:
        yield
        return
    try:
        if fcntl:
            fcntl.flock(fd, fcntl.LOCK_EX)
        else:
            fd.seek(0)
            msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
        yield
    finally:
        if fcntl:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except (OSError, IOError):
                pass
        elif msvcrt:
            try:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
            except (OSError, IOError):
                pass
        fd.close()


def load_store() -> Dict[str, Dict[str, float]]:
    """Read the entire frecency map. Returns ``{}`` on missing/corrupt.

    Each value is ``{"w": <weight float>, "t": <epoch seconds float>}``.
    """
    path = _store_file()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("file_frecency: failed to read %s: %s", path, e)
        return {}
    if not isinstance(data, dict):
        return {}
    clean: Dict[str, Dict[str, float]] = {}
    for k, v in data.items():
        if (
            isinstance(v, dict)
            and isinstance(v.get("w"), (int, float))
            and isinstance(v.get("t"), (int, float))
        ):
            clean[str(k)] = {"w": float(v["w"]), "t": float(v["t"])}
    return clean


def save_store(data: Dict[str, Dict[str, float]]) -> None:
    """Write the map atomically. Best-effort — errors logged, not raised."""
    path = _store_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), prefix=".file_frecency_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, sort_keys=True, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:  # noqa: BLE001 — telemetry write is best-effort
        logger.debug("file_frecency: failed to write %s: %s", path, e, exc_info=True)


# ---------------------------------------------------------------------------
# Read-through cache (hot path)
# ---------------------------------------------------------------------------
# The completion pickers call score_many() once per keystroke. load_store() is
# a disk read + json.loads + an O(entries) validation pass, so calling it per
# frame scales the picker's per-keystroke cost with the store size — exactly
# the asymmetry the file-list caches (5s TTL) already avoid. This is a short
# read-through cache for the SCORING path only: record()/prune_missing() keep
# using the uncached load_store() under the lock (they need a fresh
# read-modify-write) and invalidate this cache on write.
_STORE_TTL_S = 2.0
_store_cache: Optional[Dict[str, Dict[str, float]]] = None
_store_cache_mtime: float = -1.0
_store_cache_time: float = 0.0


def load_store_cached() -> Dict[str, Dict[str, float]]:
    """Return the store via a short TTL + mtime read-through cache.

    Within ``_STORE_TTL_S`` the cached parse is reused directly. After the TTL
    we stat the file: if its mtime is unchanged we refresh the timer and skip
    the re-parse entirely; only a changed (or first-seen) file pays for a fresh
    ``load_store()``. Safe to call on the per-keystroke hot path.
    """
    global _store_cache, _store_cache_mtime, _store_cache_time
    now = time.monotonic()
    if _store_cache is not None and now - _store_cache_time < _STORE_TTL_S:
        return _store_cache
    try:
        mtime = _store_file().stat().st_mtime
    except OSError:
        mtime = -1.0
    if _store_cache is not None and mtime == _store_cache_mtime:
        _store_cache_time = now  # unchanged on disk — reuse parse, reset timer
        return _store_cache
    _store_cache = load_store()
    _store_cache_mtime = mtime
    _store_cache_time = now
    return _store_cache


def _invalidate_store_cache() -> None:
    """Drop the read-through cache so the next score reflects a just-written
    store immediately (called by record() after save_store())."""
    global _store_cache, _store_cache_mtime
    _store_cache = None
    _store_cache_mtime = -1.0


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------
def _abs_key(path: str | Path) -> Optional[str]:
    """Normalize a path to the absolute string used as the store key."""
    try:
        return str(Path(path).expanduser().resolve())
    except (OSError, RuntimeError, ValueError):
        return None


def _decayed(weight: float, t_last: float, now: float, lam: float) -> float:
    """Exponentially-decayed weight at time ``now``.

    Single source of truth for the frecency decay so ``record``/``score``/
    ``score_many`` can't drift apart. The ``dt <= 0`` guard returns the stored
    weight unchanged, defending against clock skew / non-monotonic timestamps
    (a backwards jump must never inflate the score).
    """
    dt = now - t_last
    if dt <= 0:
        return weight
    return weight * math.exp(-lam * dt)


def _age(data: Dict[str, Dict[str, float]], max_entries: int, max_total: float,
         now: float, lam: float) -> None:
    """Bound the store, in place. Two independent caps, neither of which can
    wipe the store:

    1. **Count cap (primary).** If more than ``max_entries`` paths are tracked,
       drop the lowest-*frecency* entries (decayed score at ``now``) until the
       count is back at ``max_entries``. Frecent files are kept, cold ones are
       forgotten. This is a hard count bound, so memory can't grow without
       limit regardless of the weight distribution.
    2. **Weight cap (secondary).** If the summed raw weight still exceeds
       ``max_total`` (a few very-hot files), scale all weights down
       proportionally so the new total is ``max_total``. Unlike the old
       scale-then-threshold scheme this never drops entries, so a uniform
       flood can't collapse the whole store to empty — it just rescales.
    """
    # 1. Count cap — rank by decayed score, keep the top max_entries.
    if max_entries > 0 and len(data) > max_entries:
        ranked = sorted(
            data.items(),
            key=lambda kv: _decayed(kv[1]["w"], kv[1]["t"], now, lam),
            reverse=True,
        )
        for key, _ in ranked[max_entries:]:
            del data[key]

    # 2. Weight cap — proportional rescale only, never a threshold drop.
    if max_total > 0:
        total = 0.0
        for v in data.values():
            total += v["w"]
        if total > max_total:
            factor = max_total / total
            for v in data.values():
                v["w"] *= factor


def record(path: str | Path, *, now: Optional[float] = None) -> None:
    """Record a usage hit for *path*. Best-effort, no-op when disabled.

    Called from the universal `@`-ref resolver after a referenced path is
    confirmed to exist, so it counts real usage across CLI/TUI/Desktop.
    """
    if not is_enabled():
        return
    key = _abs_key(path)
    if key is None:
        return
    t = time.time() if now is None else now
    lam = _lambda()
    cfg = _frecency_config()
    try:
        with _store_lock():
            data = load_store()
            entry = data.get(key)
            if entry is None:
                data[key] = {"w": 1.0, "t": t}
            else:
                decayed = _decayed(entry["w"], entry["t"], t, lam)
                data[key] = {"w": decayed + 1.0, "t": max(t, entry["t"])}
            _age(data, int(cfg["max_entries"]), float(cfg["max_total"]), t, lam)
            save_store(data)
            _invalidate_store_cache()
    except Exception as e:  # noqa: BLE001
        logger.debug("file_frecency.record(%s) failed: %s", key, e, exc_info=True)


def score(path: str | Path, *, now: Optional[float] = None,
          store: Optional[Dict[str, Dict[str, float]]] = None) -> float:
    """Return the current frecency score for *path* (0.0 if untracked/disabled).

    Pass a preloaded ``store`` (via :func:`load_store`) when scoring many paths
    in a tight loop (the picker hot path) to avoid re-reading the file per call.
    """
    if not is_enabled():
        return 0.0
    key = _abs_key(path)
    if key is None:
        return 0.0
    data = store if store is not None else load_store_cached()
    entry = data.get(key)
    if entry is None:
        return 0.0
    t = time.time() if now is None else now
    return _decayed(entry["w"], entry["t"], t, _lambda())


def score_many(paths, *, now: Optional[float] = None) -> Dict[str, float]:
    """Score a batch of paths against one store read. Keys are the input paths
    (as given), values are frecency scores. Untracked paths score 0.0.

    This is the picker entry point: load the store once (cached), score every
    candidate. Called per keystroke by the completion pickers, so it reads
    through :func:`load_store_cached` to avoid a disk read + JSON parse on
    every frame.
    """
    if not is_enabled():
        return {p: 0.0 for p in paths}
    data = load_store_cached()
    t = time.time() if now is None else now
    lam = _lambda()
    out: Dict[str, float] = {}
    for p in paths:
        key = _abs_key(p)
        entry = data.get(key) if key is not None else None
        out[p] = _decayed(entry["w"], entry["t"], t, lam) if entry is not None else 0.0
    return out


def prune_missing() -> int:
    """Drop entries whose file no longer exists. Returns count removed.

    Lazy maintenance — safe to call occasionally (e.g. at session start). Not
    on the hot path. Best-effort.
    """
    removed = 0
    try:
        with _store_lock():
            data = load_store()
            for key in list(data.keys()):
                if not os.path.exists(key):
                    del data[key]
                    removed += 1
            if removed:
                save_store(data)
                _invalidate_store_cache()
    except Exception as e:  # noqa: BLE001
        logger.debug("file_frecency.prune_missing failed: %s", e, exc_info=True)
    return removed
