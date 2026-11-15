"""Per-worker systemd scope ``MemoryMax`` bound for ProcessRegistry's local executors."""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("tools.process_registry")

_MIN_WORKER_MEMORY_MAX_BYTES = 64 * 1024 * 1024
_DEFAULT_WORKER_MEMORY_MAX_BYTES = 1024 * 1024 * 1024
_WORKER_MEMORY_MAX_CAP_BYTES = 4 * 1024 * 1024 * 1024


def _parse_worker_memory_mb(value) -> Optional[int]:
    """Whole MiB (YAML int or digit string) -> bytes; ``None`` for anything else.
    Floats and bools are rejected outright: ``int(8192.5)`` would silently truncate
    and ``True`` is an ``int`` subclass."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    try:
        parsed = int(value) * 1024 * 1024
    except ValueError:
        return None
    return parsed if parsed >= _MIN_WORKER_MEMORY_MAX_BYTES else None


def _configured_worker_memory_max_bytes() -> Optional[int]:
    """``terminal.worker_memory_max_mb`` in bytes, or ``None`` for ``auto``/invalid.
    Read per spawn so a config edit applies to the next worker without a restart."""
    from tools.process_registry import ProcessRegistry

    try:
        value = ProcessRegistry._config_value("terminal", "worker_memory_max_mb", "auto")
    except Exception:  # unreadable config must never block a spawn; fall back to auto
        logger.warning("Could not read terminal.worker_memory_max_mb; using auto", exc_info=True)
        return None
    if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
        return None
    parsed = _parse_worker_memory_mb(value)
    if parsed is None:
        logger.warning(
            "Ignoring invalid terminal.worker_memory_max_mb=%r; "
            "expected 'auto' or a whole number of MiB >= %d",
            value, _MIN_WORKER_MEMORY_MAX_BYTES // (1024 * 1024))
    return parsed


def _enclosing_cgroup_memory_max_bytes() -> Optional[int]:
    """This process's finite cgroup-v2 ``memory.max``, or ``None`` when unbounded/unreadable."""
    try:
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
            if line.startswith("0::"):
                relative = line.partition("::")[2].lstrip("/")
                raw_limit = (
                    Path("/sys/fs/cgroup") / relative / "memory.max"
                ).read_text(encoding="utf-8-sig").strip()
                if raw_limit.isdigit() and int(raw_limit) >= _MIN_WORKER_MEMORY_MAX_BYTES:
                    return int(raw_limit)
                break
    except (OSError, ValueError):
        pass
    return None


def _worker_memory_max_bytes() -> int:
    """Finite per-worker cgroup limit that can never exceed the enclosing slice.

    ``terminal.worker_memory_max_mb: auto`` (default) is the min of the gateway's
    cgroup-v2 ``memory.max`` and half of physical RAM, capped at 4 GiB. An explicit
    MiB value replaces that auto bound, so a large host can give heavy workers more
    than 4 GiB, but it is still clamped by a finite enclosing ``memory.max`` and by
    physical RAM.

    ``TERMINAL_LOCAL_MEMORY_MAX_MB`` is honored only when it *tightens* the result,
    so this isolation composes with the local-memory guard (PR #57121) instead of
    giving it a second way to widen the bound.
    """
    override_bound: Optional[int] = None
    override = os.getenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "").strip()
    if override:
        try:
            parsed = int(override) * 1024 * 1024
        except ValueError:
            parsed = -1
        if parsed >= _MIN_WORKER_MEMORY_MAX_BYTES:
            override_bound = parsed
        else:
            logger.warning(
                "Ignoring invalid TERMINAL_LOCAL_MEMORY_MAX_MB=%r; "
                "expected an integer representing at least %d MiB",
                override, _MIN_WORKER_MEMORY_MAX_BYTES // (1024 * 1024))
    candidates: list[int] = []
    cgroup_bound = _enclosing_cgroup_memory_max_bytes()
    if cgroup_bound is not None:
        candidates.append(cgroup_bound)
    configured_bound = _configured_worker_memory_max_bytes()
    try:
        physical_bytes = int(os.sysconf("SC_PHYS_PAGES")) * int(
            os.sysconf("SC_PAGE_SIZE")
        )
        if configured_bound is not None:
            candidates.append(max(_MIN_WORKER_MEMORY_MAX_BYTES, physical_bytes))
        else:
            candidates.append(min(
                _WORKER_MEMORY_MAX_CAP_BYTES,
                max(_MIN_WORKER_MEMORY_MAX_BYTES, physical_bytes // 2),
            ))
    except (OSError, ValueError, TypeError):
        pass
    if configured_bound is not None:
        candidates.append(configured_bound)
    safe_bound = min(candidates) if candidates else _DEFAULT_WORKER_MEMORY_MAX_BYTES
    return min(override_bound, safe_bound) if override_bound else safe_bound
