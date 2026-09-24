"""Host resources the screen needs before it starts: memory.

A headed desktop is a big allocation relative to a small gateway host: Xvnc + Xfce idle at ~220 MiB
and the browser a human opens during a takeover adds 500 MiB to 1 GiB (measured in the official image:
gateway 304 MiB → +Xfce 520 MiB → +Chromium on one page 1,073 MiB). On a 4 GB container the loser
of that squeeze is Chromium, killed mid-login, or the gateway itself. So ``runtime.start()`` refuses
while the host cannot spare ``bot_desktop.min_free_memory_mb`` and ``display.status`` says why, which
the pane shows in place of the Start button.

"Available" is the tighter of two numbers: what the cgroup still allows (a container limit; v2
``memory.max``/``memory.current``, v1 ``memory.limit_in_bytes``/``memory.usage_in_bytes``) and what
the kernel reports free for the whole machine (``MemAvailable``) — a limit of 8 GB on a 4 GB host is
not 8 GB.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_CGROUP_V2 = Path("/sys/fs/cgroup")
_CGROUP_V1 = Path("/sys/fs/cgroup/memory")
_MEMINFO = Path("/proc/meminfo")
_MIB = 1024 * 1024
DEFAULT_MIN_FREE_MB = 1536


@dataclass
class MemoryInfo:
    available_mb: Optional[int]  # None when nothing readable (non-Linux, odd sandboxes)
    limit_mb: Optional[int]  # the cgroup limit when there is one, else the machine's MemTotal


def _read_int(path: Path) -> Optional[int]:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None  # "max" → None


def _meminfo() -> dict[str, int]:
    try:
        lines = _MEMINFO.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    out: dict[str, int] = {}
    for line in lines:
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[key] = int(parts[0]) * 1024  # kB
    return out


def _stat_value(path: Path, key: str) -> Optional[int]:
    """One ``<key> <bytes>`` line out of a cgroup ``memory.stat``."""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            name, _, rest = line.partition(" ")
            if name == key:
                return int(rest.strip())
    except (OSError, ValueError):
        return None
    return None


def _cgroup_limit_and_usage() -> tuple[Optional[int], Optional[int]]:
    """The cgroup's limit and its *working set* — usage minus reclaimable page cache.

    ``memory.current`` counts page cache, so a container reads several hundred MB above idle right after a
    desktop stops with every process gone. Charging that would make the gate tighten over uptime, so we
    subtract ``inactive_file``, the working-set convention kubelet uses.
    """
    limit = _read_int(_CGROUP_V2 / "memory.max")
    usage = _read_int(_CGROUP_V2 / "memory.current")
    cache = _stat_value(_CGROUP_V2 / "memory.stat", "inactive_file")
    if limit is None and usage is None:
        limit = _read_int(_CGROUP_V1 / "memory.limit_in_bytes")
        usage = _read_int(_CGROUP_V1 / "memory.usage_in_bytes")
        cache = _stat_value(_CGROUP_V1 / "memory.stat", "total_inactive_file")
        if limit is not None and limit >= 1 << 60:  # v1 "unlimited" is a huge sentinel
            limit = None
    if usage is not None and cache:
        usage = max(usage - cache, 0)
    return limit, usage


def memory_info() -> MemoryInfo:
    info = _meminfo()
    host_available = info.get("MemAvailable")
    host_total = info.get("MemTotal")
    limit, usage = _cgroup_limit_and_usage()
    candidates = [x for x in (host_available, (limit - usage) if limit is not None and usage is not None else None)
                  if x is not None]
    available = min(candidates) if candidates else None
    limit_bytes = limit if limit is not None else host_total
    return MemoryInfo(available_mb=None if available is None else max(0, available // _MIB),
                      limit_mb=None if limit_bytes is None else limit_bytes // _MIB)


def min_free_mb() -> int:
    """``bot_desktop.min_free_memory_mb``; 0 disables the gate. A hosted deployment sets it in the
    instance's config.yaml (or the managed overlay), not an env var."""
    from hermes_cli.config import load_config_readonly
    cfg = load_config_readonly().get("bot_desktop") or {}
    try:
        return max(0, int(cfg.get("min_free_memory_mb", DEFAULT_MIN_FREE_MB)))
    except (TypeError, ValueError):
        return DEFAULT_MIN_FREE_MB


def tight_headroom_mb(floor: Optional[int] = None) -> int:
    """Above the floor but below this, a start is allowed and logged: the desktop fits, a few browser tabs
    would not. Derived from the floor so raising the floor cannot silently retire the warning."""
    floor = min_free_mb() if floor is None else floor
    return floor + floor // 3


def memory_blocker(info: Optional[MemoryInfo] = None, need: Optional[int] = None) -> Optional[str]:
    """Why the screen must not start now, or None. Unknown memory is not a blocker: a host we cannot
    read is not a host we know to be small. ``need`` lets a caller that also wants
    :func:`tight_headroom_mb` read the floor once instead of loading the config twice."""
    need = min_free_mb() if need is None else need
    if need == 0:
        return None
    info = info or memory_info()
    if info.available_mb is None or info.available_mb >= need:
        return None
    limit = f" of {info.limit_mb} MB" if info.limit_mb else ""
    return (f"Not enough free memory for a desktop: {info.available_mb} MB available{limit}, "
            f"{need} MB needed (bot_desktop.min_free_memory_mb)")
