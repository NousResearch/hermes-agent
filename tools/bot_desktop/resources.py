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
import os
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


def _cgroup_limit_and_usage() -> tuple[Optional[int], Optional[int]]:
    limit = _read_int(_CGROUP_V2 / "memory.max")
    usage = _read_int(_CGROUP_V2 / "memory.current")
    if limit is None and usage is None:
        limit = _read_int(_CGROUP_V1 / "memory.limit_in_bytes")
        usage = _read_int(_CGROUP_V1 / "memory.usage_in_bytes")
        if limit is not None and limit >= 1 << 60:  # v1 "unlimited" is a huge sentinel
            limit = None
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
    # A deployment may qualify a bounded screen service without changing the
    # customer's persistent config.  The fleet candidate sets this explicitly;
    # ordinary installations retain the conservative config default.
    override = os.environ.get("HERMES_BOT_DESKTOP_MIN_FREE_MEMORY_MB")
    auto_start = os.environ.get("HERMES_BOT_DESKTOP_AUTO_START", "").lower() in {"1", "true", "yes"}
    if override is not None and auto_start:
        try:
            return max(0, int(override))
        except ValueError:
            pass
    from hermes_cli.config import load_config_readonly
    cfg = load_config_readonly().get("bot_desktop") or {}
    try:
        return max(0, int(cfg.get("min_free_memory_mb", DEFAULT_MIN_FREE_MB)))
    except (TypeError, ValueError):
        return DEFAULT_MIN_FREE_MB


def memory_blocker(info: Optional[MemoryInfo] = None) -> Optional[str]:
    """Why the screen must not start now, or None. Unknown memory is not a blocker: a host we cannot
    read is not a host we know to be small."""
    need = min_free_mb()
    if need == 0:
        return None
    info = info or memory_info()
    if info.available_mb is None or info.available_mb >= need:
        return None
    limit = f" of {info.limit_mb} MB" if info.limit_mb else ""
    return (f"Not enough free memory for a desktop: {info.available_mb} MB available{limit}, "
            f"{need} MB needed (bot_desktop.min_free_memory_mb)")
