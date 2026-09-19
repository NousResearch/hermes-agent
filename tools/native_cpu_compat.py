"""CPU compatibility guards for optional native audio dependencies."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Optional

_REQUIRED_X86_64_LOCAL_VOICE_FLAGS = frozenset({"sse4_1", "sse4_2"})


def _linux_cpu_flags(cpuinfo_text: str) -> set[str]:
    """Return the union of Linux ``/proc/cpuinfo`` flags."""
    flags: set[str] = set()
    for line in cpuinfo_text.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip().lower() in {"flags", "features"}:
            flags.update(value.strip().lower().split())
    return flags


def x86_64_local_voice_native_unsupported_reason(
    *,
    system: Optional[str] = None,
    machine: Optional[str] = None,
    cpuinfo_text: Optional[str] = None,
) -> Optional[str]:
    """Explain why local native voice dependencies are unsafe on this host.

    Older Linux x86_64 CPUs can import-crash NumPy / CTranslate2 wheels with
    SIGILL before Python can catch an exception. Non-Linux or non-x86_64 hosts
    fail open; Linux hosts with unreadable cpuinfo also fail open so unusual
    containers are not blocked by an inconclusive probe.
    """
    system = (system or platform.system()).strip().lower()
    machine = (machine or platform.machine()).strip().lower()
    if system != "linux" or machine not in {"x86_64", "amd64"}:
        return None
    if cpuinfo_text is None:
        try:
            cpuinfo_text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    flags = _linux_cpu_flags(cpuinfo_text)
    if not flags:
        return None
    missing = sorted(_REQUIRED_X86_64_LOCAL_VOICE_FLAGS - flags)
    if not missing:
        return None
    return (
        "local voice native dependencies require CPU flags "
        f"{', '.join(sorted(_REQUIRED_X86_64_LOCAL_VOICE_FLAGS))}; "
        f"this x86_64 CPU is missing {', '.join(missing)}"
    )
