"""CPU compatibility guards for native wheels that assume the x86-64-v2 baseline.

NumPy 2.4 raised its x86-64 cpu-baseline to x86-64-v2 (SSE4.1/SSE4.2/POPCNT) and
ctranslate2's wheels dispatch above that too, so on pre-v2 cores (AMD E2/Bobcat,
pre-Nehalem Intel) the first ``import numpy`` or ``WhisperModel`` load executes an
unsupported instruction — SIGILL, a signal Python cannot catch, which kills the
whole ``hermes serve`` process (#109771). Callers refuse before the native import.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Optional

# numpy 2.4's x86-64-v2 baseline is SSE4.1 + SSE4.2 + POPCNT; ctranslate2's wheels
# dispatch above that. All three must be advertised or the import can SIGILL.
_REQUIRED_X86_64_LOCAL_VOICE_FLAGS = ("sse4_1", "sse4_2", "popcnt")


def _linux_cpu_flags(cpuinfo_text: str) -> set[str]:
    """Union of the ``flags``/``features`` lines of a Linux ``/proc/cpuinfo``."""
    flags: set[str] = set()
    for line in cpuinfo_text.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip().lower() in {"flags", "features"}:
            flags.update(value.strip().lower().replace(",", " ").split())
    return flags


def x86_64_local_voice_native_unsupported_reason(
    *,
    system: Optional[str] = None,
    machine: Optional[str] = None,
    cpuinfo_text: Optional[str] = None,
) -> Optional[str]:
    """Why the SIMD-baseline native wheels (numpy 2.4 / ctranslate2) cannot load here, or None.

    Only x86-64 Linux is probed (via ``/proc/cpuinfo``): every supported arm64 host
    and every Intel Mac satisfies the x86-64-v2 baseline, and Windows has no
    portable flag source worth the false-positive risk. An unreadable cpuinfo
    fails open so unusual containers are not blocked by an inconclusive probe.
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
    missing = [flag for flag in _REQUIRED_X86_64_LOCAL_VOICE_FLAGS if flag not in flags]
    if not missing:
        return None
    return (
        "local voice native wheels (numpy 2.4, ctranslate2) require the x86-64-v2 CPU "
        f"baseline ({' '.join(_REQUIRED_X86_64_LOCAL_VOICE_FLAGS)}); this CPU lacks "
        f"{', '.join(missing)} — importing them kills the process with SIGILL rather "
        "than raising an exception. Use a cloud STT/TTS provider, a local whisper CLI "
        "(HERMES_LOCAL_STT_COMMAND), or a v2-capable CPU."
    )
