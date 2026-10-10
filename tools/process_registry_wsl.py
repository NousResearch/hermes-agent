"""WSL launcher-chain detection for process-registry entries (#120546)."""

import os
import shlex

_WSL_LAUNCHER_NAMES = frozenset({"wsl", "wsl.exe"})

_WSL_CHAIN_NOTE = (
    "Spawned via a wsl[.exe] launcher: the recorded host PID is the short-lived "
    "launcher, not the Linux-side workers. Inspect them with `wsl -e ps` / "
    "`wsl --list --running` from the host."
)


def _is_wsl_launcher_command(command: str) -> bool:
    """True when *command* routes through a ``wsl[.exe]`` launcher chain (#120546).

    The host PID recorded for such a spawn belongs to the short-lived launcher;
    grandchildren inside the VM outlive it, so the entry must say so instead of
    letting host-side hunting fail silently.
    """
    if not isinstance(command, str) or not command.strip():
        return False
    candidates = []
    try:
        candidates.append((shlex.split(command, posix=True) or [""])[0])
    except ValueError:
        pass
    # POSIX shlex eats Windows backslashes (``C:\...\wsl.exe``), so also try
    # the naive first token where path separators survive.
    words = command.strip().split()
    if words:
        candidates.append(words[0])
    for first in candidates:
        base = os.path.basename(first.replace("\\", "/")).strip("'\"").lower()
        if base in _WSL_LAUNCHER_NAMES:
            return True
    return False
