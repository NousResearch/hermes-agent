"""Chromium versions against a persistent profile's last writer.

Chromium does not support downgrades: a build older than the one that last wrote a profile crashes on the
newer data at startup (a CHECK, SIGTRAP), and a build that survives may rewrite it. The profile names its last
writer in ``<user-data-dir>/Last Version``. A persistent profile outlives its executable choice: the Bot
Desktop starts on a system Chromium and moves to pm's packaged build once that is installed, and pm's pin can
go back on a rollback. So a launch over a persistent profile compares the two first, and a choice between
binaries prefers one that can open the profile.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Optional

_LAST_VERSION_FILE = "Last Version"
_VERSION_PROBE_TIMEOUT_S = 15
_MAJOR = re.compile(r"(\d+)\.\d+\.\d+")

_probed: dict[tuple[str, int], Optional[int]] = {}


def profile_major(user_data_dir: str) -> Optional[int]:
    """Major version of the Chromium that last wrote ``user_data_dir``; ``None`` for a profile it never opened."""
    try:
        text = Path(user_data_dir, _LAST_VERSION_FILE).read_text(encoding="utf-8-sig")
    except OSError:
        return None
    match = _MAJOR.match(text.strip())
    return int(match.group(1)) if match else None


def executable_major(executable: str) -> Optional[int]:
    """Major version ``executable --version`` reports (``Chromium 155.0.8059.79 ...``), cached per binary
    and mtime; ``None`` when it prints none (Windows ``chrome.exe`` prints nothing)."""
    try:
        key = (os.path.realpath(executable), os.stat(executable).st_mtime_ns)
    except OSError:
        return None
    if key not in _probed:
        try:
            out = subprocess.run([executable, "--version"], capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", stdin=subprocess.DEVNULL, timeout=_VERSION_PROBE_TIMEOUT_S,
                                 check=False).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        match = _MAJOR.search(out)
        _probed[key] = int(match.group(1)) if match else None
    return _probed[key]


def can_open(executable: str, user_data_dir: str) -> bool:
    """False only when both versions are known and ``executable`` is older than the profile's last writer."""
    wrote = profile_major(user_data_dir)
    if wrote is None:
        return True
    runs = executable_major(executable)
    return runs is None or runs >= wrote


def downgrade_refusal(executable: str, user_data_dir: str) -> Optional[str]:
    """The launch error for a Chromium older than the profile, or ``None`` when the launch may go ahead."""
    if can_open(executable, user_data_dir):
        return None
    return (f"The browser profile {user_data_dir} was last used by Chromium {profile_major(user_data_dir)}, and "
            f"{executable} is Chromium {executable_major(executable)}. An older Chromium crashes on a newer "
            "profile, so Hermes will not start it. Point browser.executable_path at a Chromium of that version "
            "or newer, or set browser.user_data_dir to a new directory to start with an empty profile.")
