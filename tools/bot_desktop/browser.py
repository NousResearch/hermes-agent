"""The bot's browser on its Bot Desktop: one executable, one persistent user-data-dir per profile.

The agent drives Chromium through agent-browser; a human who takes over clicks the dock's Browser
icon. Both must be THE SAME browser — same binary, same ``--user-data-dir`` — or the human logs in
to a jar the bot never sees. Chromium's singleton makes a second launch on the same user-data-dir
open a window in the running instance, which is exactly the hand-over we want.
"""

from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

from tools.bot_desktop import runtime

_SYSTEM_BROWSERS = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")


def profile_dir() -> Path:
    """User-data-dir the bot's browser uses on this profile's screen (``AGENT_BROWSER_PROFILE`` wins)."""
    override = os.environ.get("AGENT_BROWSER_PROFILE", "").strip()
    if override and os.path.isabs(override):
        return Path(override)
    return runtime.state_dir() / "browser-profile"


def executable() -> Optional[str]:
    """The Chromium agent-browser launches: an explicit ``AGENT_BROWSER_EXECUTABLE_PATH``, else the newest
    Playwright Chromium it bundles, else a system Chrome/Chromium. ``None`` when there is none."""
    explicit = os.environ.get("AGENT_BROWSER_EXECUTABLE_PATH", "").strip()
    if explicit and os.access(explicit, os.X_OK):
        return explicit
    from tools.browser_tool_install import _chromium_search_roots
    candidates = sorted(
        (p for root in _chromium_search_roots() for p in glob.glob(os.path.join(root, "chromium-*", "chrome-linux*", "chrome"))),
        key=os.path.getmtime, reverse=True)
    for exe in candidates:
        if os.access(exe, os.X_OK):
            return exe
    return next((shutil.which(name) for name in _SYSTEM_BROWSERS if shutil.which(name)), None)


def dock_launch() -> Optional[Tuple[str, str]]:
    """``(executable, user_data_dir)`` for the dock's Browser icon, or ``None`` when no Chromium exists."""
    exe = executable()
    return (exe, str(profile_dir())) if exe else None


def env_for_agent(env: dict) -> dict:
    """Pin agent-browser to the screen's browser identity unless the user pinned their own."""
    env.setdefault("AGENT_BROWSER_PROFILE", str(profile_dir()))
    exe = executable()
    if exe:
        env.setdefault("AGENT_BROWSER_EXECUTABLE_PATH", exe)
    return env
