"""Raise the agent's visible browser window to the foreground.

A headed session runs a real Chromium on the user's desktop; a window that opens behind the
chat app makes the agent look idle. This module finds the browser process that owns the
session's CDP port and raises its main window, so the FIRST action (navigation/click) puts
the window on top and later actions re-assert it instead of letting it drift behind.

Windows-first (``netstat`` -> owning pid -> ``EnumWindows`` -> ``SetForegroundWindow``);
on other platforms it tries ``xdotool`` (X11) and otherwise stays a silent no-op. Every step
is best-effort: a failure here must never change a tool result.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

#: Commands that can put a fresh window/page on screen (or start the browser at all).
FOCUS_COMMANDS = frozenset({"open", "click", "dblclick", "back", "forward", "reload", "goto_url", "press"})

#: A raise per command would fight the user for focus; a short floor keeps it to the
#: navigation/click that actually changed what is on screen.
_RAISE_THROTTLE_S = 1.0

_last_raise: Dict[str, float] = {}


def _run(argv: List[str], timeout: float = 5.0) -> str:
    """stdout of a short helper command, or "" (never raises)."""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, stdin=subprocess.DEVNULL)
        return proc.stdout or ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _listening_pids_for_port(port: int, runner=_run) -> List[int]:
    """PIDs of processes listening on ``port`` (Windows ``netstat`` / POSIX ``lsof``)."""
    pids: List[int] = []
    if not port:
        return pids
    if sys.platform.startswith("win"):
        pattern = re.compile(r"(?:^|\s)(?:127\.0\.0\.1|0\.0\.0\.0|\[::1\]|\[::\]):%d\s+\S+\s+LISTENING\s+(\d+)" % port)
        for line in runner(["netstat", "-ano", "-p", "tcp"]).splitlines():
            # netstat prints ``0.0.0.0:9222`` on IPv4 and ``[::]:9222`` on IPv6.
            normalized = line.replace("[::]:", "0.0.0.0:")
            match = pattern.search(" " + normalized.strip())
            if match:
                pid = int(match.group(1))
                if pid and pid not in pids:
                    pids.append(pid)
        return pids
    out = runner(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"])
    for token in out.split():
        if token.isdigit() and int(token) not in pids:
            pids.append(int(token))
    return pids


def _window_handles_for_pid(pid: int) -> List[int]:
    """Top-level, visible, titled window handles owned by ``pid`` (Windows only)."""
    if not sys.platform.startswith("win"):
        return []
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        handles: List[int] = []

        def _callback(hwnd, _lparam):
            owner = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if owner.value == pid and user32.IsWindowVisible(hwnd) and user32.GetWindowTextLengthW(hwnd) > 0:
                handles.append(int(hwnd))
            return True

        enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        user32.EnumWindows(enum_proc(_callback), 0)
        return handles
    except Exception as exc:  # pragma: no cover - platform/ctypes defensive
        logger.debug("window enumeration failed for pid %s: %s", pid, exc)
        return []


def _raise_window(handle: int) -> bool:
    """Bring ``handle`` to the foreground (Windows) / activate it (X11)."""
    if sys.platform.startswith("win"):
        try:
            import ctypes
            user32 = ctypes.windll.user32
            user32.ShowWindow(handle, 9)              # SW_RESTORE: also un-minimizes
            user32.AllowSetForegroundWindow(-1)       # ASFW_ANY
            user32.BringWindowToTop(handle)
            user32.SetForegroundWindow(handle)
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("SetForegroundWindow failed for %s: %s", handle, exc)
            return False
    return bool(_run(["xdotool", "windowactivate", str(handle)], timeout=3.0).strip() != "")


def _session_cdp_port(session_name: str) -> int:
    """CDP port of the agent-browser session's browser, or 0 when it cannot be resolved."""
    try:
        from tools.browser_tool_real_profile import _agent_browser_get_cdp
        cdp = _agent_browser_get_cdp(session_name)
    except Exception as exc:
        logger.debug("CDP port lookup failed for session %s: %s", session_name, exc)
        return 0
    match = re.search(r":(\d+)$", (cdp or "").rstrip("/"))
    return int(match.group(1)) if match else 0


def bring_session_window_to_front(session_name: str, *, throttle: bool = True) -> bool:
    """Raise the browser window of ``session_name``. Best-effort; False when nothing was raised."""
    if not session_name:
        return False
    now = time.monotonic()
    if throttle and now - _last_raise.get(session_name, 0.0) < _RAISE_THROTTLE_S:
        return False
    port = _session_cdp_port(session_name)
    if not port:
        return False
    raised = False
    for pid in _listening_pids_for_port(port):
        handles = _window_handles_for_pid(pid)
        if handles:
            raised = _raise_window(handles[0]) or raised
    if raised:
        _last_raise[session_name] = now
    return raised


def after_browser_command(command: str, session_info: Dict[str, Any], task_id: str = "") -> None:
    """Raise the window for a command that can change what is on screen; never raises.

    Cloud/CDP sessions are skipped: their browser is not on this desktop (and a
    ``/browser connect`` window already belongs to the user).
    """
    try:
        if command not in FOCUS_COMMANDS:
            return
        if session_info.get("cdp_url"):
            return
        from tools import browser_tool_cloud as _cloud
        if not _cloud._is_headed_mode() or _cloud._get_browser_engine() == "lightpanda":
            return
        if os.environ.get("HERMES_BROWSER_KEEP_BEHIND") == "1":  # opt-out for shared desktops
            return
        bring_session_window_to_front(str(session_info.get("session_name") or ""))
    except Exception as exc:  # pragma: no cover - defensive: focus must never break a call
        logger.debug("browser window raise failed for task=%s (%s): %s", task_id, command, exc)