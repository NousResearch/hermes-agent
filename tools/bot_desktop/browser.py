"""The bot's browser on its Bot Desktop: one executable, one persistent user-data-dir per profile.

The agent drives Chromium through agent-browser; a human who takes over clicks the dock's Browser
icon. Both must be THE SAME browser — same binary, same ``--user-data-dir`` — or the human logs in
to a jar the bot never sees. Chromium's singleton makes a second launch on the same user-data-dir
open a window in the running instance, which is exactly the hand-over we want — in ONE direction. When the
human's dock instance is already up, agent-browser's own launch is forwarded to it and dies without a
DevTools endpoint, so the dock exposes a debugging port and the agent ATTACHES to it (see
:func:`running_instance_cdp_port`) instead of launching.
"""

from __future__ import annotations

import glob
import os
import shutil
import socket
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
    Playwright Chromium it bundles, else a system Chrome/Chromium. ``None`` when there is none.

    Non-root under ``kernel.apparmor_restrict_unprivileged_userns=1`` (Ubuntu 23.10+) flips the order:
    Playwright's bundle has no setuid ``chrome_sandbox`` and dies 'FATAL: No usable sandbox!' there, while
    a distro chromium ships the helper. The bundle stays the answer when it is the only browser — a dock
    icon that fails loudly beats a non-root ``--no-sandbox``.
    """
    explicit = os.environ.get("AGENT_BROWSER_EXECUTABLE_PATH", "").strip()
    if explicit and os.access(explicit, os.X_OK):
        return explicit
    finders = [_playwright_executable, _system_executable]
    if not _is_root() and _userns_restricted():
        finders.reverse()
    return next((exe for find in finders if (exe := find())), None)


def _playwright_executable() -> Optional[str]:
    from tools.browser_tool_install import _chromium_search_roots
    candidates = sorted(
        (p for root in _chromium_search_roots() for p in glob.glob(os.path.join(root, "chromium-*", "chrome-linux*", "chrome"))),
        key=os.path.getmtime, reverse=True)
    return next((exe for exe in candidates if os.access(exe, os.X_OK)), None)


def _system_executable() -> Optional[str]:
    return next((shutil.which(name) for name in _SYSTEM_BROWSERS if shutil.which(name)), None)


def _is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def _userns_restricted() -> bool:
    from tools.browser_tool_session import apparmor_restricts_unprivileged_userns
    return apparmor_restricts_unprivileged_userns()


def dock_launch() -> Optional[Tuple[str, str]]:
    """``(executable, user_data_dir)`` for the dock's Browser icon, or ``None`` when no Chromium exists."""
    exe = executable()
    return (exe, str(profile_dir())) if exe else None


def dock_argv(exe: str, user_data_dir: str) -> list[str]:
    """Command the dock's Browser icon runs. ``--remote-debugging-port=0`` makes a human-started
    instance attachable (Chromium writes the chosen port to ``<user-data-dir>/DevToolsActivePort``);
    first-run / default-browser dialogs would sit between the human and the bot's tabs."""
    # --test-type hides the "Chrome for Testing is only for automated testing" and unsupported-flag
    # (--no-sandbox as root) infobars, which otherwise sit at the top of the human's takeover view.
    # Root gets the same sandbox-bypass flags agent-browser starts this binary with (one policy).
    from tools.browser_tool_session import CHROMIUM_SANDBOX_BYPASS_ARGS
    return [exe, f"--user-data-dir={user_data_dir}", "--remote-debugging-port=0", "--no-first-run",
            "--no-default-browser-check", "--test-type", *(CHROMIUM_SANDBOX_BYPASS_ARGS if _is_root() else ())]


def dock_exec_line(exe: str, user_data_dir: str) -> str:
    """The ``Exec=`` line of the dock's ``.desktop`` entry. Every argument is double-quoted per the
    Desktop Entry spec (a browser under ``/opt/Google Chrome/`` or a profile under a spaced HERMES_HOME
    otherwise splits into garbage): inside the quotes ``" ` $ \\`` are backslash-escaped, and because the
    value is itself a string field, each of those backslashes is escaped once more."""
    def quote(arg: str) -> str:
        quoted = "".join("\\" + ch if ch in '"`$\\' else ch for ch in arg)
        return '"' + quoted.replace("\\", "\\\\") + '"'
    return "Exec=" + " ".join(quote(arg) for arg in dock_argv(exe, user_data_dir))


def running_instance_cdp_port(user_data_dir: str, *, exclude_session: Optional[str] = None) -> Optional[int]:
    """DevTools port of a Chromium currently running on ``user_data_dir``, or ``None``.

    Both files outlive a crashed or closed Chromium: ``SingletonLock`` is a symlink to ``host-pid`` and
    ``DevToolsActivePort`` keeps the last port, so the pid must be alive AND the port must accept a
    connection before it is trusted. An instance agent-browser launched for ``exclude_session`` itself is
    reported as ``None``: its daemon already owns that browser, and handing it ``--cdp`` would make it
    close the browser as a config change and then attach to the port that just died with it.
    """
    try:
        with open(os.path.join(user_data_dir, "DevToolsActivePort"), encoding="utf-8") as fh:
            port_line = fh.readline().strip()
        target = os.readlink(os.path.join(user_data_dir, "SingletonLock"))
    except OSError:
        return None
    _host, _, pid_text = target.rpartition("-")
    if not (port_line.isdigit() and pid_text.isdigit()) or not _pid_alive(int(pid_text)):
        return None
    if exclude_session and _launched_by_session(int(pid_text)) == exclude_session:
        return None
    port = int(port_line)
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            pass
    except OSError:
        return None
    return port


def _launched_by_session(chromium_pid: int) -> Optional[str]:
    """``AGENT_BROWSER_SESSION`` of the agent-browser daemon that spawned ``chromium_pid``, or ``None``
    for a human-started (dock) instance. Chromium itself gets a scrubbed environment, so the daemon's
    ``/proc/<ppid>/environ`` is the marker (Linux-only, same user)."""
    try:
        with open(f"/proc/{chromium_pid}/status", encoding="utf-8") as fh:
            ppid = next((int(line.split()[1]) for line in fh if line.startswith("PPid:")), 0)
        with open(f"/proc/{ppid}/environ", "rb") as fh:
            raw = fh.read()
    except (OSError, ValueError):
        return None
    for item in raw.split(b"\0"):
        key, sep, value = item.partition(b"=")
        if sep and key == b"AGENT_BROWSER_SESSION":
            return value.decode("utf-8", "replace") or None
    return None


def _pid_alive(pid: int) -> bool:
    import psutil
    return psutil.pid_exists(pid)


def env_for_agent(env: dict) -> dict:
    """Pin agent-browser to the screen's browser identity unless the user pinned their own."""
    env.setdefault("AGENT_BROWSER_PROFILE", str(profile_dir()))
    exe = executable()
    if exe:
        env.setdefault("AGENT_BROWSER_EXECUTABLE_PATH", exe)
    return env
