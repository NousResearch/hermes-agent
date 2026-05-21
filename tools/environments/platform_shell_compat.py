"""Cross-platform shell / subprocess helpers for local and persistent environments.

Fork-specific Windows behavior and Unix process-group semantics live here so
upstream changes to LocalEnvironment / PersistentShellMixin merge with smaller
conflict surface — adjust this module when OS APIs diverge.
"""

from __future__ import annotations

import os
import platform
import signal
import subprocess
import tempfile
from collections.abc import Callable
from typing import Literal

HermesTempKind = Literal["local", "persistent"]

# Standard PATH entries for environments with minimal PATH (e.g. systemd services).
# Includes macOS Homebrew paths (/opt/homebrew/* for Apple Silicon).
_SANE_PATH = (
    "/opt/homebrew/bin:/opt/homebrew/sbin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)


def is_windows() -> bool:
    return platform.system() == "Windows"


def hermes_temp_prefix(kind: HermesTempKind, session_id: str) -> str:
    """Return a forward-slash temp path prefix for Hermes IPC files (no trailing dash)."""
    label = "hermes-local" if kind == "local" else "hermes-persistent"
    if is_windows():
        base = tempfile.gettempdir().replace("\\", "/")
        return f"{base}/{label}-{session_id}"
    return f"/tmp/{label}-{session_id}"


def get_popen_preexec_fn() -> Callable[[], int] | None:
    """Child preexec for new session / process group; None on Windows."""
    return None if is_windows() else os.setsid


def apply_sane_path_if_unix(run_env: dict) -> None:
    """Append _SANE_PATH on Unix when PATH lacks /usr/bin; no-op on Windows."""
    if is_windows():
        return
    existing_path = run_env.get("PATH", "")
    if "/usr/bin" not in existing_path.split(":"):
        run_env["PATH"] = f"{existing_path}:{_SANE_PATH}" if existing_path else _SANE_PATH


def kill_shell_children(shell_pid: int | None) -> None:
    """Kill child processes of the persistent shell (best-effort)."""
    if shell_pid is None:
        return
    try:
        if is_windows():
            subprocess.run(
                ["taskkill", "/F", "/FI", f"PPID eq {shell_pid}"],
                capture_output=True,
                timeout=5,
            )
        else:
            subprocess.run(
                ["pkill", "-P", str(shell_pid)],
                capture_output=True,
                timeout=5,
            )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


def terminate_process_on_user_interrupt(proc: subprocess.Popen, *, wait_timeout: float = 1.0) -> None:
    """Stop a shell command invoked with process groups (Unix) or terminate (Windows)."""
    try:
        if is_windows():
            proc.terminate()
        else:
            killpg = getattr(os, "killpg")
            sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
            pgid = os.getpgid(proc.pid)
            killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=wait_timeout)
            except subprocess.TimeoutExpired:
                killpg(pgid, sigkill)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def terminate_process_on_timeout(proc: subprocess.Popen) -> None:
    """SIGTERM process group on Unix; single process terminate on Windows."""
    try:
        if is_windows():
            proc.terminate()
        else:
            killpg = getattr(os, "killpg")
            killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.kill()
