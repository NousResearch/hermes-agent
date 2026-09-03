"""Small platform lifecycle primitives used by the bridge."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import BinaryIO


class AlreadyRunningError(RuntimeError):
    """Raised when another bridge process already owns the singleton lock."""


class SingletonLock:
    """Non-blocking, process-scoped lock backed by a one-byte lock file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._handle: BinaryIO | None = None

    def __enter__(self) -> "SingletonLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            handle.close()
            raise AlreadyRunningError("Orca/Hermes bridge is already running") from exc
        self._handle = handle
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def hidden_process_flags() -> int:
    """Return creation flags that keep short-lived bridge helpers invisible."""
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def detached_process_flags() -> int:
    """Return flags suitable for a long-lived background bridge on Windows."""
    return hidden_process_flags() | int(getattr(subprocess, "DETACHED_PROCESS", 0))


def show_qwen_notification() -> None:
    """Show a fixed, token-free Windows notification when Codex falls back."""
    if sys.platform != "win32":
        return
    script = (
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$n=New-Object System.Windows.Forms.NotifyIcon;"
        "$n.Icon=[System.Drawing.SystemIcons]::Information;"
        "$n.BalloonTipTitle='Hermes account fallback';"
        "$n.BalloonTipText='Codex accounts are unavailable. Hermes is using Qwen.';"
        "$n.Visible=$true;$n.ShowBalloonTip(5000);Start-Sleep -Seconds 6;$n.Dispose()"
    )
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=hidden_process_flags(),
        timeout=10,
        check=False,
    )
