"""Install the Bot Desktop packages on the gateway host from a Desktop client.

The install runs the distro command from ``runtime.install_command()`` (apt/dnf/pacman) as a child
process on THIS host. Privilege comes from the same masked ``sudo.request`` card the terminal tool
raises: ``sudo -n true`` is probed first (NOPASSWD / cached timestamp hosts never see a prompt); when
a password is needed the caller-supplied ``ask_password`` blocks on the card and the value is written
to sudo's stdin (``-S``) exactly once, never logged, never placed on the command line. Output lines
stream through ``on_line`` so the pane can show apt's progress; the return value is the exit code.

One install per profile at a time; a second request while one runs is refused.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import threading
from typing import Callable, Optional

from hermes_constants import hermes_home_key
from tools.bot_desktop import runtime

logger = logging.getLogger(__name__)

_install_lock = threading.Lock()
_running: set[str] = set()


class InstallBusy(RuntimeError):
    pass


def assert_not_running() -> None:
    with _install_lock:
        if hermes_home_key() in _running:
            raise InstallBusy("an install is already running for this profile")


def install_packages(*, ask_password: Callable[[], str], on_line: Callable[[str], None],
                     timeout_seconds: float = 900.0) -> int:
    """Run the package install; returns the process exit code (0 = success, ``-1`` = cancelled)."""
    if not runtime.is_supported_host():
        raise RuntimeError("Bot Desktop runs on Linux gateway hosts only")
    cmd = runtime.install_command()
    if cmd is None:
        raise RuntimeError("no supported package manager (apt-get, dnf, pacman) found on this host")
    key = hermes_home_key()
    with _install_lock:
        if key in _running:
            raise InstallBusy("an install is already running for this profile")
        _running.add(key)
    try:
        return _run(cmd, ask_password=ask_password, on_line=on_line, timeout_seconds=timeout_seconds)
    finally:
        with _install_lock:
            _running.discard(key)


def _sudo_nopasswd() -> bool:
    try:
        return subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=3,
                              stdin=subprocess.DEVNULL).returncode == 0
    except Exception:
        return False


def _run(cmd: str, *, ask_password: Callable[[], str], on_line: Callable[[str], None],
         timeout_seconds: float) -> int:
    argv = shlex.split(cmd)
    assert argv[0] == "sudo", cmd
    stdin_payload: Optional[str] = None
    if not _sudo_nopasswd():
        password = ask_password() or ""
        if not password:
            on_line("install cancelled: no sudo password provided")
            return -1
        # -S: read the password from stdin; -p '': no prompt text mixed into the streamed output.
        argv = ["sudo", "-S", "-p", "", *argv[1:]]
        stdin_payload = password + "\n"
    on_line(f"$ {cmd}")
    env = {"DEBIAN_FRONTEND": "noninteractive", "LC_ALL": "C.UTF-8"}
    proc = subprocess.Popen(  # windows-footgun: ok — Linux-only (is_supported_host)
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={**os.environ, **env}, text=True, encoding="utf-8", errors="replace", start_new_session=True)
    try:
        if stdin_payload is not None:
            proc.stdin.write(stdin_payload)  # type: ignore[union-attr]
        proc.stdin.close()  # type: ignore[union-attr]
    except OSError:
        pass
    timer = threading.Timer(timeout_seconds, proc.kill)
    timer.start()
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            on_line(line.rstrip("\n"))
        return proc.wait()
    finally:
        timer.cancel()
