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

import contextlib
import logging
import os
import selectors
import shlex
import shutil
import signal
import subprocess
import threading
import time
from typing import Callable, Optional

from hermes_constants import hermes_home_key
from tools.bot_desktop import runtime

logger = logging.getLogger(__name__)

_install_lock = threading.Lock()
_running: set[str] = set()


NO_SUDO = -2  # install_packages: unprivileged host without sudo; the on_line stream carried the command to run as root


class InstallBusy(RuntimeError):
    pass


def assert_not_running() -> None:
    with _install_lock:
        if hermes_home_key() in _running:
            raise InstallBusy("an install is already running for this profile")


def claim() -> str:
    """Atomically take this profile's install slot; raises :class:`InstallBusy` when taken. A caller that
    claims before handing off to a worker passes ``claimed=True`` to :func:`install_packages`, which then
    owns releasing it — a check-then-spawn pair (``assert_not_running`` + later claim on the worker) lets
    two Install clicks both pass the check."""
    key = hermes_home_key()
    with _install_lock:
        if key in _running:
            raise InstallBusy("an install is already running for this profile")
        _running.add(key)
    return key


def release(key: str) -> None:
    with _install_lock:
        _running.discard(key)


def install_packages(*, ask_password: Callable[[], str], on_line: Callable[[str], None],
                     timeout_seconds: float = 900.0, claimed: bool = False) -> int:
    """Run the package install; returns the process exit code (0 = success, ``-1`` = cancelled,
    :data:`NO_SUDO` = unprivileged host without sudo — the command to run by hand was streamed).
    ``claimed=True``: the caller already holds the slot via :func:`claim`; it is released here either way."""
    key = hermes_home_key() if claimed else None
    try:
        if not runtime.is_supported_host():
            raise RuntimeError("Bot Desktop runs on Linux gateway hosts only")
        cmd = runtime.install_command()
        if cmd is None:
            raise RuntimeError("no supported package manager (apt-get, dnf, pacman) found on this host")
        if key is None:
            key = claim()
        return _run(cmd, ask_password=ask_password, on_line=on_line, timeout_seconds=timeout_seconds)
    finally:
        if key is not None:
            release(key)


def _sudo_nopasswd() -> bool:
    try:
        return subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=3,
                              stdin=subprocess.DEVNULL).returncode == 0
    except Exception:
        return False


def _run(cmd: str, *, ask_password: Callable[[], str], on_line: Callable[[str], None],
         timeout_seconds: float) -> int:
    argv = shlex.split(cmd)
    stdin_payload: Optional[str] = None
    if argv[0] == "sudo":
        if shutil.which("sudo") is None:
            # Minimal containers ship no sudo: a password card would be a dead end. Hand the human the
            # exact command for the host instead.
            on_line(f"install needs root and this host has no sudo; run on the host as root: {cmd[len('sudo '):]}")
            return NO_SUDO
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
        env={**os.environ, **env}, start_new_session=True)
    try:
        if stdin_payload is not None:
            proc.stdin.write(stdin_payload.encode("utf-8"))  # type: ignore[union-attr]
        proc.stdin.close()  # type: ignore[union-attr]
    except OSError:
        pass
    try:
        if _drain_until(proc, on_line, time.monotonic() + timeout_seconds):
            return proc.wait()
        _kill_group(proc)
        on_line(f"install timed out after {timeout_seconds:.0f}s; the package manager may still be running as root")
        return proc.returncode if proc.returncode is not None else -9
    finally:
        proc.stdout.close()  # type: ignore[union-attr]


_TERM_GRACE_SECONDS = 5.0


def _drain_until(proc: subprocess.Popen, on_line: Callable[[str], None], deadline: float) -> bool:
    """Stream ``proc.stdout`` lines to ``on_line`` until EOF (``True``) or ``deadline`` (``False``).

    Readiness-polled rather than a blocking ``for line in proc.stdout``: from an unprivileged Hermes no
    signal reaches a root-owned apt/dnf child, and that child keeps the pipe's write end open, so a
    blocking read would never see EOF and the profile's install slot would be held forever.
    """
    fd = proc.stdout.fileno()  # type: ignore[union-attr]
    buf = b""
    with selectors.DefaultSelector() as sel:
        sel.register(fd, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if not sel.select(timeout=min(remaining, 1.0)):
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                if buf:
                    on_line(buf.decode("utf-8", "replace"))
                return True
            *lines, buf = (buf + chunk).split(b"\n")
            for line in lines:
                on_line(line.decode("utf-8", "replace"))


def _kill_group(proc: subprocess.Popen) -> None:
    """The package manager runs in its own session (start_new_session); killing only sudo would leave
    apt/dnf running as root with the dpkg lock while the slot is released, so the whole group goes: TERM
    first so dpkg can finish its transaction, KILL after the grace. Best effort — as non-root neither
    signal reaches a root-owned child, which is why the caller never waits on EOF."""
    for sig, grace in ((signal.SIGTERM, _TERM_GRACE_SECONDS), (signal.SIGKILL, 1.0)):  # windows-footgun: ok — Linux-only (is_supported_host)
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, sig)  # windows-footgun: ok — Linux-only (is_supported_host)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=grace)
            return
