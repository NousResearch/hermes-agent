"""Owned POSIX process-group cleanup for finite command providers."""

from __future__ import annotations

import os
import signal
import subprocess
import time


class CommandProcessScope:
    """Own the wait status of a process started in a new POSIX session.

    This scope must be the leader's only waiter until cleanup finishes.
    """

    def __init__(self, proc: subprocess.Popen[str]) -> None:
        if os.name != "posix":
            raise ValueError("CommandProcessScope requires POSIX")
        self._proc = proc
        self._pid = proc.pid
        self._exited = False
        self._owns_group = True
        self._returncode: int | None = None

    def exited(self) -> bool:
        """Return completion without consuming the leader's wait status."""
        if self._proc.returncode is not None:
            self._owns_group = False
            self._exited = True
            return True
        if self._exited:
            return True
        try:
            status = os.waitid(
                os.P_PID, self._pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError:
            self._owns_group = False
            raise
        self._exited = status is not None
        return self._exited

    def _signal(self, *, group: bool, value: int) -> None:
        while True:
            try:
                if group:
                    os.killpg(self._pid, value)  # windows-footgun: ok, POSIX-only scope.
                else:
                    os.kill(self._pid, value)
                return
            except InterruptedError:
                continue
            except ProcessLookupError:
                return
            except PermissionError:
                if not group or not self.exited() or not self._owns_group:
                    raise
                if not self._group_is_quiescent():
                    raise
                return

    def _group_is_quiescent(self) -> bool:
        try:
            import psutil
        except ImportError:
            return False
        for member in psutil.process_iter():
            try:
                if os.getpgid(member.pid) != self._pid:
                    continue
                if member.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                    return False
            except (ProcessLookupError, psutil.NoSuchProcess):
                continue
            except (PermissionError, psutil.AccessDenied):
                return False
        return True

    def close(self, *, preserve_children: bool = False) -> int:
        """Reap after cleanup, or keep children of a normally completed hook."""
        if self._returncode is not None:
            return self._returncode
        completed = False
        try:
            completed = self.exited()
            if not completed:
                self._signal(group=False, value=signal.SIGTERM)
                deadline = time.monotonic() + 2
                while not self.exited():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    time.sleep(min(0.01, remaining))
        finally:
            try:
                # Keep the leader unreaped until the last group signal so its PID cannot be reused.
                if self._owns_group and not (preserve_children and completed):
                    self._signal(group=True, value=signal.SIGKILL)  # windows-footgun: ok, POSIX-only scope.
            finally:
                self._returncode = self._proc.wait(timeout=2)
        return self._returncode
