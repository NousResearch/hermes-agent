"""Environment-backed ProcessRegistry group ownership and filesystem helpers."""
from __future__ import annotations

import logging
import tempfile
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from tools.process_registry import ProcessSession

logger = logging.getLogger("tools.process_registry")


class RemoteProcessMixin:
    @staticmethod
    def _env_temp_dir(env: Any) -> str:
        """Return the writable sandbox temp dir for env-backed background tasks."""
        get_temp_dir = getattr(env, "get_temp_dir", None)
        if callable(get_temp_dir):
            try:
                temp_dir = get_temp_dir()
                if isinstance(temp_dir, str) and temp_dir.startswith("/"):
                    return temp_dir.rstrip("/") or "/"
            except Exception as exc:
                logger.debug("Could not resolve environment temp dir: %s", exc)
        return tempfile.gettempdir()

    @staticmethod
    def _remote_group_members(session: ProcessSession) -> list[int]:
        """Exact non-zombie PGID membership; unsupported/unreadable ps fails closed.

        A successful kill command only acknowledges signal delivery. Neither it nor
        kill -0 (which includes zombies) proves that the owned group is empty.
        """
        if not isinstance(session.pid, int) or session.pid <= 1:
            raise RuntimeError("Remote cleanup unconfirmed: invalid process group")
        try:
            result = session.env_ref.execute("LC_ALL=C ps -eo pid=,pgid=,stat=", timeout=5)
        except Exception as exc:
            raise RuntimeError("Remote cleanup unconfirmed: process-group probe unavailable") from exc
        if not isinstance(result, dict) or result.get("returncode", result.get("exit_code")) != 0:
            raise RuntimeError("Remote cleanup unconfirmed: process-group probe failed")
        rows = result.get("output", "").splitlines()
        if not rows:
            raise RuntimeError("Remote cleanup unconfirmed: empty process table")
        members = []
        for row in rows:
            fields = row.split()
            if (len(fields) != 3 or not fields[0].isdigit() or not fields[1].isdigit()
                    or not fields[2][0].isalpha()):
                raise RuntimeError("Remote cleanup unconfirmed: unsupported process table")
            pid, pgid = map(int, fields[:2])
            if pgid == session.pid and not fields[2].startswith("Z"):
                members.append(pid)
        return members

    def _terminate_remote_group(self, session: ProcessSession) -> None:
        """Bounded TERM/KILL escalation, requiring exact group-death confirmation."""
        # Probe before signalling, including validation of the stored group ID.
        members = self._remote_group_members(session)
        for sig, grace in (("TERM", 0.5), ("KILL", self._KILL_SETTLE_SECONDS)):
            if not members:
                return
            # A nonzero kill can race with exit; only the subsequent probe is proof.
            session.env_ref.execute(f"kill -{sig} -- -{session.pid} 2>/dev/null", timeout=5)
            deadline = time.monotonic() + grace
            while True:
                members = self._remote_group_members(session)
                if not members:
                    return
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
        raise RuntimeError("Remote cleanup unconfirmed: live process-group members remain")
