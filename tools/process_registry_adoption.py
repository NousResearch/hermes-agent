"""Registry ownership of already-running foreground process streams."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tools.process_registry import ProcessSession


class ProcessAdoptionMixin:
    def adopt_local(
        self, proc: subprocess.Popen, *, command: str, cwd: Optional[str], task_id: str = "",
        session_key: str = "", owner_task_id: str = "", output_so_far: str = "",
        notify_on_complete: bool = True) -> ProcessSession:
        """Continue draining the original foreground pipe, seeded with its already-captured output."""
        session = self._new_session(command, task_id, owner_task_id, session_key, cwd)
        session.process = proc
        session.pid = proc.pid
        session.host_start_time = self._safe_host_start_time(proc.pid)
        session.notify_on_complete = notify_on_complete
        handed_pgid = getattr(proc, "_hermes_owned_pgid", None)
        self._take_handoff_group(session, proc)
        if output_so_far:
            session.append_output(output_so_far)
        try:
            self._track_started(session, self._reader_loop, f"proc-reader-{session.id}")
        except Exception:
            # The caller resumes its foreground wait: give it back its group, still covered.
            self._return_handoff_group(session, proc, handed_pgid)
            raise
        return session

    def adopt_sandbox(
        self, record, proc: subprocess.Popen, *, command: str, cwd: Optional[str],
        task_id: str = "", owner_task_id: str = "", session_key: str = "",
        output_so_far: str = "",
    ) -> ProcessSession:
        """Take over the original exec stream while retaining verified sandbox group ownership."""
        session = self._new_session(command, task_id, owner_task_id, session_key, cwd,
                                    env_ref=record.env, pid_scope="sandbox")
        session.pid = record.pid()
        session.process = proc
        session.host_start_time = self._safe_host_start_time(proc.pid)
        session.notify_on_complete = True
        session.append_output(output_so_far)
        record.term_grace = self._daemon_term_grace_seconds()

        def read(session):
            try:
                self._reader_loop(session)
            finally:
                record.env._foreground_commands.pop(id(proc), None)

        record.adopted = True
        try:
            self._track_started(session, read, f"proc-reader-{session.id}")
        except BaseException:
            record.adopted = False
            raise
        return session
