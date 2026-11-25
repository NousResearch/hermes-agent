"""Which background processes run inside a terminal environment: observing them and their teardown.

A live process belongs to the environment object it was spawned through (``env_ref``).
A handoff changes its ``task_id`` to the new owner's cache key without moving the process.
Only processes without an ``env_ref`` (local and host-recovered ones) fall back to their
``task_id``. ``owner_task_id`` is the raw spawning id, not an environment identity.

The poller lives here too: it observes those same processes through that same environment
object, one ``execute()`` per tick.
"""

import shlex
import time
from typing import Any

from tools.environments.command_context import command_context, exact_command
from tools.process_registry_env_log import fallback_exit_code, log_delta_command, parse_log_delta_reply


def _in_env(session, key: str, env: Any) -> bool:
    if session.env_ref is not None:
        return session.env_ref is env
    return session.task_id == key


class ProcessEnvScopeMixin:
    """Environment-scoped queries for ``ProcessRegistry`` (which supplies ``_lock``, ``_running``,
    ``_any_running`` and ``_finish_exited``)."""

    def has_running_in_env(self, key: str, env: Any) -> bool:
        """Whether a still-running process lives in the environment cached under ``key`` as
        ``env``. Tearing that environment down would kill or orphan it."""
        return self._any_running(lambda s: _in_env(s, key, env))

    def settle_env_closed(self, env: Any) -> int:
        """Settle every running process polled through ``env`` as lost, once ``env`` is closed.

        A sandbox process is observed only through its spawning environment object; once that
        object is closed the poller can no longer read its output or exit code, and some backends
        report the failure as output rather than raising, so the poller would retry forever.
        Marking the session exited stops the poller at its next check and publishes the result.
        A process that exits between the snapshot and its settlement keeps its own result. A
        backend whose processes outlive cleanup() (``cleanup_stops_processes = False``) is left
        alone: its poller still reaches them and collects the real exit.
        Returns the number of sessions settled as lost."""
        # ``None`` would match every local session (no env_ref). Duck-typed flag: plugin backends
        # and test doubles need not subclass BaseEnvironment.
        if env is None or not getattr(env, "cleanup_stops_processes", True):
            return 0
        with self._lock:
            sessions = [s for s in self._running.values() if s.env_ref is env and not s.exited]
        return sum(self._finish_exited(s, -1, reason="lost", source="env_closed") for s in sessions)

    def _env_poller_loop(self, session: Any, env: Any, log_path: str, pid_path: str, exit_path: str):
        """Background thread: poll a sandbox log file for non-local backends."""
        # Set here, not inherited: ``_track_started`` starts this thread under a copy of the
        # CALLER's context, which may still carry the spawn's stamp and approval; these reads
        # are Hermes's own polls, not the gated command.
        with command_context("background_poll", job_id=session.id, task_id=session.task_id,
                             session_key=session.session_key):
            self._poll_env_log(session, env, log_path, pid_path, exit_path)

    def _poll_env_log(self, session: Any, env: Any, log_path: str, pid_path: str, exit_path: str):
        """One ``execute()`` per tick until the process exits: the read command also reports the
        process status (see ``log_delta_command``). Sandboxes rate-limit exec starts."""
        q = shlex.quote
        # Byte offset already read from the log (bytes, not chars: the shell counts bytes).
        prev_output_bytes = 0
        while not session.exited:
            time.sleep(2)
            try:
                # Read only the bytes written since the last poll, plus the process status.
                poll = log_delta_command(q(log_path), prev_output_bytes, q(pid_path), q(exit_path))
                with exact_command(poll):  # absolute paths only: "/" survives the job deleting its cwd
                    raw = env.execute(poll, cwd="/", timeout=10).get("output", "")
                reply = parse_log_delta_reply(raw)
                delta = reply.delta if reply else ""
                if reply:
                    if reply.offset < prev_output_bytes:
                        # Log rotated/truncated: what we hold no longer lines up. Restart.
                        with session._lock:
                            if session.exited:
                                return
                            session.output_buffer = ""
                    prev_output_bytes = reply.size
                if delta:
                    # A kill that landed after this fetch already published its output
                    # snapshot: leave the output as the kill reported it (as the PTY reader does).
                    with session._lock:
                        if session.exited:
                            return
                        session.output_buffer += delta
                        if len(session.output_buffer) > session.max_output_chars:
                            session.output_buffer = session.output_buffer[-session.max_output_chars:]
                    self._check_watch_patterns(session, delta)
                    self._emit_output(session, delta)

                # Unreadable reply: separate liveness and exit-code reads, as before the combined read.
                exit_code = reply.exit_code if reply else fallback_exit_code(env, q(pid_path), q(exit_path))
                if exit_code is not None:
                    self._finish_exited(session, exit_code)
                    return
            except Exception:  # health: allow BLE001 -- any execute() error means the sandbox is gone
                # Environment might be gone (sandbox reaped, etc.). A kill that already
                # settled the session keeps its result.
                self._finish_exited(session, -1, reason="lost", source="backend_lost")
                return
