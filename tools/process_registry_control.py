"""Query/control/checkpoint methods for ProcessRegistry.

Behavior-neutral extraction from tools.process_registry.
"""

import json
import logging
import os
import signal
import time
from typing import Any, Dict, List, Optional

from agent.redact import redact_sensitive_text
from tools.process_registry import (
    FINISHED_TTL_SECONDS,
    MAX_PROCESSES,
    ProcessSession,
    _IS_WINDOWS,
)

logger = logging.getLogger(__name__)


def _owner():
    from tools import process_registry as owner

    return owner


def _checkpoint_path():
    return _owner().CHECKPOINT_PATH


def _stop_systemd_unit(unit_name: str) -> bool:
    return _owner()._stop_systemd_unit(unit_name)


class ProcessRegistryControlMixin:
    # Minimum characters of the random suffix required for prefix resolution.
    # Short prefixes ("p", "pr", "proc_1") are too collision-prone to act on.
    _MIN_PREFIX_CHARS = 4

    def get(self, session_id: str) -> Optional[ProcessSession]:
        """Get a session by ID (running or finished).

        Accepts either the full ID or a unique ID prefix (inspired by Factory
        Droid's task-ID prefixes, and the same UX as ``git``/``docker`` short
        hashes): ``proc_4dae`` — or just the bare suffix ``4dae`` — resolves
        to ``proc_4dae56ca81f6`` when exactly one session matches. Ambiguous
        or too-short prefixes resolve to None (callers already report
        "No process with ID ..."), never to an arbitrary pick.
        """
        with self._lock:
            session = self._running.get(session_id) or self._finished.get(session_id)
        if session is None:
            session = self._resolve_prefix(session_id)
        return self._refresh_detached_session(session)

    def _resolve_prefix(self, session_id: str) -> Optional[ProcessSession]:
        """Resolve a unique session-ID prefix to its session, else None.

        Exact lookups happen in :meth:`get` before this runs, so a full ID
        never pays the scan. Matching is prefix-only (no substring) and
        requires a unique hit; a bare suffix without the ``proc_`` lead is
        normalized so users can paste just the hex tail.
        """
        if not session_id or not isinstance(session_id, str):
            return None
        query = session_id.strip()
        if not query:
            return None
        # Allow the bare suffix form: "4dae56" -> "proc_4dae56".
        if not query.startswith("proc_"):
            query = f"proc_{query}"
        suffix = query[len("proc_"):]
        if len(suffix) < self._MIN_PREFIX_CHARS:
            return None
        with self._lock:
            matches = [
                s
                for store in (self._running, self._finished)
                for sid, s in store.items()
                if sid.startswith(query)
            ]
        if len(matches) == 1:
            return matches[0]
        return None

    def _reconcile_local_exit(self, session: "ProcessSession") -> None:
        """Reconcile session.exited against the real child process state.

        The reader thread (`_reader_loop`) sets `session.exited = True` only
        in its `finally` block, which runs when `stdout.read()` returns EOF.
        If the direct `Popen` child has exited but a descendant process (e.g.
        a daemon spawned by `hermes update` restarting the gateway) is still
        holding the stdout pipe open, the reader blocks forever and poll()
        keeps returning "running" indefinitely (issue #17327 — 74 polls over
        7 minutes on Feishu).

        This helper closes that window: when `session.exited` is still False
        but the direct child's `Popen.poll()` reports an exit code, drain any
        readable bytes non-blocking and flip `session.exited`. The orphaned
        reader thread remains stuck on its blocking `read()` but is a daemon
        thread and will be reaped with the process.

        Safe no-op on sessions without a local `Popen` (env/PTY), already-
        exited sessions, and detached-recovered sessions.
        """
        if session is None or session.exited:
            return
        proc = getattr(session, "process", None)
        if proc is None:
            return
        try:
            rc = proc.poll()
        except Exception:
            return
        if rc is None:
            return  # Direct child still running — reader block is legitimate.

        # Direct child exited. Try to drain any bytes the reader hasn't
        # consumed yet. This is best-effort: if the pipe is held open by a
        # descendant, the non-blocking read returns what's immediately
        # available and we stop.
        drained = ""
        stdout = getattr(proc, "stdout", None)
        if stdout is not None and not _IS_WINDOWS:
            try:
                import fcntl
                fd = stdout.fileno()
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
                try:
                    chunk = stdout.read()
                    if chunk:
                        drained = chunk if isinstance(chunk, str) else chunk.decode("utf-8", errors="replace")
                except (BlockingIOError, OSError, ValueError):
                    pass
                finally:
                    try:
                        fcntl.fcntl(fd, fcntl.F_SETFL, flags)
                    except Exception:
                        pass
            except Exception as e:
                logger.debug("Non-blocking drain failed for %s: %s", session.id, e)

        with session._lock:
            if drained:
                session.output_buffer += drained
                if len(session.output_buffer) > session.max_output_chars:
                    session.output_buffer = session.output_buffer[-session.max_output_chars:]
            session.exited = True
            if session.completion_reason != "killed":
                session.exit_code = rc
                session.completion_reason = "exited"
        logger.info(
            "Reconciled session %s: direct child exited with code %s but reader "
            "was still blocked (orphaned pipe). Flipped to exited.",
            session.id, rc,
        )
        self._move_to_finished(session)

    def poll(self, session_id: str) -> dict:
        """Check status and get new output for a background process."""
        from tools.ansi_strip import strip_ansi

        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}

        # Reconcile against real child state before reading session.exited.
        # Guards against orphaned-pipe reader hangs (issue #17327).
        self._reconcile_local_exit(session)

        with session._lock:
            output_preview = strip_ansi(session.output_buffer[-1000:]) if session.output_buffer else ""

        result = {
            "session_id": session.id,
            "command": session.command,
            "status": "exited" if session.exited else "running",
            "pid": session.pid,
            "uptime_seconds": int(time.time() - session.started_at),
            "output_preview": output_preview,
        }
        if session.exited:
            result["exit_code"] = session.exit_code
            result["completion_reason"] = session.completion_reason
            result["termination_source"] = session.termination_source
            # NOTE: poll() is a read-only status query and deliberately does
            # NOT mark the session _completion_consumed. wait()/read_log()
            # represent actual output consumption and do mark it. Marking
            # consumed here would let a status check silently suppress the
            # notify_on_complete watcher's autonomous delivery turn (#10156).
            #
            # We DO record it in _poll_observed so the CLI's inline drain still
            # dedups (the agent already saw the exit in this turn's poll result)
            # without affecting the gateway/tui watchers, which only consult
            # _completion_consumed.
            self._poll_observed.add(session_id)
        if session.detached:
            result["detached"] = True
            result["note"] = "Process recovered after restart -- output history unavailable"
        return result

    def read_log(self, session_id: str, offset: int | None = None, limit: int = 200) -> dict:
        """Read the full output log with optional pagination by lines."""
        from tools.ansi_strip import strip_ansi

        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}

        with session._lock:
            full_output = strip_ansi(session.output_buffer)

        lines = full_output.splitlines()
        total_lines = len(lines)

        # Default (offset=None): last N lines. An explicit offset=0 means
        # "start from the first line" — previously it was conflated with
        # the default and silently returned the TAIL instead of the head
        # (same falsy-coercion class as the wait() timeout guard; salvaged
        # from PR #60004, credit @isheng-eqi).
        if offset is None and limit > 0:
            selected = lines[-limit:]
            observed_completion_output = bool(selected) or total_lines == 0
        else:
            offset = offset or 0
            selected = lines[offset:offset + limit]
            stop = slice(offset, offset + limit).indices(total_lines)[1]
            observed_completion_output = (
                total_lines == 0 or (bool(selected) and stop == total_lines)
            )

        result = {
            "session_id": session.id,
            "command": session.command,
            "status": "exited" if session.exited else "running",
            "output": "\n".join(selected),
            "total_lines": total_lines,
            "showing": f"{len(selected)} lines",
        }
        if session.exited and observed_completion_output:
            self._completion_consumed.add(session_id)
        return result

    def wait(self, session_id: str, timeout: int = None) -> dict:
        """
        Block until a process exits, timeout, or interrupt.

        Args:
            session_id: The process to wait for.
            timeout: Max seconds to block. Falls back to TERMINAL_TIMEOUT config.

        Returns:
            dict with status ("exited", "timeout", "interrupted", "not_found")
            and output snapshot.
        """
        from tools.ansi_strip import strip_ansi
        from tools.interrupt import is_interrupted as _is_interrupted

        try:
            default_timeout = int(os.getenv("TERMINAL_TIMEOUT", "180"))
        except (ValueError, TypeError):
            default_timeout = 180
        max_timeout = default_timeout
        requested_timeout = timeout
        timeout_note = None

        # Reject non-positive timeouts — the schema declares minimum=1, but
        # not every caller enforces schemas before dispatch. timeout=0 is
        # falsy, so without this guard it silently fell through
        # (`0 or max_timeout`) to the DEFAULT wait instead of erroring.
        # Salvaged from PR #60004 (credit @isheng-eqi).
        if requested_timeout is not None and requested_timeout <= 0:
            return {
                "status": "error",
                "error": f"timeout must be positive (got {requested_timeout})",
            }

        if requested_timeout and requested_timeout > max_timeout:
            effective_timeout = max_timeout
            timeout_note = (
                f"Requested wait of {requested_timeout}s was clamped "
                f"to configured limit of {max_timeout}s"
            )
        else:
            effective_timeout = requested_timeout or max_timeout

        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}

        deadline = time.monotonic() + effective_timeout

        while time.monotonic() < deadline:
            session = self._refresh_detached_session(session)
            if session is None:
                return {"status": "not_found", "error": f"No process with ID {session_id}"}
            # Reconcile against real child state — guards against orphaned-
            # pipe reader hangs where the reader is blocked but the direct
            # child has already exited (issue #17327).
            self._reconcile_local_exit(session)
            if session.exited:
                self._completion_consumed.add(session_id)
                result = {
                    "status": "exited",
                    "command": session.command,
                    "exit_code": session.exit_code,
                    "completion_reason": session.completion_reason,
                    "termination_source": session.termination_source,
                    "output": strip_ansi(session.output_buffer[-2000:]),
                }
                if timeout_note:
                    result["timeout_note"] = timeout_note
                return result

            if _is_interrupted():
                result = {
                    "status": "interrupted",
                    "command": session.command,
                    "output": strip_ansi(session.output_buffer[-1000:]),
                    "note": "User sent a new message -- wait interrupted",
                }
                if timeout_note:
                    result["timeout_note"] = timeout_note
                return result

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            session._completion_event.wait(timeout=min(1.0, remaining))

        result = {
            "status": "timeout",
            "command": session.command,
            "output": strip_ansi(session.output_buffer[-1000:]),
            # A wait window elapsing is NOT a failure — 511 exact-duplicate
            # process calls in a production window show models re-issuing
            # identical waits after misreading this result as an error.
            "process_running": True,
        }
        uptime = time.time() - session.started_at if session.started_at else None
        base_note = (
            f"Wait window of {effective_timeout}s elapsed — the process is "
            "still running. This is not an error."
        )
        if uptime is not None:
            base_note += f" Uptime: {int(uptime)}s."
        if session.notify_on_complete:
            base_note += (
                " notify_on_complete is set: you will be notified on exit — "
                "do more work instead of waiting again."
            )
        else:
            base_note += (
                " Poll again later or use terminal(background=true, "
                "notify_on_complete=true) next time for automatic notification."
            )
        if timeout_note:
            result["timeout_note"] = f"{timeout_note}. {base_note}"
        else:
            result["timeout_note"] = base_note
        return result

    def kill_process(
        self,
        session_id: str,
        *,
        source: str = "process.kill",
        consume_output: bool = True,
    ) -> dict:
        """Kill a background process and return its output snapshot.

        ``consume_output`` is true for explicit tool/RPC kills because their
        caller observes the returned output. Bulk cleanup passes false: it
        discards each result and therefore must not suppress an autonomous
        output-bearing completion notification. Exception: abandoned-turn
        reaping (``kill_started_since``) is bulk cleanup that deliberately
        passes true — a killed abandoned process must not enqueue a synthetic
        follow-up that revives work the timeout/interrupt stopped.
        """
        from tools.ansi_strip import strip_ansi

        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}

        if session.exited:
            # Even if the main process already exited, a double-forked
            # descendant may still be alive in the systemd scope (#70716,
            # reviewer gap #2 — the ``already_exited`` early return skipped
            # unit cleanup).  Stop the scope to reap any survivors.
            if session.systemd_unit:
                _stop_systemd_unit(session.systemd_unit)
            with session._lock:
                result = {
                    "status": "already_exited",
                    "command": session.command,
                    "exit_code": session.exit_code,
                    "completion_reason": session.completion_reason,
                    "termination_source": session.termination_source,
                    "output": strip_ansi(session.output_buffer[-2000:]),
                }
            # Only suppress the autonomous turn after its output is present in
            # the explicit kill result, matching wait/log consumption.
            if consume_output:
                self._completion_consumed.add(session_id)
            return result

        # Kill via PTY, Popen (local), or env execute (non-local)
        try:
            if session._pty:
                # PTY process -- terminate via ptyprocess
                try:
                    session._pty.terminate(force=True)
                except Exception:
                    if session.pid:
                        os.kill(session.pid, signal.SIGTERM)
            elif session.process:
                # Local process -- kill the process tree. On Windows this
                # must be taskkill /T /F; Popen.terminate() only kills the
                # shell wrapper and leaves Git Bash descendants behind.
                self._terminate_host_pid(session.process.pid, session.host_start_time)
            elif session.env_ref and session.pid:
                # Non-local -- kill inside sandbox
                session.env_ref.execute(f"kill {session.pid} 2>/dev/null", timeout=5)
            elif session.detached and session.pid_scope == "host" and session.pid:
                # Identity check, not bare liveness: if the PID is gone OR was
                # recycled onto an unrelated process, treat our process as
                # exited and never tree-kill the stranger.  If this recovered
                # session also carries an owned systemd scope, stop that scope
                # before returning: a daemonized descendant may still be alive
                # there even though the wrapper PID exited or was recycled
                # across the gateway restart (#70716, teknium1 review).
                if not self._host_pid_is_ours(session.pid, session.host_start_time):
                    if session.systemd_unit:
                        _stop_systemd_unit(session.systemd_unit)
                    with session._lock:
                        session.exited = True
                        session.exit_code = None
                        output = strip_ansi(session.output_buffer[-2000:])
                    if consume_output:
                        self._completion_consumed.add(session_id)
                    self._move_to_finished(session)
                    return {
                        "status": "already_exited",
                        "exit_code": session.exit_code,
                        "output": output,
                    }
                self._terminate_host_pid(session.pid, session.host_start_time)
            else:
                return {
                    "status": "error",
                    "error": (
                        "Recovered process cannot be killed after restart because "
                        "its original runtime handle is no longer available"
                    ),
                }

            # If the worker was spawned in its own systemd scope (#70716),
            # stop the entire unit to reap any double-forked descendants that
            # were reparented inside the scope and survived the PID signal
            # above (reviewer gap #2).  ``systemctl --user stop`` sends
            # SIGTERM to every process in the cgroup and escalates to SIGKILL
            # after TimeoutStopSec.  This is additive — the PID-based kill
            # above already handled the main process; this catches stragglers.
            if session.systemd_unit:
                _stop_systemd_unit(session.systemd_unit)
            # Capture output before marking consumed, then mark consumed before
            # exposing ``exited`` to watcher tasks. This closes the delayed
            # notification race without discarding the terminal transcript.
            with session._lock:
                output = strip_ansi(session.output_buffer[-2000:])
                if consume_output:
                    self._completion_consumed.add(session_id)
                session.exited = True
                session.exit_code = -15  # SIGTERM
                session.completion_reason = "killed"
                session.termination_source = source
            self._move_to_finished(session)
            self._write_checkpoint()
            return {
                "status": "killed",
                "session_id": session.id,
                "completion_reason": session.completion_reason,
                "termination_source": session.termination_source,
                "output": output,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def write_stdin(self, session_id: str, data: str) -> dict:
        """Send raw data to a running process's stdin (no newline appended)."""
        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}
        if session.exited:
            return {"status": "already_exited", "error": "Process has already finished"}

        # PTY mode -- write through pty handle.
        if hasattr(session, '_pty') and session._pty:
            try:
                # pywinpty expects str on Windows; ptyprocess expects bytes on POSIX.
                if _IS_WINDOWS:
                    pty_data = data.decode("utf-8") if isinstance(data, bytes) else str(data)
                else:
                    # surrogateescape: a PTY is a byte stream — round-trip the
                    # original bytes instead of crashing on surrogate content.
                    pty_data = data.encode("utf-8", "surrogateescape") if isinstance(data, str) else data
                session._pty.write(pty_data)
                return {"status": "ok", "bytes_written": len(data)}
            except Exception as e:
                return {"status": "error", "error": str(e)}

        # Popen mode -- write through stdin pipe
        if not session.process or not session.process.stdin:
            return {"status": "error", "error": "Process stdin not available (non-local backend or stdin closed)"}
        try:
            session.process.stdin.write(data)
            session.process.stdin.flush()
            return {"status": "ok", "bytes_written": len(data)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def submit_stdin(self, session_id: str, data: str = "") -> dict:
        """Send data + newline to a running process's stdin (like pressing Enter).

        On a Windows PTY session the Enter key is a carriage return: ConPTY
        cooked input treats ``\\r`` as end-of-line, and a bare ``\\n`` written
        through pywinpty is NOT delivered as a line terminator — the child's
        blocking line read (Python ``readline()``, Go ``bufio.Scanner`` as in
        ``gh auth login``'s "Press Enter to open the browser" prompt) simply
        never returns and the process hangs while looking healthy. Verified
        empirically via pywinpty 2.0.15: ``\\n`` -> no line, ``\\r`` /
        ``\\r\\n`` -> line delivered. Use ``\\r\\n`` so the child sees both the
        Enter keypress and a conventional newline; POSIX PTYs and Popen pipes
        keep the plain ``\\n``.
        """
        session = self.get(session_id)
        is_windows_pty = bool(
            _IS_WINDOWS and session is not None
            and getattr(session, "_pty", None)
        )
        line_ending = "\r\n" if is_windows_pty else "\n"
        return self.write_stdin(session_id, data + line_ending)

    def request_close_terminal(self, session_id: str) -> dict:
        """Ask the desktop GUI to close the read-only terminal tab mirroring this
        background process.

        This does NOT kill the process — it only drops the view. Output keeps
        streaming into the (capped) buffer and the user can reopen the tab from
        the status stack. Desktop-only: returns an error if no UI close sink is
        wired (e.g. CLI / messaging)."""
        sink = self.on_close
        if sink is None:
            return {
                "status": "error",
                "error": "close_terminal is only available in the Hermes desktop app.",
            }
        # The session may already be finished (or pruned) — the tab can still
        # linger and be closed, so a missing session is not an error here.
        session = self.get(session_id)
        try:
            sink(session, session_id)
        except Exception as e:
            return {"status": "error", "error": str(e)}
        return {
            "status": "ok",
            "closed": session_id,
            "note": (
                "Closed the read-only terminal tab. The process was not killed; "
                "its output remains available and the user can reopen the tab "
                "from the status stack."
            ),
        }

    def close_stdin(self, session_id: str) -> dict:
        """Close a running process's stdin / send EOF without killing the process."""
        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}
        if session.exited:
            return {"status": "already_exited", "error": "Process has already finished"}

        if hasattr(session, '_pty') and session._pty:
            try:
                session._pty.sendeof()
                return {"status": "ok", "message": "EOF sent"}
            except Exception as e:
                return {"status": "error", "error": str(e)}

        if not session.process or not session.process.stdin:
            return {"status": "error", "error": "Process stdin not available (non-local backend or stdin closed)"}
        try:
            session.process.stdin.close()
            return {"status": "ok", "message": "stdin closed"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def count_running(self) -> int:
        """Return the count of currently-running background processes.

        Cheap O(1) read of the running dict, suitable for status-bar polling
        on every render tick. CPython dict ``len()`` is atomic; callers do not
        need to hold ``self._lock``. Reflects ``_running`` only: sessions are
        moved to ``_finished`` when their subprocess exits.
        """
        try:
            return len(self._running)
        except Exception:
            return 0

    def list_sessions(self, task_id: str = None, session_key: str = None) -> list:
        """List all running and recently-finished processes.

        When ``task_id`` is given, processes for that task are included. When
        ``session_key`` is also given, session-scoped background processes
        (``background: true``) registered under that gateway session are
        surfaced too, even if they belong to a different task — so the agent
        can discover a forgotten preview server that is blocking session
        reset (#29177). Such cross-task entries are flagged with
        ``"session_scoped": true``.
        """
        with self._lock:
            all_sessions = list(self._running.values()) + list(self._finished.values())

        all_sessions = [self._refresh_detached_session(s) for s in all_sessions]

        if task_id or session_key:
            all_sessions = [
                s for s in all_sessions
                if (task_id and s.task_id == task_id)
                or (session_key and s.session_key == session_key)
            ]

        result = []
        for s in all_sessions:
            entry = {
                "session_id": s.id,
                "command": s.command[:200],
                "cwd": s.cwd,
                "pid": s.pid,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(s.started_at)),
                "uptime_seconds": int(time.time() - s.started_at),
                "status": "exited" if s.exited else "running",
                "output_preview": s.output_buffer[-200:] if s.output_buffer else "",
            }
            # Flag processes surfaced only because they share the gateway
            # session (not the current task) — these are the long-lived
            # background processes a user may have forgotten about (#29177).
            if task_id and session_key and s.task_id != task_id and s.session_key == session_key:
                entry["session_scoped"] = True
            # Trigger metadata so a goal-loop judge can decide to wait on this
            # process's OWN signal (a watch-pattern match or completion), not
            # just its exit. A watcher with watch_patterns may never exit.
            if s.watch_patterns and not s._watch_disabled:
                entry["watch_patterns"] = list(s.watch_patterns)
                entry["watch_hit"] = s._watch_hits > 0
            if s.notify_on_complete:
                entry["notify_on_complete"] = True
            if s.exited:
                entry["exit_code"] = s.exit_code
            if s.detached:
                entry["detached"] = True
            result.append(entry)
        return result

    # ----- Session/Task Queries (for gateway integration) -----

    def has_active_processes(self, task_id: str) -> bool:
        """Check if there are active (running) processes for a task_id."""
        with self._lock:
            sessions = list(self._running.values())

        for session in sessions:
            self._refresh_detached_session(session)

        with self._lock:
            return any(
                s.task_id == task_id and not s.exited
                for s in self._running.values()
            )

    def has_active_for_session(
        self, session_key: str, max_active_age: Optional[float] = None,
    ) -> bool:
        """Check if there are active processes for a gateway session key.

        When *max_active_age* is set (seconds), processes that started more
        than that many seconds ago are **ignored** — they are still running
        but are considered stale and must not block session idle / daily
        reset.  This prevents a forgotten ``http.server`` (or any long-lived
        preview process) from permanently freezing the session lifecycle.

        Args:
            session_key: Gateway session key to check.
            max_active_age: If set, ignore processes older than this many
                seconds.  ``None`` retains the legacy behaviour (any running
                process blocks).
        """
        with self._lock:
            sessions = list(self._running.values())

        for session in sessions:
            self._refresh_detached_session(session)

        now = time.time()
        with self._lock:
            return any(
                s.session_key == session_key
                and not s.exited
                and (max_active_age is None or (now - s.started_at) < max_active_age)
                for s in self._running.values()
            )

    def has_any_active(self) -> bool:
        """Whether ANY background process is still running (across all sessions).

        Used by scale-to-zero idle detection (gateway/scale_to_zero): a gateway
        with a live background process (terminal background=true) is NOT idle and
        must not be suspended, or the process is lost. Refreshes detached
        sessions first so a finished-but-unreaped process reads as inactive.
        """
        with self._lock:
            sessions = list(self._running.values())

        for session in sessions:
            self._refresh_detached_session(session)

        with self._lock:
            return any(not s.exited for s in self._running.values())

    def snapshot_running_ids(self, task_id: str) -> frozenset[str]:
        """Capture running process IDs owned by ``task_id``.

        Gateway turns use this as a boundary marker: if a turn times out, only
        processes absent from its starting snapshot belong to the abandoned
        turn. Older session processes must survive because background tasks
        intentionally span successful turns.
        """
        with self._lock:
            return frozenset(
                s.id
                for s in self._running.values()
                if s.task_id == task_id and not s.exited
            )

    def kill_started_since(
        self,
        task_id: str,
        baseline_ids,
        *,
        source: str,
    ) -> int:
        """Kill processes created for ``task_id`` after a prior snapshot.

        ``consume_output`` is forced on: abandoned-turn output must not
        enqueue a synthetic follow-up that revives work the timeout
        deliberately stopped.
        """
        return self.kill_all(
            task_id,
            exclude_ids=frozenset(baseline_ids or ()),
            source=source,
            consume_output=True,
        )

    def kill_all(
        self,
        task_id: Optional[str] = None,
        *,
        exclude_ids: frozenset = frozenset(),
        source: str = "kill_all",
        consume_output: bool = False,
    ) -> int:
        """Kill all running processes, optionally filtered by task_id. Returns count killed."""
        with self._lock:
            targets = [
                s for s in self._running.values()
                if (task_id is None or s.task_id == task_id)
                and s.id not in exclude_ids
                and not s.exited
            ]

        killed = 0
        for session in targets:
            result = self.kill_process(
                session.id,
                source=source,
                consume_output=consume_output,
            )
            if result.get("status") in {"killed", "already_exited"}:
                killed += 1
        return killed

    # ----- Cleanup / Pruning -----

    def _prune_if_needed(self):
        """Remove oldest finished sessions if over MAX_PROCESSES. Must hold _lock."""
        # First prune expired finished sessions
        now = time.time()
        expired = [
            sid for sid, s in self._finished.items()
            if (now - s.started_at) > FINISHED_TTL_SECONDS
        ]
        for sid in expired:
            del self._finished[sid]
            self._completion_consumed.discard(sid)
            self._poll_observed.discard(sid)

        # If still over limit, remove oldest finished
        total = len(self._running) + len(self._finished)
        if total >= MAX_PROCESSES and self._finished:
            oldest_id = min(self._finished, key=lambda sid: self._finished[sid].started_at)
            del self._finished[oldest_id]
            self._completion_consumed.discard(oldest_id)
            self._poll_observed.discard(oldest_id)

        # Drop any _completion_consumed / _poll_observed entries whose sessions
        # are no longer tracked at all — belt-and-suspenders against
        # module-lifetime growth on registry lookup paths that don't reach the
        # dict prunes.
        tracked = self._running.keys() | self._finished.keys()
        stale = self._completion_consumed - tracked
        if stale:
            self._completion_consumed -= stale
        stale_polls = self._poll_observed - tracked
        if stale_polls:
            self._poll_observed -= stale_polls

    # ----- Checkpoint (crash recovery) -----

    def _write_checkpoint(
        self,
        extra_entries: Optional[List[Dict[str, Any]]] = None,
    ):
        """Write running process metadata to checkpoint file atomically."""
        try:
            with self._lock:
                entries = []
                for s in self._running.values():
                    if not s.exited:
                        # Lazily backfill the kernel start time for host PIDs so
                        # recovery after restart can detect PID recycling even
                        # for sessions spawned before this field existed.
                        if s.host_start_time is None and s.pid_scope == "host" and s.pid:
                            s.host_start_time = self._safe_host_start_time(s.pid)
                        entries.append({
                            "session_id": s.id,
                            # Redact inline credentials before persisting to
                            # disk — the checkpoint file lives under
                            # ~/.hermes/processes.json with the raw command
                            # (issue #77484). Recovery only uses command for
                            # display/logging (the process is already running;
                            # adoption re-validates the PID, never re-runs the
                            # command), so masking is lossless.
                            "command": redact_sensitive_text(s.command, code_file=True),
                            "pid": s.pid,
                            "pid_scope": s.pid_scope,
                            "host_start_time": s.host_start_time,
                            "systemd_unit": s.systemd_unit,
                            "cwd": s.cwd,
                            "started_at": s.started_at,
                            "task_id": s.task_id,
                            "owner_task_id": s.owner_task_id or s.task_id,
                            "session_key": s.session_key,
                            "watcher_platform": s.watcher_platform,
                            "watcher_chat_id": s.watcher_chat_id,
                            "watcher_user_id": s.watcher_user_id,
                            "watcher_user_name": s.watcher_user_name,
                            "watcher_thread_id": s.watcher_thread_id,
                            "watcher_message_id": s.watcher_message_id,
                            "watcher_interval": s.watcher_interval,
                            "parent_session_id": s.parent_session_id,
                            "notify_on_complete": s.notify_on_complete,
                            "watch_patterns": s.watch_patterns,
                        })
                if extra_entries:
                    tracked_ids = {item.get("session_id") for item in entries}
                    entries.extend(
                        item
                        for item in extra_entries
                        if item.get("session_id") not in tracked_ids
                    )

            # Atomic write to avoid corruption on crash
            from utils import atomic_json_write
            atomic_json_write(_checkpoint_path(), entries)
        except Exception as e:
            logger.debug("Failed to write checkpoint file: %s", e, exc_info=True)

    def recover_from_checkpoint(self) -> int:
        """
        On gateway startup, probe PIDs from checkpoint file.

        Returns the number of processes recovered as detached.
        """
        if not _checkpoint_path().exists():
            return 0

        try:
            entries = json.loads(_checkpoint_path().read_text(encoding="utf-8"))
        except Exception:
            return 0

        recovered = 0
        unresolved_scope_entries: List[Dict[str, Any]] = []
        for entry in entries:
            pid = entry.get("pid")
            if not pid:
                continue

            pid_scope = entry.get("pid_scope", "host")
            if pid_scope != "host":
                # Sandbox-backed processes keep only in-sandbox PIDs in the
                # checkpoint, which are not meaningful to the restarted host
                # process once the original environment handle is gone.
                logger.info(
                    "Skipping recovery for non-host process: %s (pid=%s, scope=%s)",
                    entry.get("command", "unknown")[:60],
                    pid,
                    pid_scope,
                )
                continue

            # The PID must be alive AND still the same process we spawned. A
            # bare liveness check is unsafe: across a restart (especially a
            # reboot or long uptime) the kernel may have recycled this number
            # onto an unrelated process — adopting it would let a later kill or
            # watcher tree-kill a stranger (e.g. a browser). Re-validate the
            # kernel start time recorded in the checkpoint.
            recorded_start = entry.get("host_start_time")
            if not self._host_pid_is_ours(pid, recorded_start):
                if self._is_host_pid_alive(pid):
                    logger.info(
                        "Not recovering session %s: pid %d is alive but its "
                        "start time no longer matches — PID was recycled onto "
                        "an unrelated process; refusing to adopt it.",
                        entry.get("session_id", "?"), pid,
                    )
                systemd_unit = entry.get("systemd_unit", "")
                if systemd_unit and not _stop_systemd_unit(systemd_unit):
                    logger.warning(
                        "Could not reap persisted scope %s for dead wrapper pid %s; "
                        "retaining checkpoint entry for the next startup",
                        systemd_unit,
                        pid,
                    )
                    unresolved_scope_entries.append(entry)
                continue

            session = ProcessSession(
                id=entry["session_id"],
                command=entry.get("command", "unknown"),
                task_id=entry.get("task_id", ""),
                owner_task_id=entry.get("owner_task_id", "") or entry.get("task_id", ""),
                session_key=entry.get("session_key", ""),
                pid=pid,
                host_start_time=recorded_start,
                pid_scope=pid_scope,
                systemd_unit=entry.get("systemd_unit", ""),
                cwd=entry.get("cwd"),
                started_at=entry.get("started_at", time.time()),
                detached=True,  # Can't read output, but can report status + kill
                watcher_platform=entry.get("watcher_platform", ""),
                watcher_chat_id=entry.get("watcher_chat_id", ""),
                watcher_user_id=entry.get("watcher_user_id", ""),
                watcher_user_name=entry.get("watcher_user_name", ""),
                watcher_thread_id=entry.get("watcher_thread_id", ""),
                watcher_message_id=entry.get("watcher_message_id", ""),
                watcher_interval=entry.get("watcher_interval", 0),
                parent_session_id=entry.get("parent_session_id", ""),
                notify_on_complete=entry.get("notify_on_complete", False),
                watch_patterns=entry.get("watch_patterns", []),
            )
            with self._lock:
                self._running[session.id] = session
            recovered += 1
            logger.info("Recovered detached process: %s (pid=%d)", session.command[:60], pid)

            # Re-enqueue watcher so gateway can resume notifications
            if session.watcher_interval > 0:
                self.pending_watchers.append({
                    "session_id": session.id,
                    "check_interval": session.watcher_interval,
                    "session_key": session.session_key,
                    "platform": session.watcher_platform,
                    "chat_id": session.watcher_chat_id,
                    "user_id": session.watcher_user_id,
                    "user_name": session.watcher_user_name,
                    "thread_id": session.watcher_thread_id,
                    "message_id": session.watcher_message_id,
                    "notify_on_complete": session.notify_on_complete,
                    "parent_session_id": session.parent_session_id,
                })

        self._write_checkpoint(extra_entries=unresolved_scope_entries)

        return recovered
