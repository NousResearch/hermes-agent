"""Runtime/spawn/notification methods for ProcessRegistry.

Behavior-neutral extraction from tools.process_registry. The compatibility owner
retains ProcessSession, process_registry, tool schema, and public imports.
"""

import codecs
import json
import logging
import os
import shlex
import signal
import subprocess
import threading
import time
import uuid
from typing import Any, Optional

from hermes_cli._subprocess_compat import windows_hide_flags
from tools.environments.local import _resolve_safe_cwd, _sanitize_subprocess_env
from tools.process_registry import (
    ProcessSession,
    WATCH_GLOBAL_COOLDOWN_SECONDS,
    WATCH_GLOBAL_MAX_PER_WINDOW,
    WATCH_GLOBAL_WINDOW_SECONDS,
    WATCH_LIFETIME_MAX_HITS,
    WATCH_MIN_INTERVAL_SECONDS,
    WATCH_STRIKE_LIMIT,
    _IS_LINUX,
    _IS_WINDOWS,
    _build_systemd_scope_argv,
)

logger = logging.getLogger(__name__)


def _owner():
    from tools import process_registry as owner

    return owner


def _find_shell():
    return _owner()._find_shell()


def _is_supervised_gateway_process() -> bool:
    return _owner()._is_supervised_gateway_process()


def _systemd_run_user_scope_available() -> bool:
    return _owner()._systemd_run_user_scope_available()


def _stop_systemd_unit(unit_name: str) -> bool:
    return _owner()._stop_systemd_unit(unit_name)


def _redact_process_result(result: dict) -> dict:
    return _owner()._redact_process_result(result)


def format_process_notification(evt: dict):
    return _owner().format_process_notification(evt)


class ProcessRegistryRuntimeMixin:
    _SHELL_NOISE_SUBSTRINGS = (
        "bash: cannot set terminal process group",
        "bash: no job control in this shell",
        "no job control in this shell",
        "cannot set terminal process group",
        "tcsetattr: Inappropriate ioctl for device",
    )
    @staticmethod
    def _clean_shell_noise(text: str) -> str:
        """Strip shell startup warnings from the beginning of output."""
        lines = text.split("\n")
        while lines and any(noise in lines[0] for noise in ProcessRegistry._SHELL_NOISE_SUBSTRINGS):
            lines.pop(0)
        return "\n".join(lines)

    def _emit_output(self, session: ProcessSession, chunk: str) -> None:
        """Forward a freshly-read chunk to the live-output sink, if one is set.
        Called from reader threads; never raise into the read loop."""
        sink = self.on_output
        if sink is None or not chunk:
            return
        try:
            sink(session, chunk)
        except Exception:
            pass

    def _check_watch_patterns(self, session: ProcessSession, new_text: str) -> None:
        """Scan new output for watch patterns and queue notifications.

        Called from reader threads with new_text being the freshly-read chunk.

        Per-session rate limit: at most ONE watch-match notification per
        WATCH_MIN_INTERVAL_SECONDS. Any match arriving inside the cooldown
        window is dropped and counts as ONE strike for that window. After
        WATCH_STRIKE_LIMIT consecutive strike windows, watch_patterns is
        disabled for this session and the session is promoted to
        notify_on_complete semantics — one notification when the process
        actually exits, no more mid-process spam.

        Independently, WATCH_LIFETIME_MAX_HITS caps the total number of
        matches ever delivered for a session, so a pattern that keeps
        recurring at a cadence just above the cooldown (e.g. a service
        restarted repeatedly over a day) still gets disabled instead of
        forcing a full-context agent turn indefinitely.
        """
        if not session.watch_patterns or session._watch_disabled:
            return
        # Suppress-after-exit: once the reader loop has declared the process
        # exited, any late chunk we still see is post-exit noise. Dropping these
        # prevents the "stale notifications delivered minutes after the process
        # ended" spam when completion_queue consumers run async.
        if session.exited:
            return

        # Scan new text line-by-line for pattern matches
        matched_lines = []
        matched_pattern = None
        for line in new_text.splitlines():
            for pat in session.watch_patterns:
                if pat in line:
                    matched_lines.append(line.rstrip())
                    if matched_pattern is None:
                        matched_pattern = pat
                    break  # one match per line is enough

        if not matched_lines:
            return

        now = time.time()
        should_disable = False
        lifetime_exhausted = False
        with session._lock:
            # Case 1: still inside the cooldown from the last emission.
            # Count this as a strike for the current window (only once per window)
            # and drop the event. If we've hit the strike limit, disable watch
            # and promote to notify_on_complete.
            if session._watch_cooldown_until and now < session._watch_cooldown_until:
                session._watch_suppressed += len(matched_lines)
                if not session._watch_strike_candidate:
                    # First drop in this window — count one strike.
                    session._watch_strike_candidate = True
                    session._watch_consecutive_strikes += 1
                    if session._watch_consecutive_strikes >= WATCH_STRIKE_LIMIT:
                        session._watch_disabled = True
                        # Promote to notify_on_complete so the agent still gets
                        # exactly one notification when the process actually ends.
                        session.notify_on_complete = True
                        should_disable = True
                return_early = True
            else:
                # Case 2: cooldown has expired.
                # Decide whether this window was a "clean" one (no drops) or a
                # strike window. If no strike candidate was set during the prior
                # cooldown, reset the consecutive-strike counter — we're back to
                # healthy emission cadence.
                if (
                    session._watch_cooldown_until
                    and not session._watch_strike_candidate
                ):
                    session._watch_consecutive_strikes = 0
                session._watch_strike_candidate = False

                # Emit the notification and start a new cooldown window.
                session._watch_last_emit_at = now
                session._watch_cooldown_until = now + WATCH_MIN_INTERVAL_SECONDS
                session._watch_hits += 1
                suppressed = session._watch_suppressed
                session._watch_suppressed = 0
                return_early = False
                # Lifetime cap: this match is delivered (it already earned it),
                # but disable further ones regardless of how cleanly spaced
                # they are — see WATCH_LIFETIME_MAX_HITS above.
                lifetime_exhausted = session._watch_hits >= WATCH_LIFETIME_MAX_HITS
                if lifetime_exhausted:
                    session._watch_disabled = True
                    session.notify_on_complete = True

        if return_early:
            if should_disable:
                # Emit exactly one "watch disabled, falling back to notify_on_complete"
                # summary event so the agent/user sees why things went quiet.
                self.completion_queue.put({
                    "session_id": session.id,
                    "session_key": session.session_key,
                    "task_id": session.task_id,
                    "owner_task_id": session.owner_task_id or session.task_id,
                    "command": session.command,
                    "type": "watch_disabled",
                    "suppressed": session._watch_suppressed,
                    "platform": session.watcher_platform,
                    "chat_id": session.watcher_chat_id,
                    "user_id": session.watcher_user_id,
                    "user_name": session.watcher_user_name,
                    "thread_id": session.watcher_thread_id,
                    "message_id": session.watcher_message_id,
                    "message": (
                        f"Watch patterns disabled for process {session.id} — "
                        f"{WATCH_STRIKE_LIMIT} consecutive rate-limit windows triggered "
                        f"(min spacing {WATCH_MIN_INTERVAL_SECONDS}s). "
                        f"Falling back to notify_on_complete semantics; you'll get "
                        f"exactly one notification when the process exits."
                    ),
                })
            return

        # Trim matched output to a reasonable size
        output = "\n".join(matched_lines[:20])
        if len(output) > 2000:
            output = output[:2000] + "\n...(truncated)"

        # Global circuit breaker — across all sessions (secondary safety net).
        if not self._global_watch_admit(now):
            if lifetime_exhausted:
                # The final match was dropped by the global breaker, but the
                # session is already disabled — still tell the user why things
                # went quiet (the strike path emits its summary unconditionally
                # too).
                self._emit_lifetime_watch_disabled(session)
            return

        notification = {
            "session_id": session.id,
            "session_key": session.session_key,
            "task_id": session.task_id,
            "owner_task_id": session.owner_task_id or session.task_id,
            "command": session.command,
            "type": "watch_match",
            "pattern": matched_pattern,
            "output": output,
            "suppressed": suppressed,
            "platform": session.watcher_platform,
            "chat_id": session.watcher_chat_id,
            "user_id": session.watcher_user_id,
            "user_name": session.watcher_user_name,
            "thread_id": session.watcher_thread_id,
            "message_id": session.watcher_message_id,
        }
        _redact_process_result(notification)
        self.completion_queue.put(notification)

        if lifetime_exhausted:
            # Same "why things went quiet" summary as the strike-limit path,
            # queued right after the final delivered match.
            self._emit_lifetime_watch_disabled(session)

    def _emit_lifetime_watch_disabled(self, session: ProcessSession) -> None:
        """Queue the watch_disabled summary for the lifetime-cap path (#93513)."""
        self.completion_queue.put({
            "session_id": session.id,
            "session_key": session.session_key,
            "task_id": session.task_id,
            "owner_task_id": session.owner_task_id or session.task_id,
            "command": session.command,
            "type": "watch_disabled",
            "suppressed": 0,
            "platform": session.watcher_platform,
            "chat_id": session.watcher_chat_id,
            "user_id": session.watcher_user_id,
            "user_name": session.watcher_user_name,
            "thread_id": session.watcher_thread_id,
            "message_id": session.watcher_message_id,
            "message": (
                f"Watch patterns disabled for process {session.id} — "
                f"reached the lifetime cap of {WATCH_LIFETIME_MAX_HITS} delivered "
                f"matches. Falling back to notify_on_complete semantics; you'll get "
                f"exactly one notification when the process exits."
            ),
        })

    def _global_watch_admit(self, now: float) -> bool:
        """Return True if this watch_match event is allowed through the global breaker.

        Semantics:
        - If we're currently in a cooldown period, drop the event and count it.
        - Otherwise, slide the rolling window and check the global cap.
        - If the cap is exceeded, trip the breaker for WATCH_GLOBAL_COOLDOWN_SECONDS
          and emit ONE summary event so the agent/user sees "N notifications were
          suppressed" instead of getting them individually.
        - When the cooldown ends, emit a release summary and reset counters.
        """
        with self._global_watch_lock:
            # Handle cooldown expiry first so we can emit the release summary.
            if self._global_watch_tripped_until and now >= self._global_watch_tripped_until:
                suppressed = self._global_watch_suppressed_during_trip
                self._global_watch_tripped_until = 0.0
                self._global_watch_suppressed_during_trip = 0
                self._global_watch_window_start = now
                self._global_watch_window_hits = 0
                if suppressed > 0:
                    # Queue a summary event outside the lock (below).
                    release_msg = {
                        "session_id": "",
                        "session_key": "",
                        "command": "",
                        "type": "watch_overflow_released",
                        "suppressed": suppressed,
                        "message": (
                            f"Watch-pattern notifications resumed. "
                            f"{suppressed} match event(s) were suppressed during the flood."
                        ),
                        "platform": "",
                        "chat_id": "",
                        "user_id": "",
                        "user_name": "",
                        "thread_id": "",
                    }
                else:
                    release_msg = None
            else:
                release_msg = None

            # Still in cooldown — drop and count.
            if self._global_watch_tripped_until and now < self._global_watch_tripped_until:
                self._global_watch_suppressed_during_trip += 1
                admit = False
                trip_now = None
            else:
                # Slide the window.
                if now - self._global_watch_window_start >= WATCH_GLOBAL_WINDOW_SECONDS:
                    self._global_watch_window_start = now
                    self._global_watch_window_hits = 0

                if self._global_watch_window_hits >= WATCH_GLOBAL_MAX_PER_WINDOW:
                    # Trip the breaker.
                    self._global_watch_tripped_until = now + WATCH_GLOBAL_COOLDOWN_SECONDS
                    self._global_watch_suppressed_during_trip += 1
                    trip_now = now
                    admit = False
                else:
                    self._global_watch_window_hits += 1
                    trip_now = None
                    admit = True

        # Queue summary events outside the lock.
        if release_msg is not None:
            self.completion_queue.put(release_msg)
        if trip_now is not None:
            self.completion_queue.put({
                "session_id": "",
                "session_key": "",
                "command": "",
                "type": "watch_overflow_tripped",
                "message": (
                    f"Watch-pattern overflow: >{WATCH_GLOBAL_MAX_PER_WINDOW} "
                    f"notifications in {WATCH_GLOBAL_WINDOW_SECONDS}s across all processes. "
                    f"Suppressing further watch_match events for "
                    f"{WATCH_GLOBAL_COOLDOWN_SECONDS}s."
                ),
                "platform": "",
                "chat_id": "",
                "user_id": "",
                "user_name": "",
                "thread_id": "",
            })
        return admit

    @staticmethod
    def _is_host_pid_alive(pid: Optional[int]) -> bool:
        """Best-effort liveness check for host-visible PIDs."""
        if not pid:
            return False
        # ``os.kill(pid, 0)`` is NOT a no-op on Windows (bpo-14484) — use
        # the cross-platform existence check.
        from gateway.status import _pid_exists
        return _pid_exists(pid)

    @staticmethod
    def _safe_host_start_time(pid: Optional[int]) -> Optional[int]:
        """Kernel start ticks for a host PID, or None when unavailable."""
        if not pid:
            return None
        try:
            from gateway.status import get_process_start_time
            return get_process_start_time(pid)
        except Exception:
            return None

    @classmethod
    def _host_pid_is_ours(cls, pid: Optional[int], expected_start: Optional[int]) -> bool:
        """True only if ``pid`` is alive AND still the process we spawned.

        The kernel recycles PID/PGID numbers once a process exits and is reaped,
        so a stored PID can later name an *unrelated* process — observed in the
        wild as a recycled number landing on a desktop browser's session leader,
        which our tree-kill then SIGTERMs (Firefox dying at irregular intervals).
        We compare the kernel start time captured at spawn against the live one;
        a mismatch means the number was recycled and must never be signalled.

        When no baseline was captured (legacy checkpoints, or platforms without
        ``/proc``) we degrade to a bare liveness check rather than refusing to
        act, preserving prior best-effort behaviour.
        """
        if not cls._is_host_pid_alive(pid):
            return False
        if expected_start is None:
            return True
        return cls._safe_host_start_time(pid) == expected_start

    def _refresh_detached_session(self, session: Optional[ProcessSession]) -> Optional[ProcessSession]:
        """Update recovered host-PID sessions when the underlying process has exited."""
        if session is None or session.exited or not session.detached or session.pid_scope != "host":
            return session

        # Identity-aware liveness: a recycled PID (alive but a different process
        # than we spawned) must be treated as "our process exited", so it is
        # moved to finished and can never be tree-killed by a later kill().
        if self._host_pid_is_ours(session.pid, session.host_start_time):
            return session

        with session._lock:
            if session.exited:
                return session
            session.exited = True
            # Recovered sessions no longer have a waitable handle, so the real
            # exit code is unavailable once the original process object is gone.
            session.exit_code = None

        self._move_to_finished(session)
        return session

    @staticmethod
    def _proc_alive(proc) -> bool:
        """True if a psutil.Process is running and not a zombie.

        A zombie is already dead (just unreaped), so there's nothing to SIGKILL.
        """
        try:
            import psutil
            if not proc.is_running():
                return False
            return proc.status() != psutil.STATUS_ZOMBIE
        except Exception:
            return False

    @staticmethod
    def _daemon_term_grace_seconds() -> float:
        """Grace window (s) between SIGTERM and escalated SIGKILL.

        Read from ``terminal.daemon_term_grace_seconds`` in config.yaml; floored
        at 0 (0 disables escalation). Falls back to the DEFAULT_CONFIG value if
        config is unreadable, so callers always get a sane number.
        """
        try:
            from hermes_cli.config import read_raw_config, cfg_get, DEFAULT_CONFIG
            cfg = read_raw_config()
            val = cfg_get(cfg, "terminal", "daemon_term_grace_seconds")
            if val is None:
                val = DEFAULT_CONFIG["terminal"]["daemon_term_grace_seconds"]
            return max(float(val), 0.0)
        except Exception:
            return 2.0

    @classmethod
    def _terminate_host_pid(cls, pid: int, expected_start: Optional[int] = None) -> None:
        """Terminate a host-visible PID and its descendants.

        ``expected_start`` is the kernel start time captured when we spawned the
        process. When provided, it is re-validated against the live PID before
        any signal is sent; a mismatch (or a dead PID) means the number was
        recycled onto an unrelated process and we refuse to touch it, so a stale
        background-session PID can never tree-kill a browser or other stranger.

        POSIX: walks the process tree with ``psutil`` and SIGTERMs
        children before the parent so subprocess trees (e.g. Chromium
        renderers/GPU helpers spawned by an ``agent-browser`` daemon)
        don't get reparented to init and survive cleanup.  After a bounded
        grace window (``terminal.daemon_term_grace_seconds``) any tree member
        that ignored SIGTERM — a daemon stalled in its signal handler — is
        escalated to SIGKILL so it can't leak indefinitely.  Set the grace to
        0 to disable escalation (SIGTERM only).

        Windows: shells out to ``taskkill /PID <pid> /T /F``. This is
        the documented Microsoft primitive for tree-kill and matches the
        existing convention in ``gateway.status.terminate_pid``.  ``/F`` is
        already a hard kill, so no separate escalation step is needed.  We
        can't reuse the POSIX psutil path on Windows because:

          1. Windows doesn't maintain a Unix-style process tree —
             ``psutil.Process.children(recursive=True)`` walks PPID
             links that go stale when intermediate processes exit, so
             enumeration is best-effort and misses orphaned descendants.
          2. ``psutil.Process.terminate()`` on Windows is
             ``TerminateProcess()`` which kills only the target handle
             and is a hard kill — there is no Windows equivalent of a
             SIGTERM that cascades through a process group. (See the
             warning in ``gateway/status.py::terminate_pid``: "os.kill
             with SIGTERM is not equivalent to a tree-killing hard stop"
             on Windows.) Headless Chromium has no GUI window, so the
             softer ``taskkill /T`` without ``/F`` won't reach it either.

        ``psutil`` is a hard dependency (see ``pyproject.toml``); the
        bare-``os.kill`` fallback covers OSError / PermissionError on
        POSIX and a missing ``taskkill.exe`` on Windows (effectively
        unreachable on real Windows installs, but cheap insurance).
        """
        if expected_start is not None and not cls._host_pid_is_ours(pid, expected_start):
            # PID was recycled (start time changed) or is gone — never signal a
            # stranger. A leaked orphan is strictly preferable to killing e.g.
            # a browser whose session leader reused this dead session's PID.
            logger.warning(
                "Refusing to terminate host pid %d: start-time mismatch — "
                "PID was recycled onto an unrelated process.", pid,
            )
            return
        if _IS_WINDOWS:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    text=True, encoding='utf-8', errors='replace',
                    timeout=10,
                    creationflags=windows_hide_flags(),
                    stdin=subprocess.DEVNULL,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                try:
                    os.kill(pid, signal.SIGTERM)
                except (OSError, ProcessLookupError, PermissionError):
                    pass
            return

        import psutil
        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        except (OSError, PermissionError):
            try:
                os.kill(pid, signal.SIGTERM)
            except (OSError, ProcessLookupError, PermissionError):
                pass
            return

        # Snapshot the whole tree (children before parent) and SIGTERM each.
        try:
            targets = parent.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            targets = []
        targets.append(parent)

        for proc in targets:
            try:
                proc.terminate()
            except psutil.NoSuchProcess:
                pass
            except (psutil.AccessDenied, OSError):
                pass

        # Escalate to SIGKILL for anything that ignored SIGTERM within the
        # grace window — a daemon stalled in its signal handler would otherwise
        # leak indefinitely.
        grace = cls._daemon_term_grace_seconds()
        if grace <= 0:
            return
        # Sleep out the grace window, then independently re-probe every target
        # and SIGKILL any survivor.  We deliberately do NOT trust
        # ``psutil.wait_procs``'s gone/alive partition here: it reaps via
        # ``Process.wait()`` and can mis-partition when a target transitions
        # through a zombie state or when reaping is racy across a parent/child
        # tree, which left survivors un-killed.  A direct liveness re-probe is
        # deterministic.
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not any(cls._proc_alive(_p) for _p in targets):
                break
            time.sleep(0.05)
        for proc in targets:
            try:
                if not cls._proc_alive(proc):
                    continue
                proc.kill()  # SIGKILL on POSIX
                logger.info(
                    "Escalated to SIGKILL for pid %d (ignored SIGTERM within "
                    "%.1fs grace)", proc.pid, grace,
                )
            except psutil.NoSuchProcess:
                pass
            except (psutil.AccessDenied, OSError):
                pass

    # ----- Spawn -----

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
        return "/tmp"

    def spawn_local(
        self,
        command: str,
        cwd: str = None,
        task_id: str = "",
        session_key: str = "",
        env_vars: dict = None,
        use_pty: bool = False,
        owner_task_id: str = "",
    ) -> ProcessSession:
        """
        Spawn a background process locally.

        Only for TERMINAL_ENV=local. Other backends use spawn_via_env().

        Args:
            use_pty: If True, use a pseudo-terminal via ptyprocess for interactive
                     CLI tools (Codex, Claude Code, Python REPL). Falls back to
                     subprocess.Popen if ptyprocess is not installed.
        """
        # Guard against the `A && B &` subshell-wait trap (issue #68915).
        # Bash parses ``A && B &`` as ``(A && B) &`` — a subshell that holds
        # the stdout pipe open forever when B is a long-running server.
        # The rewriter wraps it to ``A && { B & }`` so no subshell fork.
        # Lazy import avoids circular dependency (terminal_tool imports this).
        from tools.terminal_tool import _rewrite_compound_background as _rewrite_bg

        safe_command = _rewrite_bg(command)

        session = ProcessSession(
            id=f"proc_{uuid.uuid4().hex[:12]}",
            command=command,
            task_id=task_id,
            owner_task_id=owner_task_id or task_id,
            session_key=session_key,
            cwd=_resolve_safe_cwd(cwd or os.getcwd()),
            started_at=time.time(),
        )

        pty_scope_attempted = False
        if use_pty:
            # Try PTY mode for interactive CLI tools
            try:
                if _IS_WINDOWS:
                    from winpty import PtyProcess as _PtyProcessCls
                else:
                    from ptyprocess import PtyProcess as _PtyProcessCls
                user_shell = _find_shell()
                pty_env = _sanitize_subprocess_env(os.environ, env_vars)
                pty_env["PYTHONUNBUFFERED"] = "1"
                # PTY mode is a real TTY, so pager-happy tools (git log/diff,
                # man) WILL page and hang waiting for `q` — default them to
                # cat, honoring any pager the user already exported.
                pty_env.setdefault("GIT_PAGER", "cat")
                pty_env.setdefault("PAGER", "cat")
                pty_argv = [user_shell, "-lic", f"set +m; {safe_command}"]

                # Cgroup isolation for PTY mode (#70716, reviewer gap #1):
                # Wrap the PTY command in a systemd scope so interactive
                # executors get their own cgroup, same as pipe mode.
                pty_in_supervised_gateway = (
                    _IS_LINUX and _is_supervised_gateway_process()
                )
                pty_use_systemd_scope = (
                    pty_in_supervised_gateway and _systemd_run_user_scope_available()
                )

                if pty_use_systemd_scope:
                    pty_argv = _build_systemd_scope_argv(
                        pty_argv,
                        unit_suffix=session.id,
                    )
                    session.systemd_unit = f"hermes-worker-{session.id}.scope"
                    pty_scope_attempted = True
                elif pty_in_supervised_gateway:
                    logger.debug(
                        "PTY background executor not isolated in a "
                        "systemd scope (systemd-run --user unavailable); "
                        "worker shares the gateway cgroup."
                    )

                pty_proc = _PtyProcessCls.spawn(
                    pty_argv,
                    cwd=session.cwd,
                    env=pty_env,
                    dimensions=(30, 120),
                )
                session.pid = pty_proc.pid
                session.host_start_time = self._safe_host_start_time(session.pid)
                # Store the pty handle on the session for read/write
                session._pty = pty_proc

                # PTY reader thread
                reader = threading.Thread(
                    target=self._pty_reader_loop,
                    args=(session,),
                    daemon=True,
                    name=f"proc-pty-reader-{session.id}",
                )
                session._reader_thread = reader
                reader.start()

                with self._lock:
                    self._prune_if_needed()
                    self._running[session.id] = session

                self._write_checkpoint()
                return session

            except ImportError:
                logger.warning("ptyprocess not installed, falling back to pipe mode")
            except Exception as e:
                logger.warning("PTY spawn failed (%s), falling back to pipe mode", e)
                if pty_scope_attempted and session.systemd_unit:
                    if not _stop_systemd_unit(session.systemd_unit):
                        raise RuntimeError(
                            "PTY scope could not be reaped; refusing pipe fallback "
                            "to avoid duplicate command execution"
                        ) from e
                    session.systemd_unit = ""

        # Standard Popen path (non-PTY or PTY fallback)
        # Use the user's login shell for consistency with LocalEnvironment --
        # ensures rc files are sourced and user tools are available.
        user_shell = _find_shell()
        # Force unbuffered output for Python scripts so progress is visible
        # during background execution (libraries like tqdm/datasets buffer when
        # stdout is a pipe, hiding output from process(action="poll")).
        bg_env = _sanitize_subprocess_env(os.environ, env_vars)
        bg_env["PYTHONUNBUFFERED"] = "1"
        _popen_kwargs = {"creationflags": windows_hide_flags()} if _IS_WINDOWS else {}

        # Cgroup isolation (#70716): when running in the live, supervised
        # systemd gateway, wrap the worker in its own transient systemd
        # scope so it gets a separate cgroup.  An OOM in the worker then
        # kills only the worker instead of taking down the whole gateway
        # cgroup (and the messaging control plane with it). This applies to
        # both pipe mode and the PTY path above.
        shell_argv = [user_shell, "-lic", f"set +m; {safe_command}"]
        in_supervised_gateway = _IS_LINUX and _is_supervised_gateway_process()
        use_systemd_scope = (
            in_supervised_gateway and _systemd_run_user_scope_available()
        )

        if use_systemd_scope:
            unit_suffix = (
                f"{session.id}-pipe-fallback" if pty_scope_attempted else session.id
            )
            spawn_argv = _build_systemd_scope_argv(
                shell_argv,
                unit_suffix=unit_suffix,
            )
            session.systemd_unit = f"hermes-worker-{unit_suffix}.scope"
            # CRITICAL (#70716 regression): systemd-run --scope does NOT give
            # the worker a new session — the invoked process keeps the
            # parent's session and inherits its controlling terminal.  From an
            # interactive TUI this drops the worker into the same session as
            # the foreground process group: background spawns then stop the
            # whole session (observed as 5 dead TUIs in state T / "Arrêté").
            # start_new_session=True gives systemd-run (and the scoped worker
            # below it) a private session.  Cgroup isolation is preserved:
            # the scope is attached to the invoked process, not to the
            # spawning session.
            popen_start_new_session = True
        else:
            spawn_argv = shell_argv
            popen_start_new_session = True
            if in_supervised_gateway:
                # Running under a supervisor but could not get a private
                # cgroup — the worker shares the gateway cgroup, so an OOM
                # in the worker can still kill the whole gateway (#70716).
                logger.debug(
                    "Local background executor not isolated in a systemd scope "
                    "(in_supervised_gateway=%s, systemd-run --user available=%s); "
                    "worker shares the gateway cgroup.",
                    in_supervised_gateway,
                    _systemd_run_user_scope_available(),
                )

        proc = subprocess.Popen(
            spawn_argv,
            text=True,
            cwd=session.cwd,
            env=bg_env,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=popen_start_new_session,
            **_popen_kwargs,
        )

        session.process = proc
        session.pid = proc.pid
        session.host_start_time = self._safe_host_start_time(session.pid)

        try:
            # Start output reader thread
            reader = threading.Thread(
                target=self._reader_loop,
                args=(session,),
                daemon=True,
                name=f"proc-reader-{session.id}",
            )
            session._reader_thread = reader
            reader.start()

            with self._lock:
                self._prune_if_needed()
                self._running[session.id] = session

            self._write_checkpoint()
        except Exception:
            # Post-Popen setup failed — kill the orphaned subprocess (and any
            # descendants spawned via setsid) before re-raising so they do not
            # leak as untracked background processes.
            try:
                if session.systemd_unit:
                    # The worker runs in its own systemd scope and, since the
                    # #70716 session-isolation fix, its own session.  Stop the
                    # scope (kills every process in the worker cgroup), then
                    # terminate the systemd-run wrapper PID as fallback.
                    # Never killpg: scope teardown is the authoritative
                    # cleanup for the worker cgroup.
                    _stop_systemd_unit(session.systemd_unit)
                    self._terminate_host_pid(proc.pid, session.host_start_time)
                elif not _IS_WINDOWS:
                    try:
                        kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
                        os.killpg(os.getpgid(proc.pid), kill_signal)  # windows-footgun: ok - guarded by _IS_WINDOWS above
                    except (ProcessLookupError, PermissionError, OSError):
                        proc.kill()
                else:
                    proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
            raise

        return session

    def spawn_via_env(
        self,
        env: Any,
        command: str,
        cwd: str = None,
        task_id: str = "",
        session_key: str = "",
        timeout: int = 10,
        owner_task_id: str = "",
    ) -> ProcessSession:
        """
        Spawn a background process through a non-local environment backend.

        For Docker/Singularity/Modal/Daytona/SSH: runs the command inside the sandbox
        using the environment's execute() interface. We wrap the command to
        capture the in-sandbox PID and redirect output to a log file inside
        the sandbox, then poll the log via subsequent execute() calls.

        This is less capable than local spawn (no live stdout pipe, no stdin),
        but it ensures the command runs in the correct sandbox context.
        """
        session = ProcessSession(
            id=f"proc_{uuid.uuid4().hex[:12]}",
            command=command,
            task_id=task_id,
            owner_task_id=owner_task_id or task_id,
            session_key=session_key,
            cwd=cwd,
            started_at=time.time(),
            env_ref=env,
            pid_scope="sandbox",
        )

        # Run the command in the sandbox with output capture
        temp_dir = self._env_temp_dir(env)
        log_path = f"{temp_dir}/hermes_bg_{session.id}.log"
        pid_path = f"{temp_dir}/hermes_bg_{session.id}.pid"
        exit_path = f"{temp_dir}/hermes_bg_{session.id}.exit"
        quoted_command = shlex.quote(command)
        quoted_temp_dir = shlex.quote(temp_dir)
        quoted_log_path = shlex.quote(log_path)
        quoted_pid_path = shlex.quote(pid_path)
        quoted_exit_path = shlex.quote(exit_path)
        bg_command = (
            f"mkdir -p {quoted_temp_dir} && "
            f"( nohup bash -lc {quoted_command} > {quoted_log_path} 2>&1; "
            f"rc=$?; printf '%s\\n' \"$rc\" > {quoted_exit_path} ) & "
            f"echo $! > {quoted_pid_path} && cat {quoted_pid_path}"
        )

        try:
            result = env.execute(
                bg_command,
                timeout=timeout,
                rewrite_compound_background=False,
            )
            output = result.get("output", "").strip()
            # Try to extract the PID from the output
            for line in output.splitlines():
                line = line.strip()
                if line.isdigit():
                    session.pid = int(line)
                    break
            # If the wrapper couldn't produce a PID (for example, syntax
            # error or broken redirect), treat it as a failed launch instead
            # of exposing a fake running session.
            if session.pid is None:
                session.exited = True
                session.exit_code = int(result.get("returncode", -1))
                if session.exit_code == 0:
                    session.exit_code = -1
                session.completion_reason = "failed_start"
                session.termination_source = "failed_start"
                session.output_buffer = result.get("output", "").strip()
        except Exception as e:
            session.exited = True
            session.exit_code = -1
            session.completion_reason = "failed_start"
            session.termination_source = "failed_start"
            session.output_buffer = f"Failed to start: {e}"

        if not session.exited:
            # Start a poller thread that periodically reads the log file
            reader = threading.Thread(
                target=self._env_poller_loop,
                args=(session, env, log_path, pid_path, exit_path),
                daemon=True,
                name=f"proc-poller-{session.id}",
            )
            session._reader_thread = reader
            reader.start()

        with self._lock:
            self._prune_if_needed()
            if not session.exited:
                self._running[session.id] = session

        if not session.exited:
            self._write_checkpoint()

        return session

    # ----- Reader / Poller Threads -----

    def _reader_loop(self, session: ProcessSession):
        """Background thread: read stdout from a local Popen process.

        IMPORTANT: avoid ``TextIOWrapper.read(4096)`` here. On pipes that call can
        block until EOF (or a large buffer fills), which makes "live" output land
        in one burst at process exit. ``buffer.read1(4096)`` yields incremental
        chunks as bytes become available, then we decode to text.

        Orphaned-pipe guard (issue #68915): when the user's command backgrounds
        a long-lived process (``node server.js &``, ``sleep 300 &``), that
        grandchild inherits the write end of our stdout pipe via ``fork()``.
        The direct ``bash`` child exits promptly, but the pipe never reaches
        EOF while the grandchild lives — so a blocking read would park this
        thread forever, ``session.exited`` would never flip, and
        ``notify_on_complete`` would never fire (``_reconcile_local_exit``
        only runs lazily from poll()/wait(), which an autonomous notification
        can't rely on). On POSIX we therefore ``select()`` with a short poll
        interval and stop draining shortly after the direct child exits, even
        if the pipe hasn't EOF'd — mirroring the foreground fix in
        ``tools/environments/base.py::_wait_for_process`` (#8340). Windows
        pipes don't support select(); the blocking path is kept there and the
        lazy reconcile in poll()/wait() remains the safety net.
        """
        first_chunk = True
        # Incremental decoder: raw pipe reads can split a multibyte UTF-8
        # character across two read1() chunks. A stateless per-chunk
        # ``bytes.decode(errors="replace")`` turns both halves into U+FFFD
        # mojibake. The incremental decoder holds the partial sequence until
        # the continuation bytes arrive — same treatment the foreground path
        # already has in ``tools/environments/base.py::_wait_for_process``.
        # (Ported from openclaw/openclaw#112325.)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        def _append_chunk(chunk: str):
            nonlocal first_chunk
            if first_chunk:
                chunk = self._clean_shell_noise(chunk)
                first_chunk = False
            with session._lock:
                session.output_buffer += chunk
                if len(session.output_buffer) > session.max_output_chars:
                    session.output_buffer = session.output_buffer[-session.max_output_chars:]
            self._check_watch_patterns(session, chunk)
            self._emit_output(session, chunk)

        try:
            proc = session.process
            if proc is None or proc.stdout is None:
                return
            stdout = proc.stdout

            raw_read = getattr(getattr(stdout, "buffer", None), "read1", None)

            # Resolve a real OS fd for the select() path. Mocked streams
            # (unit tests, adapters) may lack fileno() — fall back to the
            # historical blocking loop for those.
            fd = None
            if raw_read is not None and not _IS_WINDOWS:
                fileno = getattr(stdout, "fileno", None)
                try:
                    candidate = fileno() if callable(fileno) else None
                except Exception:
                    candidate = None
                if isinstance(candidate, int) and candidate >= 0:
                    fd = candidate

            if fd is not None:
                import select as _select

                idle_after_exit = 0
                while True:
                    try:
                        ready, _, _ = _select.select([fd], [], [], 0.2)
                    except (ValueError, OSError):
                        break  # fd already closed
                    if ready:
                        raw = raw_read(4096)
                        if not raw:
                            break  # true EOF — all writers closed
                        chunk = decoder.decode(raw)
                        if chunk:
                            _append_chunk(chunk)
                        idle_after_exit = 0
                    elif proc.poll() is not None:
                        # Direct child is gone and the pipe was idle for
                        # ~200ms. Give it a few more cycles to catch any
                        # buffered tail, then stop — otherwise we would wait
                        # forever on a pipe held open by an orphaned
                        # grandchild (issue #68915).
                        idle_after_exit += 1
                        if idle_after_exit >= 3:
                            break
            else:
                while True:
                    if raw_read is not None:
                        raw = raw_read(4096)
                        if not raw:
                            break
                        chunk = decoder.decode(raw)
                        if not chunk:
                            continue  # partial multibyte sequence — wait for more bytes
                    else:
                        # Fallback for mocked/alternate streams without a buffered raw
                        # interface. This may be less "live", but keeps compatibility.
                        chunk = stdout.read(4096)
                        if not chunk:
                            break

                    _append_chunk(chunk)
        except Exception as e:
            logger.debug("Process stdout reader ended: %s", e)
        finally:
            # Flush any bytes still pending in the incremental decoder (a
            # truncated multibyte sequence at EOF becomes one U+FFFD instead
            # of being dropped silently).
            try:
                tail = decoder.decode(b"", final=True)
                if tail:
                    _append_chunk(tail)
            except Exception:
                pass
            # Always reap the child to prevent zombie processes.
            try:
                session.process.wait(timeout=5)
            except Exception as e:
                logger.debug("Process wait timed out or failed: %s", e)
            session.exited = True
            if session.completion_reason != "killed":
                session.exit_code = session.process.returncode
                session.completion_reason = "exited"
            self._move_to_finished(session)

    @staticmethod
    def _log_delta_command(quoted_log_path: str, offset: int) -> str:
        """Build the shell command that reads only new bytes from a log file.

        The old version ran ``cat`` on the whole file every poll, so a job
        that keeps writing pays for its entire output again and again. Over a
        long run that turns into a lot of wasted traffic on the docker/SSH
        channel, since only the new part is ever used.

        The command prints one header line, ``"<size> <offset>"``, then the
        bytes between ``offset`` and ``size``. Reading the size first and
        cutting the tail at that same size keeps the two numbers in step, so
        a file that grows while the command runs never sends a byte twice.
        A file that shrank was rotated or truncated, so the offset drops back
        to 0 and the reader starts over.

        The end of the window is pulled back to a UTF-8 character boundary:
        the backend decodes each ``execute()`` result on its own, so a
        multibyte character straddling two polls would otherwise come back
        as replacement characters (and break watch patterns near the seam).
        Up to 3 trailing continuation bytes are held for the next poll; the
        header reports the trimmed size so the offset stays consistent.
        """
        return (
            f"O={offset}; "
            f"S=$({{ wc -c < {quoted_log_path}; }} 2>/dev/null | tr -dc '0-9'); "
            f"S=${{S:-0}}; "
            f'if [ "$S" -lt "$O" ]; then O=0; fi; '
            # Hold back an INCOMPLETE trailing UTF-8 sequence for the next
            # poll. Scan back up to 3 continuation bytes (octal 200-277) to
            # the lead byte; if the lead byte's declared length (3xx=2, 34x-35x
            # =3, 36x-37x=4) exceeds the bytes present, trim to before it.
            # Complete sequences and ASCII tails are left untouched.
            f'N=0; P=$S; while [ "$P" -gt "$O" ] && [ "$N" -lt 3 ]; do '
            f"B=$(tail -c +$P {quoted_log_path} 2>/dev/null | head -c 1 | od -An -to1 | tr -dc '0-9'); "
            f'case "$B" in 2[0-7][0-7]) P=$((P-1)); N=$((N+1));; *) break;; esac; done; '
            f'if [ "$N" -gt 0 ] || [ "$P" -eq "$S" ]; then '
            f"B=$(tail -c +$P {quoted_log_path} 2>/dev/null | head -c 1 | od -An -to1 | tr -dc '0-9'); "
            f'case "$B" in 3[0-3][0-7]) L=2;; 3[4-5][0-7]) L=3;; 3[6-7][0-7]) L=4;; *) L=1;; esac; '
            f'if [ "$L" -gt $((N+1)) ]; then S=$((P-1)); fi; fi; '
            f'echo "$S $O"; '
            f'if [ "$S" -gt "$O" ]; then '
            f"tail -c +$((O+1)) {quoted_log_path} 2>/dev/null | head -c $((S-O)); fi"
        )

    def _env_poller_loop(
        self, session: ProcessSession, env: Any, log_path: str, pid_path: str, exit_path: str
    ):
        """Background thread: poll a sandbox log file for non-local backends."""
        quoted_log_path = shlex.quote(log_path)
        quoted_pid_path = shlex.quote(pid_path)
        quoted_exit_path = shlex.quote(exit_path)
        # Byte offset already read from the log. Bytes, not characters: the
        # shell counts bytes, and a log with non-ASCII text has more bytes
        # than characters.
        prev_output_bytes = 0
        while not session.exited:
            time.sleep(2)  # Poll every 2 seconds
            try:
                # Read only the bytes written since the last poll.
                result = env.execute(
                    self._log_delta_command(quoted_log_path, prev_output_bytes),
                    timeout=10,
                )
                raw = result.get("output", "")
                header, _, delta = raw.partition("\n")
                try:
                    size_str, offset_str = header.split()
                    new_size = int(size_str)
                    used_offset = int(offset_str)
                except ValueError:
                    # No usable header (command failed, shell missing a tool).
                    # Skip this poll rather than act on a half-read value.
                    new_size = None
                    used_offset = None
                    delta = ""
                if new_size is not None and used_offset is not None:
                    if used_offset < prev_output_bytes:
                        # The log was rotated or truncated, so what we hold no
                        # longer lines up with the file. Drop it and restart.
                        with session._lock:
                            session.output_buffer = ""
                    prev_output_bytes = new_size
                if delta:
                    with session._lock:
                        session.output_buffer += delta
                        if len(session.output_buffer) > session.max_output_chars:
                            session.output_buffer = session.output_buffer[-session.max_output_chars:]
                    self._check_watch_patterns(session, delta)
                    self._emit_output(session, delta)

                # Check if process is still running
                check = env.execute(
                    f"kill -0 \"$(cat {quoted_pid_path} 2>/dev/null)\" 2>/dev/null; echo $?",
                    timeout=5,
                )
                check_output = check.get("output", "").strip()
                if check_output and check_output.splitlines()[-1].strip() != "0":
                    # Process has exited -- get exit code captured by the wrapper shell.
                    exit_result = env.execute(
                        f"cat {quoted_exit_path} 2>/dev/null",
                        timeout=5,
                    )
                    exit_str = exit_result.get("output", "").strip()
                    try:
                        session.exit_code = int(exit_str.splitlines()[-1].strip())
                    except (ValueError, IndexError):
                        session.exit_code = -1
                    session.exited = True
                    if session.completion_reason != "killed":
                        session.completion_reason = "exited"
                    self._move_to_finished(session)
                    return

            except Exception:
                # Environment might be gone (sandbox reaped, etc.)
                session.exited = True
                session.exit_code = -1
                session.completion_reason = "lost"
                session.termination_source = "backend_lost"
                self._move_to_finished(session)
                return

    def _pty_reader_loop(self, session: ProcessSession):
        """Background thread: read output from a PTY process."""
        pty = session._pty
        # PTY reads can split a multibyte UTF-8 character across chunks just
        # like pipe reads — hold partial sequences until the rest arrives.
        # (Ported from openclaw/openclaw#112325.)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        def _append_text(text: str):
            with session._lock:
                session.output_buffer += text
                if len(session.output_buffer) > session.max_output_chars:
                    session.output_buffer = session.output_buffer[-session.max_output_chars:]
            self._check_watch_patterns(session, text)
            self._emit_output(session, text)

        try:
            while pty.isalive():
                try:
                    chunk = pty.read(4096)
                    if chunk:
                        # ptyprocess returns bytes; pywinpty returns str
                        text = chunk if isinstance(chunk, str) else decoder.decode(chunk)
                        if text:
                            _append_text(text)
                except EOFError:
                    break
                except Exception:
                    break
        except Exception as e:
            logger.debug("PTY stdout reader ended: %s", e)

        # Flush any partial multibyte sequence held by the decoder.
        try:
            tail = decoder.decode(b"", final=True)
            if tail:
                _append_text(tail)
        except Exception:
            pass

        # Process exited
        try:
            pty.wait()
        except Exception as e:
            logger.debug("PTY wait timed out or failed: %s", e)
        session.exited = True
        if session.completion_reason != "killed":
            session.exit_code = pty.exitstatus if hasattr(pty, 'exitstatus') else -1
            session.completion_reason = "exited"
        self._move_to_finished(session)

    def _move_to_finished(self, session: ProcessSession):
        """Move a session from running to finished.

        Idempotent: if the session was already moved (e.g. kill_process raced
        with the reader thread), the second call is a no-op — no duplicate
        completion notification is enqueued.
        """
        with self._lock:
            was_running = self._running.pop(session.id, None) is not None
            self._finished[session.id] = session
        session._completion_event.set()
        self._write_checkpoint()

        # Only enqueue completion notification on the FIRST move.  Without
        # this guard, kill_process() and the reader thread can both call
        # _move_to_finished(), producing duplicate [IMPORTANT: ...] messages.
        if was_running and session.notify_on_complete:
            from tools.ansi_strip import strip_ansi
            output_tail = strip_ansi(session.output_buffer[-2000:]) if session.output_buffer else ""
            notification = {
                "type": "completion",
                "session_id": session.id,
                "session_key": session.session_key,
                "task_id": session.task_id,
                "owner_task_id": session.owner_task_id or session.task_id,
                "command": session.command,
                "exit_code": session.exit_code,
                "completion_reason": session.completion_reason,
                "termination_source": session.termination_source,
                "output": output_tail,
                # Stable producer identity across checkpoint recovery; unlike
                # a consumer-observed completion timestamp, this does not vary
                # based on which watcher notices exit first.
                "started_at": session.started_at,
            }
            _redact_process_result(notification)
            self.completion_queue.put(notification)

    # ----- Query Methods -----

    def is_completion_consumed(self, session_id: str) -> bool:
        """Check if a completion notification was already consumed via wait/log."""
        return session_id in self._completion_consumed

    def is_session_waiting(self, session_id: str) -> bool:
        """Whether a goal loop parked on this session should still be parked.

        Used by the goal-loop wait barrier (``hermes_cli.goals``) to support
        waiting on a process's OWN trigger, not just its exit. A session is
        "still waiting" when:
          - it is still running, AND
          - if it has ``watch_patterns``, none has matched yet (so a
            long-lived watcher that fires a trigger mid-run — and may never
            exit — unblocks the moment its pattern hits, not on exit).

        Returns False (don't wait) when the session has exited, its watch
        pattern has already fired, or the session is unknown — so a stale or
        already-triggered barrier can never wedge the loop.
        """
        if not session_id:
            return False
        with self._lock:
            session = self._running.get(session_id) or self._finished.get(session_id)
        if session is None:
            return False
        # Refresh detached/remote state so .exited is current.
        try:
            self._refresh_detached_session(session)
        except Exception:
            pass
        if session.exited:
            return False
        # Watch-pattern process: the trigger is a pattern match, not exit.
        # Once any match has been delivered, the wait is satisfied even though
        # the process keeps running (server/daemon/watcher case).
        if session.watch_patterns and not session._watch_disabled:
            if session._watch_hits > 0:
                return False
        return True

    def wait_for_pending_completions(
        self,
        task_id: Optional[str] = None,
        *,
        timeout: float | None = None,
        poll_interval: float = 1.0,
    ) -> dict:
        """Bounded wait for tracked ``notify_on_complete`` background processes.

        One-shot CLI runs (``hermes -q/-Q/-z``) exit as soon as their single
        turn ends.  Any background process the turn spawned with
        ``notify_on_complete=True`` — a bounded task whose completion the
        caller explicitly cares about — still holds a stdout pipe owned by
        the dying parent, so it is killed by SIGPIPE on its next write a few
        seconds later.  Bot Mode handoff REPLIES are the visible casualty
        (#90879): a recipient invoked as ``hermes -p <bot> chat -Q
        --query-file ...`` dispatches its reply via ``message_agent`` /
        ``bot_relay`` exactly this way, then exits, and the reply process is
        destroyed ~3s later.  The sender waits forever for a reply that was
        already killed.

        Called from the one-shot exit paths so the parent lingers (bounded)
        until those deliveries actually finish.  This fixes the class — ANY
        bounded background task in a one-shot run, not just DMs: bot_mode_dm
        deliveries, bot_relay waiter processes, and plain
        ``terminal(background=true, notify_on_complete=true)`` jobs.

        Only ``notify_on_complete`` processes are waited on. Plain background
        processes (servers, daemons, watch-pattern monitors) carry no
        completion contract and are not the parent's to wait for.

        Args:
            task_id: restrict to processes spawned for this task; ``None``
                waits on every tracked process (a one-shot CLI process hosts
                exactly one agent, so its registry is private to that run).
            timeout: max seconds to linger. ``None`` reads
                ``terminal.oneshot_completion_wait_seconds`` from config
                (default 600). ``<= 0`` disables the wait entirely.
            poll_interval: per-pass event-wait bound; each pass re-reconciles
                child state so an orphaned-pipe exit (#17327) can't wedge the
                linger for the full timeout.

        Returns:
            ``{"waited": [...], "completed": [...], "timed_out": [...]}``
            (session ids). All lists empty when there was nothing to wait on.
        """
        if timeout is None:
            timeout = self._oneshot_completion_wait_seconds()
        result: dict = {"waited": [], "completed": [], "timed_out": []}
        with self._lock:
            pending = [
                s
                for s in self._running.values()
                if s.notify_on_complete
                and not s.exited
                and (task_id is None or s.task_id == task_id)
            ]
        if not pending or timeout <= 0:
            return result
        result["waited"] = [s.id for s in pending]
        logger.info(
            "One-shot exit lingering (bounded %ss) for %d notify_on_complete "
            "background process(es): %s",
            timeout,
            len(pending),
            ", ".join(s.id for s in pending),
        )
        deadline = time.monotonic() + max(float(timeout), 0.0)
        interval = max(float(poll_interval), 0.05)
        try:
            from tools.interrupt import is_interrupted as _is_interrupted
        except Exception:
            def _is_interrupted() -> bool:
                return False
        interrupted = False
        for session in pending:
            try:
                while not session.exited:
                    if interrupted or _is_interrupted():
                        interrupted = True
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    # Reconcile first: catches direct-child exits whose reader
                    # is blocked on a pipe held open by a descendant (#17327)
                    # and detached/env sessions, so the event actually fires.
                    try:
                        self._reconcile_local_exit(session)
                        self._refresh_detached_session(session)
                    except Exception:
                        pass
                    if session.exited:
                        break
                    session._completion_event.wait(min(remaining, interval))
            except KeyboardInterrupt:
                # User aborted the linger — stop waiting on everything but
                # never let the interrupt skip the caller's durable teardown
                # (session flush, end_session) that follows this wait.
                interrupted = True
            if session.exited:
                result["completed"].append(session.id)
            else:
                result["timed_out"].append(session.id)
        if result["timed_out"]:
            logger.warning(
                "One-shot exit linger timed out after %ss with %d background "
                "process(es) still running: %s — they may be killed when this "
                "process exits.",
                timeout,
                len(result["timed_out"]),
                ", ".join(result["timed_out"]),
            )
        return result

    @staticmethod
    def _oneshot_completion_wait_seconds() -> float:
        """Bounded linger (s) for one-shot exits with pending notify_on_complete
        processes.  Read from ``terminal.oneshot_completion_wait_seconds``;
        0 disables. Falls back to the DEFAULT_CONFIG value (600) when config
        is unreadable so callers always get a sane bound.
        """
        try:
            from hermes_cli.config import DEFAULT_CONFIG, cfg_get, read_raw_config
            cfg = read_raw_config()
            val = cfg_get(cfg, "terminal", "oneshot_completion_wait_seconds")
            if val is None:
                val = DEFAULT_CONFIG["terminal"]["oneshot_completion_wait_seconds"]
            return max(float(val), 0.0)
        except Exception:
            return 600.0

    def _drain_should_skip(
        self, session_id: str, *, skip_poll_observed: bool = True
    ) -> bool:
        """Whether this drain should skip a completion event for this session.

        Skips when the agent has either truly consumed the output (wait/log →
        ``_completion_consumed``) or observed the exit inline via poll()
        (``_poll_observed``).  In both cases the CLI agent already has the
        result this turn, so injecting a [SYSTEM: ...] completion would be a
        duplicate (#8228).  The gateway/tui watchers do NOT use this — they
        check only ``is_completion_consumed`` so a read-only poll never
        suppresses their autonomous delivery turn (#10156).
        """
        return session_id in self._completion_consumed or (
            skip_poll_observed and session_id in self._poll_observed
        )

    @staticmethod
    def _surface_child_process_notifications() -> bool:
        """Whether subagent-owned process notifications surface in the parent.

        Read from ``delegation.surface_child_process_notifications`` in
        config.yaml (default false = suppress). On any config read error the
        DEFAULT applies (suppress) — never crash the drain loop.
        """
        try:
            from hermes_cli.config import DEFAULT_CONFIG, cfg_get, read_raw_config
            cfg = read_raw_config()
            val = cfg_get(cfg, "delegation", "surface_child_process_notifications")
            if val is None:
                val = DEFAULT_CONFIG["delegation"][
                    "surface_child_process_notifications"
                ]
            return bool(val)
        except Exception:
            return False

    def drain_notifications(
        self,
        session_key: str = "",
        owns_event=None,
        *,
        skip_poll_observed: bool = True,
    ) -> "list[tuple[dict, str]]":
        """Pop all pending notification events and return formatted pairs.

        Returns a list of (raw_event, formatted_text) tuples.
        Skips completion events the agent already consumed via wait/log or
        observed inline via poll() (see ``_drain_should_skip``). Gateway/TUI
        callers pass ``skip_poll_observed=False`` because read-only polling must
        not suppress autonomous delivery there.

        When a routing filter is supplied, addressed notifications must not be
        drained into the wrong session. Async-delegation events always require
        conversation payload; ordinary notifications require routing when they
        carry ``session_key`` or ``origin_ui_session_id`` metadata. Two filter
        modes are supported, strongest first:

        - ``owns_event(evt) -> bool``: positive-proof ownership callback.
          When provided, a routed event is consumed ONLY if the callback
          returns True; everything else is re-queued for its owner.
          The TUI passes its compression-chain-aware ownership check here so
          a post-compression session still claims its own pre-compression
          dispatches.
        - ``session_key``: plain key equality (CLI and other single-session
          callers). Non-matching addressed events are re-queued.

        With neither set, all events are consumed (legacy single-session
        behavior, backward compatible). Ownerless ordinary notifications also
        retain that legacy behavior even when a filter is provided. When a
        filter is provided, ownerless async-delegation events remain
        fail-closed and require positive proof.
        """
        results: "list[tuple[dict, str]]" = []
        requeue: "list[dict]" = []
        # Lazily-read flag for subagent-owned process notifications
        # (delegation.surface_child_process_notifications, default false).
        # Read at most once per drain, and only when an sa- event shows up.
        surface_child: "bool | None" = None
        while not self.completion_queue.empty():
            try:
                evt = self.completion_queue.get_nowait()
            except Exception:
                break
            # Positive-proof ownership beats bare key equality. Delegation
            # payloads always require proof; ordinary events require it once
            # they carry routing metadata. Ownerless ordinary events preserve
            # legacy single-session delivery.
            is_async_delegation = evt.get("type") == "async_delegation"
            evt_session_key = str(evt.get("session_key") or "")
            evt_origin_sid = str(evt.get("origin_ui_session_id") or "")
            requires_positive_proof = is_async_delegation or bool(
                evt_session_key or evt_origin_sid
            )
            if owns_event is not None and requires_positive_proof:
                try:
                    owned = bool(owns_event(evt))
                except Exception:
                    owned = False  # fail closed — never leak on a broken check
                if not owned:
                    requeue.append(evt)
                    continue
            elif session_key and requires_positive_proof:
                if evt_session_key != session_key:
                    requeue.append(evt)
                    continue
            elif is_async_delegation and evt.get("restored"):
                # Durable restore can enqueue previous-process payloads into a
                # fresh registry. An unfiltered legacy drain cannot prove
                # ownership, so leave those events queued for the owner.
                requeue.append(evt)
                continue
            # Local consumed/observed state may suppress only events this
            # session owns (or legacy ownerless ordinary events). Routing must
            # happen first so a foreign session cannot drop the owner's event.
            _evt_sid = evt.get("session_id", "")
            if evt.get("type") == "completion" and self._drain_should_skip(
                _evt_sid, skip_poll_observed=skip_poll_observed
            ):
                continue

            # Subagent-owned process notifications are suppressed from the
            # parent conversation by default — the child's consolidated
            # delegation result is the deliverable; "npm ci finished" walls
            # mid-chat are noise. Ownership is judged on owner_task_id (the
            # RAW spawning task id): the container key in task_id is
            # deliberately collapsed to "default"/the session key by
            # _resolve_container_task_id, which previously let child events
            # bypass this gate. Dropped, NOT requeued (children never drain
            # notify events, so requeueing would pin them in the queue
            # forever). Type 'async_delegation' is the delegation result
            # itself and is NEVER suppressed.
            _evt_task_id = str(
                evt.get("owner_task_id") or evt.get("task_id") or ""
            )
            if not is_async_delegation and _evt_task_id.startswith("sa-"):
                if surface_child is None:
                    surface_child = self._surface_child_process_notifications()
                if not surface_child:
                    logger.debug(
                        "Suppressed subagent-owned process notification "
                        "(delegation.surface_child_process_notifications=false): "
                        "type=%s session_id=%s task_id=%s",
                        evt.get("type", "completion"),
                        _evt_sid,
                        _evt_task_id,
                    )
                    continue

            text = format_process_notification(evt)
            if text:
                results.append((evt, text))
        for evt in requeue:
            self.completion_queue.put(evt)
        return results


ProcessRegistry = ProcessRegistryRuntimeMixin
