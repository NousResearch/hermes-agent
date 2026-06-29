"""
Process Registry -- In-memory registry for managed background processes.

Tracks processes spawned via terminal(background=true), providing:
  - Output buffering (rolling 200KB window)
  - Status polling and log retrieval
  - Blocking wait with interrupt support
  - Process killing
  - Crash recovery via JSON checkpoint file
  - Session-scoped tracking for gateway reset protection

Background processes execute THROUGH the environment interface -- nothing
runs on the host machine unless TERMINAL_ENV=local. For Docker, Singularity,
Modal, Daytona, and SSH backends, the command runs inside the sandbox.

Usage:
    from tools.process_registry import process_registry

    # Spawn a background process (called from terminal_tool)
    session = process_registry.spawn(env, "pytest -v", task_id="task_123")

    # Poll for status
    result = process_registry.poll(session.id)

    # Block until done
    result = process_registry.wait(session.id, timeout=300)

    # Kill it
    process_registry.kill(session.id)
"""

import json
import logging
import os
import platform
import shlex
import signal
import subprocess
import threading
import time
import uuid

_IS_WINDOWS = platform.system() == "Windows"
from tools.environments.local import _find_shell, _resolve_safe_cwd, _sanitize_subprocess_env
from hermes_cli._subprocess_compat import windows_hide_flags
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)


# Checkpoint file for crash recovery (gateway only)
CHECKPOINT_PATH = get_hermes_home() / "processes.json"

# Limits
MAX_OUTPUT_CHARS = 200_000      # 200KB rolling output buffer
FINISHED_TTL_SECONDS = 1800     # Keep finished processes for 30 minutes
MAX_PROCESSES = 64              # Max concurrent tracked processes (LRU pruning)
MAX_ACTIVE_PROCESS_AGE = 86400  # 24h default — see session_reset.bg_process_max_age_hours (#29177)

# Watch pattern rate limiting — PER SESSION.
# Hard rule: at most ONE watch-match notification every WATCH_MIN_INTERVAL_SECONDS.
# Any match arriving inside that cooldown window is dropped and counted as a strike.
# After WATCH_STRIKE_LIMIT consecutive strike windows, watch_patterns for that
# session is permanently disabled and the session falls back to notify_on_complete
# semantics (one notification when the process actually exits).
WATCH_MIN_INTERVAL_SECONDS = 15   # Minimum spacing between consecutive watch matches
WATCH_STRIKE_LIMIT = 3            # Strikes in a row → disable watch + promote to notify_on_complete

# Global circuit breaker — across all sessions. Secondary safety net so concurrent
# siblings can't collectively flood the user even when each is under its own cap.
WATCH_GLOBAL_MAX_PER_WINDOW = 15
WATCH_GLOBAL_WINDOW_SECONDS = 10
WATCH_GLOBAL_COOLDOWN_SECONDS = 30


def format_uptime_short(seconds: int) -> str:
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    mins, secs = divmod(s, 60)
    if mins < 60:
        return f"{mins}m {secs}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h {mins}m"


@dataclass
class ProcessSession:
    """A tracked background process with output buffering."""
    id: str                                     # Unique session ID ("proc_xxxxxxxxxxxx")
    command: str                                 # Original command string
    task_id: str = ""                           # Task/sandbox isolation key
    session_key: str = ""                       # Gateway session key (for reset protection)
    pid: Optional[int] = None                   # OS process ID
    process: Optional[subprocess.Popen] = None  # Popen handle (local only)
    env_ref: Any = None                         # Reference to the environment object
    cwd: Optional[str] = None                   # Working directory
    started_at: float = 0.0                     # time.time() of spawn (wall clock)
    host_start_time: Optional[int] = None       # kernel start ticks (/proc/<pid>/stat f22) — PID-reuse guard
    exited: bool = False                        # Whether the process has finished
    exit_code: Optional[int] = None             # Exit code (None if still running)
    completion_reason: str = "exited"           # exited|killed|lost|failed_start|already_exited
    termination_source: str = ""                # process.kill|kill_all|backend_lost|failed_start
    output_buffer: str = ""                     # Rolling output (last MAX_OUTPUT_CHARS)
    max_output_chars: int = MAX_OUTPUT_CHARS
    detached: bool = False                      # True if recovered from crash (no pipe)
    pid_scope: str = "host"                     # "host" for local/PTY PIDs, "sandbox" for env-local PIDs
    # Watcher/notification metadata (persisted for crash recovery)
    watcher_platform: str = ""
    watcher_chat_id: str = ""
    watcher_user_id: str = ""
    watcher_user_name: str = ""
    watcher_thread_id: str = ""
    watcher_message_id: str = ""                # Triggering message id — reply anchor for topic routing
    watcher_interval: int = 0                   # 0 = no watcher configured
    notify_on_complete: bool = False             # Queue agent notification on exit
    # Watch patterns — trigger agent notification when output matches any pattern
    watch_patterns: List[str] = field(default_factory=list)
    _watch_hits: int = field(default=0, repr=False)          # total matches delivered
    _watch_suppressed: int = field(default=0, repr=False)    # matches dropped by rate limit
    _watch_disabled: bool = field(default=False, repr=False) # permanently killed after strike limit
    # Per-session rate limit state: at most one match every WATCH_MIN_INTERVAL_SECONDS.
    # When an emission happens, _watch_cooldown_until is set to now + interval and
    # _watch_strike_candidate becomes True. The next match to arrive before that
    # deadline counts as one strike (regardless of how many matches were dropped in
    # between — a strike is a window, not a match). After WATCH_STRIKE_LIMIT strikes
    # in a row, watch_patterns is disabled and the session promotes to
    # notify_on_complete.
    _watch_last_emit_at: float = field(default=0.0, repr=False)
    _watch_cooldown_until: float = field(default=0.0, repr=False)
    _watch_strike_candidate: bool = field(default=False, repr=False)
    _watch_consecutive_strikes: int = field(default=0, repr=False)
    _completion_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _reader_thread: Optional[threading.Thread] = field(default=None, repr=False)
    _pty: Any = field(default=None, repr=False)  # ptyprocess handle (when use_pty=True)


class ProcessRegistry:
    """
    In-memory registry of running and finished background processes.

    Thread-safe. Accessed from:
      - Executor threads (terminal_tool, process tool handlers)
      - Gateway asyncio loop (watcher tasks, session reset checks)
      - Cleanup thread (sandbox reaping coordination)
    """

    _SHELL_NOISE_SUBSTRINGS = (
        "bash: cannot set terminal process group",
        "bash: no job control in this shell",
        "no job control in this shell",
        "cannot set terminal process group",
        "tcsetattr: Inappropriate ioctl for device",
    )

    def __init__(self):
        self._running: Dict[str, ProcessSession] = {}
        self._finished: Dict[str, ProcessSession] = {}
        self._lock = threading.Lock()

        # Side-channel for check_interval watchers (gateway reads after agent run)
        self.pending_watchers: List[Dict[str, Any]] = []

        # Notification queue — unified queue for all background process events.
        # Completion notifications (notify_on_complete) and watch pattern matches
        # both land here, distinguished by "type" field.  CLI process_loop and
        # gateway drain this after each agent turn to auto-trigger new turns.
        import queue as _queue_mod
        self.completion_queue: _queue_mod.Queue = _queue_mod.Queue()

        # Track sessions whose completion was already consumed by the agent
        # via wait/log.  Drain loops AND gateway/tui watchers skip notifications
        # for these — a blocking wait() or a full read_log() means the agent
        # has the output in hand and is acting on it this turn.
        self._completion_consumed: set = set()

        # Track sessions the agent merely *observed* exited via poll().  poll()
        # is a read-only status check, so it does NOT mark _completion_consumed
        # (that would let a status check suppress the gateway/tui watcher's
        # autonomous delivery turn — #10156).  But on the CLI the poll result
        # is returned inline in the same turn, so the idle/post-turn drain must
        # still skip the queued completion to avoid a duplicate [SYSTEM: ...]
        # injection (the bug #8228 originally fixed).  drain_notifications()
        # consults this set; the gateway/tui watchers deliberately do NOT.
        self._poll_observed: set = set()

        # Global watch-match circuit breaker — across all sessions.
        # Prevents sibling processes from collectively flooding the user even
        # when each stays under its own per-session cap.
        self._global_watch_lock = threading.Lock()
        self._global_watch_window_start: float = 0.0
        self._global_watch_window_hits: int = 0
        self._global_watch_tripped_until: float = 0.0
        self._global_watch_suppressed_during_trip: int = 0
        # Live-output sink set by a driver (e.g. the desktop gateway): called from
        # reader threads with (session, chunk) to stream output to a UI in
        # real time, instead of polling the output tail.
        self.on_output = None
        # Close-view sink set by a driver (desktop gateway): called with
        # (session_or_none, process_id) when the agent asks to close a read-only
        # terminal tab. Distinct from kill — the process keeps running; only the
        # UI view is dropped (the user can reopen it from the status stack).
        self.on_close = None

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

        if return_early:
            if should_disable:
                # Emit exactly one "watch disabled, falling back to notify_on_complete"
                # summary event so the agent/user sees why things went quiet.
                self.completion_queue.put({
                    "session_id": session.id,
                    "session_key": session.session_key,
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
            return

        self.completion_queue.put({
            "session_id": session.id,
            "session_key": session.session_key,
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
                    text=True,
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
    ) -> ProcessSession:
        """
        Spawn a background process locally.

        Only for TERMINAL_ENV=local. Other backends use spawn_via_env().

        Args:
            use_pty: If True, use a pseudo-terminal via ptyprocess for interactive
                     CLI tools (Codex, Claude Code, Python REPL). Falls back to
                     subprocess.Popen if ptyprocess is not installed.
        """
        session = ProcessSession(
            id=f"proc_{uuid.uuid4().hex[:12]}",
            command=command,
            task_id=task_id,
            session_key=session_key,
            cwd=_resolve_safe_cwd(cwd or os.getcwd()),
            started_at=time.time(),
        )

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
                pty_proc = _PtyProcessCls.spawn(
                    [user_shell, "-lic", f"set +m; {command}"],
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

        proc = subprocess.Popen(
            [user_shell, "-lic", f"set +m; {command}"],
            text=True,
            cwd=session.cwd,
            env=bg_env,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
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
                if not _IS_WINDOWS:
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
        """
        first_chunk = True
        try:
            stdout = session.process.stdout
            if stdout is None:
                return

            raw_read = getattr(getattr(stdout, "buffer", None), "read1", None)
            while True:
                if raw_read is not None:
                    raw = raw_read(4096)
                    if not raw:
                        break
                    chunk = raw.decode("utf-8", errors="replace")
                else:
                    # Fallback for mocked/alternate streams without a buffered raw
                    # interface. This may be less "live", but keeps compatibility.
                    chunk = stdout.read(4096)
                    if not chunk:
                        break

                if first_chunk:
                    chunk = self._clean_shell_noise(chunk)
                    first_chunk = False
                with session._lock:
                    session.output_buffer += chunk
                    if len(session.output_buffer) > session.max_output_chars:
                        session.output_buffer = session.output_buffer[-session.max_output_chars:]
                self._check_watch_patterns(session, chunk)
                self._emit_output(session, chunk)
        except Exception as e:
            logger.debug("Process stdout reader ended: %s", e)
        finally:
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

    def _env_poller_loop(
        self, session: ProcessSession, env: Any, log_path: str, pid_path: str, exit_path: str
    ):
        """Background thread: poll a sandbox log file for non-local backends."""
        quoted_log_path = shlex.quote(log_path)
        quoted_pid_path = shlex.quote(pid_path)
        quoted_exit_path = shlex.quote(exit_path)
        prev_output_len = 0  # track delta for watch pattern scanning
        while not session.exited:
            time.sleep(2)  # Poll every 2 seconds
            try:
                # Read new output from the log file
                result = env.execute(f"cat {quoted_log_path} 2>/dev/null", timeout=10)
                new_output = result.get("output", "")
                if new_output:
                    # Compute delta for watch pattern scanning
                    delta = new_output[prev_output_len:] if len(new_output) > prev_output_len else ""
                    prev_output_len = len(new_output)
                    with session._lock:
                        session.output_buffer = new_output
                        if len(session.output_buffer) > session.max_output_chars:
                            session.output_buffer = session.output_buffer[-session.max_output_chars:]
                    if delta:
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
        try:
            while pty.isalive():
                try:
                    chunk = pty.read(4096)
                    if chunk:
                        # ptyprocess returns bytes
                        text = chunk if isinstance(chunk, str) else chunk.decode("utf-8", errors="replace")
                        with session._lock:
                            session.output_buffer += text
                            if len(session.output_buffer) > session.max_output_chars:
                                session.output_buffer = session.output_buffer[-session.max_output_chars:]
                        self._check_watch_patterns(session, text)
                        self._emit_output(session, text)
                except EOFError:
                    break
                except Exception:
                    break
        except Exception as e:
            logger.debug("PTY stdout reader ended: %s", e)

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
            self.completion_queue.put({
                "type": "completion",
                "session_id": session.id,
                "session_key": session.session_key,
                "command": session.command,
                "exit_code": session.exit_code,
                "completion_reason": session.completion_reason,
                "termination_source": session.termination_source,
                "output": output_tail,
            })

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

    def _drain_should_skip(self, session_id: str) -> bool:
        """Whether the CLI drain should skip a completion event for this session.

        Skips when the agent has either truly consumed the output (wait/log →
        ``_completion_consumed``) or observed the exit inline via poll()
        (``_poll_observed``).  In both cases the CLI agent already has the
        result this turn, so injecting a [SYSTEM: ...] completion would be a
        duplicate (#8228).  The gateway/tui watchers do NOT use this — they
        check only ``is_completion_consumed`` so a read-only poll never
        suppresses their autonomous delivery turn (#10156).
        """
        return session_id in self._completion_consumed or session_id in self._poll_observed

    def drain_notifications(self) -> "list[tuple[dict, str]]":
        """Pop all pending notification events and return formatted pairs.

        Returns a list of (raw_event, formatted_text) tuples.
        Skips completion events the agent already consumed via wait/log or
        observed inline via poll() (see ``_drain_should_skip``).
        """
        results = []
        while not self.completion_queue.empty():
            try:
                evt = self.completion_queue.get_nowait()
            except Exception:
                break
            _evt_sid = evt.get("session_id", "")
            if evt.get("type") == "completion" and self._drain_should_skip(_evt_sid):
                continue
            text = format_process_notification(evt)
            if text:
                results.append((evt, text))
        return results

    def get(self, session_id: str) -> Optional[ProcessSession]:
        """Get a session by ID (running or finished)."""
        with self._lock:
            session = self._running.get(session_id) or self._finished.get(session_id)
        return self._refresh_detached_session(session)

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

    def read_log(self, session_id: str, offset: int = 0, limit: int = 200) -> dict:
        """Read the full output log with optional pagination by lines."""
        from tools.ansi_strip import strip_ansi

        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}

        with session._lock:
            full_output = strip_ansi(session.output_buffer)

        lines = full_output.splitlines()
        total_lines = len(lines)

        # Default: last N lines
        if offset == 0 and limit > 0:
            selected = lines[-limit:]
        else:
            selected = lines[offset:offset + limit]

        result = {
            "session_id": session.id,
            "command": session.command,
            "status": "exited" if session.exited else "running",
            "output": "\n".join(selected),
            "total_lines": total_lines,
            "showing": f"{len(selected)} lines",
        }
        if session.exited:
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
        }
        if timeout_note:
            result["timeout_note"] = timeout_note
        else:
            result["timeout_note"] = f"Waited {effective_timeout}s, process still running"
        return result

    def kill_process(self, session_id: str, *, source: str = "process.kill") -> dict:
        """Kill a background process."""
        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}

        if session.exited:
            return {
                "status": "already_exited",
                "exit_code": session.exit_code,
            }

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
                # Local process -- kill the process tree
                try:
                    if _IS_WINDOWS:
                        session.process.terminate()
                    else:
                        import psutil
                        try:
                            parent = psutil.Process(session.process.pid)
                            for child in parent.children(recursive=True):
                                try:
                                    child.terminate()
                                except psutil.NoSuchProcess:
                                    pass
                            parent.terminate()
                        except psutil.NoSuchProcess:
                            pass
                except (ProcessLookupError, PermissionError):
                    session.process.kill()
            elif session.env_ref and session.pid:
                # Non-local -- kill inside sandbox
                session.env_ref.execute(f"kill {session.pid} 2>/dev/null", timeout=5)
            elif session.detached and session.pid_scope == "host" and session.pid:
                # Identity check, not bare liveness: if the PID is gone OR was
                # recycled onto an unrelated process, treat our process as
                # exited and never tree-kill the stranger.
                if not self._host_pid_is_ours(session.pid, session.host_start_time):
                    with session._lock:
                        session.exited = True
                        session.exit_code = None
                    self._move_to_finished(session)
                    return {
                        "status": "already_exited",
                        "exit_code": session.exit_code,
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
                    pty_data = data.encode("utf-8") if isinstance(data, str) else data
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
        """Send data + newline to a running process's stdin (like pressing Enter)."""
        return self.write_stdin(session_id, data + "\n")

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

    def kill_all(self, task_id: str = None) -> int:
        """Kill all running processes, optionally filtered by task_id. Returns count killed."""
        with self._lock:
            targets = [
                s for s in self._running.values()
                if (task_id is None or s.task_id == task_id) and not s.exited
            ]

        killed = 0
        for session in targets:
            result = self.kill_process(session.id, source="kill_all")
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

    def _write_checkpoint(self):
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
                            "command": s.command,
                            "pid": s.pid,
                            "pid_scope": s.pid_scope,
                            "host_start_time": s.host_start_time,
                            "cwd": s.cwd,
                            "started_at": s.started_at,
                            "task_id": s.task_id,
                            "session_key": s.session_key,
                            "watcher_platform": s.watcher_platform,
                            "watcher_chat_id": s.watcher_chat_id,
                            "watcher_user_id": s.watcher_user_id,
                            "watcher_user_name": s.watcher_user_name,
                            "watcher_thread_id": s.watcher_thread_id,
                            "watcher_message_id": s.watcher_message_id,
                            "watcher_interval": s.watcher_interval,
                            "notify_on_complete": s.notify_on_complete,
                            "watch_patterns": s.watch_patterns,
                        })
            
            # Atomic write to avoid corruption on crash
            from utils import atomic_json_write
            atomic_json_write(CHECKPOINT_PATH, entries)
        except Exception as e:
            logger.debug("Failed to write checkpoint file: %s", e, exc_info=True)

    def recover_from_checkpoint(self) -> int:
        """
        On gateway startup, probe PIDs from checkpoint file.

        Returns the number of processes recovered as detached.
        """
        if not CHECKPOINT_PATH.exists():
            return 0

        try:
            entries = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        except Exception:
            return 0

        recovered = 0
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
                continue

            session = ProcessSession(
                id=entry["session_id"],
                command=entry.get("command", "unknown"),
                task_id=entry.get("task_id", ""),
                session_key=entry.get("session_key", ""),
                pid=pid,
                host_start_time=recorded_start,
                pid_scope=pid_scope,
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
                })

        self._write_checkpoint()

        return recovered


# Module-level singleton
process_registry = ProcessRegistry()


def _format_age(seconds: float) -> str:
    """Human-friendly elapsed string ('18m', '2h3m', '45s')."""
    try:
        s = int(max(0, seconds))
    except (TypeError, ValueError):
        return "?"
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m" if s == 0 else f"{m}m{s}s"
    h, m = divmod(m, 60)
    return f"{h}h" if m == 0 else f"{h}h{m}m"


def _format_async_delegation(evt: dict) -> str:
    """Format an async-delegation completion into a self-contained re-injection.

    Carries the FULL original task source (goal, the context the parent
    supplied, toolsets, role, model) plus dispatch time, status, and the
    complete result summary. When this re-enters the conversation the agent
    may be deep in unrelated context and won't remember why the subagent
    existed, so the block is written to stand entirely on its own — enough to
    use the result OR re-dispatch if the world has moved on.
    """
    import time as _time

    deleg_id = evt.get("delegation_id", "unknown")
    goal = evt.get("goal", "") or ""
    context = evt.get("context")
    toolsets = evt.get("toolsets")
    role = evt.get("role") or "leaf"
    model = evt.get("model") or "?"
    status = evt.get("status") or "completed"
    summary = evt.get("summary")
    error = evt.get("error")
    api_calls = evt.get("api_calls", 0)
    duration = evt.get("duration_seconds", "?")
    dispatched_at = evt.get("dispatched_at")
    completed_at = evt.get("completed_at") or _time.time()

    # ----- Batch (fan-out) completion: consolidated multi-task block -----
    # A whole delegate_task fan-out dispatched as one background unit finishes
    # together and carries a per-task `results` list. Render every subagent's
    # summary in one block so the model gets the consolidated outcome at once.
    batch_results = evt.get("results")
    if evt.get("is_batch") or isinstance(batch_results, list):
        results = batch_results or []
        goals = evt.get("goals") or []
        n = len(results) if results else len(goals)
        total_dur = evt.get("total_duration_seconds", duration)
        lines = [
            f"[ASYNC DELEGATION BATCH COMPLETE — {deleg_id}]",
            f"A background fan-out of {n} subagent(s) you dispatched earlier "
            "has finished. All ran in parallel and waited on each other; their "
            "consolidated results are below. You may have moved on since "
            "dispatching — act on these or re-dispatch if things have changed.",
            "",
        ]
        if isinstance(dispatched_at, (int, float)):
            ts = _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(dispatched_at))
            age = f" ({_format_age(completed_at - dispatched_at)} ago)"
            lines.append(f"Dispatched: {ts}{age}")
        if context:
            lines.append(f"Context you provided: {context}")
        if toolsets:
            lines.append(f"Toolsets: {', '.join(toolsets)}")
        lines.append(f"Role: {role}   Model: {model}   Total duration: {total_dur}s")
        if error and not results:
            lines.append("--- ERROR ---")
            lines.append(f"The batch did not complete successfully: {error}")
            return "\n".join(lines)
        for r in sorted(results, key=lambda x: x.get("task_index", 0)):
            idx = r.get("task_index", 0)
            r_status = r.get("status", "?")
            r_summary = r.get("summary")
            r_error = r.get("error")
            r_goal = goals[idx] if idx < len(goals) else r.get("goal", "")
            icon = "✓" if r_status in ("completed", "success") else "✗"
            lines.append("")
            header = f"--- {icon} TASK {idx + 1}/{n}"
            if r_goal:
                header += f": {r_goal}"
            header += f"  (status={r_status}"
            if r.get("api_calls"):
                header += f", api_calls={r['api_calls']}"
            if r.get("duration_seconds") is not None:
                header += f", {r['duration_seconds']}s"
            header += ") ---"
            lines.append(header)
            if r_status in ("completed", "success") and r_summary:
                lines.append(r_summary)
            elif r_summary:
                if r_error:
                    lines.append(f"({r_status}: {r_error})")
                lines.append("Partial output:")
                lines.append(r_summary)
            else:
                lines.append(
                    f"(no summary — status={r_status}"
                    + (f": {r_error}" if r_error else "")
                    + ")"
                )
        return "\n".join(lines)

    age = ""
    if isinstance(dispatched_at, (int, float)):
        age = f" ({_format_age(completed_at - dispatched_at)} ago)"

    lines = [
        f"[ASYNC DELEGATION COMPLETE — {deleg_id}]",
        "A background subagent you dispatched earlier has finished. You may "
        "have moved on since dispatching it; the full task source is below so "
        "you can act on the result or re-dispatch if things have changed.",
        "",
    ]
    if isinstance(dispatched_at, (int, float)):
        ts = _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(dispatched_at))
        lines.append(f"Dispatched: {ts}{age}")
    lines.append(f"Original goal: {goal}")
    if context:
        lines.append(f"Context you provided: {context}")
    if toolsets:
        lines.append(f"Toolsets: {', '.join(toolsets)}")
    lines.append(f"Role: {role}   Model: {model}")
    lines.append(f"Status: {status}   API calls: {api_calls}   Duration: {duration}s")
    lines.append("--- RESULT ---")
    if status in ("completed", "success") and summary:
        lines.append(summary)
    elif status == "interrupted":
        lines.append(
            "The subagent was interrupted before completing"
            + (f": {error}" if error else ".")
        )
        if summary:
            lines.append("Partial output:")
            lines.append(summary)
    else:
        # error / timeout / failed
        lines.append(
            f"The subagent did not complete successfully (status={status})."
            + (f"\n{error}" if error else "")
        )
        if summary:
            lines.append("Partial output:")
            lines.append(summary)
    return "\n".join(lines)


def format_process_notification(evt: dict) -> "str | None":
    """Format a process notification event into a [IMPORTANT: ...] message.

    Handles completion events (notify_on_complete), watch pattern matches,
    and watch disabled events from the unified completion_queue.
    """
    evt_type = evt.get("type", "completion")
    _sid = evt.get("session_id", "unknown")
    _cmd = evt.get("command", "unknown")

    if evt_type == "watch_disabled":
        return f"[IMPORTANT: {evt.get('message', '')}]"

    if evt_type == "watch_match":
        _pat = evt.get("pattern", "?")
        _out = evt.get("output", "")
        _sup = evt.get("suppressed", 0)
        text = (
            f"[IMPORTANT: Background process {_sid} matched "
            f"watch pattern \"{_pat}\".\n"
            f"Command: {_cmd}\n"
            f"Matched output:\n{_out}"
        )
        if _sup:
            text += f"\n({_sup} earlier matches were suppressed by rate limit)"
        text += "]"
        return text

    if evt_type == "async_delegation":
        return _format_async_delegation(evt)

    _exit = evt.get("exit_code", "?")
    _out = evt.get("output", "")
    _reason = evt.get("completion_reason") or "exited"
    _source = evt.get("termination_source") or ""
    _signal = ""
    if _exit in {-15, 143, "-15", "143"}:
        _signal = ", SIGTERM"
    if _reason == "killed":
        _status = f"terminated by {_source or 'Hermes'}"
    elif _reason == "lost":
        _status = "marked lost because the process backend disappeared"
    elif _reason == "failed_start":
        _status = "failed to start"
    elif _exit == 0:
        _status = "completed normally"
    else:
        _status = "exited"
    return (
        f"[IMPORTANT: Background process {_sid} {_status} "
        f"(exit code {_exit}{_signal}).\n"
        f"Command: {_cmd}\n"
        f"Output:\n{_out}]"
    )


# ---------------------------------------------------------------------------
# Registry -- the "process" tool schema + handler
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error

PROCESS_SCHEMA = {
    "name": "process",
    "description": (
        "Manage background processes started with terminal(background=true). "
        "Actions: 'list' (show all), 'poll' (check status + new output), "
        "'log' (full output with pagination), 'wait' (block until done or timeout), "
        "'kill' (terminate), 'write' (send raw stdin data without newline), "
        "'submit' (send data + Enter, for answering prompts), 'close' (close stdin/send EOF)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "poll", "log", "wait", "kill", "write", "submit", "close"],
                "description": "Action to perform on background processes"
            },
            "session_id": {
                "type": "string",
                "description": "Process session ID (from terminal background output). Required for all actions except 'list'."
            },
            "data": {
                "type": "string",
                "description": "Text to send to process stdin (for 'write' and 'submit' actions)"
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds to block for 'wait' action. Returns partial output on timeout.",
                "minimum": 1
            },
            "offset": {
                "type": "integer",
                "description": "Line offset for 'log' action (default: last 200 lines)"
            },
            "limit": {
                "type": "integer",
                "description": "Max lines to return for 'log' action",
                "minimum": 1
            }
        },
        "required": ["action"]
    }
}


def _redact_process_result(result: dict) -> dict:
    """Redact secrets from background-process output before it reaches the
    model, session.db, and CLI display.

    Mirrors the foreground ``terminal`` redaction (terminal_tool.py) so the
    two surfaces can't diverge — issue #43025 (background output was returned
    verbatim). Respects ``security.redact_secrets`` (no force): output fields
    pass through ``redact_terminal_output`` which picks ``code_file`` based on
    the recorded command (env dumps get the ENV-assignment pass). The command
    string itself is also redacted in case it carried an inline credential.
    """
    if not isinstance(result, dict):
        return result
    from agent.redact import redact_sensitive_text, redact_terminal_output

    command = result.get("command") or ""
    for field in ("output", "output_preview"):
        value = result.get(field)
        if isinstance(value, str) and value:
            result[field] = redact_terminal_output(value, command)
    if isinstance(result.get("command"), str) and result["command"]:
        result["command"] = redact_sensitive_text(result["command"], code_file=True)
    return result


def _handle_process(args, **kw):
    task_id = kw.get("task_id")
    action = args.get("action", "")
    # Coerce to string — some models send session_id as an integer
    session_id = str(args.get("session_id", "")) if args.get("session_id") is not None else ""

    if action == "list":
        # Surface session-scoped background processes (e.g. a forgotten
        # preview server) in addition to this task's own — they share the
        # gateway session_key and can block session reset (#29177).
        try:
            from tools.approval import get_current_session_key
            session_key = get_current_session_key(default="") or ""
        except Exception:
            session_key = ""
        return json.dumps(
            {"processes": process_registry.list_sessions(task_id=task_id, session_key=session_key or None)},
            ensure_ascii=False,
        )
    elif action in {"poll", "log", "wait", "kill", "write", "submit", "close"}:
        if not session_id:
            return tool_error(f"session_id is required for {action}")
        if action == "poll":
            return json.dumps(_redact_process_result(process_registry.poll(session_id)), ensure_ascii=False)
        elif action == "log":
            return json.dumps(_redact_process_result(process_registry.read_log(
                session_id, offset=args.get("offset", 0), limit=args.get("limit", 200))), ensure_ascii=False)
        elif action == "wait":
            return json.dumps(_redact_process_result(process_registry.wait(session_id, timeout=args.get("timeout"))), ensure_ascii=False)
        elif action == "kill":
            return json.dumps(process_registry.kill_process(session_id), ensure_ascii=False)
        elif action == "write":
            return json.dumps(process_registry.write_stdin(session_id, str(args.get("data", ""))), ensure_ascii=False)
        elif action == "submit":
            return json.dumps(process_registry.submit_stdin(session_id, str(args.get("data", ""))), ensure_ascii=False)
        elif action == "close":
            return json.dumps(process_registry.close_stdin(session_id), ensure_ascii=False)
    return tool_error(f"Unknown process action: {action}. Use: list, poll, log, wait, kill, write, submit, close")


registry.register(
    name="process",
    toolset="terminal",
    schema=PROCESS_SCHEMA,
    handler=_handle_process,
    emoji="⚙️",
)
