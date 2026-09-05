"""Process Registry -- in-memory registry for background processes spawned via
terminal(background=true): rolling 200KB output buffer, poll/log/wait/kill, JSON
checkpoint for crash recovery, session-scoped tracking for gateway reset protection.
Nothing runs on the host unless TERMINAL_ENV=local; other backends run in their sandbox.
"""

import codecs
from contextlib import contextmanager, suppress
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
from pathlib import Path

_IS_WINDOWS = platform.system() == "Windows"
# systemd transient scopes exist only on Linux; gate every scope-path branch on this
# (not merely "not Windows") so macOS and other POSIX platforms never touch systemd.
# See #70716.
_IS_LINUX = platform.system() == "Linux"
from tools.environments.local import _find_shell, _resolve_safe_cwd, _sanitize_subprocess_env
from hermes_cli._subprocess_compat import windows_hide_flags
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hermes_cli.config import get_hermes_home

from agent.redact import redact_sensitive_text
from tools.process_registry_notifications import format_process_notification

logger = logging.getLogger(__name__)

# Crash-recovery checkpoint (gateway only)
CHECKPOINT_PATH = get_hermes_home() / "processes.json"

MAX_OUTPUT_CHARS = 200_000      # rolling output buffer
FINISHED_TTL_SECONDS = 1800     # keep finished processes 30 minutes
MAX_PROCESSES = 64              # max tracked processes (LRU pruning)

# Output tail published in the checkpoint for OTHER gateway processes to render
# (messaging gateway spawns the job, Desktop shows it). Bounded well below
# MAX_OUTPUT_CHARS: this is a display preview rewritten on disk while the process
# runs, not the authoritative log, and it is what process.list already ships.
CHECKPOINT_OUTPUT_TAIL_CHARS = 4_000
# Extra left context redacted BEFORE the published tail is cut out of it. Slicing
# first and redacting after decapitates any secret straddling the cut
# (``DB_PASSWORD=`` on one side, the value on the other): the pattern no longer
# matches and the value ships in the clear. Redacting a window this much wider
# means the cut can only ever land inside text that was already scanned with full
# left context — orders of magnitude more than the longest credential.
CHECKPOINT_REDACT_CONTEXT_CHARS = 12_000
# Floor between two output-driven checkpoint rewrites. A chunk landing inside the
# window arms one trailing write, so a burst costs two rewrites instead of one
# per chunk and the last line is still published.
CHECKPOINT_OUTPUT_MIN_INTERVAL_SECONDS = 1.0

# Watch-pattern rate limiting, PER SESSION: one watch-match notification per
# WATCH_MIN_INTERVAL_SECONDS; a match inside the cooldown is dropped and counts as one
# strike per window; WATCH_STRIKE_LIMIT consecutive strike windows permanently disable
# watching and fall back to notify_on_complete semantics.
WATCH_MIN_INTERVAL_SECONDS = 15
WATCH_STRIKE_LIMIT = 3
# Lifetime cap, independent of strikes: a pattern recurring just above the cooldown never
# strikes yet forces a full-context agent turn each time; watch_patterns is "ONLY for
# rare one-shot signals", so after this many deliveries fall back to notify_on_complete.
# MAX_ACTIVE_PROCESS_AGE = 86400  # 24h default — see session_reset.bg_process_max_age_hours (#29177)
# A process whose pattern recurs at a cadence just above WATCH_MIN_INTERVAL_SECONDS (e.g. a service
# restarted repeatedly over a day) never trips the consecutive-strike limit, since each match lands in its
# own clean cooldown window, yet still forces a full-context agent turn every single time (#93513).
# watch_patterns is documented as "ONLY for rare one-shot mid-process signals", so once a session has
# delivered this many matches over its whole life we disable it and fall back to notify_on_complete, same as
# the strike-limit path.
WATCH_LIFETIME_MAX_HITS = 8
# Global circuit breaker across all sessions so concurrent siblings can't collectively
# flood the user even when each is under its own cap.
WATCH_GLOBAL_MAX_PER_WINDOW = 15
WATCH_GLOBAL_WINDOW_SECONDS = 10
WATCH_GLOBAL_COOLDOWN_SECONDS = 30


# --- systemd cgroup isolation for gateway-spawned local executors ------------------
# Under a systemd gateway with MemoryMax, local background commands inherit the gateway's
# cgroup, so a memory-heavy executor can get the ENTIRE gateway killed by systemd-oomd;
# ``systemd-run --user --scope`` gives the worker its own transient cgroup. Usability is
# probed once (binary present but user D-Bus absent in system services/containers).
# A memory-heavy executor (Codex, tests, Node) can push the whole cgroup past MemoryMax and trigger
# systemd-oomd to kill the ENTIRE gateway — taking down the messaging control plane and silently losing the
# active turn. We probe *once* whether ``systemd-run --user --scope`` is actually usable (the binary can
# exist on the PATH while the user D-Bus session is unavailable — common for system services and
# containers), and cache the result for the process lifetime. See #70716.
_SYSTEMD_SCOPE_AVAILABLE: Optional[bool] = None
_SYSTEMD_SCOPE_PROBE_LOCK = threading.Lock()
_SYSTEMD_SCOPE_PROBED_AT = 0.0
_SYSTEMD_SCOPE_FAILURE_TTL_SECONDS = 60.0
_MIN_WORKER_MEMORY_MAX_BYTES = 64 * 1024 * 1024
_DEFAULT_WORKER_MEMORY_MAX_BYTES = 1024 * 1024 * 1024
_WORKER_MEMORY_MAX_CAP_BYTES = 4 * 1024 * 1024 * 1024


def _worker_memory_max_bytes() -> int:
    """Finite per-worker cgroup limit that can never widen host risk.
    ``TERMINAL_LOCAL_MEMORY_MAX_MB`` is honored only when it *tightens* the safe
    bound (min of the gateway's cgroup-v2 ``memory.max`` and half of physical RAM,
    capped at 4 GiB), so an oversized override cannot exceed the enclosing slice.

    The proposed local-memory-guard environment override is honored when it tightens the safe bound, so this
    isolation composes with PR #57121 instead of inventing a second knob.
    """
    override_bound: Optional[int] = None
    override = os.getenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "").strip()
    if override:
        try:
            parsed = int(override) * 1024 * 1024
        except ValueError:
            parsed = -1
        if parsed >= _MIN_WORKER_MEMORY_MAX_BYTES:
            override_bound = parsed
        else:
            logger.warning(
                "Ignoring invalid TERMINAL_LOCAL_MEMORY_MAX_MB=%r; "
                "expected an integer representing at least %d MiB",
                override, _MIN_WORKER_MEMORY_MAX_BYTES // (1024 * 1024))
    candidates: List[int] = []
    with suppress(OSError, ValueError):
        lines = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
        v2 = next((ln for ln in lines if ln.startswith("0::")), None)
        if v2 is not None:
            relative = v2.partition("::")[2].lstrip("/")
            raw_limit = (Path("/sys/fs/cgroup") / relative / "memory.max").read_text(encoding="utf-8").strip()
            if raw_limit.isdigit() and int(raw_limit) >= _MIN_WORKER_MEMORY_MAX_BYTES:
                candidates.append(int(raw_limit))
    with suppress(OSError, ValueError, TypeError):
        physical_bytes = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
        candidates.append(min(_WORKER_MEMORY_MAX_CAP_BYTES, max(_MIN_WORKER_MEMORY_MAX_BYTES, physical_bytes // 2)))
    safe_bound = min(candidates) if candidates else _DEFAULT_WORKER_MEMORY_MAX_BYTES
    return min(override_bound, safe_bound) if override_bound else safe_bound


def _systemd_scope_argv(binary: str, unit_name: str, *argv: str) -> List[str]:
    """``systemd-run --user --scope`` argv shared by the probe and real spawns.
    ``--collect`` self-cleans the scope after exit; ``--unit`` names it for systemctl."""
    return [
        binary, "--user", "--scope", "--quiet", "--unit", unit_name, "--collect",
        "--property", "MemoryAccounting=yes",
        "--property", f"MemoryMax={_worker_memory_max_bytes()}",
        "--property", "OOMPolicy=kill",
        "--", *argv,
    ]


def _systemd_scope_cached() -> Optional[bool]:
    """Cached probe verdict, or None when a (re)probe is due. True is permanent; False
    expires after ``_SYSTEMD_SCOPE_FAILURE_TTL_SECONDS`` so a D-Bus blip isn't sticky."""
    if _SYSTEMD_SCOPE_AVAILABLE is True:
        return True
    stale = time.monotonic() - _SYSTEMD_SCOPE_PROBED_AT >= _SYSTEMD_SCOPE_FAILURE_TTL_SECONDS
    return None if _SYSTEMD_SCOPE_AVAILABLE is None or stale else False


def _systemd_run_user_scope_available() -> bool:
    """True if ``systemd-run --user --scope`` can create a cgroup.
    ``shutil.which`` alone is insufficient: system services and containers may lack
    the user D-Bus bus even with the binary on PATH (every spawn would fail with
    ``Failed to connect to user bus``), so a cheap ``/bin/true`` probe is run and cached."""
    global _SYSTEMD_SCOPE_AVAILABLE, _SYSTEMD_SCOPE_PROBED_AT
    verdict = _systemd_scope_cached()
    if verdict is not None:
        return verdict
    # Double-checked locking: a concurrent first-use spawn must not observe a temporary
    # False mid-probe, or it would launch back inside the gateway cgroup.
    with _SYSTEMD_SCOPE_PROBE_LOCK:
        verdict = _systemd_scope_cached()
        if verdict is not None:
            return verdict
        available = False
        if _IS_LINUX:
            try:
                import shutil

                binary = shutil.which("systemd-run")
                if binary:
                    # Unique unit avoids collisions; the timeout bounds D-Bus.
                    probe_unit = f"hermes-probe-scope-{os.getpid()}-{uuid.uuid4().hex[:8]}"
                    result = subprocess.run(
                        _systemd_scope_argv(binary, probe_unit, "/bin/true"), capture_output=True, timeout=3,
                    )
                    available = result.returncode == 0
                    if not available:
                        logger.debug(
                            "systemd-run --user --scope probe failed (rc=%s): %s",
                            result.returncode, (result.stderr or b"").decode("utf-8", "replace").strip(),
                        )
            except Exception as exc:
                logger.debug("systemd-run --user --scope probe error: %s", exc)
        _SYSTEMD_SCOPE_AVAILABLE = available
        _SYSTEMD_SCOPE_PROBED_AT = time.monotonic()
        return available


def _is_supervised_gateway_process() -> bool:
    """Whether this process is the live, supervised Hermes gateway itself.
    Supervisor markers and ``_HERMES_GATEWAY`` are inherited by every descendant (and
    importing ``gateway.run`` sets the latter), so also require ownership of the live
    gateway PID file — scopes are for the gateway, not terminal children or CLIs."""
    if os.environ.get("_HERMES_GATEWAY") != "1":
        return False
    try:
        from gateway.restart import is_gateway_supervisor_process
        from gateway.status import get_running_pid

        return is_gateway_supervisor_process() and get_running_pid(cleanup_stale=False) == os.getpid()
    except Exception as exc:
        logger.debug("Could not verify supervised gateway process identity: %s", exc)
        return False


def _build_systemd_scope_argv(shell_argv: List[str], unit_suffix: str) -> List[str]:
    """Wrap *shell_argv* in a ``systemd-run --user --scope`` invocation with its own
    memory accounting, so an OOM in the worker cannot kill the gateway cgroup.

    ``--collect`` makes the transient scope self-clean after exit; ``--unit`` gives it a recognisable name
    for ``systemctl --user status`` / journalctl. See #70716.
    """
    import shutil

    binary = shutil.which("systemd-run")
    if binary is None:
        # Caller should have probed availability; never pass None into Popen anyway.
        return shell_argv
    return _systemd_scope_argv(binary, f"hermes-worker-{unit_suffix}", *shell_argv)


def restart_safe_gateway_child_argv(
    command: List[str], *, unit_suffix: str
) -> List[str]:
    """Place a managed-systemd gateway child outside the gateway cgroup.

    Children that must survive an intentional gateway restart cannot rely on
    ``start_new_session`` alone: systemd still kills every process in the
    service cgroup.  In that topology, require a transient user scope and fail
    closed if it cannot be established.  Standalone processes, non-systemd
    supervisors, and non-Linux hosts retain the direct command.
    """
    if not _IS_LINUX:
        return command
    if not _is_supervised_gateway_process() or not os.environ.get("INVOCATION_ID"):
        return command
    if not _systemd_run_user_scope_available():
        raise RuntimeError(
            "cannot create restart-safe systemd scope for gateway child: "
            "systemd-run --user --scope is unavailable"
        )
    scoped = _build_systemd_scope_argv(command, unit_suffix=unit_suffix)
    if scoped == command:
        raise RuntimeError(
            "cannot create restart-safe systemd scope for gateway child: "
            "systemd-run disappeared after the availability probe"
        )
    return scoped


def _stop_systemd_unit(unit_name: str) -> bool:
    """Stop a transient systemd user scope by unit name.
    Reaps the *entire* cgroup — catching double-forked descendants reparented to init
    inside the scope that survive a plain PID signal (SIGTERM all, SIGKILL after
    ``TimeoutStopSec``). True if stopped or already gone; False if ``systemctl`` is
    unavailable or the stop failed.

    See #70716.
    """
    import shutil

    binary = shutil.which("systemctl")
    if binary is None:
        return False
    try:
        result = subprocess.run([binary, "--user", "stop", unit_name], capture_output=True, timeout=15,
                                stdin=subprocess.DEVNULL)
        if result.returncode != 0:
            stderr = (result.stderr or b"").decode(errors="replace").strip()
            if any(marker in stderr.lower() for marker in ("not loaded", "not found", "does not exist")):
                return True
            logger.debug("systemctl --user stop %s exited %d: %s", unit_name, result.returncode, stderr)
            return False
        return True
    except Exception as exc:
        logger.debug("systemctl --user stop %s failed: %s", unit_name, exc)
        return False


def format_uptime_short(seconds: int) -> str:
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    mins, secs = divmod(s, 60)
    if mins < 60:
        return f"{mins}m {secs}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h {mins}m"


def _flock(fh, *, lock: bool) -> None:
    """Exclusive whole-file lock/unlock on ``fh`` (fcntl on POSIX, msvcrt on Windows).
    Same primitive the other cross-process state files use (``gateway/status.py``,
    ``cron/scheduler.py``, ``hermes_cli/active_sessions.py``)."""
    if _IS_WINDOWS:
        import msvcrt
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK if lock else msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX if lock else fcntl.LOCK_UN)


@contextmanager
def _checkpoint_lock():
    """Hold the inter-process checkpoint lock; yields True when it is actually held.

    ``processes.json`` has one writer per gateway PROCESS, so an unsynchronized
    read-modify-write drops whatever another gateway published between our read and
    our replace. Callers therefore get an honest verdict instead of a silent
    downgrade: a WRITE must skip when this yields False (our own update is
    republished by the next write; another owner's row, once dropped, is gone), while
    a read may proceed — it destroys nothing, and ``atomic_json_write`` means the
    worst case is a slightly stale view rather than a torn one.
    """
    path = CHECKPOINT_PATH.with_name(CHECKPOINT_PATH.name + ".lock")
    handle = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "a+b")
        _flock(handle, lock=True)
    except Exception as exc:
        logger.warning("Process checkpoint lock unavailable: %s", exc)
        if handle is not None:
            with suppress(Exception):
                handle.close()
        handle = None
    try:
        yield handle is not None
    finally:
        if handle is not None:
            with suppress(Exception):
                _flock(handle, lock=False)
            with suppress(Exception):
                handle.close()


def _not_found(session_id: str) -> dict:
    return {"status": "not_found", "error": f"No process with ID {session_id}"}


def _output_tail(session: "ProcessSession", n: int) -> str:
    """Last *n* chars of the session output with ANSI sequences stripped."""
    from tools.ansi_strip import strip_ansi

    return strip_ansi(session.output_buffer[-n:])


def _checkpoint_output_tail(text: str) -> str:
    """Bounded, force-redacted output tail as published in the shared checkpoint.

    Redact FIRST over a window wider than what is published (see
    ``CHECKPOINT_REDACT_CONTEXT_CHARS``), then cut the tail out of the redacted
    text: every character that survives into the checkpoint was scanned with the
    full left context a pattern needs, so a credential sitting across the cut is
    masked instead of arriving headless and unmatched. The final slice bounds the
    result whatever a mask expanded to.
    """
    window = text[-(CHECKPOINT_OUTPUT_TAIL_CHARS + CHECKPOINT_REDACT_CONTEXT_CHARS):]
    return redact_sensitive_text(window, force=True, code_file=False)[-CHECKPOINT_OUTPUT_TAIL_CHARS:]


@dataclass
class ProcessSession:
    """A tracked background process with output buffering."""
    id: str                                     # "proc_xxxxxxxxxxxx"
    command: str
    task_id: str = ""                           # Task/sandbox isolation key (CONTAINER key,
                                                # may be collapsed by _resolve_container_task_id)
    owner_task_id: str = ""                     # RAW spawning task id ("sa-..."); ownership
                                                # checks must use this, not task_id
    session_key: str = ""                       # Gateway session key (reset protection)
    pid: Optional[int] = None
    process: Optional[subprocess.Popen] = None  # Popen handle (local only)
    env_ref: Any = None                         # Environment object (sandbox spawns)
    cwd: Optional[str] = None
    started_at: float = 0.0                     # time.time() of spawn
    host_start_time: Optional[int] = None       # kernel start ticks (/proc/<pid>/stat f22) — PID-reuse guard
    exited: bool = False
    exit_code: Optional[int] = None             # None while running
    completion_reason: str = "exited"           # exited|killed|lost|failed_start|already_exited
    termination_source: str = ""                # process.kill|kill_all|backend_lost|failed_start
    output_buffer: str = ""                     # Rolling tail (last max_output_chars)
    max_output_chars: int = MAX_OUTPUT_CHARS
    detached: bool = False                      # Recovered from checkpoint (no pipe)
    # Mirror of a process owned by ANOTHER gateway process, imported from the
    # shared checkpoint for display only: never signalled, never notified on,
    # never re-published (the owner is the sole author of its checkpoint entry).
    peer_mirror: bool = False
    pid_scope: str = "host"                     # "host" for local/PTY PIDs, "sandbox" for env-local PIDs
    systemd_unit: str = ""                      # transient scope unit name when spawned under systemd-run
    # Watcher/notification routing (persisted for crash recovery)
    # systemd_unit: str = ""                      # transient scope unit name when spawned under systemd-run
    # (#70716)
    watcher_platform: str = ""
    watcher_chat_id: str = ""
    watcher_user_id: str = ""
    watcher_user_name: str = ""
    watcher_thread_id: str = ""
    watcher_message_id: str = ""                # Triggering message id — reply anchor for topic routing
    watcher_interval: int = 0                   # 0 = no watcher configured
    # Session-db id of the spawning conversation; lets the gateway drop completions whose
    # session was closed at a user boundary (/new) instead of injecting into the NEW one.
    parent_session_id: str = ""
    notify_on_complete: bool = False            # Queue agent notification on exit
    watch_patterns: List[str] = field(default_factory=list)
    _watch_hits: int = field(default=0, repr=False)          # total matches delivered
    _watch_suppressed: int = field(default=0, repr=False)    # matches dropped by rate limit
    _watch_disabled: bool = field(default=False, repr=False) # permanently killed after strike limit
    # Rate-limit window state (see WATCH_*). A strike is a WINDOW with drops, not a drop.
    _watch_cooldown_until: float = field(default=0.0, repr=False)
    _watch_strike_candidate: bool = field(default=False, repr=False)
    _watch_consecutive_strikes: int = field(default=0, repr=False)
    _completion_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _reader_thread: Optional[threading.Thread] = field(default=None, repr=False)
    _pty: Any = field(default=None, repr=False)  # ptyprocess handle (use_pty=True)

    def append_output(self, text: str) -> None:
        """Append to the rolling output buffer under the session lock, keeping the tail."""
        with self._lock:
            self.output_buffer += text
            if len(self.output_buffer) > self.max_output_chars:
                self.output_buffer = self.output_buffer[-self.max_output_chars:]

    def mark_exited(self, exit_code, reason: str = "exited", source: str = "") -> None:
        """Record an exit. A kill that raced the observer already recorded its own
        exit_code/reason; never overwrite it."""
        self.exited = True
        if self.completion_reason != "killed":
            self.exit_code = exit_code
            self.completion_reason = reason
            if source:
                self.termination_source = source


# Watcher routing fields, in event-dict key order (``watcher_<key>`` on the session).
_WATCHER_ROUTE_KEYS = ("platform", "chat_id", "user_id", "user_name", "thread_id", "message_id")
# Session fields persisted verbatim in the crash-recovery checkpoint (plus
# ``session_id``; ``command`` is redacted and ``owner_task_id`` defaulted on write).
_CHECKPOINT_FIELDS = (
    "command", "pid", "pid_scope", "host_start_time", "systemd_unit", "cwd",
    "started_at", "task_id", "owner_task_id", "session_key",
    *(f"watcher_{k}" for k in _WATCHER_ROUTE_KEYS), "watcher_interval",
    "parent_session_id", "notify_on_complete", "watch_patterns")
_CHECKPOINT_DEFAULTS = {
    f.name: ([] if f.name == "watch_patterns" else f.default)
    for f in ProcessSession.__dataclass_fields__.values()
    if f.name in _CHECKPOINT_FIELDS
}


class ProcessRegistry:
    """In-memory registry of running and finished background processes.
    Thread-safe: accessed from executor threads (terminal_tool, process handlers),
    the gateway asyncio loop (watchers, reset checks) and the cleanup thread."""

    _SHELL_NOISE_SUBSTRINGS = (
        "no job control in this shell", "cannot set terminal process group",
        "tcsetattr: Inappropriate ioctl for device")

    def __init__(self):
        self._running: Dict[str, ProcessSession] = {}
        self._finished: Dict[str, ProcessSession] = {}
        # Read-only mirrors of other gateway processes' live rows. Deliberately a
        # separate store: everything that reasons about work THIS process owns
        # (kill_all, scale-to-zero, session reset, checkpoint authorship) keeps
        # reading _running only.
        self._peers: Dict[str, ProcessSession] = {}
        # Identity stamped on every checkpoint row we publish. ``owner_token`` says
        # "this row is mine" exactly (no inference); ``owner_pid`` + start time let
        # another gateway ask "is that writer still alive" without trusting a PID
        # the kernel may have recycled onto a stranger.
        self._owner_token = uuid.uuid4().hex
        self._owner_pid = os.getpid()
        self._owner_start_time = self._safe_host_start_time(self._owner_pid)
        self._lock = threading.Lock()
        # Serializes this registry's own checkpoint writes end to end (snapshot →
        # redact → merge → replace). The inter-process lock cannot order two writers
        # inside ONE process: it would just pick which stale ordering wins.
        self._checkpoint_write_lock = threading.Lock()
        self._output_checkpoint_lock = threading.Lock()
        self._last_output_checkpoint_at = 0.0
        self._output_checkpoint_timer: Optional[threading.Timer] = None
        # Side-channel for check_interval watchers (gateway reads after agent run)
        self.pending_watchers: List[Dict[str, Any]] = []
        # Unified queue for all background events (distinguished by "type"); the CLI
        # process_loop and the gateway drain it after each agent turn to trigger new turns.
        import queue as _queue_mod
        self.completion_queue: _queue_mod.Queue = _queue_mod.Queue()
        # Rehydrate durable delegation completions once, at registry startup.
        try:
            from tools.async_delegation import restore_undelivered_completions
            restore_undelivered_completions(self.completion_queue)
        except Exception as exc:
            logger.warning("Could not restore async delegation completions: %s", exc)
        # Completions the agent already consumed via wait()/read_log() (output in
        # hand): drain loops AND gateway/tui watchers skip them.
        self._completion_consumed: set = set()
        # Sessions merely *observed* exited via poll(). poll() is read-only and must NOT
        # mark consumed (a status check would suppress the watcher's autonomous delivery
        # turn), but the CLI has the poll result inline in the same turn, so
        # drain_notifications() skips these to avoid a duplicate [SYSTEM: ...];
        # gateway/tui watchers deliberately ignore this set.
        # See #8228.
        self._poll_observed: set = set()
        # Global watch-match circuit breaker across all sessions.
        self._global_watch_lock = threading.Lock()
        self._global_watch_window_start = self._global_watch_tripped_until = 0.0
        self._global_watch_window_hits = self._global_watch_suppressed_during_trip = 0
        # Driver-installed sinks (desktop gateway): on_output(session, chunk) streams
        # live output from reader threads; on_close(session_or_none, process_id) drops
        # a read-only terminal tab without killing the process.
        self.on_output = None
        self.on_close = None

    @staticmethod
    def _clean_shell_noise(text: str) -> str:
        """Strip shell startup warnings from the beginning of output."""
        lines = text.split("\n")
        while lines and any(noise in lines[0] for noise in ProcessRegistry._SHELL_NOISE_SUBSTRINGS):
            lines.pop(0)
        return "\n".join(lines)

    def _emit_output(self, session: ProcessSession, chunk: str) -> None:
        """Forward a chunk to the live-output sink; called from reader threads, never raises."""
        sink = self.on_output
        if sink is None or not chunk:
            return
        with suppress(Exception):
            sink(session, chunk)

    def _checkpoint_live_output(self) -> None:
        """Publish the redacted output tail so peer gateway processes can render it.
        Rate-limited to ``CHECKPOINT_OUTPUT_MIN_INTERVAL_SECONDS``: a rewrite per
        chunk turns a chatty dev server into continuous disk churn. A chunk that
        lands inside the window arms a single trailing write instead of being
        dropped, so the last line of a burst is never stranded off the checkpoint."""
        with self._output_checkpoint_lock:
            wait = self._last_output_checkpoint_at + CHECKPOINT_OUTPUT_MIN_INTERVAL_SECONDS - time.monotonic()
            if wait > 0:
                if self._output_checkpoint_timer is None:
                    self._output_checkpoint_timer = timer = threading.Timer(
                        wait, self._flush_output_checkpoint)
                    timer.daemon = True
                    timer.start()
                return
            self._last_output_checkpoint_at = time.monotonic()
        self._write_checkpoint()

    def _flush_output_checkpoint(self) -> None:
        """Trailing write armed by :meth:`_checkpoint_live_output` (timer thread)."""
        with self._output_checkpoint_lock:
            self._output_checkpoint_timer = None
            self._last_output_checkpoint_at = time.monotonic()
        self._write_checkpoint()

    def _check_watch_patterns(self, session: ProcessSession, new_text: str) -> None:
        """Scan a freshly-read chunk for watch patterns and queue notifications.
        Per-session rate limiting (see WATCH_* constants): one match per cooldown
        window, a match inside the window is one strike, WATCH_STRIKE_LIMIT consecutive
        strikes or WATCH_LIFETIME_MAX_HITS total deliveries disable watching and
        promote the session to notify_on_complete."""
        if not session.watch_patterns or session._watch_disabled:
            return
        # Late chunks after the reader declared exit are post-exit noise; dropping them
        # avoids stale notifications minutes after the process ended.
        if session.exited:
            return
        hits = [  # (first matching pattern, line) — one match per line
            (next(p for p in session.watch_patterns if p in line), line.rstrip())
            for line in new_text.splitlines() if any(p in line for p in session.watch_patterns)]
        if not hits:
            return
        matched_pattern = hits[0][0]
        matched_lines = [line for _, line in hits]
        now = time.time()
        with session._lock:
            if session._watch_cooldown_until and now < session._watch_cooldown_until:
                # Inside the cooldown: drop, count one strike per window, disable +
                # promote once the strike limit is hit.
                session._watch_suppressed += len(matched_lines)
                if session._watch_strike_candidate:
                    return
                session._watch_strike_candidate = True
                session._watch_consecutive_strikes += 1
                if session._watch_consecutive_strikes < WATCH_STRIKE_LIMIT:
                    return
                session._watch_disabled = True
                # Promote so the agent still gets exactly one notification on exit,
                # plus exactly one summary so it sees why things went quiet.
                session.notify_on_complete = True
                self._emit_watch_disabled(
                    session, session._watch_suppressed,
                    f"{WATCH_STRIKE_LIMIT} consecutive rate-limit windows triggered "
                    f"(min spacing {WATCH_MIN_INTERVAL_SECONDS}s). ")
                return
            # Cooldown expired. A prior window with no drops resets the
            # consecutive-strike counter (healthy cadence again).
            if session._watch_cooldown_until and not session._watch_strike_candidate:
                session._watch_consecutive_strikes = 0
            session._watch_strike_candidate = False
            # Emit and start a new cooldown window.
            session._watch_cooldown_until = now + WATCH_MIN_INTERVAL_SECONDS
            session._watch_hits += 1
            suppressed = session._watch_suppressed
            session._watch_suppressed = 0
            # Lifetime cap: this match is still delivered, but no further ones.
            lifetime_exhausted = session._watch_hits >= WATCH_LIFETIME_MAX_HITS
            if lifetime_exhausted:
                session._watch_disabled = True
                session.notify_on_complete = True
        output = "\n".join(matched_lines[:20])
        if len(output) > 2000:
            output = output[:2000] + "\n...(truncated)"
        if self._global_watch_admit(now):
            notification = {
                **self._watch_event_base(session),
                "type": "watch_match",
                "pattern": matched_pattern,
                "output": output,
                "suppressed": suppressed,
            }
            _redact_process_result(notification)
            self.completion_queue.put(notification)
        # Even when the breaker drops the final match, still explain the silence.
        if lifetime_exhausted:
            self._emit_watch_disabled(
                session, 0, f"reached the lifetime cap of {WATCH_LIFETIME_MAX_HITS} delivered matches. ",
            )

    def _emit_watch_disabled(self, session: ProcessSession, suppressed: int, why: str) -> None:
        """Queue the one-shot watch_disabled summary (strike-limit or lifetime-cap path)."""
        self.completion_queue.put({
            **self._watch_event_base(session),
            "type": "watch_disabled",
            "suppressed": suppressed,
            "message": (
                f"Watch patterns disabled for process {session.id} — {why}"
                f"Falling back to notify_on_complete semantics; you'll get "
                f"exactly one notification when the process exits."),
        })

    @staticmethod
    def _watch_event_base(session: ProcessSession) -> dict:
        """Session identity + watcher routing fields shared by every watch event."""
        return {
            "session_id": session.id,
            "session_key": session.session_key,
            "task_id": session.task_id,
            "owner_task_id": session.owner_task_id or session.task_id,
            "command": session.command,
            **{key: getattr(session, f"watcher_{key}") for key in _WATCHER_ROUTE_KEYS},
        }

    @staticmethod
    def _global_watch_event(type_: str, message: str, **extra) -> dict:
        """Unaddressed (all-sessions) watch breaker event."""
        return {
            "session_id": "", "session_key": "", "command": "", "type": type_, **extra,
            "message": message,
            "platform": "", "chat_id": "", "user_id": "", "user_name": "", "thread_id": "",
        }

    def _global_watch_admit(self, now: float) -> bool:
        """True if this watch_match may pass the global breaker.
        In cooldown: drop and count. Otherwise slide the rolling window; exceeding
        the cap trips the breaker for WATCH_GLOBAL_COOLDOWN_SECONDS with ONE
        "tripped" summary, and the cooldown's end emits ONE "released" summary."""
        events = []  # summary events, queued outside the lock
        with self._global_watch_lock:
            # Handle cooldown expiry first so we can emit the release summary.
            if self._global_watch_tripped_until and now >= self._global_watch_tripped_until:
                suppressed = self._global_watch_suppressed_during_trip
                self._global_watch_tripped_until = 0.0
                self._global_watch_suppressed_during_trip = 0
                self._global_watch_window_start, self._global_watch_window_hits = now, 0
                if suppressed > 0:
                    events.append(self._global_watch_event(
                        "watch_overflow_released",
                        f"Watch-pattern notifications resumed. "
                        f"{suppressed} match event(s) were suppressed during the flood.",
                        suppressed=suppressed))
            if self._global_watch_tripped_until and now < self._global_watch_tripped_until:
                # Still in cooldown — drop and count.
                self._global_watch_suppressed_during_trip += 1
                admit = False
            else:
                if now - self._global_watch_window_start >= WATCH_GLOBAL_WINDOW_SECONDS:
                    self._global_watch_window_start, self._global_watch_window_hits = now, 0
                admit = self._global_watch_window_hits < WATCH_GLOBAL_MAX_PER_WINDOW
                if admit:
                    self._global_watch_window_hits += 1
                else:
                    self._global_watch_tripped_until = now + WATCH_GLOBAL_COOLDOWN_SECONDS
                    self._global_watch_suppressed_during_trip += 1
                    events.append(self._global_watch_event(
                        "watch_overflow_tripped",
                        f"Watch-pattern overflow: >{WATCH_GLOBAL_MAX_PER_WINDOW} "
                        f"notifications in {WATCH_GLOBAL_WINDOW_SECONDS}s across all processes. "
                        f"Suppressing further watch_match events for "
                        f"{WATCH_GLOBAL_COOLDOWN_SECONDS}s."))
        for msg in events:
            self.completion_queue.put(msg)
        return admit

    @staticmethod
    def _is_host_pid_alive(pid: Optional[int]) -> bool:
        """Best-effort liveness check for host-visible PIDs."""
        if not pid:
            return False
        # ``os.kill(pid, 0)`` is NOT a no-op on Windows (bpo-14484) — use the
        # cross-platform existence check.
        from gateway.status import _pid_exists
        return _pid_exists(pid)

    @staticmethod
    def _safe_host_start_time(pid: Optional[int]) -> Optional[int]:
        """Kernel start ticks for a host PID, or None when unavailable."""
        try:
            from gateway.status import get_process_start_time
            return get_process_start_time(pid) if pid else None
        except Exception:
            return None

    @classmethod
    def _host_pid_is_ours(cls, pid: Optional[int], expected_start: Optional[int]) -> bool:
        """True only if ``pid`` is alive AND still the process we spawned.
        The kernel recycles PIDs, so a stored number can later name an unrelated
        process (seen in the wild: a browser's session leader tree-killed). The kernel
        start time captured at spawn must match the live one; with no baseline
        (legacy checkpoints, no ``/proc``) degrade to a bare liveness check."""
        return cls._is_host_pid_alive(pid) and (
            expected_start is None or cls._safe_host_start_time(pid) == expected_start)

    def _refresh_detached_session(self, session: Optional[ProcessSession]) -> Optional[ProcessSession]:
        """Update recovered host-PID sessions when the underlying process has exited."""
        # A peer mirror's lifecycle is re-derived from the checkpoint on every
        # sync; moving it to _finished here would file another process's row into
        # OUR stores (and back into the checkpoint we write).
        if session is None or session.exited or session.peer_mirror:
            return session
        if not session.detached or session.pid_scope != "host":
            return session
        # A recycled PID (alive but not ours) counts as "our process exited" so a
        # later kill() can never tree-kill the stranger.
        if self._host_pid_is_ours(session.pid, session.host_start_time):
            return session
        with session._lock:
            if session.exited:
                return session
            # No waitable handle survives recovery, so the real exit code is unknown.
            session.exited, session.exit_code = True, None
        self._move_to_finished(session)
        return session

    @staticmethod
    def _proc_alive(proc) -> bool:
        """True if a psutil.Process is running and not a zombie (already dead, just unreaped)."""
        try:
            import psutil
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except Exception:
            return False

    @staticmethod
    def _config_value(section: str, key: str, fallback):
        """``config.yaml`` value for ``section.key``, else the DEFAULT_CONFIG value.
        Raises if config is unreadable; callers wrap with their own hard fallback so
        registry code paths never crash on a broken config file."""
        from hermes_cli.config import DEFAULT_CONFIG, cfg_get, read_raw_config

        val = cfg_get(read_raw_config(), section, key)
        return DEFAULT_CONFIG[section][key] if val is None else val

    @staticmethod
    def _config_seconds(key: str, fallback: float) -> float:
        """``terminal.<key>`` as a non-negative float (0 disables); *fallback* if unreadable."""
        try:
            return max(float(ProcessRegistry._config_value("terminal", key, fallback)), 0.0)
        except Exception:
            return fallback

    @staticmethod
    def _daemon_term_grace_seconds() -> float:
        """Grace (s) between SIGTERM and escalated SIGKILL; 0 disables escalation."""
        return ProcessRegistry._config_seconds("daemon_term_grace_seconds", 2.0)

    @classmethod
    def _terminate_host_pid(cls, pid: int, expected_start: Optional[int] = None) -> None:
        """Terminate a host-visible PID and its descendants.
        ``expected_start`` (kernel start time at spawn) is re-validated first: a mismatch
        or dead PID means the number was recycled onto a stranger and we refuse to touch
        it — a leaked orphan beats tree-killing someone's browser. POSIX: psutil SIGTERMs
        children before the parent (so trees aren't reparented to init and survive), then
        SIGKILLs survivors after ``terminal.daemon_term_grace_seconds``. Windows:
        ``taskkill /T /F`` (psutil's stale PPID links miss orphans there); ``os.kill``
        is the fallback."""
        if expected_start is not None and not cls._host_pid_is_ours(pid, expected_start):
            logger.warning(
                "Refusing to terminate host pid %d: start-time mismatch — "
                "PID was recycled onto an unrelated process.", pid)
            return

        def _sigterm_quietly():
            with suppress(OSError, ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGTERM)
        if _IS_WINDOWS:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True,
                    encoding='utf-8', errors='replace', timeout=10, creationflags=windows_hide_flags(),
                    stdin=subprocess.DEVNULL)
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                _sigterm_quietly()
            return
        import psutil
        gone = (psutil.NoSuchProcess, psutil.AccessDenied, OSError)
        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        except (OSError, PermissionError):
            _sigterm_quietly()
            return
        # Snapshot the whole tree (children before parent) and SIGTERM each.
        try:
            targets = parent.children(recursive=True)
        except gone:
            targets = []
        targets.append(parent)
        for proc in targets:
            with suppress(gone):
                proc.terminate()
        # Escalate to SIGKILL for anything that ignored SIGTERM within the grace window.
        # ``psutil.wait_procs``' gone/alive partition is deliberately NOT trusted: it
        # reaps via ``Process.wait()`` and mis-partitions across zombie transitions in a
        # parent/child tree, leaving survivors un-killed. Re-probing every target is
        # deterministic.
        grace = cls._daemon_term_grace_seconds()
        if grace <= 0:
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and any(cls._proc_alive(_p) for _p in targets):
            time.sleep(0.05)
        for proc in targets:
            with suppress(gone):
                if cls._proc_alive(proc):
                    proc.kill()  # SIGKILL on POSIX
                    logger.info("Escalated to SIGKILL for pid %d (ignored SIGTERM within %.1fs grace)", proc.pid, grace)

    # ----- Spawn -----

    @staticmethod
    def _new_session(command, task_id, owner_task_id, session_key, cwd, **extra) -> ProcessSession:
        return ProcessSession(
            id=f"proc_{uuid.uuid4().hex[:12]}", command=command, task_id=task_id,
            owner_task_id=owner_task_id or task_id, session_key=session_key, cwd=cwd,
            started_at=time.time(), **extra)

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

    def _scope_argv(self, session: ProcessSession, safe_command: str, unit_suffix: str, label: str) -> List[str]:
        """Login-shell argv for *safe_command* (parity with LocalEnvironment: rc files
        sourced, user tools on PATH), wrapped in a transient systemd scope when we are
        the supervised gateway (own cgroup: an OOM kills only the worker, not the
        gateway and its messaging control plane)."""
        argv = [_find_shell(), "-lic", f"set +m; {safe_command}"]
        # This applies to both pipe mode and the PTY path above. See #70716.
        in_supervised_gateway = _IS_LINUX and _is_supervised_gateway_process()
        if in_supervised_gateway and _systemd_run_user_scope_available():
            session.systemd_unit = f"hermes-worker-{unit_suffix}.scope"
            return _build_systemd_scope_argv(argv, unit_suffix=unit_suffix)
        if in_supervised_gateway:
            # Under a supervisor but no private cgroup: a worker OOM can still take
            # the whole gateway down.
            logger.debug(
                "%s background executor not isolated in a systemd scope "
                "(systemd-run --user unavailable); worker shares the gateway cgroup.", label)
        return argv

    @staticmethod
    def _spawn_env(env_vars: dict) -> dict:
        """Sanitized child env; PYTHONUNBUFFERED so tqdm/datasets-style buffering
        doesn't hide progress from process(action="poll")."""
        env = _sanitize_subprocess_env(os.environ, env_vars)
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _track_started(self, session: ProcessSession, reader_target, reader_name: str, extra_args=()) -> None:
        """Start the output reader thread, register the session and checkpoint it."""
        reader = threading.Thread(target=reader_target, args=(session, *extra_args), daemon=True, name=reader_name)
        session._reader_thread = reader
        reader.start()
        with self._lock:
            self._prune_if_needed()
            self._running[session.id] = session
        self._write_checkpoint()

    def _spawn_local_pty(self, session: ProcessSession, safe_command: str, env_vars: dict) -> ProcessSession:
        """PTY spawn for interactive CLI tools (Codex, Claude Code, REPLs).
        Raises ImportError when no PTY backend is installed and re-raises any spawn
        failure; ``spawn_local`` falls back to pipe mode in both cases."""
        if _IS_WINDOWS:
            from winpty import PtyProcess as _PtyProcessCls
        else:
            from ptyprocess import PtyProcess as _PtyProcessCls
        pty_env = self._spawn_env(env_vars)
        # A PTY is a real TTY, so pager-happy tools (git log/diff, man) WILL page and
        # hang waiting for `q` — default them to cat, honoring any pager the user set.
        pty_env.setdefault("GIT_PAGER", "cat")
        pty_env.setdefault("PAGER", "cat")
        pty_argv = self._scope_argv(session, safe_command, session.id, "PTY")
        pty_proc = _PtyProcessCls.spawn(pty_argv, cwd=session.cwd, env=pty_env, dimensions=(30, 120))
        session.pid = pty_proc.pid
        session.host_start_time = self._safe_host_start_time(session.pid)
        session._pty = pty_proc
        self._track_started(session, self._pty_reader_loop, f"proc-pty-reader-{session.id}")
        return session

    def spawn_local(
        self, command: str, cwd: str = None, task_id: str = "", session_key: str = "",
        env_vars: dict = None, use_pty: bool = False, owner_task_id: str = "") -> ProcessSession:
        """Spawn a background process locally (TERMINAL_ENV=local; other backends use
        spawn_via_env()). ``use_pty`` requests a pseudo-terminal via ptyprocess/pywinpty
        for interactive CLIs, falling back to a plain pipe when unavailable or failing."""
        # Bash parses ``A && B &`` as ``(A && B) &`` — a subshell that holds our stdout
        # pipe open forever when B is a long-running server. The rewriter turns it into
        # ``A && { B & }``. Lazy import: terminal_tool imports this module.
        # Guard against the `A && B &` subshell-wait trap (issue #68915).
        from tools.terminal_tool_sudo import _rewrite_compound_background as _rewrite_bg

        safe_command = _rewrite_bg(command)
        session = self._new_session(command, task_id, owner_task_id, session_key, _resolve_safe_cwd(cwd or os.getcwd()))
        pty_scope_attempted = False
        if use_pty:
            try:
                return self._spawn_local_pty(session, safe_command, env_vars)
            except ImportError:
                logger.warning("ptyprocess not installed, falling back to pipe mode")
            except Exception as e:
                logger.warning("PTY spawn failed (%s), falling back to pipe mode", e)
                if session.systemd_unit:
                    pty_scope_attempted = True
                    if not _stop_systemd_unit(session.systemd_unit):
                        raise RuntimeError(
                            "PTY scope could not be reaped; refusing pipe fallback "
                            "to avoid duplicate command execution"
                        ) from e
                    session.systemd_unit = ""
        # Pipe path (non-PTY or PTY fallback).
        _popen_kwargs = {"creationflags": windows_hide_flags()} if _IS_WINDOWS else {}
        unit_suffix = f"{session.id}-pipe-fallback" if pty_scope_attempted else session.id
        spawn_argv = self._scope_argv(session, safe_command, unit_suffix, "Local")
        # start_new_session is REQUIRED with systemd-run --scope too: the scope does not
        # give the worker a new session, so from an interactive TUI the worker would
        # share the foreground process group and background spawns would stop the whole
        # session (observed as dead TUIs in state T). Cgroup isolation is unaffected —
        # the scope attaches to the invoked process, not the spawning session.
        proc = subprocess.Popen(
            spawn_argv, text=True, cwd=session.cwd, env=self._spawn_env(env_vars), encoding="utf-8",
            errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True, **_popen_kwargs)
        session.process = proc
        session.pid = proc.pid
        session.host_start_time = self._safe_host_start_time(session.pid)
        try:
            self._track_started(session, self._reader_loop, f"proc-reader-{session.id}")
        except Exception:
            self._reap_untracked(session, proc)
            raise
        return session

    def _reap_untracked(self, session: ProcessSession, proc: subprocess.Popen) -> None:
        """Post-Popen setup failed: kill the orphaned subprocess (and any setsid
        descendants) so nothing leaks untracked."""
        with suppress(Exception):
            if session.systemd_unit:
                # Scope teardown is the authoritative cleanup for the worker cgroup
                # (never killpg here); the wrapper PID is terminated as fallback.
                _stop_systemd_unit(session.systemd_unit)
                # The worker runs in its own systemd scope and, since the #70716 session-isolation fix, its
                # own session. Stop the scope (kills every process in the worker cgroup), then terminate the
                # systemd-run wrapper PID as fallback.
                self._terminate_host_pid(proc.pid, session.host_start_time)
            elif not _IS_WINDOWS:
                try:
                    kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
                    os.killpg(os.getpgid(proc.pid), kill_signal)  # windows-footgun: ok - guarded by _IS_WINDOWS above
                except (ProcessLookupError, PermissionError, OSError):
                    proc.kill()
            else:
                proc.kill()
        with suppress(Exception):
            proc.wait(timeout=5)

    def spawn_via_env(
        self, env: Any, command: str, cwd: str = None, task_id: str = "", session_key: str = "",
        timeout: int = 10, owner_task_id: str = "") -> ProcessSession:
        """Spawn a background process inside a non-local backend's sandbox.
        The command is wrapped to capture its in-sandbox PID and redirect output to a
        log file that later execute() calls poll. No live pipe or stdin, but it runs in
        the correct sandbox context."""
        session = self._new_session(command, task_id, owner_task_id, session_key, cwd, env_ref=env, pid_scope="sandbox")
        temp_dir = self._env_temp_dir(env)
        log_path, pid_path, exit_path = (f"{temp_dir}/hermes_bg_{session.id}.{ext}" for ext in ("log", "pid", "exit"))
        q = shlex.quote
        bg_command = (
            f"mkdir -p {q(temp_dir)} && "
            f"( nohup bash -lc {q(command)} > {q(log_path)} 2>&1; "
            f"rc=$?; printf '%s\\n' \"$rc\" > {q(exit_path)} ) & "
            f"echo $! > {q(pid_path)} && cat {q(pid_path)}")
        try:
            result = env.execute(bg_command, timeout=timeout, rewrite_compound_background=False)
            output = result.get("output", "").strip()
            session.pid = next((int(ln) for ln in map(str.strip, output.splitlines()) if ln.isdigit()), None)
            # No PID from the wrapper (syntax error, broken redirect): a failed launch,
            # not a fake running session.
            if session.pid is None:
                session.mark_exited(int(result.get("returncode", -1)) or -1, "failed_start", "failed_start")
                session.output_buffer = output
        except Exception as e:
            session.mark_exited(-1, "failed_start", "failed_start")
            session.output_buffer = f"Failed to start: {e}"
        if session.exited:
            with self._lock:
                self._prune_if_needed()
        else:
            self._track_started(
                session, self._env_poller_loop, f"proc-poller-{session.id}", (env, log_path, pid_path, exit_path))
        return session

    # ----- Reader / Poller Threads -----

    def _reader_loop(self, session: ProcessSession):
        """Background thread: read stdout from a local Popen process.
        ``buffer.read1(4096)`` not ``TextIOWrapper.read(4096)``: on pipes the latter
        blocks until EOF, landing "live" output in one burst at exit. Orphaned-pipe
        guard: a backgrounded grandchild (``node server.js &``) inherits our pipe's write
        end so EOF never arrives while it lives, which would park this thread and never
        fire ``notify_on_complete``; on POSIX we ``select()`` and stop draining shortly
        after the direct child exits (mirrors ``environments/base.py::_wait_for_process``).
        Windows pipes lack select(), so the lazy ``_reconcile_local_exit`` is the net.

        Windows pipes don't support select(); the blocking path is kept there and the lazy reconcile in
        poll()/wait() remains the safety net. See #68915, #8340.
        """
        first_chunk = True
        # A split multibyte UTF-8 char would become U+FFFD with stateless decoding; the
        # incremental decoder holds the partial sequence until the rest arrives.
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        # Incremental decoder: raw pipe reads can split a multibyte UTF-8 character across two read1()
        # chunks. A stateless per-chunk ``bytes.decode(errors="replace")`` turns both halves into U+FFFD
        # mojibake. The incremental decoder holds the partial sequence until the continuation bytes arrive —
        # same treatment the foreground path already has in
        # ``tools/environments/base.py::_wait_for_process``. (Ported from openclaw/openclaw#112325.)
        def _append_chunk(chunk: str):
            nonlocal first_chunk
            if first_chunk:
                chunk = self._clean_shell_noise(chunk)
                first_chunk = False
            self._ingest_output(session, chunk)
        try:
            proc = session.process
            if proc is None or proc.stdout is None:
                return
            stdout = proc.stdout
            raw_read = getattr(getattr(stdout, "buffer", None), "read1", None)

            def _read_once():
                """One 4 KiB read: decoded text ('' for a partial multibyte tail), None at EOF."""
                if raw_read is None:  # mocked/alternate streams without a raw buffer: less "live"
                    return stdout.read(4096) or None
                raw = raw_read(4096)
                return decoder.decode(raw) if raw else None
            # select() needs a real OS fd; mocked streams (tests, adapters) may lack
            # fileno() and use the blocking read instead.
            try:
                fd = stdout.fileno() if raw_read is not None and not _IS_WINDOWS else None
            except Exception:
                fd = None
            if not (isinstance(fd, int) and fd >= 0):
                fd = None
            if fd is not None:
                import select as _select
            idle_after_exit = 0
            while True:
                if fd is not None:
                    try:
                        ready, _, _ = _select.select([fd], [], [], 0.2)
                    except (ValueError, OSError):
                        break  # fd already closed
                    if not ready:
                        # Direct child gone and pipe idle ~200ms: a few more cycles for a
                        # buffered tail, then stop rather than wait forever on an orphaned
                        # grandchild's pipe.
                        if proc.poll() is not None:
                            # See #68915.
                            idle_after_exit += 1
                        if idle_after_exit >= 3:
                            break
                        continue
                chunk = _read_once()
                if chunk is None:
                    break  # true EOF — all writers closed
                if chunk:
                    _append_chunk(chunk)
                idle_after_exit = 0
        except Exception as e:
            logger.debug("Process stdout reader ended: %s", e)
        finally:
            self._finish_reader(
                session, decoder, _append_chunk, "Process",
                lambda: session.process.wait(timeout=5), lambda: session.process.returncode)

    def _finish_reader(self, session, decoder, append, label, wait, exit_code) -> None:
        """Reader-thread teardown: flush the decoder (a truncated multibyte tail becomes
        one U+FFFD instead of vanishing), reap the child (no zombies), record the exit."""
        with suppress(Exception):
            tail = decoder.decode(b"", final=True)
            if tail:
                append(tail)
        try:
            wait()
        except Exception as e:
            logger.debug("%s wait timed out or failed: %s", label, e)
        self._finish_exited(session, exit_code())

    @staticmethod
    def _log_delta_command(quoted_log_path: str, offset: int) -> str:
        """Shell command that reads only the log bytes written since ``offset``
        (``cat``-ing the whole file every poll re-sends all output over docker/SSH).

        Prints one header line ``"<size> <offset>"`` then the bytes in [offset, size).
        The size is read first and the tail cut at that same size, so a growing file
        never sends a byte twice; a file that shrank was rotated/truncated, so the
        offset drops to 0 and the reader starts over. The window end is pulled back
        to a UTF-8 character boundary (the backend decodes each ``execute()`` result
        on its own, so a straddling multibyte char would become U+FFFD and break watch
        patterns at the seam): up to 3 trailing continuation bytes are held for the
        next poll and the header reports the trimmed size."""
        return (
            f"O={offset}; "
            f"S=$({{ wc -c < {quoted_log_path}; }} 2>/dev/null | tr -dc '0-9'); "
            f"S=${{S:-0}}; "
            f'if [ "$S" -lt "$O" ]; then O=0; fi; '
            # Scan back up to 3 continuation bytes (octal 200-277) to the lead byte; if
            # the lead's declared length (3xx=2, 34x-35x=3, 36x-37x=4) exceeds the bytes
            # present, trim to before it. Complete sequences and ASCII tails untouched.
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

    def _env_poller_loop(self, session: ProcessSession, env: Any, log_path: str, pid_path: str, exit_path: str):
        """Background thread: poll a sandbox log file for non-local backends."""
        q = shlex.quote
        # Byte offset already read from the log (bytes, not chars: the shell counts bytes).
        prev_output_bytes = 0
        while not session.exited:
            time.sleep(2)
            try:
                # Read only the bytes written since the last poll.
                raw = env.execute(self._log_delta_command(q(log_path), prev_output_bytes),
                                  timeout=10).get("output", "")
                header, _, delta = raw.partition("\n")
                try:
                    size_str, offset_str = header.split()
                    new_size = int(size_str)
                    used_offset = int(offset_str)
                except ValueError:
                    # No usable header (command failed, shell missing a tool): skip this
                    # poll rather than act on a half-read value.
                    new_size = None
                    used_offset = None
                    delta = ""
                if new_size is not None:
                    if used_offset < prev_output_bytes:
                        # Log rotated/truncated: what we hold no longer lines up. Restart.
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
                    self._checkpoint_live_output()

                check = env.execute(
                    f"kill -0 \"$(cat {q(pid_path)} 2>/dev/null)\" 2>/dev/null; echo $?", timeout=5)
                check_output = check.get("output", "").strip()
                if check_output and check_output.splitlines()[-1].strip() != "0":
                    # Exited -- read the exit code captured by the wrapper shell.
                    exit_str = env.execute(f"cat {q(exit_path)} 2>/dev/null", timeout=5).get("output", "").strip()
                    try:
                        exit_code = int(exit_str.splitlines()[-1].strip())
                    except (ValueError, IndexError):
                        exit_code = -1
                    session.exit_code = exit_code  # unlike mark_exited, a raced kill still takes this code
                    self._finish_exited(session, exit_code)
                    return
            except Exception:
                # Environment might be gone (sandbox reaped, etc.)
                session.exited, session.exit_code = True, -1
                session.completion_reason, session.termination_source = "lost", "backend_lost"
                self._move_to_finished(session)
                return

    def _pty_reader_loop(self, session: ProcessSession):
        """Background thread: read output from a PTY process."""
        pty = session._pty
        # Same split-multibyte handling as _reader_loop.
        # PTY reads can split a multibyte UTF-8 character across chunks just like pipe reads — hold partial
        # sequences until the rest arrives. (Ported from openclaw/openclaw#112325.)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while pty.isalive():
                try:
                    chunk = pty.read(4096)
                    if chunk:
                        # ptyprocess returns bytes; pywinpty returns str
                        text = chunk if isinstance(chunk, str) else decoder.decode(chunk)
                        if text:
                            self._ingest_output(session, text)
                except Exception:  # EOFError included
                    break
        except Exception as e:
            logger.debug("PTY stdout reader ended: %s", e)
        self._finish_reader(
            session, decoder, lambda t: self._ingest_output(session, t), "PTY",
            pty.wait, lambda: pty.exitstatus if hasattr(pty, 'exitstatus') else -1)

    def _ingest_output(self, session: ProcessSession, text: str) -> None:
        """Buffer a freshly-read chunk, then scan watch patterns and stream it live."""
        session.append_output(text)
        self._check_watch_patterns(session, text)
        self._emit_output(session, text)
        self._checkpoint_live_output()

    def _finish_exited(self, session: ProcessSession, exit_code) -> None:
        """Mark a reader-observed exit (a raced kill keeps its own code/reason) and finish."""
        session.mark_exited(exit_code)
        self._move_to_finished(session)

    def _move_to_finished(self, session: ProcessSession):
        """Move a session from running to finished.
        Idempotent: kill_process() and the reader thread can both call this; only
        the FIRST move enqueues the completion notification, so no duplicates."""
        with self._lock:
            was_running = self._running.pop(session.id, None) is not None
            self._finished[session.id] = session
        session._completion_event.set()
        self._write_checkpoint()
        if was_running and session.notify_on_complete:
            notification = {
                "type": "completion",
                "session_id": session.id,
                "session_key": session.session_key,
                "task_id": session.task_id,
                "owner_task_id": session.owner_task_id or session.task_id,
                "command": session.command,
                **self._exit_fields(session),
                "output": _output_tail(session, 2000),
                # Stable producer identity across checkpoint recovery (unlike a
                # consumer-observed completion timestamp).
                "started_at": session.started_at,
            }
            _redact_process_result(notification)
            self.completion_queue.put(notification)

    @staticmethod
    def _exit_fields(session: ProcessSession) -> dict:
        return {
            "exit_code": session.exit_code,
            "completion_reason": session.completion_reason,
            "termination_source": session.termination_source,
        }

    # ----- Query Methods -----

    def is_completion_consumed(self, session_id: str) -> bool:
        """Check if a completion notification was already consumed via wait/log."""
        return session_id in self._completion_consumed

    def is_session_waiting(self, session_id: str) -> bool:
        """Whether a goal loop (``hermes_cli.goals`` wait barrier) should stay parked on
        this session: still running AND, with ``watch_patterns``, none matched yet (a
        long-lived watcher unblocks on its trigger, not on exit). Unknown/exited/
        already-fired sessions return False so a stale barrier can never wedge the loop."""
        with self._lock:
            session = (self._running.get(session_id) or self._finished.get(session_id)) if session_id else None
        if session is None:
            return False
        with suppress(Exception):
            self._refresh_detached_session(session)
        return not session.exited and not (
            session.watch_patterns and not session._watch_disabled and session._watch_hits > 0)

    def wait_for_pending_completions(
        self, task_id: Optional[str] = None, *, timeout: float | None = None, poll_interval: float = 1.0,
    ) -> dict:
        """Bounded linger for ``notify_on_complete`` background processes at one-shot exit.
        A one-shot CLI run (``hermes -q/-Q/-z``) exits when its turn ends; a background
        process it spawned still holds a stdout pipe owned by the dying parent and dies of
        SIGPIPE seconds later (Bot Mode handoff replies were the visible casualty). Only
        ``notify_on_complete`` processes carry a completion contract — servers/daemons/
        watchers aren't the parent's to wait for. ``task_id=None`` waits on every tracked
        process; ``timeout=None`` reads ``terminal.oneshot_completion_wait_seconds`` (``<= 0``
        disables). Each pass re-reconciles child state so an orphaned-pipe exit can't wedge
        the linger. Returns ``{"waited", "completed", "timed_out"}`` id lists.

        Bot Mode handoff REPLIES are the visible casualty (#90879): a recipient invoked as ``hermes -p <bot>
        chat -Q --query-file ...`` dispatches its reply via ``message_agent`` / ``bot_relay`` exactly this
        way, then exits, and the reply process is destroyed ~3s later. The sender waits forever for a reply
        that was already killed.
        See #17327.
        """
        if timeout is None:
            timeout = self._oneshot_completion_wait_seconds()
        result: dict = {"waited": [], "completed": [], "timed_out": []}
        with self._lock:
            pending = [
                s for s in self._running.values()
                if s.notify_on_complete and not s.exited and (task_id is None or s.task_id == task_id)
            ]
        if not pending or timeout <= 0:
            return result
        result["waited"] = [s.id for s in pending]
        logger.info(
            "One-shot exit lingering (bounded %ss) for %d notify_on_complete "
            "background process(es): %s",
            timeout, len(pending), ", ".join(s.id for s in pending))
        deadline = time.monotonic() + max(float(timeout), 0.0)
        interval = max(float(poll_interval), 0.05)
        try:
            from tools.interrupt import is_interrupted as _is_interrupted
        except Exception:
            _is_interrupted = lambda: False  # noqa: E731
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
                    # Reconcile first so orphaned-pipe and detached exits fire the event.
                    with suppress(Exception):
                        # Reconcile first: catches direct-child exits whose reader is blocked on a pipe held
                        # open by a descendant (#17327) and detached/env sessions, so the event actually
                        # fires.
                        # Reconcile against real child state before reading session.exited. Guards against
                        # orphaned-pipe reader hangs (issue #17327).
                        # Reconcile against real child state — guards against orphaned- pipe reader hangs
                        # where the reader is blocked but the direct child has already exited (issue
                        # #17327).
                        self._reconcile_local_exit(session)
                        self._refresh_detached_session(session)
                    if session.exited:
                        break
                    session._completion_event.wait(min(remaining, interval))
            except KeyboardInterrupt:
                # Stop waiting, but never let the interrupt skip the caller's durable
                # teardown (session flush, end_session) that follows.
                interrupted = True
            result["completed" if session.exited else "timed_out"].append(session.id)
        if result["timed_out"]:
            logger.warning(
                "One-shot exit linger timed out after %ss with %d background "
                "process(es) still running: %s — they may be killed when this "
                "process exits.",
                timeout, len(result["timed_out"]), ", ".join(result["timed_out"]))
        return result

    @staticmethod
    def _oneshot_completion_wait_seconds() -> float:
        """Linger (s) for one-shot exits with pending notify_on_complete processes; 0 disables."""
        return ProcessRegistry._config_seconds("oneshot_completion_wait_seconds", 600.0)

    def _drain_should_skip(self, session_id: str, *, skip_poll_observed: bool = True) -> bool:
        """Skip a completion the CLI agent already has this turn — consumed via wait/log
        or observed inline via poll(). Gateway/tui watchers check only
        ``is_completion_consumed`` so a read-only poll never suppresses their turn.

        Skips when the agent has either truly consumed the output (wait/log → ``_completion_consumed``) or
        observed the exit inline via poll() (``_poll_observed``). In both cases the CLI agent already has
        the result this turn, so injecting a [SYSTEM: ...] completion would be a duplicate (#8228).
        """
        return session_id in self._completion_consumed or (skip_poll_observed and session_id in self._poll_observed)

    @staticmethod
    def _surface_child_process_notifications() -> bool:
        """``delegation.surface_child_process_notifications``; False on any config
        error — never crash the drain loop."""
        try:
            return bool(ProcessRegistry._config_value("delegation", "surface_child_process_notifications", False))
        except Exception:
            return False

    @staticmethod
    def _owns_event(evt: dict, session_key: str, owns_event, is_async_delegation: bool) -> bool:
        """Routing verdict for one drained event (see drain_notifications); False = requeue."""
        evt_session_key = str(evt.get("session_key") or "")
        requires_positive_proof = is_async_delegation or bool(evt_session_key or evt.get("origin_ui_session_id"))
        if owns_event is not None and requires_positive_proof:
            try:
                return bool(owns_event(evt))
            except Exception:
                return False  # fail closed — never leak on a broken check
        if session_key and requires_positive_proof:
            return evt_session_key == session_key
        # Restored payloads from a previous process: an unfiltered drain cannot prove
        # ownership, so leave them for the owner.
        return not (is_async_delegation and evt.get("restored"))

    def drain_notifications(
        self, session_key: str = "", owns_event=None, *, skip_poll_observed: bool = True,
    ) -> "list[tuple[dict, str]]":
        """Pop all pending events and return ``(raw_event, formatted_text)`` pairs.
        Skips completions per ``_drain_should_skip`` (gateway/TUI pass
        ``skip_poll_observed=False``). Routing (``_owns_event``): async-delegation events
        always need ownership proof, ordinary events once they carry ``session_key`` or
        ``origin_ui_session_id``; ``owns_event(evt)`` (strongest; the TUI passes a
        compression-chain-aware check) consumes ONLY on True, ``session_key`` uses plain
        equality; non-owned events are re-queued for their owner. No filter consumes
        everything (legacy single-session) except restored delegation payloads (fail-closed)."""
        results: "list[tuple[dict, str]]" = []
        requeue: "list[dict]" = []
        # delegation.surface_child_process_notifications, read at most once per drain
        # and only when an sa- event shows up.
        surface_child: "bool | None" = None
        while not self.completion_queue.empty():
            try:
                evt = self.completion_queue.get_nowait()
            except Exception:
                break
            is_async_delegation = evt.get("type") == "async_delegation"
            if not self._owns_event(evt, session_key, owns_event, is_async_delegation):
                requeue.append(evt)
                continue
            # Routing happened first so a foreign session cannot drop the owner's
            # event via its own consumed/observed state.
            _evt_sid = evt.get("session_id", "")
            if evt.get("type") == "completion" and self._drain_should_skip(
                _evt_sid, skip_poll_observed=skip_poll_observed):
                continue
            # Subagent-owned process notifications are suppressed by default — the
            # child's delegation result is the deliverable. Judge ownership on
            # owner_task_id (RAW spawning id; task_id is the container key, collapsed
            # by _resolve_container_task_id). Dropped, NOT requeued: children never
            # drain, so a requeue would pin the event forever. 'async_delegation'
            # is the result itself and is NEVER suppressed.
            _evt_task_id = str(evt.get("owner_task_id") or evt.get("task_id") or "")
            if not is_async_delegation and _evt_task_id.startswith("sa-"):
                if surface_child is None:
                    surface_child = self._surface_child_process_notifications()
                if not surface_child:
                    logger.debug(
                        "Suppressed subagent-owned process notification "
                        "(delegation.surface_child_process_notifications=false): "
                        "type=%s session_id=%s task_id=%s",
                        evt.get("type", "completion"), _evt_sid, _evt_task_id)
                    continue
            if text := format_process_notification(evt):
                results.append((evt, text))
        for evt in requeue:
            self.completion_queue.put(evt)
        return results

    # Minimum suffix chars for prefix resolution; "p"/"proc_1" are too collision-prone.
    _MIN_PREFIX_CHARS = 4

    def get(self, session_id: str) -> Optional[ProcessSession]:
        """Session by full ID or unique prefix (``proc_4dae`` / bare ``4dae``, like git
        short hashes); ambiguous or too-short prefixes resolve to None, never a guess."""
        with self._lock:
            session = (self._running.get(session_id) or self._finished.get(session_id)
                       or self._peers.get(session_id))
        return self._refresh_detached_session(session if session is not None else self._resolve_prefix(session_id))

    def _resolve_prefix(self, session_id: str) -> Optional[ProcessSession]:
        """Resolve a unique session-ID prefix (a bare hex tail is normalized to
        ``proc_<tail>``); :meth:`get` tries exact first."""
        query = session_id.strip() if isinstance(session_id, str) else ""
        if not query:
            return None
        if not query.startswith("proc_"):
            query = f"proc_{query}"
        if len(query) - len("proc_") < self._MIN_PREFIX_CHARS:
            return None
        with self._lock:
            matches = [
                s for store in (self._running, self._finished, self._peers)
                for sid, s in store.items() if sid.startswith(query)
            ]
        return matches[0] if len(matches) == 1 else None

    def _reconcile_local_exit(self, session: "ProcessSession") -> None:
        """Reconcile ``session.exited`` against the real child state.
        The reader flips ``exited`` only at EOF; when the direct child has exited but a
        descendant (e.g. a daemon from ``hermes update``) holds the pipe open, poll()
        would report "running" forever. If ``Popen.poll()`` has an exit code, drain
        readable bytes non-blocking and flip ``exited``. No-op for env/PTY, exited and
        detached sessions.

        The reader thread (`_reader_loop`) sets `session.exited = True` only in its `finally` block, which
        runs when `stdout.read()` returns EOF. If the direct `Popen` child has exited but a descendant
        process (e.g. a daemon spawned by `hermes update` restarting the gateway) is still holding the
        stdout pipe open, the reader blocks forever and poll() keeps returning "running" indefinitely (issue
        #17327 — 74 polls over 7 minutes on Feishu).
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
        # Best-effort non-blocking drain of whatever the reader hasn't consumed.
        stdout = getattr(proc, "stdout", None)
        if stdout is not None and not _IS_WINDOWS:
            try:
                import fcntl
                fd = stdout.fileno()
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
                try:
                    with suppress(BlockingIOError, OSError, ValueError):
                        chunk = stdout.read()
                        if chunk:
                            session.append_output(chunk if isinstance(chunk, str) else chunk.decode("utf-8", errors="replace"))
                finally:
                    with suppress(Exception):
                        fcntl.fcntl(fd, fcntl.F_SETFL, flags)
            except Exception as e:
                logger.debug("Non-blocking drain failed for %s: %s", session.id, e)
        with session._lock:
            session.mark_exited(rc)
        logger.info(
            "Reconciled session %s: direct child exited with code %s but reader "
            "was still blocked (orphaned pipe). Flipped to exited.",
            session.id, rc)
        self._move_to_finished(session)

    @staticmethod
    def _status_head(session: ProcessSession) -> dict:
        return {"session_id": session.id, "command": session.command, "status": "exited" if session.exited else "running"}

    def poll(self, session_id: str) -> dict:
        """Check status and get new output for a background process."""
        session = self.get(session_id)
        if session is None:
            return _not_found(session_id)
        self._reconcile_local_exit(session)  # orphaned-pipe reader guard
        with session._lock:
            output_preview = _output_tail(session, 1000)
        result = {
            **self._status_head(session), "pid": session.pid,
            "uptime_seconds": int(time.time() - session.started_at), "output_preview": output_preview}
        if session.exited:
            result.update(self._exit_fields(session))
            # Read-only: record in _poll_observed (CLI inline dedup) but NOT in
            # _completion_consumed, or a status check would suppress the watcher's
            # autonomous delivery turn. See __init__.
            self._poll_observed.add(session_id)
        if session.detached:
            result.update(detached=True, note="Process recovered after restart -- output history unavailable")
        return result

    def read_log(self, session_id: str, offset: int | None = None, limit: int = 200) -> dict:
        """Read the full output log with optional pagination by lines."""
        from tools.ansi_strip import strip_ansi

        session = self.get(session_id)
        if session is None:
            return _not_found(session_id)
        with session._lock:
            full_output = strip_ansi(session.output_buffer)
        lines = full_output.splitlines()
        total_lines = len(lines)
        # offset=None -> last N lines; an explicit offset=0 means the HEAD (don't
        # conflate the two via falsiness).
        # An explicit offset=0 means "start from the first line" — previously it was conflated with the
        # default and silently returned the TAIL instead of the head (same falsy-coercion class as the
        # wait() timeout guard; salvaged from PR #60004, credit @isheng-eqi).
        if offset is None and limit > 0:
            selected = lines[-limit:]
            observed_completion_output = bool(selected) or total_lines == 0
        else:
            offset = offset or 0
            selected = lines[offset:offset + limit]
            stop = slice(offset, offset + limit).indices(total_lines)[1]
            observed_completion_output = total_lines == 0 or (bool(selected) and stop == total_lines)
        result = {
            **self._status_head(session), "output": "\n".join(selected),
            "total_lines": total_lines, "showing": f"{len(selected)} lines"}
        if session.exited and observed_completion_output:
            self._completion_consumed.add(session_id)
        return result

    def wait(self, session_id: str, timeout: int = None) -> dict:
        """Block until the process exits, the timeout elapses, or the user interrupts.
        ``timeout`` defaults to (and is clamped by) TERMINAL_TIMEOUT. Returns a dict
        with status exited|timeout|interrupted|not_found|error and an output snapshot."""
        from tools.interrupt import is_interrupted as _is_interrupted

        try:
            max_timeout = int(os.getenv("TERMINAL_TIMEOUT", "180"))
        except (ValueError, TypeError):
            max_timeout = 180
        # The schema says minimum=1 but not every caller enforces it; timeout=0 is
        # falsy and would silently fall through to the default wait.
        if timeout is not None and timeout <= 0:
            return {"status": "error", "error": f"timeout must be positive (got {timeout})"}
        timeout_note = None
        effective_timeout = timeout or max_timeout
        if timeout and timeout > max_timeout:
            effective_timeout = max_timeout
            timeout_note = f"Requested wait of {timeout}s was clamped to configured limit of {max_timeout}s"
        session = self.get(session_id)
        if session is None:
            return _not_found(session_id)
        deadline = time.monotonic() + effective_timeout
        while time.monotonic() < deadline:
            session = self._refresh_detached_session(session)
            if session is None:
                return _not_found(session_id)
            self._reconcile_local_exit(session)  # orphaned-pipe reader guard
            result = None
            if session.exited:
                self._completion_consumed.add(session_id)
                result = self._exit_snapshot(session, "exited")
            elif _is_interrupted():
                result = {
                    "status": "interrupted", "command": session.command, "output": _output_tail(session, 1000),
                    "note": "User sent a new message -- wait interrupted"}
            if result is not None:
                if timeout_note:
                    result["timeout_note"] = timeout_note
                return result
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            session._completion_event.wait(timeout=min(1.0, remaining))
        result = {
            "status": "timeout", "command": session.command, "output": _output_tail(session, 1000),
            # Not a failure — models re-issued identical waits after misreading this as an error.
            "process_running": True}
        base_note = (
            f"Wait window of {effective_timeout}s elapsed — the process is still running. This is not an error.")
        if session.started_at:
            base_note += f" Uptime: {int(time.time() - session.started_at)}s."
        base_note += (
            " notify_on_complete is set: you will be notified on exit — do more work instead of waiting again."
            if session.notify_on_complete else
            " Poll again later or use terminal(background=true, "
            "notify_on_complete=true) next time for automatic notification.")
        result["timeout_note"] = f"{timeout_note}. {base_note}" if timeout_note else base_note
        return result

    @staticmethod
    def _exit_snapshot(session: ProcessSession, status: str) -> dict:
        """Result dict for an exited session: exit metadata + last 2000 chars of output."""
        return {
            "status": status, "command": session.command,
            **ProcessRegistry._exit_fields(session), "output": _output_tail(session, 2000)}

    def kill_process(
        self, session_id: str, *, source: str = "process.kill", consume_output: bool = True,
    ) -> dict:
        """Kill a background process and return its output snapshot.
        ``consume_output`` is true for explicit tool/RPC kills (the caller sees the
        output). Bulk cleanup passes false so it doesn't suppress an autonomous
        completion notification — except abandoned-turn reaping (``kill_started_since``),
        which passes true so a killed abandoned process can't revive stopped work."""
        session = self.get(session_id)
        if session is None:
            return _not_found(session_id)
        if session.exited:
            # A double-forked descendant may still be alive in the systemd scope even
            # though the main process exited — stop the scope to reap survivors.
            # See #70716.
            # If the worker was spawned in its own systemd scope (#70716), stop the entire unit to reap any
            # double-forked descendants that were reparented inside the scope and survived the PID signal
            # above (reviewer gap #2). ``systemctl --user stop`` sends SIGTERM to every process in the
            # cgroup and escalates to SIGKILL after TimeoutStopSec. This is additive — the PID-based kill
            # above already handled the main process; this catches stragglers.
            if session.systemd_unit:
                _stop_systemd_unit(session.systemd_unit)
            with session._lock:
                result = self._exit_snapshot(session, "already_exited")
            # Only suppress the autonomous turn after its output is present in
            # the explicit kill result, matching wait/log consumption.
            if consume_output:
                self._completion_consumed.add(session_id)
            return result
        try:
            early = self._signal_kill(session, session_id, consume_output)
            if early is not None:
                return early
            # Additive to the PID kill: stopping the scope reaps double-forked
            # descendants reparented inside the cgroup.
            if session.systemd_unit:
                _stop_systemd_unit(session.systemd_unit)
            # Capture output, mark consumed, THEN expose ``exited`` to watcher tasks —
            # closes the delayed-notification race without losing the transcript.
            with session._lock:
                output = _output_tail(session, 2000)
                if consume_output:
                    self._completion_consumed.add(session_id)
                session.exited = True
                session.exit_code = -15  # SIGTERM
                session.completion_reason = "killed"
                session.termination_source = source
            self._move_to_finished(session)
            self._write_checkpoint()
            return {
                "status": "killed", "session_id": session.id, "completion_reason": session.completion_reason,
                "termination_source": session.termination_source, "output": output}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _signal_kill(self, session: ProcessSession, session_id: str, consume_output: bool) -> Optional[dict]:
        """Deliver the kill via PTY, local Popen tree, sandbox exec or recovered host
        PID. Returns a final result dict when the kill cannot proceed (peer-owned
        process, recycled/dead recovered PID, or no runtime handle), else None."""
        if session.peer_mirror:
            # The PID belongs to another gateway process's child. We have no
            # handle on it and no way to tell a recycled number from the real
            # thing beyond the owner's own bookkeeping, so signalling it here is
            # exactly the tree-kill-a-stranger risk the identity checks exist to
            # prevent. Ask through the gateway that owns it instead.
            return {
                "status": "error",
                "error": "This process is owned by another Hermes gateway process; "
                         "stop it from the surface that started it.",
            }
        if session._pty:
            try:
                session._pty.terminate(force=True)
            except Exception:
                if session.pid:
                    os.kill(session.pid, signal.SIGTERM)
        elif session.process:
            # Tree kill: on Windows Popen.terminate() only kills the shell wrapper and
            # leaves Git Bash descendants behind.
            self._terminate_host_pid(session.process.pid, session.host_start_time)
        elif session.env_ref and session.pid:
            session.env_ref.execute(f"kill {session.pid} 2>/dev/null", timeout=5)
        elif session.detached and session.pid_scope == "host" and session.pid:
            # Identity check, not bare liveness: a gone/recycled PID means our
            # process exited — never tree-kill the stranger. Still stop an owned
            # scope: a daemonized descendant may survive the wrapper PID.
            # If this recovered session also carries an owned systemd scope, stop that scope before
            # returning: a daemonized descendant may still be alive there even though the wrapper PID exited
            # or was recycled across the gateway restart (#70716, teknium1 review).
            if not self._host_pid_is_ours(session.pid, session.host_start_time):
                if session.systemd_unit:
                    _stop_systemd_unit(session.systemd_unit)
                with session._lock:
                    session.exited = True
                    session.exit_code = None
                    output = _output_tail(session, 2000)
                if consume_output:
                    self._completion_consumed.add(session_id)
                self._move_to_finished(session)
                return {"status": "already_exited", "exit_code": session.exit_code, "output": output}
            self._terminate_host_pid(session.pid, session.host_start_time)
        else:
            return {
                # Reject non-positive timeouts — the schema declares minimum=1, but not every caller
                # enforces schemas before dispatch. timeout=0 is falsy, so without this guard it silently
                # fell through (`0 or max_timeout`) to the DEFAULT wait instead of erroring. Salvaged from
                # PR #60004 (credit @isheng-eqi).
                "status": "error",
                "error": "Recovered process cannot be killed after restart because "
                         "its original runtime handle is no longer available",
            }
        return None

    def _stdin_op(self, session_id: str, pty_op, pipe_op, ok: dict) -> dict:
        """Run a stdin operation on a running session — ``pty_op(pty)`` under PTY mode,
        else ``pipe_op(stdin)`` on the Popen pipe — and return *ok* on success."""
        session = self.get(session_id)
        if session is None:
            return _not_found(session_id)
        if session.exited:
            return {"status": "already_exited", "error": "Process has already finished"}
        try:
            if session._pty:
                pty_op(session._pty)
            elif not session.process or not session.process.stdin:
                return {"status": "error", "error": "Process stdin not available (non-local backend or stdin closed)"}
            else:
                pipe_op(session.process.stdin)
            return ok
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def write_stdin(self, session_id: str, data: str) -> dict:
        """Send raw data to a running process's stdin (no newline appended)."""

        def via_pty(pty):
            # pywinpty expects str on Windows; ptyprocess expects bytes on POSIX.
            if _IS_WINDOWS:
                pty.write(data.decode("utf-8") if isinstance(data, bytes) else str(data))
            else:
                # surrogateescape: a PTY is a byte stream — round-trip the original
                # bytes instead of crashing on surrogate content.
                pty.write(data.encode("utf-8", "surrogateescape") if isinstance(data, str) else data)

        def via_pipe(stdin):
            stdin.write(data)
            stdin.flush()
        return self._stdin_op(session_id, via_pty, via_pipe, {"status": "ok", "bytes_written": len(data)})

    def submit_stdin(self, session_id: str, data: str = "") -> dict:
        """Send data + newline to stdin (like pressing Enter).
        On a Windows PTY, Enter is a carriage return: ConPTY treats ``\\r`` as
        end-of-line and a bare ``\\n`` through pywinpty is NOT a line terminator — the
        child's blocking line read (``readline()``, Go ``bufio.Scanner``) never returns
        and the process hangs looking healthy. ``\\r\\n`` gives it both; POSIX keeps ``\\n``."""
        session = self.get(session_id)
        return self.write_stdin(session_id, data + ("\r\n" if _IS_WINDOWS and session and session._pty else "\n"))

    def request_close_terminal(self, session_id: str) -> dict:
        """Ask the desktop GUI to close this process's read-only terminal tab. Does NOT
        kill the process — output keeps buffering and the tab can be reopened from the
        status stack. Errors when no UI close sink is wired."""
        if self.on_close is None:
            return {"status": "error", "error": "close_terminal is only available in the Hermes desktop app."}
        # The session may already be finished (or pruned) — the tab can still
        # linger and be closed, so a missing session is not an error here.
        try:
            self.on_close(self.get(session_id), session_id)
        except Exception as e:
            return {"status": "error", "error": str(e)}
        return {
            "status": "ok", "closed": session_id,
            "note": "Closed the read-only terminal tab. The process was not killed; "
                    "its output remains available and the user can reopen the tab "
                    "from the status stack."}

    def close_stdin(self, session_id: str) -> dict:
        """Close a running process's stdin / send EOF without killing the process."""
        session = self.get(session_id)
        msg = "EOF sent" if session is not None and session._pty else "stdin closed"
        return self._stdin_op(
            session_id, lambda pty: pty.sendeof(), lambda stdin: stdin.close(), {"status": "ok", "message": msg})

    def count_running(self) -> int:
        """O(1) running count for status-bar polling; dict ``len()`` is atomic, no lock."""
        return len(self._running)

    def list_sessions(self, task_id: str = None, session_key: str = None) -> list:
        """Running and recently-finished processes for ``task_id`` and/or ``session_key``;
        cross-task entries sharing the gateway session (a forgotten preview server
        blocking session reset) are flagged ``"session_scoped": true``; rows mirrored
        from another gateway process (see :meth:`sync_from_checkpoint`) are flagged
        ``"peer": true`` so callers know they are display-only.

        When ``task_id`` is given, processes for that task are included. When ``session_key`` is also given,
        session-scoped background processes (``background: true``) registered under that gateway session are
        surfaced too, even if they belong to a different task — so the agent can discover a forgotten
        preview server that is blocking session reset (#29177).
        """
        with self._lock:
            all_sessions = (list(self._running.values()) + list(self._finished.values())
                            + list(self._peers.values()))
        all_sessions = [self._refresh_detached_session(s) for s in all_sessions]
        if task_id or session_key:
            all_sessions = [
                s for s in all_sessions
                if (task_id and s.task_id == task_id) or (session_key and s.session_key == session_key)
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
            # Flag processes surfaced only because they share the gateway session (not the current task) —
            # these are the long-lived background processes a user may have forgotten about (#29177).
            if task_id and session_key and s.task_id != task_id and s.session_key == session_key:
                entry["session_scoped"] = True
            # Trigger metadata for goal-loop judges (a watcher may never exit).
            if s.watch_patterns and not s._watch_disabled:
                entry.update(watch_patterns=list(s.watch_patterns), watch_hit=s._watch_hits > 0)
            if s.notify_on_complete:
                entry["notify_on_complete"] = True
            if s.exited:
                entry["exit_code"] = s.exit_code
            if s.detached:
                entry["detached"] = True
            if s.peer_mirror:
                entry["peer"] = True
            result.append(entry)
        return result

    # ----- Session/Task Queries (for gateway integration) -----

    def _any_running(self, predicate) -> bool:
        """True if any still-running session satisfies *predicate*, after refreshing
        detached sessions so a finished-but-unreaped process reads as inactive."""
        with self._lock:
            sessions = list(self._running.values())
        for session in sessions:
            self._refresh_detached_session(session)
        with self._lock:
            return any(not s.exited and predicate(s) for s in self._running.values())

    def has_active_processes(self, task_id: str) -> bool:
        """Whether any process for ``task_id`` is still running."""
        return self._any_running(lambda s: s.task_id == task_id)

    def has_active_for_session(self, session_key: str, max_active_age: Optional[float] = None) -> bool:
        """Active processes for a gateway session key. Processes older than
        ``max_active_age`` seconds are ignored as stale so a forgotten ``http.server``
        can't freeze session idle/daily reset forever; ``None`` keeps legacy behaviour
        (any running process blocks)."""
        now = time.time()
        return self._any_running(
            lambda s: s.session_key == session_key
            and (max_active_age is None or (now - s.started_at) < max_active_age))

    def has_any_active(self) -> bool:
        """Whether ANY background process is running — scale-to-zero must not
        suspend a gateway with live background work or the process is lost."""
        return self._any_running(lambda s: True)

    def snapshot_running_ids(self, task_id: str) -> frozenset[str]:
        """Running IDs owned by ``task_id`` — a turn-boundary marker: on timeout
        only processes absent from the starting snapshot belong to the abandoned
        turn; older ones intentionally span turns and must survive."""
        with self._lock:
            return frozenset(s.id for s in self._running.values() if s.task_id == task_id and not s.exited)

    def kill_started_since(self, task_id: str, baseline_ids, *, source: str) -> int:
        """Kill ``task_id`` processes created after ``baseline_ids``. Output is
        consumed so an abandoned turn can't enqueue a follow-up reviving work the
        timeout deliberately stopped."""
        return self.kill_all(task_id, exclude_ids=frozenset(baseline_ids or ()), source=source, consume_output=True)

    def kill_all(
        self, task_id: Optional[str] = None, *, exclude_ids: frozenset = frozenset(),
        source: str = "kill_all", consume_output: bool = False) -> int:
        """Kill all running processes, optionally filtered by task_id. Returns count killed."""
        with self._lock:
            targets = [
                s for s in self._running.values()
                if (task_id is None or s.task_id == task_id) and s.id not in exclude_ids and not s.exited
            ]
        return sum(
            self.kill_process(s.id, source=source, consume_output=consume_output).get("status")
            in {"killed", "already_exited"}
            for s in targets)

    # ----- Cleanup / Pruning -----

    def _prune_if_needed(self):
        """Drop expired finished sessions, then the oldest survivor while over
        MAX_PROCESSES. Must hold _lock."""
        now = time.time()
        expired = [sid for sid, s in self._finished.items() if (now - s.started_at) > FINISHED_TTL_SECONDS]
        over_cap = len(self._running) + len(self._finished) - len(expired) >= MAX_PROCESSES
        if over_cap and (survivors := [sid for sid in self._finished if sid not in expired]):
            expired.append(min(survivors, key=lambda sid: self._finished[sid].started_at))
        for sid in expired:
            del self._finished[sid]
        # Belt-and-suspenders against module-lifetime growth: forget consumed /
        # poll-observed marks for any session no longer tracked at all.
        tracked = self._running.keys() | self._finished.keys()
        self._completion_consumed &= tracked
        self._poll_observed &= tracked

    # ----- Checkpoint (crash recovery) -----

    def _row_is_mine(self, row: Dict[str, Any]) -> bool:
        """Whether this registry authored *row* (exact identity, never inference)."""
        return bool(self._owner_token) and row.get("owner_token") == self._owner_token

    def _row_owner_alive(self, row: Dict[str, Any]) -> bool:
        """Whether the gateway process that published *row* is still running.
        Unstamped legacy rows answer False — an unknown writer is treated as gone,
        which is what lets crash recovery adopt them."""
        owner_pid = row.get("owner_pid")
        return bool(owner_pid) and self._host_pid_is_ours(owner_pid, row.get("owner_start_time"))

    def _row_process_identified(self, row: Dict[str, Any]) -> bool:
        """Whether *row* still provably names the host process its author recorded.

        The single identity rule behind mirroring, adoption and pruning alike. A
        recorded ``host_start_time`` is REQUIRED, not merely compared: without a
        baseline ``_host_pid_is_ours`` degrades to bare liveness, and a session
        adopted on that basis also disables the identity check inside
        ``_terminate_host_pid`` — so a recycled PID would be tree-killed by a later
        ``process(action="kill")``. An unidentifiable row is actionable by nobody.
        """
        pid, recorded_start = row.get("pid"), row.get("host_start_time")
        return bool(pid) and recorded_start is not None and self._host_pid_is_ours(pid, recorded_start)

    def _row_is_garbage(self, row: Dict[str, Any]) -> bool:
        """Whether *row* is useless to every reader and may be dropped.

        A host row we cannot identify helps nobody — not this display, not the next
        startup's recovery, both of which now refuse it. A non-host (in-sandbox) row
        is meaningless without the environment handle its owner holds, so it survives
        only while that owner does. Everything else is preserved untouched: an
        identified live process with a dead owner is precisely what
        ``recover_from_checkpoint`` exists to adopt, so pruning on owner death alone
        would delete the feature.
        """
        if row.get("pid_scope", "host") != "host":
            return not self._row_owner_alive(row)
        return not self._row_process_identified(row)

    def _merge_foreign_rows(
        self, published: List[Dict[str, Any]], ours: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rows in the checkpoint that belong to OTHER writers and must survive our
        update.

        Dropped: rows we authored (this write republishes them, so keeping the old
        copies would resurrect anything we just retired), rows carrying an id we are
        publishing ourselves — ``recover_from_checkpoint`` adopts an orphan under a
        NEW owner stamp, and the dead owner's copy would otherwise linger as a
        duplicate — and garbage (see :meth:`_row_is_garbage`).
        """
        claimed = {row.get("session_id") for row in ours}
        return [
            row for row in published
            if not self._row_is_mine(row)
            and row.get("session_id") not in claimed
            and not self._row_is_garbage(row)
        ]

    def _write_checkpoint(self, extra_entries: Optional[List[Dict[str, Any]]] = None):
        """Publish our running processes into the shared checkpoint.

        Read-modify-write under the inter-process lock: this file is written by
        every gateway process on the profile, so our update replaces only the rows
        we authored and leaves every other live owner's rows exactly as they are.

        The whole snapshot→replace sequence also runs under an instance lock. The
        running set is read before the redaction and disk work, so two writers from
        THIS registry (a reader thread's live-output publish racing a completion)
        could otherwise land in the wrong order and put a retired row back on disk.
        """
        try:
            with self._checkpoint_write_lock:
                self._write_checkpoint_locked(extra_entries)
        except Exception as e:
            logger.debug("Failed to write checkpoint file: %s", e, exc_info=True)

    def _write_checkpoint_locked(self, extra_entries: Optional[List[Dict[str, Any]]]) -> None:
        """Body of :meth:`_write_checkpoint`; call only with ``_checkpoint_write_lock``."""
        # Snapshot under the registry lock, serialize outside it: the redaction
        # passes over each tail are far too much work to hold that lock through.
        with self._lock:
            live = [s for s in self._running.values() if not s.exited]
        entries = []
        for s in live:
            # Backfill the start time so recovery can detect PID recycling
            # even for sessions spawned before this field existed.
            if s.host_start_time is None and s.pid_scope == "host" and s.pid:
                s.host_start_time = self._safe_host_start_time(s.pid)
            entry = {"session_id": s.id, **{f: getattr(s, f) for f in _CHECKPOINT_FIELDS}}
            # Redact inline credentials before persisting (~/.hermes/processes.json).
            # Recovery uses command only for display (adoption re-validates the
            # PID, never re-runs it), so masking is lossless.
            # ``force``: this file is a disk artifact another gateway process
            # renders for the user, so it must not return raw secrets even
            # when the user turned redaction off. ``code_file=False`` keeps the
            # ENV/JSON assignment passes on — ``DB_PASSWORD=…`` in a command
            # line or a log line is exactly what must not land here.
            # See #77484.
            entry["command"] = redact_sensitive_text(s.command, force=True, code_file=False)
            entry["owner_task_id"] = s.owner_task_id or s.task_id
            # Peer gateways have their own Python heap and cannot read our
            # buffers; a bounded tail is what makes the row renderable there.
            entry["output_tail"] = _checkpoint_output_tail(s.output_buffer)
            entry.update(
                owner_token=self._owner_token, owner_pid=self._owner_pid,
                owner_start_time=self._owner_start_time)
            entries.append(entry)
        from utils import atomic_json_write
        with _checkpoint_lock() as locked:
            if not locked:
                # Skipping costs us this publication, which the next write redoes.
                # Writing anyway risks dropping another gateway's row for good.
                logger.warning(
                    "Skipping process checkpoint write: could not take the shared lock. "
                    "%d running process(es) are unpublished until the next write.", len(entries))
                return
            entries.extend(self._merge_foreign_rows(self._read_checkpoint_rows(), entries))
            if extra_entries:
                tracked_ids = {item.get("session_id") for item in entries}
                entries.extend(item for item in extra_entries if item.get("session_id") not in tracked_ids)
            atomic_json_write(CHECKPOINT_PATH, entries)

    def publish_checkpoint(self) -> None:
        """Persist the current running set now.

        Spawn callers finish a session AFTER the registry has already registered and
        checkpointed it — session lineage, watcher routing and the notify/watch flags
        are all stamped on the way back out, and all three are checkpoint fields. The
        row written at registration is therefore incomplete, and for a quiet process
        nothing else ever rewrites it. Callers publish once their metadata is final.
        """
        self._write_checkpoint()

    @staticmethod
    def _read_checkpoint_rows() -> List[Dict[str, Any]]:
        """Parse the checkpoint WITHOUT taking the inter-process lock — for callers
        that already hold it. ``[]`` when absent/unreadable/malformed."""
        if not CHECKPOINT_PATH.exists():
            return []
        try:
            entries = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
        return [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []

    def _read_checkpoint_entries(self) -> List[Dict[str, Any]]:
        """Parsed checkpoint rows, read under the inter-process lock so a concurrent
        publisher's replace can never be observed half-applied. Unlike a write, a read
        still proceeds when the lock is unavailable: it destroys nothing, and the
        atomic replace means the worst case is a slightly stale view."""
        with _checkpoint_lock():
            return self._read_checkpoint_rows()

    def sync_from_checkpoint(self) -> int:
        """Mirror OTHER gateway processes' live rows from the shared checkpoint.

        Desktop and the messaging gateway are separate Python processes with
        separate registries; the checkpoint is the only thing both see. Each call
        rebuilds the mirror set from the file, so a row the owner retired stops
        being reported here on the next poll. Returns the number of mirrors held.

        Adoption is deliberately strict: host scope only, and the PID must still
        be the process the owner recorded (``host_start_time`` must be present AND
        match). A PID alone proves nothing — the kernel recycles them — and a
        mirror we could not identify would be a row pointing at a stranger.
        Locally-owned ids are never overwritten: this process is authoritative for
        the work it spawned.
        """
        mirrors: Dict[str, ProcessSession] = {}
        for entry in self._read_checkpoint_entries():
            process_id = str(entry.get("session_id") or "")
            pid, recorded_start = entry.get("pid"), entry.get("host_start_time")
            if not process_id or entry.get("pid_scope", "host") != "host":
                continue
            if not self._row_process_identified(entry):
                continue
            if self._row_is_mine(entry):
                continue  # our own publication — mirroring it would make us a peer of ourselves
            with self._lock:
                if process_id in self._running or process_id in self._finished:
                    continue  # ours; the checkpoint copy is our own publication
                existing = self._peers.get(process_id)
            if existing is not None:
                # Keep object identity across polls (stable rows for readers) and
                # refresh only what the owner republishes.
                with existing._lock:
                    existing.output_buffer = str(entry.get("output_tail") or "")[-MAX_OUTPUT_CHARS:]
                mirrors[process_id] = existing
                continue
            mirrors[process_id] = ProcessSession(
                id=process_id,
                command=entry.get("command", "unknown"),
                task_id=entry.get("task_id", ""),
                owner_task_id=entry.get("owner_task_id", "") or entry.get("task_id", ""),
                session_key=entry.get("session_key", ""),
                parent_session_id=entry.get("parent_session_id", ""),
                pid=pid,
                host_start_time=recorded_start,
                pid_scope="host",
                systemd_unit=entry.get("systemd_unit", ""),
                cwd=entry.get("cwd"),
                started_at=entry.get("started_at", time.time()),
                output_buffer=str(entry.get("output_tail") or "")[-MAX_OUTPUT_CHARS:],
                detached=True,
                peer_mirror=True,
                # The owning gateway holds the completion contract and the watcher;
                # a mirror that notified would double-deliver to the same user.
                notify_on_complete=False,
            )
        with self._lock:
            self._peers = mirrors
            return len(self._peers)

    def recover_from_checkpoint(self) -> int:
        """On gateway startup, probe PIDs from the checkpoint file; returns how many
        were recovered as detached sessions.

        Only ORPHANED rows are adopted. A row whose publishing gateway is still
        running belongs to that gateway: adopting it would give two processes a
        handle on one child, each believing it may kill it. Those rows reach the
        user through :meth:`sync_from_checkpoint` as read-only mirrors instead.
        """
        if not CHECKPOINT_PATH.exists():
            return 0
        entries = self._read_checkpoint_entries()
        recovered = 0
        unresolved_scope_entries: List[Dict[str, Any]] = []
        for entry in entries:
            pid, pid_scope = entry.get("pid"), entry.get("pid_scope", "host")
            if not pid:
                continue
            if self._row_owner_alive(entry):
                continue
            if pid_scope != "host":  # in-sandbox PIDs mean nothing once the env handle is gone
                logger.info(
                    "Skipping recovery for non-host process: %s (pid=%s, scope=%s)",
                    entry.get("command", "unknown")[:60], pid, pid_scope)
                continue
            # Alive AND provably the same process: across a restart the kernel may
            # have recycled the PID onto a stranger, and adopting it would let a
            # later kill tree-kill e.g. a browser. A row with no recorded baseline
            # cannot clear that bar (see ``_row_process_identified``).
            if not self._row_process_identified(entry):
                if self._is_host_pid_alive(pid):
                    logger.info(
                        "Not recovering session %s: pid %d is alive but its "
                        "start time no longer matches — PID was recycled onto "
                        "an unrelated process; refusing to adopt it.",
                        entry.get("session_id", "?"), pid)
                systemd_unit = entry.get("systemd_unit", "")
                if systemd_unit and not _stop_systemd_unit(systemd_unit):
                    logger.warning(
                        "Could not reap persisted scope %s for dead wrapper pid %s; "
                        "retaining checkpoint entry for the next startup",
                        systemd_unit, pid)
                    unresolved_scope_entries.append(entry)
                continue
            fields = {f: entry.get(f, _CHECKPOINT_DEFAULTS[f]) for f in _CHECKPOINT_FIELDS}
            fields.update(
                command=entry.get("command", "unknown"),
                owner_task_id=entry.get("owner_task_id", "") or entry.get("task_id", ""),
                started_at=entry.get("started_at", time.time()))
            # detached: can't read output, but can report status + kill
            session = ProcessSession(id=entry["session_id"], detached=True, **fields)
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
                    **{key: getattr(session, f"watcher_{key}") for key in _WATCHER_ROUTE_KEYS},
                    "notify_on_complete": session.notify_on_complete,
                    "parent_session_id": session.parent_session_id,
                })
        self._write_checkpoint(extra_entries=unresolved_scope_entries)
        return recovered


process_registry = ProcessRegistry()


# --- the "process_manage" tool schema + handler -----------------------------------
from tools.registry import registry, tool_error

PROCESS_SCHEMA = {
    "name": "process_manage",
    # The enum names the verbs; the description keeps only non-obvious semantics
    # (write-vs-submit is the one real trap: a lone \n on a Windows PTY is not Enter).
    # See #95681.
    "description": (
        "Poll, wait on, or kill background terminal processes (from "
        "terminal(background=true)). "
        "poll: status + new output. log: full output, paged. wait: block "
        "until exit or timeout (partial output on timeout). write vs "
        "submit: submit appends Enter — use it to answer prompts; write "
        "sends raw bytes, no newline. close: EOF stdin. kill: terminate."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "poll", "log", "wait", "kill", "write", "submit", "close"]
            },
            "session_id": {
                "type": "string",
                "description": "From terminal background output; any unique prefix works ('4dae' for proc_4dae56ca81f6). Required except for 'list'."
            },
            "data": {
                "type": "string",
                "description": "Stdin text for write/submit."
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds for 'wait'.",
                "minimum": 1
            },
            "offset": {
                "type": "integer",
                "description": "Log line offset (default: last 200)."
            },
            "limit": {
                "type": "integer",
                "description": "Max log lines.",
                "minimum": 1
            }
        },
        "required": ["action"]
    }
}


def _redact_process_result(result: dict) -> dict:
    """Redact secrets from background-process output before it reaches the model,
    session.db and CLI, mirroring the foreground ``terminal`` redaction so the two
    surfaces can't diverge. Respects ``security.redact_secrets``; ``redact_terminal_output``
    picks ``code_file`` from the recorded command. The command itself is redacted too.

    The command string itself is also redacted in case it carried an inline credential. See #43025.
    """
    if not isinstance(result, dict):
        return result
    from agent.redact import redact_sensitive_text, redact_terminal_output

    command = result.get("command") or ""
    for key in ("output", "output_preview"):
        if isinstance(value := result.get(key), str) and value:
            result[key] = redact_terminal_output(value, command)
    if isinstance(command, str) and command:
        result["command"] = redact_sensitive_text(command, code_file=True)
    return result


def _list_processes(task_id) -> dict:
    # Also surface session-scoped background processes (e.g. a forgotten preview
    # server): they share the gateway session_key and can block session reset.
    session_key = ""
    with suppress(Exception):
        # See #29177.
        from tools.approval_context import get_current_session_key
        session_key = get_current_session_key(default="") or ""
    return {"processes": [
        _redact_process_result(p)
        for p in process_registry.list_sessions(task_id=task_id, session_key=session_key or None)]}


# action -> (handler(session_id, args) -> dict, redact output?). Output-bearing
# actions are redacted; stdin actions return only status.
_SESSION_ACTIONS = {
    "poll": (lambda sid, a: process_registry.poll(sid), True),
    "log": (lambda sid, a: process_registry.read_log(sid, offset=a.get("offset"), limit=a.get("limit", 200)), True),
    "wait": (lambda sid, a: process_registry.wait(sid, timeout=a.get("timeout")), True),
    "kill": (lambda sid, a: process_registry.kill_process(sid), True),
    "write": (lambda sid, a: process_registry.write_stdin(sid, str(a.get("data", ""))), False),
    "submit": (lambda sid, a: process_registry.submit_stdin(sid, str(a.get("data", ""))), False),
    "close": (lambda sid, a: process_registry.close_stdin(sid), False),
}


def _handle_process(args, **kw):
    action = args.get("action", "")
    # Coerce to string — some models send session_id as an integer
    session_id = str(args.get("session_id", "")) if args.get("session_id") is not None else ""
    if action == "list":
        return json.dumps(_list_processes(kw.get("task_id")), ensure_ascii=False)
    if action in _SESSION_ACTIONS:
        if not session_id:
            return tool_error(f"session_id is required for {action}")
        handler, redact = _SESSION_ACTIONS[action]
        result = handler(session_id, args)
        return json.dumps(_redact_process_result(result) if redact else result, ensure_ascii=False)
    return tool_error(f"Unknown process action: {action}. Use: list, poll, log, wait, kill, write, submit, close")


registry.register(
    name="process_manage",
    toolset="terminal",
    schema=PROCESS_SCHEMA,
    handler=_handle_process,
    emoji="⚙️",
)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

MAX_ACTIVE_PROCESS_AGE = 86400  # 24h default — see session_reset.bg_process_max_age_hours (#29177)
# ---- END PLUGIN-COMPAT ----
