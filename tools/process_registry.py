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

import codecs
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
# systemd transient scopes exist only on Linux. Gate every scope-path branch
# on this constant (not merely "not Windows") so macOS and other POSIX
# platforms provably never touch systemd code (#70716 cross-platform audit).
_IS_LINUX = platform.system() == "Linux"
from tools.environments.local import _find_shell, _resolve_safe_cwd, _sanitize_subprocess_env
from hermes_cli._subprocess_compat import windows_hide_flags
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hermes_cli.config import get_hermes_home
from tools.async_delegation_formatter import format_async_delegation
from tools.delegation_outcome import (
    delegation_evidence_fields as _delegation_evidence_fields,
    derive_result_outcome as _derive_result_outcome,
)

from agent.redact import redact_sensitive_text

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

# Lifetime cap — independent of the strike counter above. A process whose
# pattern recurs at a cadence just above WATCH_MIN_INTERVAL_SECONDS (e.g. a
# service restarted repeatedly over a day) never trips the consecutive-strike
# limit, since each match lands in its own clean cooldown window, yet still
# forces a full-context agent turn every single time (#93513). watch_patterns
# is documented as "ONLY for rare one-shot mid-process signals", so once a
# session has delivered this many matches over its whole life we disable it
# and fall back to notify_on_complete, same as the strike-limit path.
WATCH_LIFETIME_MAX_HITS = 8

# Global circuit breaker — across all sessions. Secondary safety net so concurrent
# siblings can't collectively flood the user even when each is under its own cap.
WATCH_GLOBAL_MAX_PER_WINDOW = 15
WATCH_GLOBAL_WINDOW_SECONDS = 10
WATCH_GLOBAL_COOLDOWN_SECONDS = 30


# ---------------------------------------------------------------------------
# systemd cgroup isolation for gateway-spawned local executors (#70716)
# ---------------------------------------------------------------------------
# When Hermes runs as a systemd gateway with MemoryHigh/MemoryMax limits,
# local background terminal commands inherit the gateway's cgroup.  A
# memory-heavy executor (Codex, tests, Node) can push the whole cgroup past
# MemoryMax and trigger systemd-oomd to kill the ENTIRE gateway — taking down
# the messaging control plane and silently losing the active turn.
#
# Wrapping the spawn in ``systemd-run --user --scope --unit=hermes-worker-<pid>``
# places the worker in its own transient cgroup so an OOM in the worker kills
# only the worker, not the gateway.  We probe *once* whether
# ``systemd-run --user --scope`` is actually usable (the binary can exist on
# the PATH while the user D-Bus session is unavailable — common for system
# services and containers), and cache the result for the process lifetime.

_SYSTEMD_SCOPE_AVAILABLE: Optional[bool] = None
_SYSTEMD_SCOPE_PROBE_LOCK = threading.Lock()
_SYSTEMD_SCOPE_PROBED_AT = 0.0
_SYSTEMD_SCOPE_FAILURE_TTL_SECONDS = 60.0
_MIN_WORKER_MEMORY_MAX_BYTES = 64 * 1024 * 1024
_DEFAULT_WORKER_MEMORY_MAX_BYTES = 1024 * 1024 * 1024
_WORKER_MEMORY_MAX_CAP_BYTES = 4 * 1024 * 1024 * 1024


def _worker_memory_max_bytes() -> int:
    """Return a finite per-worker cgroup limit without widening host risk.

    The proposed local-memory-guard environment override is honored when it
    tightens the safe bound, so this isolation composes with PR #57121 instead
    of inventing a second knob.  An oversized override cannot widen host risk.
    Otherwise retain the tighter of the gateway's current cgroup-v2
    ``memory.max`` and half of physical RAM, capped at 4 GiB.  This keeps the
    sibling worker outside the gateway cgroup while ensuring the worker cannot
    consume memory up to the enclosing user slice or host limit.
    """
    override_bound: Optional[int] = None
    override = os.getenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "").strip()
    if override:
        override_valid = False
        try:
            parsed = int(override) * 1024 * 1024
            if parsed >= _MIN_WORKER_MEMORY_MAX_BYTES:
                override_bound = parsed
                override_valid = True
        except ValueError:
            pass
        if not override_valid:
            logger.warning(
                "Ignoring invalid TERMINAL_LOCAL_MEMORY_MAX_MB=%r; "
                "expected an integer representing at least %d MiB",
                override,
                _MIN_WORKER_MEMORY_MAX_BYTES // (1024 * 1024),
            )

    candidates: List[int] = []
    try:
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
            if line.startswith("0::"):
                relative = line.partition("::")[2].lstrip("/")
                raw_limit = (
                    Path("/sys/fs/cgroup") / relative / "memory.max"
                ).read_text(encoding="utf-8").strip()
                if raw_limit.isdigit():
                    cgroup_limit = int(raw_limit)
                    if cgroup_limit >= _MIN_WORKER_MEMORY_MAX_BYTES:
                        candidates.append(cgroup_limit)
                break
    except (OSError, ValueError):
        pass

    try:
        physical_bytes = int(os.sysconf("SC_PHYS_PAGES")) * int(
            os.sysconf("SC_PAGE_SIZE")
        )
        physical_bound = min(
            _WORKER_MEMORY_MAX_CAP_BYTES,
            max(_MIN_WORKER_MEMORY_MAX_BYTES, physical_bytes // 2),
        )
        candidates.append(physical_bound)
    except (OSError, ValueError, TypeError):
        pass

    safe_bound = min(candidates) if candidates else _DEFAULT_WORKER_MEMORY_MAX_BYTES
    return min(override_bound, safe_bound) if override_bound else safe_bound


def _systemd_run_user_scope_available() -> bool:
    """Return True if ``systemd-run --user --scope`` can create a cgroup.

    Cached after the first probe.  ``shutil.which`` alone is insufficient:
    in system-service deployments (and containers) the user D-Bus session
    bus that ``systemd-run --user`` needs may be absent even though the
    binary is on PATH, causing every spawn to fail with
    ``Failed to connect to user bus``.  We do a cheap no-op probe
    (``systemd-run --user --scope --unit=… -- /bin/true``) and remember the
    outcome.
    """
    global _SYSTEMD_SCOPE_AVAILABLE, _SYSTEMD_SCOPE_PROBED_AT
    cached = _SYSTEMD_SCOPE_AVAILABLE
    now = time.monotonic()
    if cached is True:
        return True
    if (
        cached is False
        and now - _SYSTEMD_SCOPE_PROBED_AT < _SYSTEMD_SCOPE_FAILURE_TTL_SECONDS
    ):
        return False

    # Double-checked locking keeps concurrent first-use spawns from observing
    # a temporary False while the definitive probe is still in flight.  Such a
    # race would launch the losing workload back inside the gateway cgroup.
    with _SYSTEMD_SCOPE_PROBE_LOCK:
        cached = _SYSTEMD_SCOPE_AVAILABLE
        now = time.monotonic()
        if cached is True:
            return True
        if (
            cached is False
            and now - _SYSTEMD_SCOPE_PROBED_AT
            < _SYSTEMD_SCOPE_FAILURE_TTL_SECONDS
        ):
            return False

        available = False
        if _IS_LINUX:
            try:
                import shutil

                binary = shutil.which("systemd-run")
                if binary:
                    # Probe: create a transient scope that immediately exits.
                    # A unique unit avoids collisions; timeout bounds D-Bus.
                    probe_unit = f"hermes-probe-scope-{os.getpid()}-{uuid.uuid4().hex[:8]}"
                    result = subprocess.run(
                        [
                            binary, "--user", "--scope", "--quiet",
                            "--unit", probe_unit,
                            "--collect",
                            "--property", "MemoryAccounting=yes",
                            "--property", f"MemoryMax={_worker_memory_max_bytes()}",
                            "--property", "OOMPolicy=kill",
                            "--",
                            "/bin/true",
                        ],
                        capture_output=True,
                        timeout=3,
                    )
                    available = result.returncode == 0
                    if not available:
                        logger.debug(
                            "systemd-run --user --scope probe failed (rc=%s): %s",
                            result.returncode,
                            (result.stderr or b"").decode(
                                "utf-8", "replace"
                            ).strip(),
                        )
            except Exception as exc:
                logger.debug("systemd-run --user --scope probe error: %s", exc)

        _SYSTEMD_SCOPE_AVAILABLE = available
        _SYSTEMD_SCOPE_PROBED_AT = time.monotonic()
        return available


def _is_supervised_gateway_process() -> bool:
    """Return whether this process is in a supervised Hermes gateway runtime.

    Both supervisor markers and ``_HERMES_GATEWAY`` are inherited by every
    descendant, and importing ``gateway.run`` also sets the latter. Require
    this process to own the live gateway PID file as well. That keeps transient
    systemd scopes limited to the gateway itself instead of terminal children
    or unrelated interactive CLIs in the same supervised process tree.
    """
    if os.environ.get("_HERMES_GATEWAY") != "1":
        return False

    try:
        from gateway.restart import is_gateway_supervisor_process
        from gateway.status import get_running_pid

        return (
            is_gateway_supervisor_process()
            and get_running_pid(cleanup_stale=False) == os.getpid()
        )
    except Exception as exc:
        logger.debug("Could not verify supervised gateway process identity: %s", exc)
        return False


def _build_systemd_scope_argv(
    shell_argv: List[str],
    unit_suffix: str,
) -> List[str]:
    """Wrap *shell_argv* in a ``systemd-run --user --scope`` invocation.

    The resulting cgroup gets its own memory accounting so an OOM in the
    worker does not kill the gateway cgroup (#70716).  ``--collect`` makes
    the transient scope self-clean after exit; ``--unit`` gives it a
    recognisable name for ``systemctl --user status`` / journalctl.
    """
    import shutil

    binary = shutil.which("systemd-run")
    if binary is None:
        # Caller should have checked _systemd_run_user_scope_available();
        # guard anyway so we never pass None into Popen.
        return shell_argv
    unit_name = f"hermes-worker-{unit_suffix}"
    memory_max = _worker_memory_max_bytes()
    return [
        binary,
        "--user",
        "--scope",
        "--quiet",
        "--unit",
        unit_name,
        "--collect",
        "--property",
        "MemoryAccounting=yes",
        "--property",
        f"MemoryMax={memory_max}",
        "--property",
        "OOMPolicy=kill",
        "--",
        *shell_argv,
    ]


def _stop_systemd_unit(unit_name: str) -> bool:
    """Stop a transient systemd user scope by unit name.

    This reaps the *entire* cgroup — catching double-forked descendants that
    survive a plain PID signal because they were reparented to init inside the
    scope (issue #70716, reviewer gap #2).  ``systemctl --user stop`` sends
    SIGTERM to every process in the unit's cgroup and escalates to SIGKILL
    after the unit's ``TimeoutStopSec``.

    Returns True if the unit was successfully stopped (or was already gone),
    False if ``systemctl`` is unavailable or the stop command failed.
    """
    import shutil

    binary = shutil.which("systemctl")
    if binary is None:
        return False
    try:
        result = subprocess.run(
            [binary, "--user", "stop", unit_name],
            capture_output=True,
            timeout=15,
        )
        if result.returncode != 0:
            stderr = (result.stderr or b"").decode(errors="replace").strip()
            stderr_lower = stderr.lower()
            if any(
                marker in stderr_lower
                for marker in ("not loaded", "not found", "does not exist")
            ):
                return True
            logger.debug(
                "systemctl --user stop %s exited %d: %s",
                unit_name, result.returncode,
                stderr,
            )
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


@dataclass
class ProcessSession:
    """A tracked background process with output buffering."""
    id: str                                     # Unique session ID ("proc_xxxxxxxxxxxx")
    command: str                                 # Original command string
    task_id: str = ""                           # Task/sandbox isolation key
    owner_task_id: str = ""                     # RAW spawning task id (e.g. subagent "sa-...");
                                                # task_id is the CONTAINER key and may be collapsed
                                                # to "default"/session key by _resolve_container_task_id,
                                                # so ownership checks must use this field (#child-notify)
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
    systemd_unit: str = ""                      # transient scope unit name when spawned under systemd-run (#70716)
    # Watcher/notification metadata (persisted for crash recovery)
    watcher_platform: str = ""
    watcher_chat_id: str = ""
    watcher_user_id: str = ""
    watcher_user_name: str = ""
    watcher_thread_id: str = ""
    watcher_message_id: str = ""                # Triggering message id — reply anchor for topic routing
    watcher_interval: int = 0                   # 0 = no watcher configured
    # Session-db id of the conversation that spawned this process. Lets the
    # gateway's completion pre-flight (_classify_completion_target) drop
    # notifications whose spawning session was closed at an explicit user
    # boundary (/new), instead of injecting them into the chat's NEW session.
    parent_session_id: str = ""
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


from tools.process_registry_control import ProcessRegistryControlMixin
from tools.process_registry_runtime import ProcessRegistryRuntimeMixin

# Spawn implementation moved to ProcessRegistryRuntimeMixin; its Popen path
# retains the guarded ``start_new_session=`` contract checked by Windows safety
# tests while this compatibility owner preserves public imports.


class ProcessRegistry(ProcessRegistryRuntimeMixin, ProcessRegistryControlMixin):
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
        # Rehydrate durable delegation completions only at registry startup.
        # Consumers still inject them as fresh turns through this existing rail.
        try:
            from tools.async_delegation import restore_undelivered_completions
            restore_undelivered_completions(self.completion_queue)
        except Exception as exc:
            logger.warning("Could not restore async delegation completions: %s", exc)

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


def _model_not_found_patterns() -> "list[str]":
    """Model-not-found phrases from the failover classifier.

    Imported from ``agent.error_classifier`` so the batch renderer applies
    the SAME classification the failover path consumes — no hand-copied
    pattern list to drift. Fails open to a minimal built-in set so a
    classifier import problem never hides the per-task blocks.
    (Import approach from PR #97667 by @liuhao1024.)
    """
    try:
        from agent.error_classifier import _MODEL_NOT_FOUND_PATTERNS

        return list(_MODEL_NOT_FOUND_PATTERNS)
    except Exception:
        return ["is not a valid model", "model not found", "model_not_found"]


def _delegation_config() -> dict:
    """Load the active delegation config (model/provider/fallbacks), fail-open.

    Mirrors ``tools.delegate_tool._load_config`` so the renderer sees the same
    ``model`` / ``provider`` the dispatcher used, without importing the heavy
    delegation module at import time. Returns ``{}`` on any error so callers
    fail open to "no notice" rather than dropping the per-task blocks.
    """
    try:
        from tools.delegate_tool import _load_config as _cfg

        return _cfg() or {}
    except Exception:
        return {}


def _delegation_model_not_found(results, config) -> bool:
    """True when a result entry reflects a config-level model_not_found rejection.

    Matches when at least one entry's error/summary text contains both a
    model-not-found phrase AND the name of the currently-configured delegation
    model — so a stale task failing on a *different* (removed) model is not
    mis-attributed to the config-level root cause.
    """
    model = (config or {}).get("model")
    if not model:
        return False
    model = str(model).lower()
    for r in results or []:
        text = " ".join(
            str(part) for part in (r.get("error"), r.get("summary")) if part
        ).lower()
        if not text or model not in text:
            continue
        if any(p in text for p in _model_not_found_patterns()):
            return True
    return False


def _delegation_model_not_found_notice(results) -> "list[str] | None":
    """Build the config-level model_not_found notice lines, or None.

    Returns ``None`` unless at least one result entry shows the configured
    delegation model being rejected by its provider, in which case a short
    actionable block is returned. Every failure path fails open to ``None`` so
    a config hiccup never hides the per-task blocks. Emit once per batch.
    """
    config = _delegation_config()
    if not _delegation_model_not_found(results, config):
        return None
    model = config.get("model") or "?"
    provider = config.get("provider") or "configured provider"
    lines = [
        "⚠ SUBAGENT MODEL REJECTED: the configured Subagent Model "
        f'"{model}" was rejected by provider "{provider}" '
        "(HTTP 400: not a valid model ID).",
        "Every task in this batch failed for this reason before doing any work.",
        "Check Settings → Advanced → Subagent Model (or: "
        "hermes config get delegation.model).",
    ]
    try:
        from hermes_cli.fallback_config import get_fallback_chain

        if not get_fallback_chain(config):
            lines.append(
                "No fallback chain is configured, so no failover was attempted."
            )
    except Exception:
        pass
    return lines


def _format_async_delegation(evt: dict) -> str:
    """Compatibility delegate to the bounded async renderer."""
    return format_async_delegation(
        evt,
        format_age=_format_age,
        model_not_found_notice=_delegation_model_not_found_notice,
    )


def _delegation_attribution_line(evt: dict) -> "str | None":
    """One-line delegation attribution for a child-originated process event.

    Subagents run their terminal sessions under ``task_id == subagent_id``
    (delegate_tool._run_single_child). When a background process they started
    completes, its notification is routed to the PARENT conversation by
    design (children consume their own waits via process(wait); anything
    that outlives the child must land where a durable consumer exists).
    Without attribution the parent-facing user sees an anonymous raw output
    wall mid-conversation with no hint it came from a delegation. Resolve
    the task_id against the live + recently-finished subagent registry and
    return a short provenance line, or None for parent-owned processes.
    """
    task_id = str(evt.get("owner_task_id") or evt.get("task_id") or "")
    if not task_id.startswith("sa-"):
        return None
    try:
        from tools.delegate_tool import get_subagent_attribution

        info = get_subagent_attribution(task_id)
    except Exception:
        info = None
    if not info:
        # The task_id shape says "subagent" even when the registry entry has
        # aged out — still attribute generically rather than anonymously.
        return f"Started by subagent {task_id} (delegate_task)."
    goal = str(info.get("goal") or "").strip()
    if len(goal) > 120:
        goal = goal[:117] + "..."
    deleg = info.get("delegation_id")
    parts = [f"Started by subagent {task_id}"]
    if deleg:
        parts.append(f"of delegation {deleg}")
    line = " ".join(parts) + "."
    if goal:
        line += f' Task: "{goal}"'
    return line


def format_process_notification(evt: dict) -> "str | None":
    """Format a process notification event into a [IMPORTANT: ...] message.

    Handles completion events (notify_on_complete), watch pattern matches,
    and watch disabled events from the unified completion_queue.
    """
    evt_type = evt.get("type", "completion")
    _sid = evt.get("session_id", "unknown")
    _cmd = evt.get("command", "unknown")
    _attribution = _delegation_attribution_line(evt)

    if evt_type == "watch_disabled":
        return f"[IMPORTANT: {evt.get('message', '')}]"

    # Overflow events carry their human-readable summary in `message` —
    # without this case they fall through to the completion formatter and
    # surface as a phantom "process exited (exit code ?)" notification.
    if evt_type in ("watch_overflow_tripped", "watch_overflow_released"):
        return f"[IMPORTANT: {evt.get('message', '')}]"

    if evt_type == "watch_match":
        _pat = evt.get("pattern", "?")
        _out = evt.get("output", "")
        _sup = evt.get("suppressed", 0)
        text = (
            f"[IMPORTANT: Background process {_sid} matched "
            f"watch pattern \"{_pat}\".\n"
        )
        if _attribution:
            text += f"{_attribution}\n"
        text += (
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
    text = (
        f"[IMPORTANT: Background process {_sid} {_status} "
        f"(exit code {_exit}{_signal}).\n"
    )
    if _attribution:
        text += f"{_attribution}\n"
        # A subagent-owned process's full output belongs in the child's
        # transcript/summary, not as a raw wall in the parent conversation —
        # trim the tail hard while keeping enough to recognise failures.
        if isinstance(_out, str) and len(_out) > 600:
            _out = (
                "...(output trimmed — subagent-owned process; see the "
                "delegation's live transcript for full output)\n"
                + _out[-600:]
            )
    text += (
        f"Command: {_cmd}\n"
        f"Output:\n{_out}]"
    )
    return text


# ---------------------------------------------------------------------------
# Registry -- the "process" tool schema + handler
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error

PROCESS_SCHEMA = {
    "name": "process_manage",
    # Dieted (#95681): the action enum names the verbs; the description
    # keeps only non-obvious semantics. write-vs-submit is the tool's one
    # real trap (a lone \n on a Windows PTY is not a line terminator) —
    # that teaching gains emphasis rather than losing it.
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
            {
                "processes": [
                    _redact_process_result(p)
                    for p in process_registry.list_sessions(task_id=task_id, session_key=session_key or None)
                ]
            },
            ensure_ascii=False,
        )
    elif action in {"poll", "log", "wait", "kill", "write", "submit", "close"}:
        if not session_id:
            return tool_error(f"session_id is required for {action}")
        if action == "poll":
            return json.dumps(_redact_process_result(process_registry.poll(session_id)), ensure_ascii=False)
        elif action == "log":
            return json.dumps(_redact_process_result(process_registry.read_log(
                session_id, offset=args.get("offset"), limit=args.get("limit", 200))), ensure_ascii=False)
        elif action == "wait":
            return json.dumps(_redact_process_result(process_registry.wait(session_id, timeout=args.get("timeout"))), ensure_ascii=False)
        elif action == "kill":
            return json.dumps(
                _redact_process_result(process_registry.kill_process(session_id)),
                ensure_ascii=False,
            )
        elif action == "write":
            return json.dumps(process_registry.write_stdin(session_id, str(args.get("data", ""))), ensure_ascii=False)
        elif action == "submit":
            return json.dumps(process_registry.submit_stdin(session_id, str(args.get("data", ""))), ensure_ascii=False)
        elif action == "close":
            return json.dumps(process_registry.close_stdin(session_id), ensure_ascii=False)
    return tool_error(f"Unknown process action: {action}. Use: list, poll, log, wait, kill, write, submit, close")


registry.register(
    name="process_manage",
    toolset="terminal",
    schema=PROCESS_SCHEMA,
    handler=_handle_process,
    emoji="⚙️",
)
