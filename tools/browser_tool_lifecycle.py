"""Browser session lifecycle: inactivity janitor, orphan reaper, per-session teardown, atexit cleanup.

Split out of ``tools/browser_tool.py``. Facade-owned state is read through ``_bt`` (``tools.browser_tool``, resolved per call) — no import cycle.
"""

import contextlib
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from tools.browser_tool_origin import origin as _bt
from tools import browser_tool_cdp as _cdp
from tools import browser_tool_cloud as _cloud
from tools import browser_tool_session as _session
from tools import browser_tool_install as _install
from tools import browser_tool_real_profile as _real_profile


class BrowserCleanupReason(str, Enum):
    """Why browser ownership is being torn down."""

    TERMINAL = "terminal"
    INACTIVITY = "inactivity"
    PROVIDER_EXPIRY = "provider_expiry"
    RESTART_ROLLBACK = "restart_rollback"


class BrowserTaskState(str, Enum):
    """Process-local lifecycle for one bare browser task."""

    STARTING = "starting"
    ACTIVE = "active"
    RETIRING = "retiring"
    RETIRED = "retired"


class OrphanTargetOwnership(str, Enum):
    """Confidence in exact target ownership recovered from daemon metadata."""

    PINNED = "pinned"
    CONFIRMED_UNPINNED = "confirmed_unpinned"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class _PendingProviderCleanup:
    """Provider session identity retained until close is confirmed."""

    task_id: str
    provider: Any
    session_id: str


class _BrowserSessionRetiredError(RuntimeError):
    """Implicit lookup targeted a retired or still-retiring browser task."""


def _task_state_locked(bare_task_id: str) -> BrowserTaskState:
    """Read lifecycle state while ``_cleanup_lock`` is held."""
    if bare_task_id in _bt._retired_browser_tasks:
        return BrowserTaskState.RETIRED
    state = _bt._browser_task_states.get(bare_task_id, BrowserTaskState.ACTIVE)
    # The compatibility tombstone stays authoritative. A stale RETIRED map
    # entry cannot survive a test/hot-reload caller clearing the set.
    return BrowserTaskState.ACTIVE if state is BrowserTaskState.RETIRED else state


def _set_task_state_locked(
    bare_task_id: str,
    state: BrowserTaskState,
    *,
    cleanup_reason: Optional[BrowserCleanupReason] = None,
) -> None:
    """Write lifecycle state and keep the compatibility tombstone in sync."""
    _bt._browser_task_states[bare_task_id] = state
    if state is BrowserTaskState.RETIRED:
        _bt._retired_browser_tasks.add(bare_task_id)
    else:
        _bt._retired_browser_tasks.discard(bare_task_id)
    if cleanup_reason is None:
        if state is not BrowserTaskState.RETIRING:
            _bt._browser_task_cleanup_reasons.pop(bare_task_id, None)
    else:
        _bt._browser_task_cleanup_reasons[bare_task_id] = cleanup_reason


def _task_generation_locked(bare_task_id: str) -> int:
    return _bt._browser_task_generations.get(bare_task_id, 0)


def _advance_task_generation_locked(bare_task_id: str) -> int:
    generation = _task_generation_locked(bare_task_id) + 1
    _bt._browser_task_generations[bare_task_id] = generation
    return generation


def _task_has_active_sessions_locked(bare_task_id: str) -> bool:
    return any(
        _bt._bare_task_id_for_session_key(session_key) == bare_task_id
        for session_key in _bt._active_sessions
    )


def _task_cleanup_operation_lock(bare_task_id: str) -> Any:
    """Per-task re-entrant lock shared by commands, dialogs, and cleanup."""
    with _bt._cleanup_lock:
        return _bt._browser_task_cleanup_locks.setdefault(
            bare_task_id, threading.RLock()
        )


def _cleanup_reason_retires_task(reason: BrowserCleanupReason) -> bool:
    return reason in {
        BrowserCleanupReason.TERMINAL,
        BrowserCleanupReason.RESTART_ROLLBACK,
    }


def _coerce_cleanup_reason(reason: BrowserCleanupReason | str) -> BrowserCleanupReason:
    return reason if isinstance(reason, BrowserCleanupReason) else BrowserCleanupReason(str(reason))


def _is_browser_task_unavailable(task_id: str) -> bool:
    bare_task_id = _bt._bare_task_id_for_session_key(task_id or "default")
    with _bt._cleanup_lock:
        return _task_state_locked(bare_task_id) in {
            BrowserTaskState.RETIRING,
            BrowserTaskState.RETIRED,
        }


def _is_task_owned_shared_cdp_session(session_info: Dict[str, Any]) -> bool:
    """True only for a task-owned tab on a user-supplied shared CDP."""
    features = session_info.get("features")
    return bool(session_info.get("cdp_url")) and bool(
        isinstance(features, dict) and features.get("cdp_override") is True
    )


def _is_browser_task_retired(task_id: str) -> bool:
    bare_task_id = _bt._bare_task_id_for_session_key(task_id or "default")
    with _bt._cleanup_lock:
        return _task_state_locked(bare_task_id) is BrowserTaskState.RETIRED


def _browser_session_retired_result(task_id: str) -> Dict[str, Any]:
    bare_task_id = _bt._bare_task_id_for_session_key(task_id or "default")
    with _bt._cleanup_lock:
        state = _task_state_locked(bare_task_id)
        reason = _bt._browser_task_cleanup_reasons.get(bare_task_id)
    cleanup_pending = state is BrowserTaskState.RETIRING
    terminal = state is BrowserTaskState.RETIRED or (
        cleanup_pending
        and reason is not None
        and _cleanup_reason_retires_task(reason)
    )
    return {
        "success": False,
        "error": (
            "Browser session cleanup is still pending; normal browser commands "
            "are blocked until exact ownership is released."
            if cleanup_pending
            else (
                "Browser session ownership for this task has been retired. "
                "Call browser_navigate to start a fresh owned session."
            )
        ),
        "code": "browser_session_retired",
        "data": {
            "task_id": bare_task_id,
            "state": state.value,
            "terminal": terminal,
            "cleanup_pending": cleanup_pending,
            "cleanup_reason": reason.value if reason is not None else None,
            "recovery": "retry_cleanup" if cleanup_pending else "browser_navigate",
        },
    }


def _clear_retired_browser_task_for_navigation(task_id: str) -> bool:
    """Let explicit navigation restart a retired task; return whether it did."""
    bare_task_id = _bt._bare_task_id_for_session_key(task_id or "default")
    with _bt._cleanup_lock:
        state = _task_state_locked(bare_task_id)
        if state is BrowserTaskState.RETIRING:
            raise _BrowserSessionRetiredError(bare_task_id)
        was_retired = state is BrowserTaskState.RETIRED
        if was_retired:
            _advance_task_generation_locked(bare_task_id)
            _set_task_state_locked(bare_task_id, BrowserTaskState.STARTING)
    if was_retired:
        _bt.logger.info("Explicit navigation is restarting retired browser task: %s", bare_task_id)
    return was_retired


def _restore_retired_browser_task_after_failed_navigation(
    task_id: str, session_key: str
) -> None:
    """Rollback a failed explicit restart through the exact cleanup path."""
    cleanup_browser(session_key, reason=BrowserCleanupReason.RESTART_ROLLBACK)


def _session_expiry_timestamp(session_info: Dict[str, Any]) -> Optional[float]:
    """Provider-authoritative session expiry as epoch seconds; None when absent or
    malformed (cloud providers may omit ``expires_at``; local browsers never have one)."""
    value = session_info.get("expires_at")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None

    normalized = value.strip()
    if normalized.endswith(("Z", "z")):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        _bt.logger.warning("Ignoring invalid cloud browser session expiry timestamp")
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _session_has_expired(
    session_info: Dict[str, Any], *, now: Optional[float] = None
) -> bool:
    """Whether a cached browser session crossed its provider deadline."""
    expires_at = _session_expiry_timestamp(session_info)
    if expires_at is None:
        return False
    return (time.time() if now is None else now) >= expires_at


def _provider_cleanup_key(
    task_id: str, provider: Any, session_id: str
) -> Tuple[str, int, str]:
    return (task_id, id(provider) if provider is not None else 0, session_id)


def _remember_pending_provider_cleanup(
    task_id: str, provider: Any, session_id: str
) -> None:
    """Retain provider identity until an authoritative close succeeds."""
    bare_task_id = _bt._bare_task_id_for_session_key(task_id)
    record = _PendingProviderCleanup(bare_task_id, provider, session_id)
    with _bt._cleanup_lock:
        stale_keys = [
            key
            for key, pending in _bt._pending_provider_cleanups.items()
            if pending.task_id == bare_task_id and pending.session_id == session_id
        ]
        for key in stale_keys:
            _bt._pending_provider_cleanups.pop(key, None)
        _bt._pending_provider_cleanups[
            _provider_cleanup_key(bare_task_id, provider, session_id)
        ] = record
        # A stale creator may finish after terminal cleanup reached RETIRED.
        # Reopen cleanup only; normal browser commands remain fenced out.
        if _task_state_locked(bare_task_id) is BrowserTaskState.RETIRED:
            _set_task_state_locked(
                bare_task_id,
                BrowserTaskState.RETIRING,
                cleanup_reason=BrowserCleanupReason.TERMINAL,
            )
        _bt._session_last_activity.setdefault(bare_task_id, time.time())


def _clear_pending_provider_cleanup(task_id: str, session_id: str) -> None:
    bare_task_id = _bt._bare_task_id_for_session_key(task_id)
    with _bt._cleanup_lock:
        stale_keys = [
            key
            for key, record in _bt._pending_provider_cleanups.items()
            if record.task_id == bare_task_id and record.session_id == session_id
        ]
        for key in stale_keys:
            _bt._pending_provider_cleanups.pop(key, None)


def _attempt_provider_session_close(
    task_id: str, provider: Any, session_id: str
) -> bool:
    """Close a provider session once and retain exact retry identity on failure."""
    bare_task_id = _bt._bare_task_id_for_session_key(task_id)
    owner = provider
    if owner is None:
        try:
            owner = _cloud._get_cloud_provider()
        except Exception as exc:
            _bt.logger.warning("Could not resolve cloud browser provider for cleanup: %s", exc)
    if owner is None:
        _remember_pending_provider_cleanup(bare_task_id, provider, session_id)
        return False
    try:
        closed = owner.close_session(session_id)
    except Exception as exc:
        _bt.logger.warning("Could not close cloud browser session: %s", exc)
        _remember_pending_provider_cleanup(bare_task_id, owner, session_id)
        return False
    if closed is not True:
        _bt.logger.warning("Cloud browser provider did not confirm session close")
        _remember_pending_provider_cleanup(bare_task_id, owner, session_id)
        return False
    _clear_pending_provider_cleanup(bare_task_id, session_id)
    return True


def _dispose_unpublished_session(
    task_id: str, session_info: Any, provider: Any = None
) -> bool:
    """Dispose provider resources created by a stale or losing creator."""
    if not isinstance(session_info, dict):
        return True
    session_id = session_info.get("bb_session_id")
    if not session_id:
        # Local/shared-CDP candidates are metadata-only until first command.
        return True
    owner = session_info.get("_provider_cleanup_owner", provider)
    return _attempt_provider_session_close(task_id, owner, str(session_id))


def _best_effort(label: str, fn) -> None:
    """Run ``fn()``; log (debug) and swallow any exception — teardown must never abort."""
    try:
        fn()
    except Exception as e:
        _bt.logger.debug("%s failed: %s", label, e)


def _stop_all_lightpanda() -> None:
    from tools.browser_lightpanda import stop_all_lightpanda
    stop_all_lightpanda()


def _emergency_cleanup_all_sessions():
    """atexit: close this process's sessions, then sweep orphans left by crashed
    hermes processes — every clean exit reaps accumulated orphans, not only
    processes that used the browser tool."""
    if _bt._cleanup_done:
        return
    _bt._cleanup_done = True

    # Own sessions first so their owner_pid files are gone before the reaper scans.
    # Real-profile Chrome is launched directly (not by agent-browser), so the
    # session cleanup never reaps it.
    _best_effort("Real-profile chrome cleanup on exit", _real_profile._terminate_real_profile_chrome)
    if _bt._active_sessions or _bt._pending_provider_cleanups:
        ownership_count = len(_bt._active_sessions) + len(_bt._pending_provider_cleanups)
        _bt.logger.info("Emergency cleanup: closing %s active/pending browser ownership(s)...", ownership_count)
        try:
            cleanup_all_browsers()
        except Exception as e:
            _bt.logger.error("Emergency cleanup error: %s", e)
        finally:
            with _bt._cleanup_lock:
                _bt._active_sessions.clear()
                _bt._session_last_activity.clear()
                _bt._session_owner_homes.clear()
                _bt._cleanup_failures.clear()
                _bt._recording_sessions.clear()
    # Lightpanda servers we spawned that fell out of ``_active_sessions``.
    _best_effort("Lightpanda cleanup on exit", _stop_all_lightpanda)
    # Safe even if we never used the browser — owner_pid liveness protects daemons
    # owned by other live hermes processes.
    _best_effort("Orphan reap on exit", _reap_orphaned_browser_sessions)


@contextlib.contextmanager
def _session_owner_scope(task_id: str):
    """Run under the Hermes home + secret scope owning ``task_id``'s session (no-op if unrecorded).

    The janitor thread is process-global, so each teardown must re-enter its OWN
    profile's scope rather than inherit the spawning profile's; never falls
    through to ``os.environ``.
    """
    owner_home = _bt._session_owner_homes.get(task_id)
    if owner_home is None:
        yield
        return

    from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
    from hermes_cli.env_loader import hydrate_profile_secret_sources

    home_token = set_hermes_home_override(owner_home)
    try:
        hydrate_profile_secret_sources(Path(owner_home))
        secret_token = set_secret_scope(build_profile_secret_scope(Path(owner_home)))
        try:
            yield
        finally:
            reset_secret_scope(secret_token)
    finally:
        reset_hermes_home_override(home_token)


def _forget_session_tracking(task_id: str, *, activity: bool = True, session: bool = False) -> None:
    """Drop the janitor's bookkeeping (and optionally the session entry) for ``task_id``."""
    with _bt._cleanup_lock:
        if session:
            _bt._active_sessions.pop(task_id, None)
        if activity:
            _bt._session_last_activity.pop(task_id, None)
        _bt._session_owner_homes.pop(task_id, None)
        _bt._cleanup_failures.pop(task_id, None)


def _cleanup_inactive_browser_sessions():
    """Close sessions inactive longer than the timeout (cleanup thread).

    Each teardown runs under its owner profile's scope. A session whose cleanup
    keeps failing is force-reaped after MAX_INACTIVITY_CLEANUP_FAILURES attempts;
    only a successful cleanup clears its failure count.

    See #100738, #86402.
    """
    current_time = time.time()

    with _bt._cleanup_lock:
        sessions_to_cleanup = []
        for task_id, last_time in list(_bt._session_last_activity.items()):
            session_info = _bt._active_sessions.get(task_id)
            # Shared-CDP silence is not turn inactivity: the model may be
            # waiting on the page. Only an already-pending cleanup retry may be
            # driven by the janitor; normal ownership ends at the turn boundary.
            if (
                session_info
                and _is_task_owned_shared_cdp_session(session_info)
                and not session_info.get("_cleanup_retry_pending")
            ):
                continue
            if current_time - last_time > _bt.BROWSER_SESSION_INACTIVITY_TIMEOUT:
                sessions_to_cleanup.append(task_id)

    for task_id in sessions_to_cleanup:
        elapsed = int(current_time - _bt._session_last_activity.get(task_id, current_time))
        _bt.logger.info("Cleaning up inactive session for task: %s (inactive for %ss)", task_id, elapsed)
        try:
            with _session_owner_scope(task_id):
                cleanup_succeeded = cleanup_browser(
                    task_id, reason=BrowserCleanupReason.INACTIVITY
                )
            if cleanup_succeeded:
                _forget_session_tracking(task_id)
            else:
                # Keep ownership retryable without hammering it every tick.
                with _bt._cleanup_lock:
                    if task_id in _bt._session_last_activity:
                        _bt._session_last_activity[task_id] = current_time
        except Exception as e:
            with _bt._cleanup_lock:
                failures = _bt._cleanup_failures[task_id] = _bt._cleanup_failures.get(task_id, 0) + 1
            if failures < _bt.MAX_INACTIVITY_CLEANUP_FAILURES:
                _bt.logger.warning("Error cleaning up inactive session %s (attempt %d/%d): %s",
                               task_id, failures, _bt.MAX_INACTIVITY_CLEANUP_FAILURES, e)
                continue
            _bt.logger.error("Browser cleanup failed %d times for inactive session %s; "
                         "force-reaping: %s", failures, task_id, e)
            try:
                with _session_owner_scope(task_id):
                    _force_reap_browser_session(task_id)
            except Exception as reap_exc:
                _bt.logger.error("Force-reap of browser session %s failed: %s", task_id, reap_exc)
            finally:
                _forget_session_tracking(task_id, activity=False)


def _write_owner_pid(socket_dir: str, session_name: str) -> None:
    """Record this hermes PID in ``<socket_dir>/<session>.owner_pid`` so the orphan
    reaper can tell live-owner daemons from crashed-owner ones. Best-effort: an
    OSError falls back to the legacy ``tracked_names`` heuristic."""
    try:
        path = os.path.join(socket_dir, f"{session_name}.owner_pid")
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except OSError as exc:
        _bt.logger.debug("Could not write owner_pid file for %s: %s", session_name, exc)


def _write_shared_cdp_endpoint(
    socket_dir: str, session_name: str, cdp_url: str
) -> None:
    """Persist the browser endpoint needed for exact orphan target cleanup.

    It may contain authentication material, so the file is private. Failure is
    fail-closed: the orphan reaper retains the ownership directory instead of
    discovering or guessing a replacement target.
    """
    cdp_url = str(cdp_url or "").strip()
    if not cdp_url:
        return
    path = os.path.join(socket_dir, f"{session_name}.cdp_endpoint")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(cdp_url)
        os.chmod(path, 0o600)
    except OSError as exc:
        _bt.logger.debug("Could not persist shared-CDP endpoint for %s: %s", session_name, exc)


def _verify_reapable_browser_daemon(daemon_pid: int, socket_dir: str,
                                    session_name: str) -> bool:
    """Confirm a live PID is genuinely *this* session's agent-browser daemon (fail-closed).

    The ``.pid`` file sits in a world-writable temp dir: a planted or recycled PID
    would turn the tree-kill into an arbitrary-process DoS. Both must pass:
    (1) identity — ``agent-browser`` in name/cmdline; (2) binding — the socket dir in
    the cmdline or ``AGENT_BROWSER_SOCKET_DIR`` in its environ (the real spoof defense).
    """
    def refuse(reason: str, *args) -> bool:
        _bt.logger.warning("Refusing to reap browser daemon PID %d (session %s): " + reason,
                           daemon_pid, session_name, *args)
        return False

    try:
        import psutil
    except ImportError:  # psutil is a hard dep; defensive only
        return refuse("psutil unavailable for identity verification")

    try:
        proc = psutil.Process(daemon_pid)
        name = (proc.name() or "").lower()
        cmdline = " ".join(proc.cmdline() or []).lower()
    except psutil.NoSuchProcess:
        return False  # vanished between the liveness check and now
    except (psutil.AccessDenied, OSError) as exc:
        return refuse("could not read process identity (%s)", exc)

    if "agent-browser" not in name and "agent-browser" not in cmdline:
        return refuse("not an agent-browser process (name=%r)", name)

    socket_dir_l = socket_dir.lower()
    socket_base_l = os.path.basename(socket_dir).lower()
    bound = socket_dir_l in cmdline or (socket_base_l and socket_base_l in cmdline)
    if not bound:
        try:
            env_dir = (proc.environ() or {}).get("AGENT_BROWSER_SOCKET_DIR", "")
            bound = bool(env_dir) and os.path.normpath(env_dir) == os.path.normpath(socket_dir)
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            bound = False  # environ() can be denied even same-user; cmdline already failed — fail closed
    if not bound:
        return refuse("not bound to session socket dir %s (possible recycled PID or planted pid file)", socket_dir)
    return True


def _socket_dir_idle_seconds(socket_dir: str) -> Optional[float]:
    """Seconds since anything in ``socket_dir`` was last written; None if unknown (fail safe).
    Every command rewrites ``_stdout_<cmd>`` there — a restart-proof activity marker — and
    rewriting doesn't touch the dir mtime, so entries are scanned too."""
    try:
        latest = os.path.getmtime(socket_dir)
    except OSError:
        return None

    try:
        with os.scandir(socket_dir) as entries:
            for entry in entries:
                try:
                    latest = max(latest, entry.stat().st_mtime)
                except OSError:
                    continue
    except OSError:
        pass  # dir mtime alone is still a usable lower bound

    return max(0.0, time.time() - latest)


def _read_pid_file(path: str) -> Optional[int]:
    """Integer PID from ``path``; None when missing or corrupt."""
    try:
        return int(Path(path).read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def _owner_pid_alive(socket_dir: str, session_name: str) -> Tuple[Optional[int], Optional[bool]]:
    """Read ``<session>.owner_pid`` and report ``(pid, alive)``; ``(None, None)`` when missing/corrupt."""
    owner_pid = _read_pid_file(os.path.join(socket_dir, f"{session_name}.owner_pid"))
    if owner_pid is None:
        return None, None
    # ``os.kill(pid, 0)`` is NOT a no-op on Windows; use the cross-platform check.
    from gateway.status import _pid_exists
    return owner_pid, _pid_exists(owner_pid)


def _terminate_verified_daemon(daemon_pid: int, session_name: str, log) -> bool:
    """Tree-kill ``daemon_pid`` if it has a start-time fingerprint (so a PID swapped
    between check and kill is refused); False (logged via ``log``) when no fingerprint.
    Raises on OS errors."""
    from gateway.status import get_process_start_time
    from tools.process_registry import ProcessRegistry
    daemon_start = get_process_start_time(daemon_pid)
    if daemon_start is None:
        log("Refusing to reap browser daemon PID %d (session %s): no start-time fingerprint available",
            daemon_pid, session_name)
        return False
    ProcessRegistry._terminate_host_pid(daemon_pid, daemon_start)
    return True


def _orphan_target_ownership(
    socket_dir: str, session_name: str
) -> OrphanTargetOwnership:
    """Classify exact target ownership without guessing from bad metadata."""
    target_file = Path(socket_dir) / f"{session_name}.target"
    if not target_file.exists():
        return (
            OrphanTargetOwnership.UNKNOWN
            if session_name.startswith("cdp_")
            else OrphanTargetOwnership.CONFIRMED_UNPINNED
        )
    try:
        payload = json.loads(target_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return OrphanTargetOwnership.UNKNOWN
    if not isinstance(payload, dict):
        return OrphanTargetOwnership.UNKNOWN
    target_id = payload.get("targetId")
    if payload.get("pinned") is True and isinstance(target_id, str) and target_id:
        return OrphanTargetOwnership.PINNED
    if payload.get("pinned") is False:
        return OrphanTargetOwnership.CONFIRMED_UNPINNED
    return OrphanTargetOwnership.UNKNOWN


def _read_recorded_pinned_target_id(
    socket_dir: str, session_name: str
) -> Optional[str]:
    """Read only an explicitly pinned target id from daemon metadata."""
    try:
        payload = json.loads(
            (Path(socket_dir) / f"{session_name}.target").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict) or payload.get("pinned") is not True:
        return None
    target_id = payload.get("targetId")
    if not isinstance(target_id, str) or not target_id.strip():
        return None
    return target_id.strip()


def _orphan_has_pinned_target(socket_dir: str, session_name: str) -> bool:
    """Compatibility predicate; the reaper itself uses tri-state ownership."""
    return (
        _orphan_target_ownership(socket_dir, session_name)
        is OrphanTargetOwnership.PINNED
    )


def _close_orphaned_pinned_target(socket_dir: str, session_name: str) -> bool:
    """Close only the persisted exact target through its persisted endpoint."""
    try:
        target_id = _read_recorded_pinned_target_id(socket_dir, session_name)
        cdp_url = (
            Path(socket_dir) / f"{session_name}.cdp_endpoint"
        ).read_text(encoding="utf-8").strip()
    except (OSError, ValueError, TypeError):
        return False
    if not target_id or not cdp_url:
        return False
    return _cdp._close_shared_cdp_target_confirmed(cdp_url, target_id)


def _reap_socket_dir(socket_dir: str, session_name: str, tracked_names: set) -> bool:
    """Reap one ``agent-browser-<session>`` dir if orphaned; True when a daemon was killed.

    A live ``owner_pid`` means another hermes process owns it — leave it UNLESS untracked
    here and idle past ``BROWSER_ORPHAN_GRACE_SECONDS`` (owner-alive alone made leaked
    daemons immortal); no owner_pid (legacy) falls back to this process's tracking. A
    pidless dir is only stale after the grace period (deleting it immediately races the
    creator's first stdout open). The PID is identity-verified before any tree-kill.
    """
    owner_pid, owner_alive = _owner_pid_alive(socket_dir, session_name)
    if owner_alive is True:
        if session_name in tracked_names:
            return False
        idle_s = _socket_dir_idle_seconds(socket_dir)
        if idle_s is None or idle_s < _bt.BROWSER_ORPHAN_GRACE_SECONDS:
            return False  # unknown age or within grace — fail safe
        _bt.logger.warning(
            "Browser session %s has a live owner (PID %s) but is untracked "
            "and idle for %ds (grace %ds) — treating as leaked and reaping",
            session_name, owner_pid, int(idle_s),
            _bt.BROWSER_ORPHAN_GRACE_SECONDS)
    elif owner_alive is None and session_name in tracked_names:
        return False

    target_ownership = _orphan_target_ownership(socket_dir, session_name)
    if target_ownership is OrphanTargetOwnership.UNKNOWN:
        _bt.logger.warning(
            "Orphaned shared-CDP target metadata for session %s is missing "
            "or unreadable; retaining daemon and ownership directory",
            session_name,
        )
        return False
    pinned_target_owned = target_ownership is OrphanTargetOwnership.PINNED
    if pinned_target_owned and not _close_orphaned_pinned_target(
        socket_dir, session_name
    ):
        _bt.logger.warning(
            "Could not confirm exact pinned target close for orphaned session %s; "
            "retaining daemon and ownership metadata for retry",
            session_name,
        )
        return False

    pid_file = os.path.join(socket_dir, f"{session_name}.pid")
    if not os.path.isfile(pid_file):
        idle_s = _socket_dir_idle_seconds(socket_dir)
        if idle_s is None or idle_s < _bt.BROWSER_ORPHAN_GRACE_SECONDS:
            return False
        if pinned_target_owned:
            _bt.logger.warning(
                "Orphaned pinned target metadata for session %s has no daemon PID; "
                "retaining ownership record for manual recovery",
                session_name,
            )
            return False
        shutil.rmtree(socket_dir, ignore_errors=True)
        return False

    daemon_pid = _read_pid_file(pid_file)
    from gateway.status import _pid_exists
    if daemon_pid is None or not _pid_exists(daemon_pid):
        shutil.rmtree(socket_dir, ignore_errors=True)
        return False

    if not _verify_reapable_browser_daemon(daemon_pid, socket_dir, session_name):
        return False  # leave process and dir for a later sweep once the imposter PID is gone

    # Tree-kill so Chromium children (renderer, GPU, ...) go too.
    reaped = False
    try:
        if not _terminate_verified_daemon(daemon_pid, session_name, _bt.logger.warning):
            return False
        _bt.logger.info("Reaped orphaned browser daemon PID %d (session %s)", daemon_pid, session_name)
        reaped = True
    except (ProcessLookupError, PermissionError, OSError):
        pass
    shutil.rmtree(socket_dir, ignore_errors=True)
    return reaped


def _reap_orphaned_browser_sessions():
    """Kill agent-browser daemons whose owning hermes process is gone (an unclean exit loses
    ``_active_sessions`` but node + Chromium keep running). Scans the tmp dir for
    ``agent-browser-*`` socket dirs; safe from any context."""
    import glob

    # Lightpanda servers keep their own records (no socket dir); sweep them with the
    # same owner-liveness rule BEFORE the daemon scan, which may return early.
    def _reap_lp():
        from tools.browser_lightpanda import reap_orphaned_lightpanda
        reap_orphaned_lightpanda()
    _best_effort("Lightpanda orphan reap", _reap_lp)

    tmpdir = _bt._socket_safe_tmpdir()
    socket_dirs = []
    for prefix in ("agent-browser-h_*", "agent-browser-cdp_*", "agent-browser-hermes_*"):
        socket_dirs += glob.glob(os.path.join(tmpdir, prefix))
    if not socket_dirs:
        return

    with _bt._cleanup_lock:
        tracked_names = {info.get("session_name") for info in _bt._active_sessions.values() if info.get("session_name")}

    reaped = 0
    for socket_dir in socket_dirs:
        session_name = os.path.basename(socket_dir).removeprefix("agent-browser-")
        if session_name and _reap_socket_dir(socket_dir, session_name, tracked_names):
            reaped += 1

    if reaped:
        _bt.logger.info("Reaped %d orphaned browser session(s) from previous run(s)", reaped)


def _browser_cleanup_thread_worker():
    """Every 30s: close sessions idle past BROWSER_SESSION_INACTIVITY_TIMEOUT; reap
    orphans on startup AND every BROWSER_ORPHAN_REAP_INTERVAL seconds."""
    reap_every_cycles = max(1, round(_bt.BROWSER_ORPHAN_REAP_INTERVAL / 30))
    cycle = 0

    while _bt._cleanup_running:
        if cycle % reap_every_cycles == 0:  # cycle 0 is the startup reap
            try:
                _reap_orphaned_browser_sessions()
            except Exception as e:
                _bt.logger.warning("Orphan reap error: %s", e)
        cycle += 1

        try:
            _cleanup_inactive_browser_sessions()
        except Exception as e:
            _bt.logger.warning("Cleanup thread error: %s", e)

        for _ in range(30):  # 1s granularity so stop is quick
            if not _bt._cleanup_running:
                break
            time.sleep(1)


def _start_browser_cleanup_thread():
    """Start the background cleanup thread if not already running."""
    with _bt._cleanup_lock:
        if _bt._cleanup_thread is None or not _bt._cleanup_thread.is_alive():
            _bt._cleanup_running = True
            _bt._cleanup_thread = threading.Thread(target=_browser_cleanup_thread_worker, daemon=True,
                                                   name="browser-cleanup")
            _bt._cleanup_thread.start()
            _bt.logger.info("Started inactivity cleanup thread (timeout: %ss)", _bt.BROWSER_SESSION_INACTIVITY_TIMEOUT)


def _stop_browser_cleanup_thread():
    """Stop the background cleanup thread."""
    _bt._cleanup_running = False
    if _bt._cleanup_thread is not None:
        _bt._cleanup_thread.join(timeout=5)


def _update_session_activity(task_id: str):
    """Touch the activity timestamp and record the owning Hermes home on first sight (the
    janitor tears down under the owner's scope). Does NOT reset ``_cleanup_failures``.

    See #86402.
    """
    with _bt._cleanup_lock:
        _bt._session_last_activity[task_id] = time.time()
        _bt._session_owner_homes.setdefault(task_id, str(get_hermes_home()))


def _kill_process_tree(proc: "subprocess.Popen") -> None:
    """Best-effort kill of *proc* and every descendant; never raises.

    ``Popen.kill()`` only signals the direct child; npm/npx helpers and the detached
    daemon grandchild keep a capture pipe open so ``communicate()`` never sees EOF, so
    the whole tree must go (no grace: the caller already burned its timeout). Delegates
    to :func:`agent.deadline.kill_process_tree`, falling back to the legacy kill.

    ``Popen.kill()`` only signals the direct child PID. npm/npx routinely fork further processes
    (registry-fetch helpers, npm's own lifecycle runner, agent-browser's own detached daemon grandchild)
    that can survive a plain ``kill()`` of the top-level PID and keep a ``capture_output``-style pipe open,
    hanging the caller's ``communicate()`` past the nominal timeout — the same orphaned-pipe hazard already
    hit in production on POSIX (see ``tools/process_registry.py``'s ``_reader_loop``, issue 68915: a
    backgrounded grandchild inheriting a pipe's write end kept it from ever reaching EOF). That hazard is
    cross-platform, not Windows-specific; what *is* Windows-specific is the lack of a remedy other than
    killing the tree — anonymous pipes there don't support overlapped I/O, so there's no ``select()``-style
    non-blocking read to poll around a stuck grandchild the way POSIX can. Killing the whole process
    group/tree the child was launched into reaches those descendants on both platforms. See #68915.
    """
    try:
        from agent.deadline import kill_process_tree as _deadline_kill_tree

        _deadline_kill_tree(proc.pid)
    except Exception:
        _legacy_kill_process_tree(proc)


def _legacy_kill_process_tree(proc: "subprocess.Popen") -> None:
    """Local tree-kill (SIGTERM then SIGKILL to the process group) — fallback when
    agent.deadline is unavailable; tests pin this signal sequence."""
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           check=False, capture_output=True, stdin=subprocess.DEVNULL)
        except Exception:
            pass
        return
    # POSIX-only below (the nt guard returned), but resolve killpg/SIGKILL via
    # getattr so a future refactor dropping that guard degrades to plain kill().
    killpg = getattr(os, "killpg", None)
    if killpg is None:  # windows-footgun: ok - non-POSIX fallback
        try:
            proc.kill()
        except Exception:
            pass
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return
    for sig in (signal.SIGTERM, getattr(signal, "SIGKILL", signal.SIGTERM)):
        try:
            killpg(pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            return


def _pid_exists(pid: int) -> bool:
    """Best-effort 'is this PID alive' (cross-platform via gateway.status; zombies count as dead)."""
    if pid <= 0:
        return False
    from gateway.status import _pid_exists as _gateway_pid_exists
    return _gateway_pid_exists(pid)


def _unlink_older_than(directory: Path, pattern: str, max_age_hours: float, label: str) -> None:
    """Delete ``directory/pattern`` files older than ``max_age_hours``; never raises."""
    try:
        cutoff = time.time() - (max_age_hours * 3600)
        for f in directory.glob(pattern):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except Exception as e:
                _bt.logger.debug("Failed to clean old %s %s: %s", label, f, e)
    except Exception as e:
        _bt.logger.debug("%s cleanup error (non-critical): %s", label.capitalize(), e)


def _cleanup_old_screenshots(screenshots_dir, max_age_hours=24):
    """Prune old browser screenshots; throttled to once per hour per directory."""
    key = str(screenshots_dir)
    now = time.time()
    if now - _bt._last_screenshot_cleanup_by_dir.get(key, 0.0) < 3600:
        return
    _bt._last_screenshot_cleanup_by_dir[key] = now
    _unlink_older_than(screenshots_dir, "browser_screenshot_*.png", max_age_hours, "screenshot")


def _cleanup_old_recordings(max_age_hours=72):
    """Prune old browser recordings."""
    try:
        recordings_dir = get_hermes_home() / "browser_recordings"
    except Exception as e:
        _bt.logger.debug("Recording cleanup error (non-critical): %s", e)
        return
    if recordings_dir.exists():
        _unlink_older_than(recordings_dir, "session_*.webm", max_age_hours, "recording")


def _retry_provider_cleanup_records(
    records: list[_PendingProviderCleanup],
) -> bool:
    all_closed = True
    for record in records:
        if not _attempt_provider_session_close(
            record.task_id, record.provider, record.session_id
        ):
            all_closed = False
    return all_closed


def _begin_task_cleanup_locked(
    bare_task_id: str, reason: BrowserCleanupReason
) -> BrowserCleanupReason:
    """Enter RETIRING and advance generation before inspecting ownership."""
    state = _task_state_locked(bare_task_id)
    previous_reason = _bt._browser_task_cleanup_reasons.get(bare_task_id)
    if state is BrowserTaskState.RETIRED:
        effective_reason = BrowserCleanupReason.TERMINAL
    elif (
        state is BrowserTaskState.RETIRING
        and previous_reason is not None
        and _cleanup_reason_retires_task(previous_reason)
        and not _cleanup_reason_retires_task(reason)
    ):
        effective_reason = previous_reason
    else:
        effective_reason = reason
    _advance_task_generation_locked(bare_task_id)
    _set_task_state_locked(
        bare_task_id,
        BrowserTaskState.RETIRING,
        cleanup_reason=effective_reason,
    )
    return effective_reason


def _is_hermes_owned_local_browser_session(session_info: Dict[str, Any]) -> bool:
    """Whether headed cross-turn persistence may retain this session."""
    features = session_info.get("features")
    if isinstance(features, dict) and features.get("local") is True:
        return not session_info.get("bb_session_id") and not bool(
            features.get("cdp_override")
        )
    # Hot-reload compatibility for throwaway-local rows predating features.
    return not session_info.get("cdp_url") and not session_info.get("bb_session_id")


def _cleanup_browser_session_keys(
    task_id: str,
    session_keys: Optional[list[str]] = None,
    *,
    reason: BrowserCleanupReason | str,
    preserve_local_headed: bool = False,
) -> bool:
    """Cleanup selected ownership under one bare-task lifecycle transition."""
    reason = _coerce_cleanup_reason(reason)
    bare_task_id = _bt._bare_task_id_for_session_key(task_id or "default")
    include_camofox = session_keys is None and _bt._is_camofox_mode()

    # Browser-free turns are a true no-op: do not leak lifecycle rows/locks.
    with _bt._cleanup_lock:
        state = _task_state_locked(bare_task_id)
        if session_keys is None:
            has_selected_session = any(
                _bt._bare_task_id_for_session_key(key) == bare_task_id
                for key in _bt._active_sessions
            )
        else:
            has_selected_session = any(
                key in _bt._active_sessions for key in dict.fromkeys(session_keys)
            )
        has_pending_provider = any(
            record.task_id == bare_task_id
            for record in _bt._pending_provider_cleanups.values()
        )
        if (
            not include_camofox
            and not has_selected_session
            and not has_pending_provider
            and bare_task_id not in _bt._browser_task_cleanup_locks
            and state in {BrowserTaskState.ACTIVE, BrowserTaskState.RETIRED}
        ):
            return True

    with _task_cleanup_operation_lock(bare_task_id):
        with _bt._cleanup_lock:
            effective_reason = _begin_task_cleanup_locked(bare_task_id, reason)
            if session_keys is None:
                if _bt._is_local_sidecar_key(task_id):
                    selected_keys = [task_id] if task_id in _bt._active_sessions else []
                else:
                    selected_keys = [
                        key
                        for key, info in _bt._active_sessions.items()
                        if _bt._bare_task_id_for_session_key(key) == bare_task_id
                        and not (
                            preserve_local_headed
                            and _is_hermes_owned_local_browser_session(info)
                        )
                    ]
            else:
                selected_keys = list(dict.fromkeys(session_keys))
            if include_camofox and task_id not in selected_keys:
                selected_keys.append(task_id)
            pending_before = [
                record
                for record in _bt._pending_provider_cleanups.values()
                if record.task_id == bare_task_id
            ]

        # Fence in-flight supervisor starters for task keys without a selected
        # published session. Selected sessions stop only after exact target close.
        for supervisor_key in {task_id} - set(selected_keys):
            _cdp._stop_cdp_supervisor(supervisor_key)

        pending_retries_closed = _retry_provider_cleanup_records(pending_before)
        cleanup_results = {
            session_key: _cleanup_single_browser_session(session_key)
            for session_key in selected_keys
        }
        failed_keys = [
            key for key, cleaned in cleanup_results.items() if cleaned is False
        ]

        with _bt._cleanup_lock:
            retained_failed_keys = []
            for failed_key in failed_keys:
                retained = _bt._active_sessions.get(failed_key)
                if retained is not None:
                    retained["_cleanup_retry_pending"] = True
                    retained_failed_keys.append(failed_key)

            recorded_key = _bt._last_active_session_key.get(bare_task_id)
            if retained_failed_keys:
                if recorded_key not in retained_failed_keys:
                    _bt._last_active_session_key[bare_task_id] = retained_failed_keys[0]
            elif recorded_key in selected_keys:
                sidecar_key = f"{bare_task_id}{_bt._LOCAL_SUFFIX}"
                if (
                    task_id == bare_task_id
                    and sidecar_key not in selected_keys
                    and sidecar_key in _bt._active_sessions
                ):
                    _bt._last_active_session_key[bare_task_id] = sidecar_key
                else:
                    _bt._last_active_session_key.pop(bare_task_id, None)
            elif (
                not failed_keys
                and not _bt._is_local_sidecar_key(task_id)
                and not _task_has_active_sessions_locked(bare_task_id)
            ):
                _bt._last_active_session_key.pop(bare_task_id, None)

            has_active_sessions = _task_has_active_sessions_locked(bare_task_id)
            has_pending_provider = any(
                record.task_id == bare_task_id
                for record in _bt._pending_provider_cleanups.values()
            )
            cleanup_confirmed = (
                not failed_keys
                and pending_retries_closed
                and not has_pending_provider
            )
            if cleanup_confirmed:
                if has_active_sessions:
                    _set_task_state_locked(bare_task_id, BrowserTaskState.ACTIVE)
                elif _cleanup_reason_retires_task(effective_reason):
                    _set_task_state_locked(bare_task_id, BrowserTaskState.RETIRED)
                else:
                    _set_task_state_locked(bare_task_id, BrowserTaskState.ACTIVE)
                if not has_active_sessions:
                    for activity_key in list(_bt._session_last_activity):
                        if _bt._bare_task_id_for_session_key(activity_key) == bare_task_id:
                            _bt._session_last_activity.pop(activity_key, None)
            else:
                _set_task_state_locked(
                    bare_task_id,
                    BrowserTaskState.RETIRING,
                    cleanup_reason=effective_reason,
                )
                _bt._session_last_activity.setdefault(bare_task_id, time.time())
        return cleanup_confirmed


def cleanup_browser(
    task_id: Optional[str] = None,
    *,
    reason: BrowserCleanupReason | str = BrowserCleanupReason.TERMINAL,
) -> bool:
    """Clean all ownership for a task (primary + sidecar for bare ids)."""
    return _cleanup_browser_session_keys(task_id or "default", reason=reason)


def cleanup_browser_for_turn(task_id: Optional[str] = None) -> bool:
    """Finalize a turn while retaining only Hermes-owned local headed state."""
    task_id = task_id or "default"
    preserve_local = _cloud._is_headed_mode()
    if _bt._is_camofox_mode():
        if preserve_local:
            return True
        try:
            from tools.browser_camofox import camofox_has_session

            if not camofox_has_session(task_id):
                bare_task_id = _bt._bare_task_id_for_session_key(task_id)
                with _bt._cleanup_lock:
                    if bare_task_id not in _bt._browser_task_cleanup_locks:
                        return True
        except Exception:
            pass
    if preserve_local:
        bare_task_id = _bt._bare_task_id_for_session_key(task_id)
        with _bt._cleanup_lock:
            task_state = _task_state_locked(bare_task_id)
            owned_sessions = [
                info
                for key, info in _bt._active_sessions.items()
                if _bt._bare_task_id_for_session_key(key) == bare_task_id
            ]
            has_pending_provider = any(
                record.task_id == bare_task_id
                for record in _bt._pending_provider_cleanups.values()
            )
        if (
            task_state is BrowserTaskState.ACTIVE
            and owned_sessions
            and not has_pending_provider
            and all(_is_hermes_owned_local_browser_session(info) for info in owned_sessions)
        ):
            return True
    return _cleanup_browser_session_keys(
        task_id,
        reason=BrowserCleanupReason.TERMINAL,
        preserve_local_headed=preserve_local,
    )


def _kill_verified_daemon(socket_dir: str, session_name: str) -> bool:
    """Tree-kill the daemon in ``<socket_dir>/<session>.pid`` if verifiably ours; True when
    a kill was issued. Never raises."""
    pid_file = os.path.join(socket_dir, f"{session_name}.pid")
    if not os.path.isfile(pid_file):
        return False
    try:
        daemon_pid = int(Path(pid_file).read_text(encoding="utf-8").strip())
        if not _verify_reapable_browser_daemon(daemon_pid, socket_dir, session_name):
            _bt.logger.debug("Skipped daemon kill for %s: pid %s failed identity verification", session_name, daemon_pid)
            return False
        if not _terminate_verified_daemon(daemon_pid, session_name, lambda *_a: _bt.logger.debug(
                "Skipped daemon kill for %s: no start-time fingerprint for pid %s", session_name, daemon_pid)):
            return False
        _bt.logger.debug("Killed daemon pid %s for %s", daemon_pid, session_name)
        return True
    except (ProcessLookupError, ValueError, PermissionError, OSError):
        _bt.logger.debug("Could not kill daemon pid for %s (already dead or inaccessible)", session_name)
        return False


def _release_session_resources(
    task_id: str,
    session_info: Dict[str, Any],
    *,
    retain_provider_failure: bool = True,
) -> bool:
    """Untrack ``task_id``, close its cloud provider session, kill its daemon — the
    unconditional tail of a teardown, and the whole of the janitor's force-reap path.

    The unconditional tail of ``_cleanup_single_browser_session``; also the whole of the janitor's
    force-reap path (#100738), which skips the polite agent-browser/Camofox ``close`` that kept failing but
    must still release the cloud session and the local Chromium.
    """
    bb_session_id = session_info.get("bb_session_id")
    _forget_session_tracking(task_id, session=True)

    provider_closed = True
    if bb_session_id:
        if retain_provider_failure:
            provider_closed = _attempt_provider_session_close(
                task_id,
                session_info.get("_provider_cleanup_owner"),
                str(bb_session_id),
            )
        else:
            # Janitor last resort: after repeated cleanup exceptions, issue one
            # provider close but do not keep retry state forever.
            provider = session_info.get("_provider_cleanup_owner")
            if provider is None:
                provider = _cloud._get_cloud_provider()
            if provider is not None:
                try:
                    provider.close_session(bb_session_id)
                except Exception as exc:
                    _bt.logger.warning("Could not close cloud browser session: %s", exc)

    session_name = session_info.get("session_name", "")
    if session_name:
        socket_dir = os.path.join(_bt._socket_safe_tmpdir(), f"agent-browser-{session_name}")
        if os.path.exists(socket_dir):
            _kill_verified_daemon(socket_dir, session_name)
            shutil.rmtree(socket_dir, ignore_errors=True)
    return provider_closed


def _force_reap_browser_session(task_id: str) -> None:
    """Janitor last resort: skip the failing ``close`` round-trips, release resources directly.

    Janitor last resort after repeated cleanup failures (#100738).
    """
    with _bt._cleanup_lock:
        session_info = _bt._active_sessions.get(task_id)
        _bt._recording_sessions.discard(task_id)
    if session_info and _is_task_owned_shared_cdp_session(session_info):
        # Never destroy the only exact-close path for an external shared tab.
        if _cleanup_single_browser_session(task_id):
            with _bt._cleanup_lock:
                _bt._session_last_activity.pop(task_id, None)
        return
    _cdp._stop_cdp_supervisor(task_id)
    if session_info:
        _release_session_resources(
            task_id, session_info, retain_provider_failure=False
        )
    with _bt._cleanup_lock:
        _bt._session_last_activity.pop(task_id, None)
    bare_task_id = _bt._bare_task_id_for_session_key(task_id)
    if bare_task_id == task_id or _bt._last_active_session_key.get(bare_task_id) == task_id:
        _bt._last_active_session_key.pop(bare_task_id, None)


def _cleanup_single_browser_session(task_id: str) -> bool:
    """Reap one exact session key; False retains retryable ownership."""
    with _bt._cleanup_lock:
        session_info = _bt._active_sessions.get(task_id)

    # Close a task-owned shared-CDP page by exact ID before stopping the only
    # reliable supervisor/daemon connection or removing its metadata.
    if session_info and _is_task_owned_shared_cdp_session(session_info):
        session_name = str(session_info.get("session_name") or "")
        socket_dir = os.path.join(
            _bt._socket_safe_tmpdir(), f"agent-browser-{session_name}"
        )
        target_id = str(session_info.get("target_id") or "").strip()
        if not target_id and session_name:
            target_id = _read_recorded_pinned_target_id(socket_dir, session_name) or ""
            if target_id:
                with _bt._cleanup_lock:
                    current = _bt._active_sessions.get(task_id)
                    if current is session_info:
                        current["target_id"] = target_id
        cdp_url = str(session_info.get("cdp_url") or _cdp._get_cdp_override()).strip()
        if not target_id or not _cdp._close_shared_cdp_target_confirmed(cdp_url, target_id):
            _bt.logger.warning(
                "Exact shared-CDP target close could not be confirmed for task %s; "
                "retaining session ownership for retry",
                task_id,
            )
            return False

    _cdp._stop_cdp_supervisor(task_id)

    # Camofox: managed persistence keeps the profile (cookies) across tasks; skip the full
    # close then — the inactivity reaper still frees idle resources.
    if _bt._is_camofox_mode():
        def _camofox_cleanup():
            from tools.browser_camofox import camofox_close, camofox_soft_cleanup
            if not camofox_soft_cleanup(task_id):
                camofox_close(task_id)
        _best_effort(f"Camofox cleanup for task {task_id}", _camofox_cleanup)

    _bt.logger.debug("cleanup_browser called for task_id: %s", task_id)
    _bt.logger.debug("Active sessions: %s", list(_bt._active_sessions.keys()))

    # Look up but don't remove yet — cleanup's close command needs the entry.
    with _bt._cleanup_lock:
        session_info = _bt._active_sessions.get(task_id)

    if not session_info:
        _bt.logger.debug("No active session found for task_id: %s", task_id)
        return True

    _bt.logger.debug("Found session for task %s: bb_session_id=%s", task_id, session_info.get("bb_session_id", "unknown"))
    _bt._maybe_stop_recording(task_id)  # saves the file before close

    # Lightpanda sessions have no daemon to ``close``; an expired cloud CDP URL cannot
    # accept one and would make _get_session_info() renew the session mid-cleanup.
    if (session_info.get("features") or {}).get("lightpanda"):
        try:
            from tools.browser_lightpanda import stop_lightpanda
            stop_lightpanda(session_info.get("session_name", ""))
        except Exception as e:
            _bt.logger.warning("lightpanda stop failed for task %s: %s", task_id, e)
    elif _session_has_expired(session_info):
        _bt.logger.debug("Skipping agent-browser close for expired session %s", task_id)
    else:
        try:
            close_result = _session._run_browser_command(
                task_id, "close", [], timeout=10, _allow_cleanup=True
            )
            if not close_result.get("success"):
                _bt.logger.warning(
                    "agent-browser session close failed for task %s: %s",
                    task_id,
                    close_result.get("error", close_result),
                )
            _bt.logger.debug("agent-browser close command completed for task %s", task_id)
        except Exception as e:
            _bt.logger.warning("agent-browser close failed for task %s: %s", task_id, e)

    provider_closed = _release_session_resources(task_id, session_info)
    _bt.logger.debug("Removed task %s from active sessions", task_id)
    return provider_closed


def cleanup_all_browsers() -> None:
    """Clean up all active browser sessions (shutdown) and reset cached lookups."""
    with _bt._cleanup_lock:
        task_ids = {
            _bt._bare_task_id_for_session_key(task_id)
            for task_id in _bt._active_sessions
        }
        task_ids.update(
            record.task_id for record in _bt._pending_provider_cleanups.values()
        )
    for task_id in task_ids:
        cleanup_browser(task_id)

    try:  # tear down CDP supervisors so background threads exit
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]
        SUPERVISOR_REGISTRY.stop_all()
    except Exception:
        pass

    _install._discover_homebrew_node_dirs.cache_clear()
    # Each resolved flag flips BEFORE its cache is nulled so a concurrent reader never
    # sees ``resolved=True`` with ``cache=None``.
    for flag, cache in (
        ("_agent_browser_resolved", "_cached_agent_browser"),
        ("_pin_tab_agent_browser_resolved", "_cached_pin_tab_agent_browser"),
        ("_command_timeout_resolved", "_cached_command_timeout"),
        ("_snapshot_threshold_resolved", "_cached_snapshot_threshold"),
        ("_chromium_autoinstall_attempted", "_cached_chromium_installed"),
        ("_browser_engine_resolved", "_cached_browser_engine"),
    ):
        setattr(_bt, flag, False)
        setattr(_bt, cache, None)
    _bt._pin_tab_failure_cache = None
