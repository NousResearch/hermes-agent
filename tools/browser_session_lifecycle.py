"""Browser session lifecycle: inactivity janitor, orphan reap, per-session teardown, ``cleanup_*`` entry
points and the atexit sweep.

Teardown per session kind (CDP supervisor stopped first, so our WebSocket closes before the endpoint):
cloud → ``provider.close_session``; Lightpanda → ``stop_lightpanda``; launcher Chromium → ``stop_browser``
(a shared persistent profile only once no other session uses it); sandbox → SIGTERM inside the sandbox
once no other session uses it;
real profile → shared per profile home, terminated at exit / on that profile's consent-off; CDP override → nothing (the user's browser).
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

logger = logging.getLogger(__name__)


def _bs():
    from tools import browser_session
    return browser_session


def _session_expiry_timestamp(session_info: dict[str, Any]) -> Optional[float]:
    """Provider-authoritative expiry as epoch seconds; None when absent or malformed."""
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
        logger.warning("Ignoring invalid cloud browser session expiry timestamp")
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def session_has_expired(session_info: dict[str, Any], *, now: Optional[float] = None) -> bool:
    """Whether a cached session crossed its provider deadline."""
    expires_at = _session_expiry_timestamp(session_info)
    return expires_at is not None and (time.time() if now is None else now) >= expires_at


def _best_effort(label: str, fn) -> None:
    """Run ``fn()``; log and swallow any exception — teardown must never abort."""
    try:
        fn()
    except Exception as e:  # health: allow BLE001 -- teardown boundary: one failure must not stop the rest
        logger.debug("%s failed: %s", label, e)


def update_session_activity(cache_key: str) -> None:
    """Touch the activity timestamp and record the owning Hermes home on first sight (#86402)."""
    bs = _bs()
    with bs._cleanup_lock:
        bs._session_last_activity[cache_key] = time.time()
        bs._session_owner_homes.setdefault(cache_key, str(get_hermes_home()))


@contextlib.contextmanager
def _session_owner_scope(cache_key: str):
    """Run under the Hermes home + secret scope owning ``cache_key``'s session (no-op when unrecorded); the
    janitor is process-global, so each teardown re-enters ITS profile, never ``os.environ``."""
    owner_home = _bs()._session_owner_homes.get(cache_key)
    if owner_home is None:
        yield
        return
    from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
    from hermes_cli.env_loader import hydrate_profile_secret_sources
    home_token = set_hermes_home_override(owner_home)
    try:
        hydrate_profile_secret_sources(Path(owner_home))
        secret_token = set_secret_scope(build_profile_secret_scope(Path(owner_home)), profile_home=owner_home)
        try:
            yield
        finally:
            reset_secret_scope(secret_token)
    finally:
        reset_hermes_home_override(home_token)


def _forget_session_tracking(cache_key: str, *, activity: bool = True, session: bool = False) -> None:
    bs = _bs()
    with bs._cleanup_lock:
        if session:
            bs._active_sessions.pop(cache_key, None)
        if activity:
            bs._session_last_activity.pop(cache_key, None)
        bs._session_owner_homes.pop(cache_key, None)
        bs._cleanup_failures.pop(cache_key, None)


# ── Teardown ─────────────────────────────────────────────────────────────────────────────────────────

def _launcher_key_in_use(launcher_key: str, session_info: dict[str, Any]) -> bool:
    bs = _bs()
    with bs._cleanup_lock:
        return any(info is not session_info and info.get("launcher_key") == launcher_key
                   for info in bs._active_sessions.values())


def _close_cloud_session(session_info: dict[str, Any]) -> None:
    bb_session_id = session_info.get("bb_session_id")
    if not bb_session_id:
        return
    from tools.browser_session_cloud import get_cloud_provider
    provider = get_cloud_provider()
    if provider is None:
        return
    try:
        provider.close_session(bb_session_id)
    except Exception as e:  # health: allow BLE001 -- a provider error must never abort the rest of the teardown
        logger.warning("Could not close cloud browser session: %s", e)


def release_session_backend(cache_key: str, session_info: dict[str, Any]) -> None:
    """Release ``session_info``'s browser BY OBJECT (an unpublished session releases through this alone)."""
    _close_cloud_session(session_info)
    features = session_info.get("features") or {}
    if features.get("lightpanda"):
        from tools.browser_lightpanda import stop_lightpanda
        stop_lightpanda(session_info.get("session_name", ""))
    launcher_key = session_info.get("launcher_key")
    if not launcher_key or _launcher_key_in_use(launcher_key, session_info):
        return  # nothing launched for it, or a shared browser another live session still uses
    if features.get("sandbox"):
        from tools.browser_session_sandbox import stop_sandbox_browser
        stop_sandbox_browser(session_info)
    elif not features.get("real_profile"):  # outlives its sessions: stopped at exit / on consent-off
        from tools.browser_launcher import stop_browser
        stop_browser(launcher_key)


def cleanup_single_session(cache_key: str) -> None:
    """Tear down one session by its exact key."""
    from tools.browser_session_cdp import stop_cdp_supervisor
    stop_cdp_supervisor(cache_key)  # close our WebSocket BEFORE the backend tears the endpoint down
    bs = _bs()
    with bs._cleanup_lock:
        session_info = bs._active_sessions.pop(cache_key, None)
    _forget_session_tracking(cache_key)
    if session_info:
        release_session_backend(cache_key, session_info)
        logger.debug("Closed browser session %s", cache_key)


def cleanup_browser(task_id: Optional[str] = None) -> None:
    """Close the sessions serving ``task_id``: its exact key, its hybrid ``::local`` sidecar, and the
    served-profile-tagged keys browser_exec builds for it (``<task>@<home>``). Per-turn cleanup calls this
    with the bare task id."""
    task = task_id or "default"
    bs = _bs()
    with bs._cleanup_lock:
        if bs.is_local_sidecar_key(task):
            keys = [task] if task in bs._active_sessions else []
        else:
            keys = [key for key in bs._active_sessions if key == task or bs.owner_task_id(key) == task]
    for key in keys:
        cleanup_single_session(key)
    if not keys:
        _forget_session_tracking(task)


def cleanup_profile_browsers(home_key: str) -> None:
    """Close one profile home's sessions, CDP supervisors and harness daemons, and drop its cached config
    (``/browser connect`` on a multiplexed gateway). Other profiles' browsers stay."""
    bs = _bs()
    with bs._cleanup_lock:
        keys = [key for key, info in bs._active_sessions.items() if info.get("home_key") == home_key]
    for key in keys:
        cleanup_single_session(key)

    def _stop_supervisors():
        from tools.browser_supervisor import SUPERVISOR_REGISTRY
        SUPERVISOR_REGISTRY.stop_home(home_key)
    _best_effort("CDP supervisor stop", _stop_supervisors)

    def _stop_harness():
        from tools.browser_use_cli import stop_harness_daemons
        stop_harness_daemons(home_key)
    _best_effort("Browser Use harness daemon stop", _stop_harness)
    from tools.browser_session_cloud import reset_caches
    reset_caches(home_key)


def _cleanup_in_owner_scope(cache_key: str) -> None:
    with _session_owner_scope(cache_key):
        cleanup_single_session(cache_key)


def cleanup_all_browsers() -> None:
    """Close every session (gateway shutdown), stop every browser this process launched, reset caches. Each
    session is closed under its owning profile, as the janitor does: a cloud session belongs to that profile's
    provider and credentials."""
    bs = _bs()
    with bs._cleanup_lock:
        keys = list(bs._active_sessions)
    for key in keys:
        _best_effort(f"browser cleanup for {key}", lambda key=key: _cleanup_in_owner_scope(key))

    def _stop_supervisors():
        from tools.browser_supervisor import SUPERVISOR_REGISTRY
        SUPERVISOR_REGISTRY.stop_all()
    _best_effort("CDP supervisor stop", _stop_supervisors)

    def _stop_harness():
        from tools.browser_use_cli import stop_harness_daemons
        stop_harness_daemons()
    _best_effort("Browser Use harness daemon stop", _stop_harness)

    def _stop_launched():
        from tools.browser_launcher import stop_all_browsers
        stop_all_browsers()  # incl. the shared real-profile / persistent-profile browsers
    _best_effort("Chromium stop", _stop_launched)

    def _stop_lightpanda():
        from tools.browser_lightpanda import stop_all_lightpanda
        stop_all_lightpanda()
    _best_effort("Lightpanda stop", _stop_lightpanda)
    from tools.browser_session_cloud import reset_caches
    reset_caches()


def emergency_cleanup_all_sessions() -> None:
    """atexit: close this process's browsers, then reap orphans left by crashed Hermes processes (every
    clean exit sweeps accumulated orphans). Idempotent."""
    bs = _bs()
    if bs._cleanup_done:
        return
    bs._cleanup_done = True
    if bs._active_sessions:
        logger.info("Emergency cleanup: closing %s active session(s)...", len(bs._active_sessions))
    _best_effort("Browser cleanup on exit", cleanup_all_browsers)
    with bs._cleanup_lock:
        bs._active_sessions.clear()
        bs._session_last_activity.clear()
        bs._session_owner_homes.clear()
        bs._cleanup_failures.clear()
    _best_effort("Orphan reap on exit", reap_orphaned_browser_sessions)


def reap_orphaned_browser_sessions() -> int:
    """Stop Chromium / ``lightpanda serve`` processes whose owning Hermes process is gone (Chromium: in every
    profile this process serves, incl. each live session's owning home)."""
    from tools.browser_launcher import reap_orphaned_browsers
    from tools.browser_lightpanda import reap_orphaned_lightpanda
    bs = _bs()
    with bs._cleanup_lock:
        session_homes = list(bs._session_owner_homes.values())
    reaped = 0
    for name, reap in (("lightpanda", reap_orphaned_lightpanda),
                       ("chromium", lambda: reap_orphaned_browsers(session_homes))):
        try:
            reaped += reap()
        except OSError as exc:
            logger.debug("orphan reap (%s) failed: %s", name, exc)
    if reaped:
        logger.info("Reaped %d orphaned browser process(es) from previous run(s)", reaped)
    return reaped


# ── Inactivity janitor ───────────────────────────────────────────────────────────────────────────────

def _human_holds_shared_browser(cache_key: str) -> bool:
    bs = _bs()
    with bs._cleanup_lock:
        session_info = bs._active_sessions.get(cache_key)
    return bool(session_info) and bs.human_holds_shared_browser(session_info)


def _force_reap(cache_key: str) -> None:
    """Janitor last resort after repeated failures (#100738): drop the record, release the backend."""
    from tools.browser_session_cdp import stop_cdp_supervisor
    stop_cdp_supervisor(cache_key)
    bs = _bs()
    with bs._cleanup_lock:
        session_info = bs._active_sessions.get(cache_key)
    _forget_session_tracking(cache_key, session=True)  # untracked even if the release below fails
    if session_info:
        release_session_backend(cache_key, session_info)


def _cleanup_one_inactive(cache_key: str, elapsed: int) -> None:
    bs = _bs()
    logger.info("Cleaning up inactive session %s (inactive for %ss)", cache_key, elapsed)
    try:
        with _session_owner_scope(cache_key):
            cleanup_single_session(cache_key)
        _forget_session_tracking(cache_key)
    except Exception as e:  # health: allow BLE001 -- counted; force-reaped after MAX_INACTIVITY_CLEANUP_FAILURES
        with bs._cleanup_lock:
            failures = bs._cleanup_failures[cache_key] = bs._cleanup_failures.get(cache_key, 0) + 1
        if failures < bs.MAX_INACTIVITY_CLEANUP_FAILURES:
            logger.warning("Error cleaning up inactive session %s (attempt %d/%d): %s",
                           cache_key, failures, bs.MAX_INACTIVITY_CLEANUP_FAILURES, e)
            return
        logger.error("Browser cleanup failed %d times for %s; force-reaping: %s", failures, cache_key, e)
        try:
            with _session_owner_scope(cache_key):
                _force_reap(cache_key)
        finally:
            _forget_session_tracking(cache_key, activity=False)


def cleanup_inactive_sessions() -> None:
    """Close sessions idle past ``BROWSER_SESSION_INACTIVITY_TIMEOUT`` (janitor thread), each under its owner
    profile's scope. A human holding the Bot Desktop lease counts as activity."""
    bs = _bs()
    now = time.time()
    with bs._cleanup_lock:
        idle = [(key, int(now - last)) for key, last in bs._session_last_activity.items()
                if now - last > bs.BROWSER_SESSION_INACTIVITY_TIMEOUT]
    for key, elapsed in idle:
        with _session_owner_scope(key):
            if _human_holds_shared_browser(key):
                update_session_activity(key)
                continue
        _cleanup_one_inactive(key, elapsed)


def _janitor() -> None:
    """Every 30 s: close idle sessions; reap orphans at startup AND every BROWSER_ORPHAN_REAP_INTERVAL."""
    bs = _bs()
    reap_every = max(1, round(bs.BROWSER_ORPHAN_REAP_INTERVAL / 30))
    cycle = 0
    while bs._cleanup_running:
        if cycle % reap_every == 0:
            _best_effort("Orphan reap", reap_orphaned_browser_sessions)
        cycle += 1
        _best_effort("Inactive browser cleanup", cleanup_inactive_sessions)
        for _ in range(30):  # 1 s granularity so stop is quick
            if not bs._cleanup_running:
                break
            time.sleep(1)


def start_cleanup_thread() -> None:
    bs = _bs()
    with bs._cleanup_lock:
        if bs._cleanup_thread is None or not bs._cleanup_thread.is_alive():
            bs._cleanup_running = True
            # health: allow HX012 -- process-wide janitor: each teardown binds its own session owner scope
            bs._cleanup_thread = threading.Thread(target=_janitor, daemon=True, name="browser-cleanup")
            bs._cleanup_thread.start()
            logger.info("Started browser inactivity janitor (timeout: %ss)", bs.BROWSER_SESSION_INACTIVITY_TIMEOUT)


def stop_cleanup_thread() -> None:
    bs = _bs()
    bs._cleanup_running = False
    thread = bs._cleanup_thread
    if thread is not None:
        # A second Ctrl+C during the timed join lands here; the janitor is a daemon thread.
        try:
            thread.join(timeout=5)
        except (SystemExit, KeyboardInterrupt):
            pass
