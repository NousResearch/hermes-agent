"""Browser sessions for ``browser_exec``: one session per cache key, created in a fixed precedence and
always carrying a ``cdp_url``.

Precedence (``BU_CDP_*`` operator env is honoured earlier, by ``browser_exec``):
CDP override (live ``/browser connect`` > ``BROWSER_CDP_URL`` > ``browser.cdp_url``) > hybrid ``::local``
sidecar (private URL, never the real profile) > cloud provider (``create_session``; on failure a LOCAL
session marked ``fallback_from_cloud``) > local: real profile (consented) > Lightpanda
(``browser.engine: lightpanda``) > the packaged Chromium via ``tools.browser_launcher``.

State lives here; sibling modules (``browser_session_*``) late-import this module inside functions.
"""

from __future__ import annotations

import atexit
import logging
import threading
import uuid
from typing import Any, Callable, Optional

from hermes_cli.config import DEFAULT_CONFIG
from hermes_constants import hermes_home_key
from utils import env_int

logger = logging.getLogger(__name__)

LOCAL_SUFFIX = "::local"

# Inactivity timeout: config.yaml is authoritative; BROWSER_INACTIVITY_TIMEOUT stays a legacy env fallback.
DEFAULT_SESSION_INACTIVITY_TIMEOUT = int(DEFAULT_CONFIG.get("browser", {}).get("inactivity_timeout", 120))


def _get_session_inactivity_timeout() -> int:
    from tools.browser_session_cloud import inactivity_timeout
    return inactivity_timeout(env_int("BROWSER_INACTIVITY_TIMEOUT", DEFAULT_SESSION_INACTIVITY_TIMEOUT))


BROWSER_SESSION_INACTIVITY_TIMEOUT = _get_session_inactivity_timeout()
# Orphan reap cadence: a startup-only reap never recovers a leak that appears in a long-lived process.
BROWSER_ORPHAN_REAP_INTERVAL = 300
MAX_INACTIVITY_CLEANUP_FAILURES = 3

_active_sessions: dict[str, dict[str, Any]] = {}
_session_last_activity: dict[str, float] = {}
# Cache keys with a browser_exec call currently RUNNING. The harness daemon talks CDP straight to
# the browser, so a long call's traffic never reaches _session_last_activity; without this set the
# janitor reads wall-clock inactivity as an idle session and kills the browser mid-call.
_session_inflight: set[str] = set()
# Owner Hermes home per session: the janitor is one process-global thread, so each teardown re-enters the
# OWNING profile's scope (#86402).
_session_owner_homes: dict[str, str] = {}
_cleanup_failures: dict[str, int] = {}  # consecutive janitor failures (#100738)
_cleanup_lock = threading.Lock()  # protects every dict above
_cleanup_thread: Optional[threading.Thread] = None
_cleanup_running = False
_cleanup_done = False

# A session built while its profile published a new live endpoint is closed and rebuilt; bounded so a
# flapping connect/disconnect loop cannot keep a call building browsers forever.
_STALE_ENDPOINT_RETRIES = 2


# ── Env ──────────────────────────────────────────────────────────────────────────────────────────────

def build_browser_env() -> dict:
    """Env for every browser child (Chromium, ``lightpanda serve``, the harness CLI base): the served
    profile's child env WITHOUT credentials, the Bot Desktop DISPLAY when its screen is up, a loopback
    NO_PROXY (a macOS system proxy would capture the local CDP dial, #110565), and a short TMPDIR
    (Chrome's SingletonSocket overflows the AF_UNIX path cap under a deep scratch dir)."""
    from agent.proxy_bypass import add_loopback_no_proxy
    from hermes_constants import socket_safe_tmpdir
    from tools.bot_desktop.runtime import desktop_env
    from tools.environments.local import served_profile_child_env
    env = add_loopback_no_proxy(desktop_env(served_profile_child_env(inherit_credentials=False)))
    env["TMPDIR"] = socket_safe_tmpdir()
    return env


# ── Bot Desktop fence ────────────────────────────────────────────────────────────────────────────────

def shares_bot_desktop_browser(session_info: dict[str, Any]) -> bool:
    """Every LOCAL session is a browser Hermes launched with this profile's Bot Desktop DISPLAY, so it is the
    screen a human who took over is typing into. Cloud / user-supplied CDP sessions are another browser. A
    human lease with the screen already gone still fences (computer_use does the same)."""
    if not (session_info.get("features") or {}).get("local"):
        return False
    from tools.bot_desktop import lease as _bd_lease, runtime as _bd_runtime
    return bool(_bd_runtime.published_env().get("DISPLAY")) or _bd_lease.human_holds()


def human_holds_shared_browser(session_info: dict[str, Any]) -> bool:
    """True while a human holds the Bot Desktop lease over the browser ``session_info`` shares with them; the
    janitor treats that as activity (reaping it mid-login kills the human's session, #110064)."""
    if not shares_bot_desktop_browser(session_info):
        return False
    from tools.bot_desktop import lease as _bd_lease
    return _bd_lease.human_holds()


def run_fenced(session_info: dict[str, Any], fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run ``fn`` under the Bot Desktop lease fence when ``session_info`` is the bot's LOCAL browser.

    While a human holds the lease every action AND read against it is refused (the page may show their
    credential); the fence brackets the whole run so a takeover mid-command also voids the result.
    """
    if not shares_bot_desktop_browser(session_info):
        return fn()
    from tools.bot_desktop import lease as _bd_lease
    try:
        admitted = _bd_lease.assert_agent_may_act()
    except _bd_lease.HumanHasControl as e:
        return {"success": False, "error": str(e), "code": "human_has_control"}
    result = fn()
    if _bd_lease.get().epoch != admitted.epoch:
        return {"success": False, "code": "human_has_control",
                "error": "A human took over the bot's screen while this browser command ran; its result was "
                         "discarded. Tell the user what you need; retry once they hand back."}
    return result


def ensure_screen_for_headed_chromium() -> None:
    """Tool-call boundary for ``bot_desktop.auto_start``: a headed Chromium is about to start, so a headless
    profile gets its screen first. Headless browsing and Lightpanda never bring one up."""
    from tools.browser_session_cloud import is_headed_mode
    if is_headed_mode():
        from tools.bot_desktop.runtime import ensure_started_for_tool
        ensure_started_for_tool()


# ── Session records ──────────────────────────────────────────────────────────────────────────────────

def _session_record(prefix: str, cdp_url: Optional[str], features: dict[str, Any]) -> dict[str, Any]:
    """Fresh session dict with a random ``<prefix>_<hex10>`` session name (log / Lightpanda identity)."""
    return {"session_name": f"{prefix}_{uuid.uuid4().hex[:10]}", "bb_session_id": None,
            "cdp_url": cdp_url, "features": features}


def is_local_sidecar_key(session_key: str) -> bool:
    return session_key.endswith(LOCAL_SUFFIX)


def owner_task_id(session_key: str) -> str:
    """The task a session key belongs to: strips the ``::local`` sidecar suffix and the ``@<home>`` served
    profile tag browser_exec adds (per-turn cleanup is called with the bare task id)."""
    key = session_key[: -len(LOCAL_SUFFIX)] if is_local_sidecar_key(session_key) else session_key
    return key.rsplit("@", 1)[0] if "@" in key else key


def real_profile_note(cache_key: Optional[str]) -> Optional[str]:
    """The real-profile fallback note on ``cache_key``'s existing session; never creates or touches one."""
    with _cleanup_lock:
        return (_active_sessions.get(cache_key or "default") or {}).get("real_profile_unavailable")


def session_for_task(task_id: Optional[str]) -> dict[str, Any]:
    """The active session serving ``task_id`` (its exact key, else one whose owner is that task), or {}."""
    task = task_id or "default"
    with _cleanup_lock:
        if task in _active_sessions:
            return _active_sessions[task]
        return next((info for key, info in _active_sessions.items()
                     if owner_task_id(key) == task and not is_local_sidecar_key(key)), {})


# ── Session creation ─────────────────────────────────────────────────────────────────────────────────

def _bot_desktop_target() -> Optional[tuple[str, dict[str, Any]]]:
    """``(launcher key, launch kwargs)`` for the shared Bot Desktop profile while its host screen is up: the
    agent's Chromium and the dock's Browser icon are THE SAME browser (binary + user-data-dir), and a dock
    instance a human already started is attached, never relaunched."""
    from tools.bot_desktop import runtime as _bd_runtime
    if not _bd_runtime.published_env().get("DISPLAY"):
        return None
    from tools.bot_desktop import browser as _bd_browser
    udd = str(_bd_browser.profile_dir())
    return f"profile:{udd}", {"executable": _bd_browser.executable(), "user_data_dir": udd, "ephemeral": False,
                              "extra_args": _bd_browser.SHARED_PROFILE_ARGS}


def _launch_target(cache_key: str) -> tuple[str, dict[str, Any], bool]:
    """``(launcher key, launch kwargs, shared)``. A persistent profile (Bot Desktop, ``browser.user_data_dir``)
    is keyed by its dir so every cache key shares the one Chromium Chromium's singleton allows on it."""
    bot_desktop = _bot_desktop_target()
    if bot_desktop is not None:
        return bot_desktop[0], bot_desktop[1], True
    from tools.browser_session_cloud import user_data_dir
    udd = user_data_dir()
    if udd:
        return f"profile:{udd}", {"user_data_dir": udd, "ephemeral": False}, True
    return cache_key, {}, False


def _create_chromium_session(cache_key: str) -> dict[str, Any]:
    from tools.bot_desktop import placement, runtime
    try:
        # Policy first: a refused placement raises, a terminal placement starts its sandbox screen; neither
        # ever falls back to a host browser.
        where = runtime.tool_placement()
    except RuntimeError as exc:
        raise RuntimeError(f"The browser cannot run here: {exc}") from exc
    if where == placement.TERMINAL:
        from tools.browser_session_sandbox import create_sandbox_session
        return create_sandbox_session(cache_key)
    from tools.browser_launcher import launch_chromium
    ensure_screen_for_headed_chromium()  # first: an auto-started screen selects the Bot Desktop profile
    launcher_key, kwargs, shared = _launch_target(cache_key)
    browser, err = launch_chromium(launcher_key, **kwargs)
    if browser is None:
        raise RuntimeError(err or "the local browser could not be started")
    info = _session_record("h", browser.cdp_url, {"local": True})
    info.update(launcher_key=launcher_key, shared_profile=shared)
    logger.info("Created local browser session %s for %s (Chromium pid %s)", info["session_name"], cache_key, browser.pid)
    return info


def _create_lightpanda_session(cache_key: str) -> dict[str, Any]:
    from tools.browser_lightpanda import launch_lightpanda
    from tools.browser_session_cloud import is_local_backend
    info = _session_record("lp", None, {"local": True, "lightpanda": True})
    server, err = launch_lightpanda(info["session_name"], block_private_networks=not is_local_backend())
    if err:
        raise RuntimeError(err)
    info["cdp_url"] = server.cdp_url
    logger.info("Created Lightpanda session %s (port %s) for %s", info["session_name"], server.port, cache_key)
    return info


def _create_local_session(cache_key: str, allow_real_profile: bool = True) -> dict[str, Any]:
    """Local session. The ONE place real profile vs fresh browser is decided: when the consented profile
    can't load, the fresh browser carries ``real_profile_unavailable`` so the first result tells the model it
    is signed out. The hybrid sidecar passes ``allow_real_profile=False``: the user's cookie jar must not
    reach an arbitrary internal host the model chose."""
    fallback_note = None
    if allow_real_profile:
        from tools.browser_session_real_profile import real_profile_cdp, real_profile_fallback_note, real_profile_key
        cdp_url, err = real_profile_cdp()
        if err:
            logger.warning("real-profile unavailable for %s, using a fresh browser: %s", cache_key, err)
            fallback_note = real_profile_fallback_note(err)
        elif cdp_url:
            info = _session_record("rp", cdp_url, {"local": True, "real_profile": True})
            info.update(launcher_key=real_profile_key(), shared_profile=True)
            logger.info("Created real-profile local session %s for %s", info["session_name"], cache_key)
            return info
    from tools.browser_lightpanda import using_lightpanda_engine
    info = _create_lightpanda_session(cache_key) if using_lightpanda_engine() else _create_chromium_session(cache_key)
    if fallback_note:
        info["real_profile_unavailable"] = fallback_note
    return info


def _create_cloud_session_or_fallback(cache_key: str, provider) -> dict[str, Any]:
    """Cloud session; on failure (429, quota, outage) a LOCAL launcher session marked ``fallback_from_cloud``.
    ``cdp_url`` is resolved here because some providers return an HTTP discovery URL."""
    from tools.browser_session_cdp import resolve_cdp_override
    try:
        session_info = provider.create_session(cache_key)
        if not session_info or not isinstance(session_info, dict):
            raise ValueError(f"Cloud provider returned invalid session: {session_info!r}")
    except Exception as e:  # health: allow BLE001 -- any provider failure falls back to a local browser
        provider_name = type(provider).__name__
        logger.warning("Cloud provider %s failed (%s); falling back to local Chromium for %s",
                       provider_name, e, cache_key, exc_info=True)
        try:
            local = _create_local_session(cache_key)
        except Exception as local_error:  # health: allow BLE001 -- report both failures together
            raise RuntimeError(f"Cloud provider {provider_name} failed ({e}) and local "
                               f"fallback also failed ({local_error})") from e
        return {**local, "fallback_from_cloud": True, "fallback_reason": str(e), "fallback_provider": provider_name}
    session_info = dict(session_info)
    if session_info.get("cdp_url"):
        session_info["cdp_url"] = resolve_cdp_override(str(session_info["cdp_url"]))
    return session_info


def _create_session_for_key(cache_key: str, force_local: bool) -> dict[str, Any]:
    """Fresh session (runs OUTSIDE the lock: cloud mode makes a network call, a launch takes seconds)."""
    from tools.browser_session_cdp import get_cdp_override
    from tools.browser_session_cloud import get_cloud_provider
    override = get_cdp_override()
    if override and not force_local:
        logger.info("Created CDP browser session for %s", cache_key)
        return _session_record("cdp", override, {"cdp_override": True})
    if force_local:
        return _create_local_session(cache_key, allow_real_profile=False)
    provider = get_cloud_provider()
    if provider is None:
        return _create_local_session(cache_key)
    return _create_cloud_session_or_fallback(cache_key, provider)


def _backend_dead(session_info: dict[str, Any]) -> bool:
    """True when a local session's process is gone (Chromium crashed or was killed, ``lightpanda serve``
    exited): the next call builds a fresh one instead of handing out a dead endpoint.

    A launcher key is shared by every cache key on one persistent profile, so a live browser on the key is
    not enough: when another key relaunched it after a crash, this record still names the dead endpoint."""
    features = session_info.get("features") or {}
    if features.get("lightpanda"):
        from tools.browser_lightpanda import get_server
        server = get_server(session_info.get("session_name", ""))
        return server is None or not server.is_alive()
    if features.get("sandbox"):
        from tools.browser_session_sandbox import sandbox_browser_alive
        return not sandbox_browser_alive(session_info)
    launcher_key = session_info.get("launcher_key")
    if launcher_key:
        from tools.browser_launcher import get_browser
        browser = get_browser(launcher_key)
        return browser is None or browser.cdp_url != session_info.get("cdp_url")
    return False


def _reuse_or_evict(cache_key: str) -> Optional[dict[str, Any]]:
    from tools import browser_session_lifecycle as _lifecycle
    with _cleanup_lock:
        existing = _active_sessions.get(cache_key)
    if existing is None:
        return None
    if not _lifecycle.session_has_expired(existing) and not _backend_dead(existing):
        return existing
    logger.info("Replacing expired or dead browser session for %s", cache_key)
    _lifecycle.cleanup_single_session(cache_key)
    _lifecycle.update_session_activity(cache_key)  # teardown dropped the entry; track the replacement
    with _cleanup_lock:
        replacement = _active_sessions.get(cache_key)  # another thread may already have rebuilt it
    return replacement if replacement is not None and replacement is not existing else None


def _publish(cache_key: str, session_info: dict[str, Any], home_key: str) -> dict[str, Any]:
    session_info = dict(session_info)
    session_info.setdefault("session_key", cache_key)
    session_info.setdefault("owner_task_id", owner_task_id(cache_key))
    session_info.setdefault("home_key", home_key)  # the profile reaped on /browser connect
    _active_sessions[cache_key] = session_info
    return session_info


def get_session_info(cache_key: Optional[str] = None) -> dict[str, Any]:
    """Get or create the session for ``cache_key`` (thread-safe); starts the inactivity janitor and touches
    activity. A ``::local`` key forces a local browser even with a cloud provider configured."""
    from tools import browser_session_lifecycle as _lifecycle
    from tools.browser_session_cdp import ensure_cdp_supervisor, live_cdp_generation
    key = cache_key or "default"
    _lifecycle.start_cleanup_thread()
    _lifecycle.update_session_activity(key)
    reused = _reuse_or_evict(key)
    if reused is not None:
        return reused

    force_local = is_local_sidecar_key(key)
    home_key = hermes_home_key()
    for attempt in range(_STALE_ENDPOINT_RETRIES + 1):
        generation = live_cdp_generation(home_key)
        session_info = _create_session_for_key(key, force_local)
        with _cleanup_lock:
            if key in _active_sessions:  # created concurrently during the launch: don't leak ours
                winner = _active_sessions[key]
                stale_ours = session_info
                session_info = None
            else:
                winner = stale_ours = None
                # A /browser connect publish while ours was built may mean it resolved the old endpoint:
                # never publish it (a concurrent caller would get a browser being closed); rebuild.
                if live_cdp_generation(home_key) == generation or attempt == _STALE_ENDPOINT_RETRIES:
                    session_info = _publish(key, session_info, home_key)
                    break
        if winner is not None:
            _lifecycle.release_session_backend(key, stale_ours)
            return winner
        logger.info("Browser endpoint changed during session creation for %s; recreating", key)
        _lifecycle.release_session_backend(key, session_info)

    # Cloud / CDP-override sessions get the supervisor here (dialogs, frame tree, vault fill). Local
    # browsers are attached by browser_exec per call under the task id.
    if not (session_info.get("features") or {}).get("local"):
        ensure_cdp_supervisor(key)
    return session_info


def _atexit_cleanup() -> None:
    """atexit only — NO SIGINT/SIGTERM handlers calling sys.exit(): a SystemExit raised inside a
    prompt_toolkit key-binding callback corrupts the coroutine state and makes the process unkillable."""
    try:
        from tools.browser_session_lifecycle import emergency_cleanup_all_sessions, stop_cleanup_thread
    except ImportError:  # half-updated tree mid-`hermes update` (#112437): nothing resolvable to clean
        return
    emergency_cleanup_all_sessions()
    stop_cleanup_thread()


atexit.register(_atexit_cleanup)
