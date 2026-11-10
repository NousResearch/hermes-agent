"""User-supplied CDP endpoints (live ``/browser connect``, ``BROWSER_CDP_URL``, ``browser.cdp_url``),
dialog-policy config and the per-task CDP supervisor lifecycle."""

from __future__ import annotations

import logging
import threading
from typing import Optional

from agent.proxy_bypass import loopback_request_kwargs
from agent.redact import redact_cdp_url

logger = logging.getLogger(__name__)

# Live ``/browser connect`` endpoints, one per profile home (``hermes_home_key()``): a multiplexed
# gateway serves every profile from this process, so a process-wide value would reach them all.
_live_overrides: dict[str, str] = {}
_live_overrides_lock = threading.Lock()
# Per-profile count of live-endpoint publishes. A session records the value it saw before resolving its
# endpoint; ``get_session_info`` closes one whose profile published in between.
_live_generations: dict[str, int] = {}


def resolve_cdp_override(cdp_url: str) -> str:
    """Normalize a CDP endpoint into a concrete websocket URL.

    Full ``ws://.../devtools/browser/...`` endpoints pass through; HTTP discovery roots and bare
    ``ws://host:port`` resolve via ``/json/version`` → ``webSocketDebuggerUrl`` (falls back to the raw value).
    """
    raw = (cdp_url or "").strip()
    if not raw:
        return ""
    lowered = raw.lower()
    if "/devtools/browser/" in lowered:
        return raw
    discovery_url = raw
    if lowered.startswith(("ws://", "wss://")):
        if not (raw.count(":") == 2 and raw.rstrip("/").rsplit(":", 1)[-1].isdigit() and "/" not in raw.split(":", 2)[-1]):
            return raw
        discovery_url = ("http://" if lowered.startswith("ws://") else "https://") + raw.split("://", 1)[1]
    version_url = discovery_url if discovery_url.lower().endswith("/json/version") else discovery_url.rstrip("/") + "/json/version"
    try:
        import requests  # lazy: shared module object, so test patches still apply
        response = requests.get(version_url, timeout=10, **loopback_request_kwargs(version_url))
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # health: allow BLE001 -- any discovery failure falls back to the raw endpoint
        logger.warning("Failed to resolve CDP endpoint %s via %s: %s", redact_cdp_url(raw),
                       redact_cdp_url(version_url), redact_cdp_url(str(exc)))
        return raw
    ws_url = str(payload.get("webSocketDebuggerUrl") or "").strip()
    if ws_url:
        logger.info("Resolved CDP endpoint %s -> %s", redact_cdp_url(raw), redact_cdp_url(ws_url))
        return ws_url
    logger.warning("CDP discovery at %s did not return webSocketDebuggerUrl; using raw endpoint",
                   redact_cdp_url(version_url))
    return raw


def live_cdp_override() -> str:
    """The current profile's live ``/browser connect`` endpoint, or "".

    On a multiplexed gateway a context with no profile bound answers "" rather than the launch
    profile's endpoint, the way ``get_secret`` fails closed."""
    from agent.secret_scope import is_multiplex_active
    from hermes_constants import get_hermes_home_override, hermes_home_key
    if get_hermes_home_override() is None and is_multiplex_active():
        return ""
    with _live_overrides_lock:
        return _live_overrides.get(hermes_home_key(), "")


def set_live_cdp_override(url: Optional[str]) -> None:
    """Set (or with a falsy ``url`` clear) the current profile's live endpoint and bump its generation."""
    from hermes_constants import hermes_home_key
    key = hermes_home_key()
    with _live_overrides_lock:
        if url:
            _live_overrides[key] = url
        else:
            _live_overrides.pop(key, None)
        _live_generations[key] = _live_generations.get(key, 0) + 1


def live_cdp_generation(home_key: Optional[str] = None) -> int:
    """The current (or ``home_key``'s) profile's live-endpoint generation."""
    from hermes_constants import hermes_home_key
    with _live_overrides_lock:
        return _live_generations.get(home_key or hermes_home_key(), 0)


def env_cdp_url() -> str:
    """``BROWSER_CDP_URL`` from this profile's secret scope. A routed profile reads only its own ``.env``
    (``os.environ`` holds the launch profile's value)."""
    from agent.secret_scope import current_secret_scope, get_secret, serves_routed_profile
    if serves_routed_profile():
        value = (current_secret_scope() or {}).get("BROWSER_CDP_URL")
    else:
        value = get_secret("BROWSER_CDP_URL")
    return (value or "").strip()


def get_cdp_override_raw() -> str:
    """The *configured* CDP override, no network I/O: live ``/browser connect`` > ``BROWSER_CDP_URL`` >
    ``browser.cdp_url``. Is-it-configured gates (check_fns, local-mode checks, doctor) MUST use this:
    a 10 s discovery probe against a stale URL would stall every schema build."""
    from tools.browser_session_cloud import browser_cfg
    return live_cdp_override() or env_cdp_url() or browser_cfg("cdp_url", "", lambda v: str(v or "").strip())


def get_cdp_override() -> str:
    """Resolved CDP override URL, or "". May do HTTP discovery: call only on paths about to connect."""
    raw = get_cdp_override_raw()
    return resolve_cdp_override(raw) if raw else ""


def get_dialog_policy_config() -> tuple[str, float]:
    """``browser.dialog_policy`` + ``browser.dialog_timeout_s``; supervisor defaults when absent/invalid."""
    from tools.browser_session_cloud import browser_cfg
    from tools.browser_supervisor_dialogs import DEFAULT_DIALOG_POLICY, DEFAULT_DIALOG_TIMEOUT_S, _VALID_POLICIES
    policy = str(browser_cfg("dialog_policy", DEFAULT_DIALOG_POLICY) or DEFAULT_DIALOG_POLICY)
    if policy not in _VALID_POLICIES:
        logger.debug("Invalid browser.dialog_policy=%r; using default", policy)
        policy = DEFAULT_DIALOG_POLICY
    try:
        timeout_s = float(browser_cfg("dialog_timeout_s", DEFAULT_DIALOG_TIMEOUT_S) or DEFAULT_DIALOG_TIMEOUT_S)
    except (TypeError, ValueError):
        timeout_s = DEFAULT_DIALOG_TIMEOUT_S
    return policy, (timeout_s if timeout_s > 0 else DEFAULT_DIALOG_TIMEOUT_S)


def ensure_cdp_supervisor(task_id: str) -> None:
    """Start the CDP supervisor for ``task_id`` when an endpoint is reachable (idempotent; restarts on a
    URL change). URL precedence: the CDP override, then the session's own ``cdp_url``. A failed attach
    never breaks the session."""
    cdp_url = get_cdp_override()
    if not cdp_url:
        from tools import browser_session as _bs
        with _bs._cleanup_lock:
            session_cdp = str((_bs._active_sessions.get(task_id) or {}).get("cdp_url") or "")
        cdp_url = resolve_cdp_override(session_cdp) if session_cdp else ""
    if not cdp_url:
        return
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY
        policy, timeout_s = get_dialog_policy_config()
        SUPERVISOR_REGISTRY.get_or_start(task_id=task_id, cdp_url=cdp_url, dialog_policy=policy,
                                         dialog_timeout_s=timeout_s)
    except Exception as exc:  # health: allow BLE001 -- a failed supervisor attach must not break the session
        logger.debug("CDP supervisor attach for task=%s failed (non-fatal): %s", task_id, exc)


def stop_cdp_supervisor(task_id: str) -> None:
    """Stop ``task_id``'s CDP supervisor if one exists."""
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY
        SUPERVISOR_REGISTRY.stop(task_id)
    except Exception as exc:  # health: allow BLE001 -- teardown must never abort on a supervisor error
        logger.debug("CDP supervisor stop for task=%s failed (non-fatal): %s", task_id, exc)
