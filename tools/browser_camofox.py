"""Camofox browser backend — local anti-detection browser via REST API.

Camofox-browser is a self-hosted Node.js server wrapping Camoufox (Firefox
fork with C++ fingerprint spoofing).  It exposes a REST API that maps 1:1
to our browser tool interface: accessibility snapshots with element refs,
click/type/scroll by ref, screenshots, etc.

When ``CAMOFOX_URL`` is set (e.g. ``http://localhost:9377``), the browser
tools route through this module instead of the ``agent-browser`` CLI.

Setup::

    # Option 1: npm
    git clone https://github.com/jo-inc/camofox-browser && cd camofox-browser
    npm install && npm start   # downloads Camoufox (~300MB) on first run

    # Option 2: Docker
    docker run -p 9377:9377 -e CAMOFOX_PORT=9377 jo-inc/camofox-browser

Then set ``CAMOFOX_URL=http://localhost:9377`` in ``~/.hermes/.env``.
For Docker Camofox, optionally set ``CAMOFOX_REWRITE_LOOPBACK_URLS=true``
so page URLs like ``http://127.0.0.1:3000`` are opened inside the
container as ``http://host.docker.internal:3000``.

Multi-account operation
-----------------------
Every tool here takes an optional ``user_id``.  Camofox maps each ``userId``
to its own browser profile (its own cookies and logins), so passing a
different one per call lets a single Hermes process drive several signed-in
accounts without spawning extra Hermes profiles or instances.  Omitting it
keeps the historical single-identity behaviour.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import SplitResult, urlsplit, urlunsplit

import requests

from agent.secret_scope import get_secret
from hermes_cli.config import cfg_get, load_config, read_raw_config
from tools.browser_camofox_state import get_camofox_identity
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 30  # fallback when config is unreadable
_SNAPSHOT_MAX_CHARS = 80_000  # camofox paginates at this limit
_vnc_url: Optional[str] = None  # cached from /health response
_vnc_url_checked = False  # only probe once per process

# Cached command timeout from config (resolved lazily, like browser_tool)
_cached_cmd_timeout: Optional[int] = None
_cmd_timeout_resolved = False


def _get_command_timeout() -> int:
    """Return ``browser.command_timeout`` from config, falling back to 30s.

    Mirrors :func:`tools.browser_tool._get_command_timeout` so both the
    local browser path and the Camofox path honour the same config knob.
    Result is cached after the first call.
    """
    global _cached_cmd_timeout, _cmd_timeout_resolved
    if _cmd_timeout_resolved:
        return _cached_cmd_timeout  # type: ignore[return-value]

    _cmd_timeout_resolved = True
    result = _DEFAULT_TIMEOUT
    try:
        cfg = read_raw_config()
        val = cfg_get(cfg, "browser", "command_timeout")
        if val is not None:
            result = max(int(val), 5)  # floor at 5s
    except Exception as exc:
        logger.debug("Could not read browser.command_timeout: %s", exc)
    _cached_cmd_timeout = result
    return result


def _auth_headers() -> Dict[str, str]:
    """Return Authorization header when CAMOFOX_API_KEY is set."""
    key = (get_secret("CAMOFOX_API_KEY", "") or "").strip()
    if key:
        return {"Authorization": f"Bearer {key}"}
    return {}


def get_camofox_url() -> str:
    """Return the configured Camofox server URL, or empty string."""
    return (get_secret("CAMOFOX_URL", "") or "").rstrip("/")


def _config_cdp_url() -> str:
    """Persistent ``browser.cdp_url`` from config.yaml, or empty string.

    Read here (instead of importing ``browser_tool._get_cdp_override`` to avoid
    a circular import) so Camofox can yield to a config-based CDP override the
    same way it already yields to the ``BROWSER_CDP_URL`` env override.
    """
    try:
        from hermes_cli.config import read_raw_config

        browser_cfg = read_raw_config().get("browser", {})
        if isinstance(browser_cfg, dict):
            return str(browser_cfg.get("cdp_url", "") or "").strip()
    except Exception:
        pass
    return ""


def is_camofox_mode() -> bool:
    """True when Camofox backend is configured and no CDP override is active.

    A CDP override takes priority over Camofox so the browser tools operate on
    the real CDP browser (and a CDP backend is treated as non-local for SSRF
    checks) instead of being silently routed to Camofox. The override may come
    from the ``BROWSER_CDP_URL`` env var (set by ``/browser connect``) OR a
    persistent ``browser.cdp_url`` in config.yaml — both are honored, matching
    ``browser_tool._get_cdp_override()``'s precedence. (Previously only the env
    var suppressed Camofox, so ``CAMOFOX_URL`` + a config CDP override still
    routed navigation through Camofox.)
    """
    if os.getenv("BROWSER_CDP_URL", "").strip():
        return False
    if _config_cdp_url():
        return False
    return bool(get_camofox_url())


def check_camofox_available() -> bool:
    """Verify the Camofox server is reachable."""
    global _vnc_url, _vnc_url_checked
    url = get_camofox_url()
    if not url:
        return False
    try:
        resp = requests.get(f"{url}/health", timeout=5)
        if resp.status_code == 200 and not _vnc_url_checked:
            try:
                data = resp.json()
                vnc_port = data.get("vncPort")
                if isinstance(vnc_port, int) and 1 <= vnc_port <= 65535:
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    host = parsed.hostname or "localhost"
                    _vnc_url = f"http://{host}:{vnc_port}"
            except (ValueError, KeyError):
                pass
            _vnc_url_checked = True
        return resp.status_code == 200
    except Exception:
        return False


def get_vnc_url() -> Optional[str]:
    """Return the VNC URL if the Camofox server exposes one, or None."""
    if not _vnc_url_checked:
        check_camofox_available()
    return _vnc_url


def _get_camofox_config() -> Dict[str, Any]:
    """Return the ``browser.camofox`` config block, or an empty dict."""
    try:
        camofox_cfg = load_config().get("browser", {}).get("camofox", {})
    except Exception as exc:
        logger.warning("camofox config check failed, defaulting to disabled: %s", exc)
        return {}
    return camofox_cfg if isinstance(camofox_cfg, dict) else {}


def _managed_persistence_enabled() -> bool:
    """Return whether Hermes-managed persistence is enabled for Camofox.

    When enabled, sessions use a stable profile-scoped userId so the
    Camofox server can map it to a persistent browser profile directory.
    When disabled (default), each session gets a random userId (ephemeral).

    Controlled by ``browser.camofox.managed_persistence`` in config.yaml.
    """
    return bool(_get_camofox_config().get("managed_persistence"))


def _camofox_identity_override(task_id: Optional[str], camofox_cfg: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Return an externally configured Camofox identity, if one is set.

    Integrations that own the visible Camofox browser can set a shared user ID
    so Hermes operates in the same browser profile instead of creating a
    separate private session.
    """
    user_id = (
        (get_secret("CAMOFOX_USER_ID", "") or "").strip()
        or str(camofox_cfg.get("user_id") or "").strip()
    )
    if not user_id:
        return None

    session_key = (
        (get_secret("CAMOFOX_SESSION_KEY", "") or "").strip()
        or str(camofox_cfg.get("session_key") or "").strip()
        or f"task_{(task_id or 'default')[:16]}"
    )
    return {"user_id": user_id, "session_key": session_key}


def _env_flag(name: str) -> Optional[bool]:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return None
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    logger.debug("Ignoring invalid boolean env %s=%r", name, raw)
    return None


def _adopt_existing_tab_enabled(camofox_cfg: Dict[str, Any]) -> bool:
    """Return whether Hermes should recover an existing Camofox tab ID."""
    env_value = _env_flag("CAMOFOX_ADOPT_EXISTING_TAB")
    if env_value is not None:
        return env_value
    return bool(camofox_cfg.get("adopt_existing_tab"))


def _loopback_rewrite_enabled(camofox_cfg: Dict[str, Any]) -> bool:
    """Return whether loopback navigation URLs should be rewritten for Docker.

    ``CAMOFOX_URL`` itself often points at a host-published Docker port such as
    ``http://127.0.0.1:9377``.  That is correct for Hermes talking to the
    Camofox control API, but a page URL like ``http://127.0.0.1:3000`` is opened
    by the browser *inside* the Docker container.  In that context loopback
    points at the container, not the host running the web app.

    The rewrite is opt-in because non-Docker Camofox installs run the browser on
    the host, where loopback URLs are already correct.
    """
    env_value = _env_flag("CAMOFOX_REWRITE_LOOPBACK_URLS")
    if env_value is not None:
        return env_value
    return bool(camofox_cfg.get("rewrite_loopback_urls"))


def _loopback_rewrite_host(camofox_cfg: Dict[str, Any]) -> str:
    """Return the host alias used when rewriting loopback page URLs."""
    return (
        os.getenv("CAMOFOX_LOOPBACK_HOST_ALIAS", "").strip()
        or str(camofox_cfg.get("loopback_host_alias") or "").strip()
        or "host.docker.internal"
    )


def _is_loopback_hostname(hostname: Optional[str]) -> bool:
    """Return True for localhost/127.0.0.0/8/::1-style hostnames."""
    if not hostname:
        return False
    host = hostname.strip().strip("[]").lower()
    if host in {"localhost", "localhost.localdomain"}:
        return True
    try:
        import ipaddress

        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _rewrite_loopback_url_for_camofox(url: str) -> tuple[str, Optional[Dict[str, str]]]:
    """Rewrite loopback page URLs for Docker-hosted Camofox, if configured.

    Returns ``(rewritten_url, metadata)``.  ``metadata`` is present only when a
    rewrite happened so the tool result can disclose the change to the model.
    """
    camofox_cfg = _get_camofox_config()
    if not _loopback_rewrite_enabled(camofox_cfg):
        return url, None

    try:
        parsed = urlsplit(url)
    except ValueError:
        return url, None

    if parsed.scheme not in {"http", "https"} or not _is_loopback_hostname(parsed.hostname):
        return url, None

    alias = _loopback_rewrite_host(camofox_cfg)
    if not alias:
        return url, None

    userinfo = ""
    if parsed.username:
        userinfo = parsed.username
        if parsed.password:
            userinfo += f":{parsed.password}"
        userinfo += "@"
    host_part = f"[{alias}]" if ":" in alias and not alias.startswith("[") else alias
    port_part = f":{parsed.port}" if parsed.port else ""
    rewritten = urlunsplit(
        SplitResult(parsed.scheme, f"{userinfo}{host_part}{port_part}", parsed.path, parsed.query, parsed.fragment)
    )
    return rewritten, {
        "from": parsed.hostname or "",
        "to": alias,
        "original_url": url,
        "rewritten_url": rewritten,
    }


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
# Maps session cache key -> {"user_id": str, "tab_id": str|None, ...}
#
# The cache key is the bare ``task_id`` for the default (single-identity) path
# and ``<task_id>::camofox_user::<user_id>`` when a caller pins an explicit
# per-call identity.  Suffixing keeps today's key space byte-identical for
# every existing caller while letting one task drive several Camofox accounts
# side by side; it mirrors the ``::local`` sidecar key convention in
# ``browser_tool._is_local_sidecar_key``.
_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()

_USER_SCOPE_SEP = "::camofox_user::"

# Charset for a per-call userId.  Deliberately narrow because the value reaches
# ``DELETE /sessions/{user_id}`` as a *path segment* (see :func:`camofox_close`)
# and, unlike the config/env identity, it can originate from a model tool call:
#
#   - first char must be alphanumeric, which rejects ``.`` / ``..`` / ``-flag``.
#     ``DELETE /sessions/..`` normalizes to ``DELETE /sessions`` on some servers,
#     which would wipe every user rather than one.
#   - no ``/``, ``\``, ``%`` or whitespace, so neither path traversal nor
#     percent-encoded traversal can be expressed.
#
# Matched with ``fullmatch``: Python's ``$`` also matches just before a trailing
# newline, so ``match(...)`` would accept ``"acct\n"``.  The ``.strip()`` below
# happens to remove it today, but a security check must not depend on that.
_CALL_USER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@-]{0,63}")


class InvalidCamofoxUserId(ValueError):
    """Raised when a per-call ``user_id`` fails validation."""


def _validate_call_user_id(user_id: Any) -> str:
    """Validate and return a caller-supplied Camofox ``userId``.

    Only applies to per-call identities.  ``CAMOFOX_USER_ID`` /
    ``browser.camofox.user_id`` are operator-configured and stay unvalidated —
    tightening those retroactively would break working deployments, and they
    are not attacker-reachable the way a tool argument is.

    The type check is not defensive nit-picking: models emit type-confused tool
    arguments, and a bare truthiness gate would let ``user_id=0`` fall through
    to the *default* identity while the caller believes it addressed a specific
    account — silently producing the cross-account action this parameter exists
    to prevent.
    """
    if not isinstance(user_id, str):
        raise InvalidCamofoxUserId(
            f"Invalid user_id {user_id!r}: expected a string, got "
            f"{type(user_id).__name__}."
        )
    candidate = user_id.strip()
    if not _CALL_USER_ID_RE.fullmatch(candidate):
        raise InvalidCamofoxUserId(
            f"Invalid user_id {user_id!r}. Must be 1-64 characters, start with a "
            "letter or digit, and contain only letters, digits, and '.', '_', "
            "'-', ':', '@'."
        )
    return candidate


def _normalize_call_user_id(user_id: Any) -> Optional[str]:
    """Return the validated per-call identity, or ``None`` when none was given.

    An empty/whitespace string means "not supplied" — models routinely emit
    ``""`` for an optional string parameter.  Any *other* non-string (``0``,
    ``False``, ``12345``) is an error rather than a fallback to the default
    identity: falling back would run the action under a different account than
    the caller named, with nothing in the result to distinguish it.
    """
    if user_id is None:
        return None
    if isinstance(user_id, str) and not user_id.strip():
        return None
    return _validate_call_user_id(user_id)


def _session_cache_key(task_id: Optional[str], user_id: Optional[str]) -> str:
    """Return the ``_sessions`` key for a task, optionally scoped to a user."""
    base = task_id or "default"
    if not user_id:
        return base
    return f"{base}{_USER_SCOPE_SEP}{user_id}"


def _session_keys_for_task(task_id: Optional[str]) -> List[str]:
    """Return every live ``_sessions`` key owned by ``task_id``.

    Includes the bare key plus every per-user key scoped to that task, so task
    teardown reaps multi-account sessions instead of leaking them.  Callers must
    hold ``_sessions_lock``.
    """
    base = task_id or "default"
    prefix = f"{base}{_USER_SCOPE_SEP}"
    return [key for key in _sessions if key == base or key.startswith(prefix)]


def _adopt_existing_tab(session: Dict[str, Any]) -> Dict[str, Any]:
    """Attach process-local state to an already-open managed Camofox tab.

    Some integrations own the visible Camofox tab outside Hermes. Gateway
    restarts can leave this module's in-memory session cache empty even though
    Camofox still has that tab, so rehydrate tab_id before creating a new tab.
    """
    if session.get("tab_id") or not session.get("adopt_existing_tab"):
        return session

    if not get_camofox_url():
        return session

    try:
        tabs = _get("/tabs", params={"userId": session["user_id"]}, timeout=5).get("tabs", [])
    except Exception as exc:
        logger.debug("Camofox tab adoption failed for %s: %s", session.get("user_id"), exc)
        return session

    if not isinstance(tabs, list) or not tabs:
        return session

    session_key = session.get("session_key")
    matching_tabs = [
        tab
        for tab in tabs
        if isinstance(tab, dict) and tab.get("listItemId") == session_key
    ]
    candidates = matching_tabs or [tab for tab in tabs if isinstance(tab, dict)]
    latest = candidates[-1] if candidates else None
    tab_id = latest.get("tabId") if isinstance(latest, dict) else None
    if isinstance(tab_id, str) and tab_id:
        session["tab_id"] = tab_id
        logger.debug("Adopted existing Camofox tab %s for %s", tab_id, session.get("user_id"))

    return session


def _get_session(task_id: Optional[str], user_id: Optional[str] = None) -> Dict[str, Any]:
    """Get or create a camofox session for the given task (and optional user).

    Identity resolution, highest precedence first:

    1. ``user_id`` — an explicit per-call identity.  Lets one Hermes process
       drive several Camofox accounts side by side (issue #77273).
    2. ``CAMOFOX_USER_ID`` / ``browser.camofox.user_id`` — an externally
       managed identity shared with another app.
    3. ``managed_persistence`` — a deterministic userId derived from the Hermes
       profile so the Camofox server can map it to the same persistent browser
       profile across restarts.
    4. A random ephemeral userId.

    Sessions are cached per ``(task_id, user_id)`` so repeated calls with the
    same identity reuse the same tab.

    Raises:
        InvalidCamofoxUserId: when an explicit ``user_id`` fails validation.
            Raised before any HTTP request, so a rejected identity never
            reaches the Camofox server.
    """
    task_id = task_id or "default"
    user_id = _normalize_call_user_id(user_id)
    cache_key = _session_cache_key(task_id, user_id)
    with _sessions_lock:
        if cache_key in _sessions:
            return _adopt_existing_tab(_sessions[cache_key])

        camofox_cfg = _get_camofox_config()
        if user_id:
            # Per-call identity: treated like the externally managed path — the
            # profile belongs to a caller-named account, so destructive cleanup
            # is off and tab adoption follows the same operator opt-in.
            session = {
                "user_id": user_id,
                "tab_id": None,
                "session_key": f"task_{task_id[:16]}",
                "managed": True,
                "adopt_existing_tab": _adopt_existing_tab_enabled(camofox_cfg),
                "destroy_on_close": False,
            }
            _sessions[cache_key] = session
            return _adopt_existing_tab(session)

        identity_override = _camofox_identity_override(task_id, camofox_cfg)
        if identity_override:
            session = {
                "user_id": identity_override["user_id"],
                "tab_id": None,
                "session_key": identity_override["session_key"],
                "managed": True,
                "adopt_existing_tab": _adopt_existing_tab_enabled(camofox_cfg),
                "destroy_on_close": True,
            }
        elif bool(camofox_cfg.get("managed_persistence")):
            identity = get_camofox_identity(task_id)
            session = {
                "user_id": identity["user_id"],
                "tab_id": None,
                "session_key": identity["session_key"],
                "managed": True,
                "adopt_existing_tab": _adopt_existing_tab_enabled(camofox_cfg),
                "destroy_on_close": True,
            }
        else:
            session = {
                "user_id": f"hermes_{uuid.uuid4().hex[:10]}",
                "tab_id": None,
                "session_key": f"task_{task_id[:16]}",
                "managed": False,
                "adopt_existing_tab": False,
                "destroy_on_close": True,
            }
        _sessions[cache_key] = session
        return _adopt_existing_tab(session)


def _ensure_tab(
    task_id: Optional[str],
    url: str = "about:blank",
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Ensure a tab exists for the session, creating one if needed."""
    session = _get_session(task_id, user_id)
    if session["tab_id"]:
        return session
    base = get_camofox_url()
    resp = requests.post(
        f"{base}/tabs",
        json={
            "userId": session["user_id"],
            "listItemId": session["session_key"],
            "url": url,
        },
        timeout=_get_command_timeout(),
        headers=_auth_headers(),
    )
    resp.raise_for_status()
    data = resp.json()
    session["tab_id"] = data.get("tabId")
    return session


def _drop_sessions(
    task_id: Optional[str], user_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Remove and return session info.

    With ``user_id`` set, drops only that identity's session.  Without it, drops
    the task's default session *and* every per-user session scoped to the task
    so teardown doesn't leak multi-account state.
    """
    with _sessions_lock:
        if user_id:
            keys = [_session_cache_key(task_id, user_id)]
        else:
            keys = _session_keys_for_task(task_id)
        return [s for s in (_sessions.pop(key, None) for key in keys) if s]


def _drop_session(task_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Remove the task's default session and return it (back-compat shim)."""
    with _sessions_lock:
        return _sessions.pop(task_id or "default", None)


def list_camofox_sessions() -> List[Dict[str, Any]]:
    """Return a snapshot of the active Camofox sessions.

    One entry per live ``(task_id, user_id)`` pair — useful for skills and
    debugging multi-account runs.  Not exposed as a model tool; the resolved
    ``user_id`` is echoed in every tool result instead.
    """
    with _sessions_lock:
        items = list(_sessions.items())
    sessions = []
    for key, session in items:
        task_id, sep, _ = key.partition(_USER_SCOPE_SEP)
        sessions.append({
            "task_id": task_id,
            "user_id": session.get("user_id", ""),
            "session_key": session.get("session_key", ""),
            "tab_id": session.get("tab_id"),
            "explicit_user_id": bool(sep),
        })
    return sessions


def camofox_soft_cleanup(task_id: Optional[str] = None) -> bool:
    """Release the in-memory session without destroying the server-side context.

    When managed persistence is enabled the browser profile (and its cookies)
    must survive across agent tasks.  This helper drops only the local tracking
    entries and returns ``True``.  When managed persistence is *not* enabled it
    does nothing and returns ``False`` so the caller can fall back to
    :func:`camofox_close`.
    """
    camofox_cfg = _get_camofox_config()
    if bool(camofox_cfg.get("managed_persistence")) or _camofox_identity_override(task_id, camofox_cfg):
        for session in _drop_sessions(task_id):
            # The managed/override profile is left completely untouched — that
            # is the point of soft cleanup.  Per-call identities still get
            # their tab closed, which is non-destructive (the profile and its
            # cookies survive) and keeps multi-account runs from leaking
            # windows on the Camofox server.
            if session.get("destroy_on_close"):
                continue
            try:
                _release_session(session)
            except Exception as exc:
                logger.debug(
                    "Camofox tab release failed for %s: %s", session["user_id"], exc
                )
        logger.debug("Camofox soft cleanup for task %s (managed persistence)", task_id)
        return True
    return False


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post(path: str, body: dict, timeout: Optional[int] = None) -> dict:
    """POST JSON to camofox and return parsed response."""
    if timeout is None:
        timeout = _get_command_timeout()
    url = f"{get_camofox_url()}{path}"
    resp = requests.post(url, json=body, timeout=timeout, headers=_auth_headers())
    resp.raise_for_status()
    return resp.json()


def _get(path: str, params: dict = None, timeout: Optional[int] = None) -> dict:
    """GET from camofox and return parsed response."""
    if timeout is None:
        timeout = _get_command_timeout()
    url = f"{get_camofox_url()}{path}"
    resp = requests.get(url, params=params, timeout=timeout, headers=_auth_headers())
    resp.raise_for_status()
    return resp.json()


def _get_raw(path: str, params: dict = None, timeout: Optional[int] = None) -> requests.Response:
    """GET from camofox and return raw response (for binary data)."""
    if timeout is None:
        timeout = _get_command_timeout()
    url = f"{get_camofox_url()}{path}"
    resp = requests.get(url, params=params, timeout=timeout, headers=_auth_headers())
    resp.raise_for_status()
    return resp


def _delete(path: str, body: dict = None, timeout: Optional[int] = None,
            params: dict = None) -> dict:
    """DELETE to camofox and return parsed response."""
    if timeout is None:
        timeout = _get_command_timeout()
    url = f"{get_camofox_url()}{path}"
    resp = requests.delete(url, json=body, params=params, timeout=timeout,
                           headers=_auth_headers())
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _identity_error(message: str, session: Optional[Dict[str, Any]] = None) -> str:
    """Return a tool error that also discloses the identity it acted as.

    Success payloads echo ``user_id`` so a multi-account agent can confirm it
    did not cross accounts.  Errors must do the same: an agent that only sees
    the echo on success cannot tell "this failed" from "this ran as someone
    else", which is the failure mode the echo exists to surface.
    """
    if session and session.get("user_id"):
        return tool_error(message, success=False, user_id=session["user_id"])
    return tool_error(message, success=False)


def camofox_navigate(url: str, task_id: Optional[str] = None,
                     user_id: Optional[str] = None) -> str:
    """Navigate to a URL via Camofox."""
    session = None
    try:
        browser_url, rewrite_info = _rewrite_loopback_url_for_camofox(url)
        session = _get_session(task_id, user_id)
        if not session["tab_id"]:
            # Create tab with the target URL directly
            session = _ensure_tab(task_id, browser_url, user_id)
            data = {"ok": True, "url": browser_url}
        else:
            # Navigate existing tab — recover from stale tab 404
            try:
                data = _post(
                    f"/tabs/{session['tab_id']}/navigate",
                    {"userId": session["user_id"], "url": browser_url},
                    timeout=60,
                )
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    logger.warning(
                        "Camofox tab %s returned 404 — tab was garbage collected. "
                        "Creating a fresh tab.",
                        session["tab_id"],
                    )
                    session["tab_id"] = None
                    session = _ensure_tab(task_id, browser_url, user_id)
                    data = {"ok": True, "url": browser_url}
                else:
                    raise
        result = {
            "success": True,
            "url": data.get("url", browser_url),
            "title": data.get("title", ""),
            "user_id": session["user_id"],
        }
        if rewrite_info:
            result["requested_url"] = url
            result["url_rewrite"] = rewrite_info
            result["warning"] = (
                "Rewrote loopback URL for Docker-hosted Camofox: "
                f"{rewrite_info['from']} -> {rewrite_info['to']}"
            )
        vnc = get_vnc_url()
        if vnc:
            result["vnc_url"] = vnc
            result["vnc_hint"] = (
                "Browser is visible via VNC. "
                "Share this link with the user so they can watch the browser live."
            )

        # Auto-take a compact snapshot so the model can act immediately
        try:
            snap_data = _get(
                f"/tabs/{session['tab_id']}/snapshot",
                params={"userId": session["user_id"]},
            )
            snapshot_text = snap_data.get("snapshot", "")
            from tools.browser_tool import (
                SNAPSHOT_SUMMARIZE_THRESHOLD,
                _truncate_snapshot,
            )
            if len(snapshot_text) > SNAPSHOT_SUMMARIZE_THRESHOLD:
                snapshot_text = _truncate_snapshot(snapshot_text)
            result["snapshot"] = snapshot_text
            result["element_count"] = snap_data.get("refsCount", 0)
        except Exception:
            pass  # Navigation succeeded; snapshot is a bonus

        return json.dumps(result)
    except requests.HTTPError as e:
        return tool_error(f"Navigation failed: {e}", success=False)
    except requests.ConnectionError:
        return json.dumps({
            "success": False,
            "error": f"Cannot connect to Camofox at {get_camofox_url()}. "
                     "Is the server running? Start with: npm start (in camofox-browser dir) "
                     "or: docker run -p 9377:9377 -e CAMOFOX_PORT=9377 jo-inc/camofox-browser",
        })
    except Exception as e:
        return _identity_error(str(e), session)


def _camofox_private_page_block(session: Dict[str, Any], task_id: Optional[str], action: str) -> Optional[str]:
    """Return a blocked payload when the current Camofox page is private/internal.

    Mirrors the eval-path guard added for ``_camofox_eval`` (browser_tool.py):
    Camofox snapshot / vision / image-extraction all read current page state, so
    on a non-local backend they can leak the content of an intranet/metadata
    page the terminal itself can't reach.  The gate matches ``browser_snapshot``
    / ``browser_vision`` — only active when the SSRF guard applies (non-local
    backend, not a local sidecar, ``allow_private_urls`` unset).  Fail-open on
    probe failure, matching the sibling guards.

    Imports are deferred to call time because ``browser_tool`` imports this
    module; importing it at module load would create a circular import.
    """
    from tools.browser_tool import (
        _camofox_current_page_private_url,
        _eval_ssrf_guard_active,
    )

    if not _eval_ssrf_guard_active(task_id or "default"):
        return None
    blocked_url = _camofox_current_page_private_url(session["tab_id"], session["user_id"])
    if not blocked_url:
        return None
    return json.dumps({
        "success": False,
        "error": (
            "Blocked: page URL targets a private or internal address "
            f"({blocked_url}). Refusing to {action} on this page in this "
            "browser mode."
        ),
        "user_id": session["user_id"],
    }, ensure_ascii=False)


def camofox_snapshot(full: bool = False, task_id: Optional[str] = None,
                     user_task: Optional[str] = None,
                     user_id: Optional[str] = None) -> str:
    """Get accessibility tree snapshot from Camofox."""
    session = None
    try:
        session = _get_session(task_id, user_id)
        if not session["tab_id"]:
            return _identity_error("No browser session. Call browser_navigate first.", session)

        blocked = _camofox_private_page_block(session, task_id, "read a page snapshot")
        if blocked:
            return blocked

        data = _get(
            f"/tabs/{session['tab_id']}/snapshot",
            params={"userId": session["user_id"]},
        )

        snapshot = data.get("snapshot", "")
        refs_count = data.get("refsCount", 0)

        # Apply same summarization logic as the main browser tool
        from tools.browser_tool import (
            SNAPSHOT_SUMMARIZE_THRESHOLD,
            _extract_relevant_content,
            _truncate_snapshot,
        )

        if len(snapshot) > SNAPSHOT_SUMMARIZE_THRESHOLD:
            if user_task:
                snapshot = _extract_relevant_content(snapshot, user_task)
            else:
                snapshot = _truncate_snapshot(snapshot)

        return json.dumps({
            "success": True,
            "snapshot": snapshot,
            "element_count": refs_count,
            "user_id": session["user_id"],
        })
    except Exception as e:
        return _identity_error(str(e), session)


def camofox_click(ref: str, task_id: Optional[str] = None,
                  user_id: Optional[str] = None) -> str:
    """Click an element by ref via Camofox."""
    session = None
    try:
        session = _get_session(task_id, user_id)
        if not session["tab_id"]:
            return _identity_error("No browser session. Call browser_navigate first.", session)

        blocked = _camofox_private_page_block(session, task_id, "click")
        if blocked:
            return blocked

        # Strip @ prefix if present (our tool convention)
        clean_ref = ref.lstrip("@")

        data = _post(
            f"/tabs/{session['tab_id']}/click",
            {"userId": session["user_id"], "ref": clean_ref},
        )
        return json.dumps({
            "success": True,
            "clicked": clean_ref,
            "url": data.get("url", ""),
            "user_id": session["user_id"],
        })
    except Exception as e:
        return _identity_error(str(e), session)


def camofox_type(ref: str, text: str, task_id: Optional[str] = None,
                 user_id: Optional[str] = None) -> str:
    """Type text into an element by ref via Camofox."""
    session = None
    try:
        session = _get_session(task_id, user_id)
        if not session["tab_id"]:
            return _identity_error("No browser session. Call browser_navigate first.", session)

        blocked = _camofox_private_page_block(session, task_id, "type")
        if blocked:
            return blocked

        clean_ref = ref.lstrip("@")

        _post(
            f"/tabs/{session['tab_id']}/type",
            {"userId": session["user_id"], "ref": clean_ref, "text": text},
        )
        from agent.display import (
            redact_browser_typed_text_for_display,
            redact_tool_args_for_display,
        )

        display_text = (redact_tool_args_for_display("browser_type", {"text": text}) or {})["text"]

        response = {
            "success": True,
            # Match browser_tool.browser_type: run typed text through the
            # secret-pattern redactor so API keys / tokens don't leak into
            # tool progress or chat history.  The raw text is still typed into
            # the page; only the returned display value is redacted.
            "typed": display_text,
            "element": clean_ref,
            "user_id": session["user_id"],
        }
        response = redact_browser_typed_text_for_display(response, text)
        return json.dumps(response)
    except Exception as e:
        from agent.display import redact_browser_typed_text_for_display

        return tool_error(redact_browser_typed_text_for_display(str(e), text), success=False)


def camofox_scroll(direction: str, task_id: Optional[str] = None,
                   user_id: Optional[str] = None) -> str:
    """Scroll the page via Camofox."""
    session = None
    try:
        session = _get_session(task_id, user_id)
        if not session["tab_id"]:
            return _identity_error("No browser session. Call browser_navigate first.", session)

        _post(
            f"/tabs/{session['tab_id']}/scroll",
            {"userId": session["user_id"], "direction": direction},
        )
        return json.dumps({
            "success": True,
            "scrolled": direction,
            "user_id": session["user_id"],
        })
    except Exception as e:
        return _identity_error(str(e), session)


def camofox_back(task_id: Optional[str] = None,
                 user_id: Optional[str] = None) -> str:
    """Navigate back via Camofox."""
    session = None
    try:
        session = _get_session(task_id, user_id)
        if not session["tab_id"]:
            return _identity_error("No browser session. Call browser_navigate first.", session)

        data = _post(
            f"/tabs/{session['tab_id']}/back",
            {"userId": session["user_id"]},
        )
        return json.dumps({
            "success": True,
            "url": data.get("url", ""),
            "user_id": session["user_id"],
        })
    except Exception as e:
        return _identity_error(str(e), session)


def camofox_press(key: str, task_id: Optional[str] = None,
                  user_id: Optional[str] = None) -> str:
    """Press a keyboard key via Camofox."""
    session = None
    try:
        session = _get_session(task_id, user_id)
        if not session["tab_id"]:
            return _identity_error("No browser session. Call browser_navigate first.", session)

        blocked = _camofox_private_page_block(session, task_id, "press")
        if blocked:
            return blocked

        _post(
            f"/tabs/{session['tab_id']}/press",
            {"userId": session["user_id"], "key": key},
        )
        return json.dumps({
            "success": True,
            "pressed": key,
            "user_id": session["user_id"],
        })
    except Exception as e:
        return _identity_error(str(e), session)


def _release_session(session: Dict[str, Any]) -> None:
    """Release one session's server-side resources.

    Two distinct operations on the Camofox API, and the difference matters:

    * ``DELETE /sessions/<userId>`` destroys the whole profile — "Closes all
      tabs and cleans up state for the given userId".  Only for identities
      Hermes owns (``destroy_on_close``).
    * ``DELETE /tabs/<tabId>?userId=`` just closes the window and leaves the
      profile, cookies, and logins intact.  This is what a caller-supplied
      identity gets: without it every distinct ``user_id`` would leak an open
      tab that nothing ever reclaims (the server's ``POST /pressure/cleanup``
      is caller-triggered, not an automatic reaper).
    """
    if session.get("destroy_on_close"):
        _delete(f"/sessions/{session['user_id']}")
        return
    tab_id = session.get("tab_id")
    if tab_id:
        _delete(f"/tabs/{tab_id}", params={"userId": session["user_id"]})


def camofox_close(task_id: Optional[str] = None,
                  user_id: Optional[str] = None) -> str:
    """Close the browser session(s) via Camofox.

    With ``user_id`` set, closes only that identity's session.  Without it,
    closes the task's default session *and* every per-user session scoped to
    the task.

    Identities supplied per call are caller-owned, so their profile survives —
    only their tab is closed.  See :func:`_release_session`.
    """
    try:
        user_id = _normalize_call_user_id(user_id)
    except InvalidCamofoxUserId as exc:
        return tool_error(str(exc), success=False)

    sessions = _drop_sessions(task_id, user_id)
    if not sessions:
        return json.dumps({"success": True, "closed": True})

    # Release each session independently. A single failing DELETE must not
    # abort the loop: _drop_sessions has already removed every entry from the
    # local map, so a sibling skipped here could never be reclaimed by anyone.
    warnings = []
    for session in sessions:
        try:
            _release_session(session)
        except Exception as exc:
            warnings.append(f"{session['user_id']}: {exc}")
            logger.debug("Camofox release failed for %s: %s", session["user_id"], exc)

    result = {
        "success": True,
        "closed": True,
        "user_ids": [s["user_id"] for s in sessions],
    }
    if warnings:
        result["warning"] = "; ".join(warnings)
    return json.dumps(result)


def camofox_get_images(task_id: Optional[str] = None,
                       user_id: Optional[str] = None) -> str:
    """Get images on the current page via Camofox.

    Extracts image information from the accessibility tree snapshot,
    since Camofox does not expose a dedicated /images endpoint.
    """
    session = None
    try:
        session = _get_session(task_id, user_id)
        if not session["tab_id"]:
            return _identity_error("No browser session. Call browser_navigate first.", session)

        blocked = _camofox_private_page_block(session, task_id, "extract page images")
        if blocked:
            return blocked

        data = _get(
            f"/tabs/{session['tab_id']}/snapshot",
            params={"userId": session["user_id"]},
        )
        snapshot = data.get("snapshot", "")

        # Parse img elements from the accessibility tree.
        # Format: img "alt text" or img "alt text" [eN]
        # URLs appear on /url: lines following img entries
        images = []
        lines = snapshot.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("- img ", "img ")):
                alt_match = re.search(r'img\s+"([^"]*)"', stripped)
                alt = alt_match.group(1) if alt_match else ""
                # Look for URL on the next line
                src = ""
                if i + 1 < len(lines):
                    url_match = re.search(r'/url:\s*(\S+)', lines[i + 1].strip())
                    if url_match:
                        src = url_match.group(1)
                if alt or src:
                    images.append({"src": src, "alt": alt})

        return json.dumps({
            "success": True,
            "images": images,
            "count": len(images),
            "user_id": session["user_id"],
        })
    except Exception as e:
        return _identity_error(str(e), session)


def camofox_vision(question: str, annotate: bool = False,
                   task_id: Optional[str] = None,
                   user_id: Optional[str] = None) -> str:
    """Take a screenshot and analyze it with vision AI via Camofox."""
    session = None
    try:
        session = _get_session(task_id, user_id)
        if not session["tab_id"]:
            return _identity_error("No browser session. Call browser_navigate first.", session)

        blocked = _camofox_private_page_block(session, task_id, "capture a screenshot")
        if blocked:
            return blocked

        # Get screenshot as binary PNG
        resp = _get_raw(
            f"/tabs/{session['tab_id']}/screenshot",
            params={"userId": session["user_id"]},
        )

        # Save screenshot to cache
        from hermes_constants import get_hermes_home
        screenshots_dir = get_hermes_home() / "browser_screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = str(screenshots_dir / f"browser_screenshot_{uuid.uuid4().hex[:8]}.png")

        with open(screenshot_path, "wb") as f:
            f.write(resp.content)

        # Encode for vision LLM
        img_b64 = base64.b64encode(resp.content).decode("utf-8")

        # Also get annotated snapshot if requested
        annotation_context = ""
        if annotate:
            try:
                snap_data = _get(
                    f"/tabs/{session['tab_id']}/snapshot",
                    params={"userId": session["user_id"]},
                )
                annotation_context = f"\n\nAccessibility tree (element refs for interaction):\n{snap_data.get('snapshot', '')[:3000]}"
            except Exception:
                pass

        # Redact secrets from annotation context before sending to vision LLM.
        # The screenshot image itself cannot be redacted, but at least the
        # text-based accessibility tree snippet won't leak secret values.
        from agent.redact import redact_sensitive_text
        annotation_context = redact_sensitive_text(annotation_context)

        # Send to vision LLM
        from agent.auxiliary_client import call_llm

        vision_prompt = (
            f"Analyze this browser screenshot and answer: {question}"
            f"{annotation_context}"
        )

        try:
            _cfg = load_config()
            _vision_cfg = cfg_get(_cfg, "auxiliary", "vision", default={})
            _vision_timeout = float(_vision_cfg.get("timeout", 120))
            _vision_temperature = float(_vision_cfg.get("temperature", 0.1))
        except Exception:
            _vision_timeout = 120.0
            _vision_temperature = 0.1

        response = call_llm(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": vision_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}",
                        },
                    },
                ],
            }],
            task="vision",
            temperature=_vision_temperature,
            timeout=_vision_timeout,
        )
        analysis = (response.choices[0].message.content or "").strip() if response.choices else ""

        # Redact secrets the vision LLM may have read from the screenshot.
        from agent.redact import redact_sensitive_text
        analysis = redact_sensitive_text(analysis)

        return json.dumps({
            "success": True,
            "analysis": analysis,
            "screenshot_path": screenshot_path,
            "user_id": session["user_id"],
        })
    except Exception as e:
        return _identity_error(str(e), session)


def camofox_console(clear: bool = False, task_id: Optional[str] = None,
                    user_id: Optional[str] = None) -> str:
    """Get console output — limited support in Camofox.

    Camofox does not expose browser console logs via its REST API.
    Returns an empty result with a note.

    ``user_id`` is validated for signature parity with the other Camofox tools
    but deliberately *not* echoed: this path never opens a session, so echoing
    it back would rubber-stamp an identity that was never resolved against the
    server — the opposite of the confirmation the echo is meant to provide.
    """
    try:
        _normalize_call_user_id(user_id)
    except InvalidCamofoxUserId as exc:
        return tool_error(str(exc), success=False)
    return json.dumps({
        "success": True,
        "console_messages": [],
        "js_errors": [],
        "total_messages": 0,
        "total_errors": 0,
        "note": "Console log capture is not available with the Camofox backend. "
                "Use browser_snapshot or browser_vision to inspect page state.",
    })



