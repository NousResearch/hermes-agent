"""Legacy-handler fallback for authority-backed WebSocket connections.

Session verbs live on the session authority; methods it answers ``-32601`` for may still have a
legacy sidecar handler (pet, wake word, connectors, config). That fallback must not widen what
the connection's ticket grants (R2-M3):

* only connections holding the authority's full interactive grant reach legacy dispatch — a
  worker-adoption ticket (``{'worker:adopt'}``) or a capability-less connection keeps the
  authority's ``-32601``;
* a ticket bound to a secondary profile keeps that profile's scope: sessionless
  ``@_profile_scoped`` handlers read :func:`connection_profile_home` instead of defaulting to the
  launch profile.
"""

from __future__ import annotations

import contextlib
import contextvars
from pathlib import Path
from typing import Any

_connection_profile_home: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "hermes_ws_connection_profile_home", default=None)


# The authority owns every session, its transcript and its admissions: a legacy handler in these
# namespaces would write the same state.db behind its receipts and revision fence (a legacy
# ``session.delete`` dropped a row the authority still held live; ``session.branch_stored`` built a
# second, legacy-owned runtime; ``session.workspace.move`` rewrote only the row, never the frozen
# policy: Desktop sends ``session.mutate operation=workspace``). Kept: store-wide reads and
# foreign-history import (new rows only).
_SESSION_NAMESPACES = ("session.", "prompt.", "message.")
_SESSION_FALLBACK_ALLOWED = frozenset({
    "session.most_recent", "session.active_list", "session.events.stats",
    "session.foreign.list", "session.foreign.preview", "session.foreign.import",
})
# Sidecar verbs whose handler acts on PROCESS-global state. In a TUI's own sidecar that state was
# that TUI's alone; on the shared owner it is every chat's, Desktop window's and served profile's:
# ``process.stop`` = ``kill_all`` over the whole registry (persisted jobs included),
# ``delegation.pause`` = the global spawn gate, ``reload.mcp`` = tear down and rediscover every MCP
# server, ``reload.env`` = rewrite ``os.environ`` from the launch profile's ``.env``, ``agents.list``
# = every chat's process commands. The owner serves the session-scoped ``process.stop`` itself
# (``gateway/session_ancillary.py``), and ``reload.env`` / ``delegation.pause`` with owner-side rules
# (``gateway/session_tui_ops.py``); listing them here keeps the sidecar handler unreachable.
# ``process.kill`` stays: its handler is already session-scoped.
_PROCESS_GLOBAL = frozenset({"process.stop", "delegation.pause", "reload.mcp", "reload.env", "agents.list"})
# ``browser.manage`` is NOT here: connect/disconnect run in the requesting profile's scope (the
# ticket's home, or the session's ``profile_home`` via ``_profile_scoped``) and the live endpoint, the
# reap and the CDP supervisors are keyed by that profile home (CLI-9, #134889), so a multiplexed
# owner serves them without reaching another profile's browser.


def legacy_fallback_allowed(actor: Any, method: str) -> bool:
    """True when the connection's grant covers the interactive purpose (the authority's own
    capability map; never a second list that could drift) and *method* is neither a session verb
    the authority owns nor a process-global sidecar verb."""
    from gateway.runtime_bootstrap import _PURPOSE_CAPABILITIES
    if method in _PROCESS_GLOBAL:
        return False
    if method.startswith(_SESSION_NAMESPACES) and method not in _SESSION_FALLBACK_ALLOWED:
        return False
    return _PURPOSE_CAPABILITIES["interactive"] <= frozenset(getattr(actor, "capabilities", ()) or ())


def connection_profile_home() -> str | None:
    """The non-launch profile home the current legacy fallback request's ticket is bound to."""
    return _connection_profile_home.get()


def _foreign_profile_home(server: Any, profile_id: Any) -> str | None:
    """The ticket's non-launch home, or None for the launch profile. Raises when the ticket names a
    home that stopped being a directory: that is a refusal, never the launch profile (N86)."""
    home = Path(str(profile_id or ""))
    if not home.is_absolute():
        return None
    if not home.is_dir():
        raise FileNotFoundError(str(home))
    if home.resolve() == Path(server._launch_home()).resolve():
        return None
    return str(home)


def _unavailable(req: dict) -> dict:
    return {"jsonrpc": "2.0", "id": req.get("id"), "error": {"code": 4064, "message": "profile unavailable"}}


def _explicit_profile_request(server: Any, req: dict) -> tuple[dict, str | None] | None:
    """The home an explicit ``profile`` param names (Desktop's host-scoped routing): the request
    plus that home ("" = launch), an error frame and None when the name does not resolve here, or
    None when the request names no profile."""
    params = req.get("params")
    if not isinstance(params, dict) or not (explicit := str(params.get("profile") or "").strip()):
        return None
    try:
        named = server._profile_home(explicit)
    except (OSError, ValueError):  # ProfileUnavailableError is a ValueError
        return _unavailable(req), None
    return req, str(named) if named else ""


def _ticket_profile_request(server: Any, req: dict, home: str) -> tuple[dict, str | None]:
    """Bind a non-launch ticket's profile into a legacy verb (N86). The verb runs in the ticket's
    home, or in the one an explicit ``profile`` names (Desktop's host-scoped routing), so handlers
    that ignore ``profile`` (billing, usage) resolve through that runtime scope instead of the
    launch home. A method whose contract declares ``profile`` and got none is handed the ticket's
    own profile name (refused when that name does not resolve to the ticket's home). Returns the
    request plus the home to scope ("" = launch), or an error frame and None."""
    from hermes_constants import profile_name_for_home
    from tui_gateway.contracts.registry import METHODS
    params = req.get("params") if isinstance(req.get("params"), dict) else {}
    refused = _unavailable(req)
    if (explicit := _explicit_profile_request(server, req)) is not None:
        return explicit
    contract = METHODS.get(req.get("method"))
    if contract is None or "profile" not in contract.params.model_fields:
        return req, home
    name = profile_name_for_home(home)
    try:
        resolved = server._profile_home(name) if name else None
    except (OSError, ValueError):
        resolved = None
    if resolved is None or Path(resolved).resolve() != Path(home).resolve():
        return refused, None
    return dict(req, params={**params, "profile": name}), home


def _own_profile_read(server: Any, req: dict, home: str) -> dict:
    """A store-wide session read for a ticket bound to the non-launch *home*. Those handlers default
    to the LAUNCH profile's store and ``session.active_list`` lists every profile's live sessions, so
    the ticket's own profile is named (or the read refused), never the launch profile's served."""
    from hermes_constants import profile_name_for_home
    params = req.get("params") if isinstance(req.get("params"), dict) else {}
    name = params.get("profile") or profile_name_for_home(home)
    try:
        resolved = server._profile_home(name) if name else None
    except (OSError, ValueError):
        resolved = None
    if resolved is None or Path(resolved).resolve() != Path(home).resolve():
        return {"jsonrpc": "2.0", "id": req.get("id"), "error": {"code": 4064, "message": "profile unavailable"}}
    if req.get("method") != "session.active_list":
        return dict(req, params={**params, "profile": name})
    with server._sessions_lock:
        live = list(server._sessions.items())
    current = str(params.get("current_session_id") or "")
    rows = [server._session_live_item(sid, session, current) for sid, session in live
            if not session.get("_finalized") and Path(session.get("profile_home") or server._launch_home()).resolve()
            == Path(home).resolve()]
    return {"jsonrpc": "2.0", "id": req.get("id"), "result": {"sessions": rows}}


def dispatch_legacy(server: Any, req: dict, transport: Any, actor: Any) -> dict | None:
    """Run ``server.dispatch`` with the ticket's profile home bound (call from a worker thread;
    the pool path copies this context, so long handlers see it too)."""
    try:
        home = _foreign_profile_home(server, getattr(actor, "profile_id", None))
    except OSError:
        return {"jsonrpc": "2.0", "id": req.get("id"), "error": {"code": 4064, "message": "profile unavailable"}}
    scope = contextlib.nullcontext()
    if home is not None and req.get("method") in _SESSION_FALLBACK_ALLOWED:
        scoped = _own_profile_read(server, req, home)
        if "result" in scoped or "error" in scoped:
            return scoped
        req = scoped
    elif home is not None:
        req, target = _ticket_profile_request(server, req, home)
        if target is None:
            return req
        # "" = an explicit ``profile`` naming the launch home: the socket runs in the ticket's
        # scope (``chat_ws.gateway_ws``), so bind the launch scope (``None`` selects it, process env
        # credentials included) or a handler that ignores ``profile`` would use the secondary's.
        scope = server._session_profile_runtime_scope({"profile_home": target or None})
    elif (req.get("method") not in _SESSION_FALLBACK_ALLOWED
          and (explicit := _explicit_profile_request(server, req)) is not None):
        # A launch-profile socket runs in the launch scope (``chat_ws.gateway_ws``), so a verb that
        # names another profile is bound to it here: a handler that ignores ``profile`` would
        # otherwise read the launch profile's secrets instead of refusing an unscoped read.
        req, target = explicit
        if target is None:
            return req
        if target:
            scope = server._session_profile_runtime_scope({"profile_home": target})
    token = _connection_profile_home.set(home)
    try:
        with scope:
            return server.dispatch(req, transport)
    finally:
        _connection_profile_home.reset(token)
