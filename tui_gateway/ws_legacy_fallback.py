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

import contextvars
from pathlib import Path
from typing import Any

_connection_profile_home: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "hermes_ws_connection_profile_home", default=None)


# The authority owns every session, its transcript and its admissions: a legacy handler in these
# namespaces would write the same state.db behind its receipts and revision fence (a legacy
# ``session.delete`` dropped a row the authority still held live; ``session.branch_stored`` built a
# second, legacy-owned runtime). Kept: store-wide reads, foreign-history import (new rows only), and
# Desktop's project move (``session.workspace.move``), which rewrites only the row's cwd/git grouping
# and has no authority verb yet.
_SESSION_NAMESPACES = ("session.", "prompt.", "message.")
_SESSION_FALLBACK_ALLOWED = frozenset({
    "session.most_recent", "session.active_list", "session.events.stats",
    "session.foreign.list", "session.foreign.preview", "session.foreign.import",
    "session.workspace.move",
})
# Sidecar verbs whose handler acts on PROCESS-global state. In a TUI's own sidecar that state was
# that TUI's alone; on the shared owner it is every chat's, Desktop window's and served profile's:
# ``process.stop`` = ``kill_all`` over the whole registry (persisted jobs included),
# ``delegation.pause`` = the global spawn gate, ``reload.mcp`` = tear down and rediscover every MCP
# server, ``reload.env`` = rewrite ``os.environ`` from the launch profile's ``.env``, ``agents.list``
# = every chat's process commands. The owner serves the session-scoped ``process.stop`` itself
# (``gateway/session_ancillary.py``); ``process.kill`` stays: its handler is already session-scoped.
_PROCESS_GLOBAL = frozenset({"process.stop", "delegation.pause", "reload.mcp", "reload.env", "agents.list"})
# Process-global ACTIONS of a verb whose other actions are fine on the shared owner (the gate above is
# method-level, and refusing the whole verb turns its safe actions into the client's -32601 version-skew
# notice). ``browser.manage`` connect/disconnect reap every task's browser and rewrite the process-wide
# ``BROWSER_CDP_URL`` that every chat's and served profile's browser tools read first (dokterdok N25);
# ``status`` and ``use`` (the profile's ``browser.backend``) stay. A missing action means ``status``.
_PROCESS_GLOBAL_ACTIONS = {"browser.manage": ({"connect", "disconnect"}, (
    "/browser connect and /browser disconnect would switch the browser for every chat on this shared "
    "gateway. Set browser.cdp_url in config.yaml instead, or use the classic CLI (hermes --cli)."))}


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
    home = Path(str(profile_id or ""))
    if not home.is_absolute() or not home.is_dir():
        return None
    if home.resolve() == Path(server._launch_home()).resolve():
        return None
    return str(home)


def dispatch_legacy(server: Any, req: dict, transport: Any, actor: Any) -> dict | None:
    """Run ``server.dispatch`` with the ticket's profile home bound (call from a worker thread;
    the pool path copies this context, so long handlers see it too)."""
    actions, refusal = _PROCESS_GLOBAL_ACTIONS.get(req.get("method"), ((), ""))
    if (req.get("params") or {}).get("action", "status") in actions:
        return {"jsonrpc": "2.0", "id": req.get("id"), "error": {"code": 4030, "message": refusal}}
    token = _connection_profile_home.set(_foreign_profile_home(server, getattr(actor, "profile_id", None)))
    try:
        return server.dispatch(req, transport)
    finally:
        _connection_profile_home.reset(token)
