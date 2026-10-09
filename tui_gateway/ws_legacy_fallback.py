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


def legacy_fallback_allowed(actor: Any) -> bool:
    """True when the connection's grant covers the interactive purpose (the authority's own
    capability map; never a second list that could drift)."""
    from gateway.runtime_bootstrap import _PURPOSE_CAPABILITIES
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
    token = _connection_profile_home.set(_foreign_profile_home(server, getattr(actor, "profile_id", None)))
    try:
        return server.dispatch(req, transport)
    finally:
        _connection_profile_home.reset(token)
