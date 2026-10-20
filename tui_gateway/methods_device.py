"""``device.capabilities``: what the client attached to a session can serve on request.

Bodies are rebound onto server.py's globals (``method_ctx.bind_module``) and reference them bare,
so ``_sessions`` / ``_session_live_transports`` / ``_ok`` are server.py's, not this module's.
"""

from __future__ import annotations

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


@method("device.capabilities")
def _(rid, params: dict) -> dict:
    """The device capabilities the session's live clients advertised in ``client.capabilities``.

    An agent calls this before offering a phone capability: an empty list means every attached
    client is a desktop window or the TUI, so ``device.request`` would be refused at once anyway.
    A capability is reported when ANY attached client advertised it — two windows on one session
    each contribute what they have.
    """
    from tui_gateway import server_requests
    from tui_gateway.ws import WSTransport

    sid = str(params.get("session_id") or "").strip()
    union: set[str] = set()
    for peer in _session_live_transports(_sessions.get(sid)):
        if isinstance(peer, WSTransport):
            union |= server_requests.device_capabilities(peer)
    return _ok(rid, {"session_id": sid, "capabilities": sorted(union)})


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
