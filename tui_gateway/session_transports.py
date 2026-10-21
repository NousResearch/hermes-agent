"""Additive transport membership for shared live sessions."""
from __future__ import annotations

import threading
import time

from tui_gateway.method_ctx import bind_module

# Leaf lock: callers may hold sessions/history locks, never acquire them here.
_session_transport_lock = threading.RLock()


def _transport_is_live_peer(transport) -> bool:
    """Exclude the process fallback sink, parked sentinel, and closed peers."""
    return (transport is not None
            and transport is not _detached_ws_transport
            and transport is not _stdio_transport
            and not isinstance(transport, (_DropTransport, StdioTransport))
            and not _transport_is_dead(transport))


def _session_transport_contains(session: dict | None, transport) -> bool:
    if not session or transport is None or _transport_is_dead(transport):
        return False
    existing = session.get("transport")
    return existing is transport or (
        isinstance(existing, FanoutTransport) and existing.contains(transport))


def _session_live_transports(session: dict | None) -> list:
    existing = (session or {}).get("transport")
    peers = existing.transports() if isinstance(existing, FanoutTransport) else [existing]
    return [peer for peer in peers if _transport_is_live_peer(peer)]


def _session_has_live_transport(session: dict | None, *, excluding=None) -> bool:
    return any(peer is not excluding for peer in _session_live_transports(session))


def _transport_auth_user_id(transport) -> str | None:
    """``<provider>:<user id>`` the WS-upgrade credential authenticated for ``transport``, or None for the legacy
    token, stdio and the PTY child's server-internal credential. The prefix keeps a basic-auth ``alice`` and an
    OIDC ``alice`` apart."""
    identity = getattr(transport, "auth_identity", None)
    if _methods_browser_control._is_authenticated_identity(identity):
        return f"{str(identity['provider']).strip()}:{str(identity['user_id']).strip()}"
    return None


def _session_auth_user_id(session: dict | None) -> str | None:
    """The login ``session`` was created under, stamped on the record as ``auth_user_id``. A second window turns
    the transport slot into a FanoutTransport, which names no login, so only a record without the slot reads
    its transport."""
    session = session or {}
    if "auth_user_id" in session:
        return session["auth_user_id"]
    return _transport_auth_user_id(session.get("transport"))


def _session_client_answers_requests(sid: str) -> bool:
    """Whether a server→client request for *sid* can be answered: False only when every live WebSocket
    client attached to the session is a build that never sent ``client.capabilities`` (Desktop / dashboard
    update separately from this backend; the stdio TUI ships with it). No attached client is still True — the
    question waits in ``open_requests`` for the reconnect replay. Compute-host relays and other non-client
    transports never count."""
    from tui_gateway import server_requests
    from tui_gateway.ws import WSTransport
    clients = [peer for peer in _session_live_transports(_sessions.get(sid)) if isinstance(peer, WSTransport)]
    return not clients or any(server_requests.answers_requests(peer) for peer in clients)


def _session_answering_clients(sid: str) -> list:
    """The live WebSocket clients attached to *sid* that advertised answering server→client requests:
    the windows whose unanimous "not shown here" settles a window-owned request (server_requests.py)."""
    from tui_gateway import server_requests
    from tui_gateway.ws import WSTransport
    return [peer for peer in _session_live_transports(_sessions.get(sid))
            if isinstance(peer, WSTransport) and server_requests.answers_requests(peer)]


# A session whose device client detached — a phone that dropped its socket, a page reload — stays
# answerable for ``device.client_seen_seconds``: the reconnect replay must still be able to answer a
# request the agent just made. The memory lives ON the session dict, so it dies with the session and
# a session that never had a phone is refused at once instead of parking the agent for the deadline.
_DEFAULT_CLIENT_SEEN_SECONDS = 120.0


def _device_client_seen_seconds() -> float:
    """``device.client_seen_seconds``: how long a detached device client may still be waited for,
    clamped to 0..3600. 0 disables the memory, so a detached session is always refused at once."""
    try:
        from hermes_cli.config import cfg_get, load_config
        seconds = float(cfg_get(load_config(), "device", "client_seen_seconds",
                                default=_DEFAULT_CLIENT_SEEN_SECONDS))
    except Exception:
        return _DEFAULT_CLIENT_SEEN_SECONDS
    return min(max(seconds, 0.0), 3600.0)


def _remember_device_client(session: dict | None, capabilities: frozenset[str]) -> None:
    """Remember what a session's device client advertised, so a reconnect can still answer."""
    if session is not None:
        session["_device_seen"] = (capabilities, time.monotonic())


def _remembered_device_capabilities(session: dict | None) -> frozenset[str]:
    """The capabilities a recently detached device client advertised for *session*; empty once expired."""
    seen = (session or {}).get("_device_seen")
    if not seen:
        return frozenset()
    capabilities, at = seen
    return capabilities if time.monotonic() - at <= _device_client_seen_seconds() else frozenset()


def _session_device_capabilities(sid: str) -> frozenset[str]:
    """Union of the device capabilities the live WebSocket clients attached to *sid* advertised.

    The honest "what can this session's client serve" answer: two windows contribute what each has,
    a desktop window contributes nothing (``device.capabilities`` RPC, ``mobile_device`` toolset
    fold-in).
    """
    from tui_gateway import server_requests
    from tui_gateway.ws import WSTransport

    union: frozenset[str] = frozenset()
    for peer in _session_live_transports(_sessions.get(sid)):
        if isinstance(peer, WSTransport):
            union |= server_requests.device_capabilities(peer)
    return union


def _session_device_capable(sid: str, capability: str) -> bool:
    """Whether a live WebSocket client attached to *sid* advertised *capability*.

    A session with no attached client is refused unless one advertised a device capability within
    ``device.client_seen_seconds`` — a bare stdio/TUI session waits for no phone. Just detached
    counts, so a reconnect replay still answers the request the agent just made.
    """
    session = _sessions.get(sid)
    advertised = _session_device_capabilities(sid)
    if advertised:
        _remember_device_client(session, advertised)
        return capability in advertised
    if _session_live_transports(session):
        return False  # a client is attached and advertised no device: refuse now, do not wait
    return capability in _remembered_device_capabilities(session)


def _warn_foreign_login(session: dict, transport) -> None:
    """Ownership is not enforced; a second login sharing a session is only logged, and the agent keeps the
    creator's user id."""
    attaching = _transport_auth_user_id(transport)
    if attaching is None:
        return
    creator = _session_auth_user_id(session)
    if creator != attaching:
        logger.warning("Session %s keeps the user id %s it was created with; a client logged in as %s attached",
                       session.get("session_key"), creator or "(none)", attaching)


def _attach_session_transport(session: dict | None, transport) -> bool:
    """Add live peers; flatten captured queued fanouts without nesting authority."""
    if not session or transport is None:
        return False
    with _session_transport_lock:
        if isinstance(transport, FanoutTransport):
            # Snapshot and attach share detach's lock: a queued fanout cannot
            # resurrect a still-open peer removed during flattening.
            attached = [_attach_session_transport(session, peer) for peer in transport.transports()]
            return any(attached)
        existing = session.get("transport")
        if _transport_is_dead(transport):
            if isinstance(existing, FanoutTransport):
                existing.detach(transport)
            return False
        if not _transport_is_live_peer(transport):
            if _session_has_live_transport(session):
                return False
            session["transport"] = transport
            return True
        if existing is transport:
            return True
        if isinstance(existing, FanoutTransport):
            if not existing.contains(transport):
                _warn_foreign_login(session, transport)
            existing.attach(transport)
            return existing.contains(transport)
        _warn_foreign_login(session, transport)
        if _transport_is_live_peer(existing):
            session["transport"] = FanoutTransport(existing, transport)
        else:
            session["transport"] = transport
        return True


def _detach_session_transport(session: dict | None, transport) -> bool:
    """Remove membership; return whether another live client prevents parking."""
    if not session:
        return False
    with _session_transport_lock:
        (session.get("viewers") or {}).pop(transport, None)
        existing = session.get("transport")
        if isinstance(existing, FanoutTransport):
            existing.detach(transport)
            viewers = session.get("viewers") or {}
            for viewer in list(viewers):
                if not existing.contains(viewer) or _transport_is_dead(viewer):
                    viewers.pop(viewer, None)
            # Keep the surviving mailbox: collapsing to a bare transport lets
            # new frames overtake its already queued terminal/control events.
        return _session_has_live_transport(session, excluding=transport)


def _detach_transport_from_sessions(transport) -> list[tuple[str, dict]]:
    """Remove even closed/pruned peers' viewer entries; return clientless slots."""
    with _sessions_lock:
        attached = []
        for sid, session in _sessions.items():
            existing = session.get("transport")
            if (existing is transport
                    or isinstance(existing, FanoutTransport) and existing.contains(transport)
                    or transport in (session.get("viewers") or {})):
                attached.append((sid, session))
    return [(sid, session) for sid, session in attached
            if not _detach_session_transport(session, transport)]


def register(server) -> None:
    bind_module(globals(), server)
