"""Additive transport membership for shared live sessions."""
from __future__ import annotations

import threading
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
            existing.attach(transport)
            return existing.contains(transport)
        elif _transport_is_live_peer(existing):
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
