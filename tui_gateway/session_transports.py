"""Additive transport membership for shared live sessions."""
from __future__ import annotations

import threading
from tui_gateway.method_ctx import bind_module

# ── multi-client fan-out ─────────────────────────────────────────────────────
#
# A session's ``transport`` slot holds either ONE transport — the historical,
# single-client shape, byte-identical to pre-fan-out behaviour — or a
# ``FanoutTransport`` wrapping several. Because the fan-out satisfies the same
# Transport protocol, ``write_json`` and every other reader of the slot are
# untouched; only the attach/detach ladder below knows the difference.
#
# ``_session_transport_lock`` serializes the read-modify-write of that one slot.
# It is a LEAF lock: nothing under it acquires ``_sessions_lock`` or a session's
# ``history_lock`` (the fan-out's own lock is likewise a leaf), so it is safe to
# take while holding either of those.
_session_transport_lock = threading.Lock()


def _transport_is_live_peer(transport) -> bool:
    """True when *transport* is a real, currently attached client.

    Excluded: the parked drop sentinel (by definition clientless) and stdio,
    which is the process-wide fallback sink — a standalone ``hermes --tui``
    writes there, but nothing "attaches" to it and a second client can never
    share it.
    """
    if transport is None:
        return False
    if transport is _detached_ws_transport or transport is _stdio_transport:
        return False
    if isinstance(transport, (_DropTransport, StdioTransport)):
        return False
    # A socket that already latched ``_closed`` is a departed client, not a live
    # peer: without this, a stale WSTransport keeps a session out of the park /
    # reap path and keeps answering "another client is still here".
    # ``_transport_is_dead`` is the deadness predicate this module shares with
    # the reaper (defined in ``session_reaper``, published onto this namespace
    # by ``method_ctx.bind_module``; module globals resolve at call time).
    return not _transport_is_dead(transport)


def _session_transport_contains(session: dict | None, transport) -> bool:
    """True when *transport* is attached to *session*, directly or via fan-out."""
    if not session or transport is None:
        return False
    existing = session.get("transport")
    if existing is transport:
        return True
    if isinstance(existing, FanoutTransport):
        return existing.contains(transport)
    return False


def _session_live_transports(session: dict | None) -> list:
    """Every live client attached to *session* (empty for a parked/stdio slot)."""
    existing = (session or {}).get("transport")
    if isinstance(existing, FanoutTransport):
        return [t for t in existing.transports() if _transport_is_live_peer(t)]
    return [existing] if _transport_is_live_peer(existing) else []


def _session_has_live_transport(session: dict | None, *, excluding=None) -> bool:
    """True when *session* still has a live client attached, ignoring *excluding*.

    ``excluding`` answers whether another live client remains once one peer is
    ignored, which is what the detach and orphan paths need: whether anything
    survives the departing transport.
    """
    return any(t is not excluding for t in _session_live_transports(session))


def _attach_session_transport(session: dict | None, transport) -> bool:
    """Attach *transport* to *session* ADDITIVELY — never steal the slot.

    A ``FanoutTransport`` argument is FLATTENED before the ladder runs, whatever
    the slot holds: each of its peers is attached as a leaf, so the slot never
    nests one fan-out inside another. The queued-prompt paths make that
    reachable — a busy submit records ``session["transport"]`` as the prompt's
    transport, which is the fan-out itself when two clients are attached, and
    the drain hands it back here after a disconnect has collapsed the slot to a
    single client. Nesting would blind every single-level reader of the slot
    (``FanoutTransport.contains``, the steer-authority scan, detach) to the
    inner peers, so a client inside the inner fan-out could never be found,
    parked, reaped, or granted authority over its own turn.

    The ladder, for the leaf newcomer that always reaches it:

    * same object already in the slot → no-op;
    * the slot already fans out → attach into it;
    * the slot is empty / stdio / the parked drop sentinel → take the slot,
      which is exactly what the old rebind did;
    * otherwise → wrap the incumbent and the newcomer in a ``FanoutTransport``
      so both clients keep streaming.

    A non-peer newcomer (stdio, drop sentinel) never displaces a live client:
    an activate/resume dispatched without a bound websocket must not silence the
    socket that owns the session.

    Returns ``True`` when *transport* is attached afterwards.
    """
    if not session or transport is None:
        return False
    if isinstance(transport, FanoutTransport):
        # Flatten OUTSIDE the lock — _session_transport_lock is not reentrant.
        # Each peer then walks the ladder on its own, so a peer whose socket
        # died while the prompt sat in the queue is dropped by the liveness rung
        # instead of being pinned back into the slot.
        attached = False
        for peer in transport.transports():
            if _attach_session_transport(session, peer):
                attached = True
        return attached
    with _session_transport_lock:
        existing = session.get("transport")
        if existing is transport:
            return True
        if isinstance(existing, FanoutTransport):
            if isinstance(transport, FanoutTransport):
                for member in transport.transports():
                    existing.attach(member)
            else:
                existing.attach(transport)
            return True
        if not _transport_is_live_peer(transport):
            if _transport_is_live_peer(existing):
                return False
            session["transport"] = transport
            return True
        if not _transport_is_live_peer(existing):
            session["transport"] = transport
            return True
        session["transport"] = FanoutTransport(existing, transport)
        return True


def _detach_session_transport(session: dict | None, transport) -> bool:
    """Detach *transport* from *session*'s slot.

    Returns ``True`` when a live client OTHER than *transport* remains — i.e.
    the session must keep streaming and must NOT be parked or reaped. A
    single-client session leaves the departing transport in the slot, exactly as
    before fan-out existed; its caller parks the drop sentinel over it.
    """
    if not session:
        return False
    with _session_transport_lock:
        (session.get("viewers") or {}).pop(transport, None)
        existing = session.get("transport")
        if isinstance(existing, FanoutTransport):
            existing.detach(transport)
            remaining = existing.transports()
            if len(remaining) == 1:
                # Collapse: a session back down to one client is indistinguishable
                # from one that never fanned out.
                session["transport"] = remaining[0]
    return _session_has_live_transport(session, excluding=transport)


def _detach_transport_from_sessions(transport) -> list[tuple[str, dict]]:
    """Detach *transport* from every session holding it.

    Returns the ``(sid, session)`` pairs left with NO live client — the ones the
    disconnect path must park or reap. Sessions that retain another attached
    client keep streaming and are not returned.
    """
    with _sessions_lock:
        attached = [
            (sid, s)
            for sid, s in _sessions.items()
            if _session_transport_contains(s, transport)
        ]
    return [
        (sid, session)
        for sid, session in attached
        if not _detach_session_transport(session, transport)
    ]



def register(server) -> None:
    bind_module(globals(), server)
