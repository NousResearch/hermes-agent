"""A session streams to EVERY attached client, not just the newest one.

Before this, ``session["transport"]`` held exactly one client and every
``prompt.submit`` / ``session.resume`` / ``session.activate`` / queued-prompt
drain rebound it — so a second client either saw nothing, stole the stream from
the first, or silenced the turn the first was reading. The slot now holds a
``FanoutTransport`` as soon as a second client attaches, and the disconnect path
detaches instead of parking whenever another client is still there.

Single-client behaviour is the control condition throughout: with one client
attached the slot holds the bare transport and every path behaves exactly as it
did before fan-out existed.
"""

import threading
import types

from tui_gateway import server
from tui_gateway.transport import FanoutTransport


class _FakeClient:
    """A connected client: records the frames it receives.

    ``ok`` False models a peer that has gone away (``write`` returns False, the
    gateway's peer-gone signal); ``boom`` models a wedged peer whose write
    raises. Both must be pruned without disturbing the healthy clients.
    """

    def __init__(self, name: str, *, ok: bool = True, boom: bool = False) -> None:
        self.name = name
        self.frames: list[dict] = []
        self.closed = False
        self._ok = ok
        self._boom = boom

    def write(self, obj: dict) -> bool:
        if self._boom:
            raise RuntimeError(f"{self.name} is wedged")
        self.frames.append(obj)
        return self._ok

    def close(self) -> None:
        self.closed = True

    def types(self) -> list[str]:
        return [(f.get("params") or {}).get("type") for f in self.frames]


def _session(**extra) -> dict:
    return {
        "agent": None,
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "transport": None,
        **extra,
    }


# ── FanoutTransport ────────────────────────────────────────────────────────


def test_fanout_delivers_to_every_attached_transport():
    a, b = _FakeClient("a"), _FakeClient("b")
    fan = FanoutTransport(a, b)

    assert fan.write({"frame": 1}) is True
    assert a.frames == b.frames == [{"frame": 1}]
    assert fan.transports() == [a, b]


def test_fanout_attach_is_idempotent_and_detach_is_by_identity():
    a, b = _FakeClient("a"), _FakeClient("b")
    fan = FanoutTransport(a)

    assert fan.attach(a) is False  # already attached
    assert fan.attach(b) is True
    assert fan.contains(a) and fan.contains(b)
    assert fan.has_transports(excluding=a) is True

    assert fan.detach(a) is True
    assert fan.detach(a) is False
    assert fan.transports() == [b]
    assert fan.has_transports(excluding=b) is False


def test_fanout_prunes_a_wedged_peer_without_disturbing_the_rest():
    """(g) A raising peer is dropped; the healthy client keeps its stream."""
    healthy, wedged = _FakeClient("healthy"), _FakeClient("wedged", boom=True)
    fan = FanoutTransport(wedged, healthy)

    assert fan.write({"frame": 1}) is True
    assert healthy.frames == [{"frame": 1}]
    assert fan.transports() == [healthy]

    # And the pruning is permanent — no retry storm against the wedged peer.
    assert fan.write({"frame": 2}) is True
    assert len(healthy.frames) == 2


def test_fanout_prunes_a_peer_that_reports_gone_and_reports_all_dead():
    gone = _FakeClient("gone", ok=False)
    fan = FanoutTransport(gone)

    # write() returned False -> peer gone -> pruned, and an empty fan-out
    # reports peer-gone exactly like a single dead transport would.
    assert fan.write({"frame": 1}) is False
    assert fan.transports() == []
    assert fan.write({"frame": 2}) is False


def test_fanout_close_releases_peers_without_closing_their_sockets():
    a = _FakeClient("a")
    fan = FanoutTransport(a)

    fan.close()

    assert fan.transports() == []
    # Each client owns its own socket; the WS handler closes it on disconnect.
    assert a.closed is False


# ── attach ladder ──────────────────────────────────────────────────────────


def test_attach_replaces_an_empty_or_stdio_slot_without_wrapping():
    """The single-client shape is unchanged: no FanoutTransport in sight."""
    a = _FakeClient("a")

    empty = _session(transport=None)
    assert server._attach_session_transport(empty, a) is True
    assert empty["transport"] is a

    stdio = _session(transport=server._stdio_transport)
    assert server._attach_session_transport(stdio, a) is True
    assert stdio["transport"] is a

    parked = _session(transport=server._detached_ws_transport)
    assert server._attach_session_transport(parked, a) is True
    assert parked["transport"] is a


def test_attach_of_the_same_transport_is_a_noop():
    a = _FakeClient("a")
    session = _session(transport=a)

    assert server._attach_session_transport(session, a) is True
    assert session["transport"] is a


def test_attach_of_a_second_client_wraps_both_and_a_third_joins_the_fanout():
    a, b, c = _FakeClient("a"), _FakeClient("b"), _FakeClient("c")
    session = _session(transport=a)

    server._attach_session_transport(session, b)
    assert isinstance(session["transport"], FanoutTransport)
    assert session["transport"].transports() == [a, b]

    server._attach_session_transport(session, c)
    assert session["transport"].transports() == [a, b, c]


def test_attach_never_lets_stdio_displace_a_live_client():
    """An unbound-context activate must not silence the websocket that owns it."""
    a = _FakeClient("a")
    session = _session(transport=a)

    assert server._attach_session_transport(session, server._stdio_transport) is False
    assert session["transport"] is a


def test_attach_flattens_a_fanout_argument_instead_of_nesting_it():
    a, b, c = _FakeClient("a"), _FakeClient("b"), _FakeClient("c")
    session = _session(transport=a)
    server._attach_session_transport(session, b)

    server._attach_session_transport(session, FanoutTransport(b, c))

    assert session["transport"].transports() == [a, b, c]


def test_attach_flattens_a_fanout_argument_onto_a_single_client_slot(monkeypatch):
    """The nesting the queued-prompt paths can actually reach.

    A busy submit pins ``t or session["transport"]`` to the queued prompt, so
    the envelope can hold the FAN-OUT that was in the slot, and the drain hands
    that straight back to attach. Flattening only when the slot ALREADY fans out
    leaves the leaf-slot case nesting one fan-out inside another, and every
    reader of the slot scans a single level — ``FanoutTransport.contains``, the
    steer-authority check, detach — so a peer in the inner fan-out is invisible
    to all of them, and a peer in BOTH levels is written to twice per frame.
    """
    a, b = _FakeClient("a"), _FakeClient("b")

    # (1) A fan-out arriving at a leaf slot is flattened into it, not wrapped.
    session = _session(transport=a)
    assert server._attach_session_transport(session, FanoutTransport(a, b)) is True
    slot = session["transport"]
    assert isinstance(slot, FanoutTransport)
    assert slot.transports() == [a, b]
    assert not any(isinstance(t, FanoutTransport) for t in slot.transports())
    assert server._session_transport_contains(session, b) is True

    # (2) The sequence that produces it. Two clients attach, so the slot holds a
    # fan-out; a busy submit captures that slot in the queued envelope; the
    # second client disconnects and the slot collapses back to the first; the
    # drain then attaches the captured fan-out to a leaf slot.
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a_, **k: None)

    drained = _session(transport=a)
    server._attach_session_transport(drained, b)
    captured = drained["transport"]
    assert isinstance(captured, FanoutTransport)
    server._enqueue_prompt(drained, "queued while busy", captured)

    assert server._detach_session_transport(drained, b) is True
    assert drained["transport"] is a

    server._sessions["flatten-sid"] = drained
    try:
        assert server._drain_queued_prompt("drain", "flatten-sid", drained) is True
        # Flat: the captured fan-out lost b to the same detach that collapsed
        # the slot, so flattening it re-attaches only a, which is already there.
        assert drained["transport"] is a
        server._emit("message.delta", "flatten-sid", {"text": "once"})
    finally:
        server._sessions.pop("flatten-sid", None)

    # Nesting would have put a inside the slot AND under it: one frame, two
    # writes to the same client.
    assert a.types() == ["message.delta"]


def test_detach_collapses_back_to_the_single_remaining_client():
    a, b = _FakeClient("a"), _FakeClient("b")
    session = _session(transport=a)
    server._attach_session_transport(session, b)

    assert server._detach_session_transport(session, a) is True
    assert session["transport"] is b
    assert server._detach_session_transport(session, b) is False


def test_detach_of_a_single_client_leaves_the_slot_for_the_caller_to_park():
    a = _FakeClient("a")
    session = _session(transport=a)

    # False == "no live client remains" — the disconnect path parks the sentinel.
    assert server._detach_session_transport(session, a) is False
    assert session["transport"] is a


def test_live_transport_predicates_ignore_stdio_and_the_drop_sentinel():
    a = _FakeClient("a")

    assert server._session_has_live_transport(_session(transport=a)) is True
    assert server._session_has_live_transport(_session(transport=a), excluding=a) is False
    assert server._session_has_live_transport(_session(transport=None)) is False
    assert (
        server._session_has_live_transport(_session(transport=server._stdio_transport))
        is False
    )
    assert (
        server._session_has_live_transport(
            _session(transport=server._detached_ws_transport)
        )
        is False
    )


def test_a_closed_socket_is_not_a_live_peer():
    """A transport that already latched ``_closed`` is a departed client.

    ``_transport_is_live_peer`` ended on a bare ``return True``, so a WSTransport
    whose socket had gone away still counted as a live peer: a session whose only
    remaining peer was that dead socket answered "another client is still here"
    and escaped both the park and the reap. ``_transport_is_dead`` is the
    module's deadness predicate; the ladder now defers to it.
    """
    dead = _FakeClient("dead")
    dead._closed = True  # the flag WSTransport latches when its socket goes away

    assert server._transport_is_live_peer(dead) is False
    assert server._session_has_live_transport(_session(transport=dead)) is False

    # And a fan-out that collapses onto a dead peer is parked, not skipped: the
    # live client leaving was the last real client.
    live_client = _FakeClient("live")
    session = _session(transport=FanoutTransport(live_client, dead))
    server._sessions["dead-peer-sid"] = session
    try:
        assert server._close_sessions_for_transport(live_client) == (0, 1)
        assert session["transport"] is server._detached_ws_transport
    finally:
        server._sessions.pop("dead-peer-sid", None)


def test_a_fanout_with_no_live_peer_is_dead():
    """``_transport_is_dead`` is the reapers' gate, and a fan-out could not fail it.

    A ``FanoutTransport`` is never the drop sentinel and its ``__slots__`` give
    it no ``_closed``, so a session that fanned out once read as alive forever:
    the TTL reaper, the LRU cap and the disconnect revalidation all ask this one
    predicate. A fan-out reaches the no-live-peer state without any disconnect
    passing through ``_close_sessions_for_transport`` — a write that returns
    False or raises prunes that peer — so the empty case is reachable, and it is
    dead.
    """
    a, b = _FakeClient("a"), _FakeClient("b")
    session = _session(transport=a)
    server._attach_session_transport(session, b)
    assert isinstance(session["transport"], FanoutTransport)
    assert server._transport_is_dead(session["transport"]) is False

    # One peer gone, one still reading: the session keeps its client.
    a._closed = True  # the flag WSTransport latches when its socket goes away
    assert server._transport_is_dead(session["transport"]) is False

    b._closed = True
    assert server._transport_is_dead(session["transport"]) is True

    # Pruning every peer leaves the fan-out empty, which is nobody attached.
    assert server._transport_is_dead(FanoutTransport()) is True


def test_steer_authority_recognizes_the_exact_client_inside_a_fanout():
    """Wrapping attached peers must not revoke the commissioning peer's authority."""
    owner, watcher, stranger = (
        _FakeClient("owner"),
        _FakeClient("watcher"),
        _FakeClient("stranger"),
    )
    session = _session(transport=FanoutTransport(owner, watcher))
    server._sessions["sid"] = session
    try:
        token = server.bind_transport(owner)
        try:
            assert server._current_session_steer_authority("sid") == (owner, session)
        finally:
            server.reset_transport(token)

        token = server.bind_transport(stranger)
        try:
            assert server._current_session_steer_authority("sid") == (None, None)
        finally:
            server.reset_transport(token)
    finally:
        server._sessions.pop("sid", None)


def test_steer_authority_is_granted_to_an_attached_watcher():
    """A client that attached to watch a session may also steer its subagents.

    This is INTENTIONAL and wider than the pre-fan-out rule, which admitted only
    whichever client happened to hold the transport slot. A mirrored session has
    no single owner in that slot, so authority is membership in it. Narrowing
    this back to the commissioning peer would need a per-subagent record of who
    commissioned it, which this change does not add; the widening is stated in
    the pull request description, and this test pins it so it cannot be changed
    silently in either direction.
    """
    owner, watcher, stranger = (
        _FakeClient("owner"),
        _FakeClient("watcher"),
        _FakeClient("stranger"),
    )
    session = _session(transport=owner)
    server._attach_session_transport(session, watcher)
    solo = _session(transport=owner)
    server._sessions["sid"] = session
    server._sessions["solo"] = solo
    try:
        token = server.bind_transport(watcher)
        try:
            assert server._current_session_steer_authority("sid") == (watcher, session)
            # Control: on a single-client session the same watcher is a stranger,
            # so the widening reaches attached clients and nobody else.
            assert server._current_session_steer_authority("solo") == (None, None)
        finally:
            server.reset_transport(token)

        token = server.bind_transport(stranger)
        try:
            assert server._current_session_steer_authority("sid") == (None, None)
        finally:
            server.reset_transport(token)
    finally:
        server._sessions.pop("sid", None)
        server._sessions.pop("solo", None)


# ── browser-control session ownership ──────────────────────────────────────
#
# The four browser.controller.* handlers gate on the session slot exactly as
# subagent.steer did, so fan-out breaks them the same way and the fix is the
# same predicate. These tests pin the gate itself: they assert on the ownership
# error message and stop at the NEXT gate ("no controller registered for this
# session"), so a later broker change cannot make them pass vacuously.

_CONTROLLER_IDENTITY = {"user_id": "user-fixture", "provider": "provider-fixture"}
_NOT_OWNED = "session is not owned by this transport"
_NO_CONTROLLER = "no controller registered for this session"


def _controller_client(name: str) -> _FakeClient:
    """A fan-out client that also carries a server-authenticated identity."""
    client = _FakeClient(name)
    client.auth_identity = dict(_CONTROLLER_IDENTITY)
    return client


def _controller_rpc(transport, method_name: str, **params) -> dict:
    return server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method_name,
            "params": params,
        },
        transport,
    )


def _error_message(response: dict) -> str | None:
    return (response.get("error") or {}).get("message")


def test_browser_control_ownership_gate_admits_a_peer_inside_a_fanout():
    """Wrapping attached peers must not revoke browser control for all of them."""
    owner, watcher, stranger = (
        _controller_client("owner"),
        _controller_client("watcher"),
        _controller_client("stranger"),
    )
    session = _session(transport=FanoutTransport(owner, watcher), profile="default")
    server._sessions["sid"] = session
    try:
        # The peer that would have registered the controller clears the
        # ownership gate and stops at the next one.
        assert _error_message(_controller_rpc(owner, "browser.controller.heartbeat", session_id="sid")) == _NO_CONTROLLER
        # Widened, exactly as the steer conversion widened steer: any attached
        # peer clears the session gate. The broker's is_owner check below it is
        # what still refuses a peer that did not register the controller.
        assert _error_message(_controller_rpc(watcher, "browser.controller.heartbeat", session_id="sid")) == _NO_CONTROLLER
        # An unattached client is still refused at the ownership gate.
        assert _error_message(_controller_rpc(stranger, "browser.controller.heartbeat", session_id="sid")) == _NOT_OWNED
    finally:
        server._sessions.pop("sid", None)


def test_browser_control_ownership_gate_is_unchanged_for_a_single_client():
    """Control: a bare slot still compares by identity on all four handlers."""
    owner, stranger = _controller_client("owner"), _controller_client("stranger")
    session = _session(transport=owner, profile="default")
    server._sessions["solo"] = session
    try:
        for method_name in (
            "browser.controller.heartbeat",
            "browser.controller.result",
            "browser.controller.detach",
        ):
            assert _error_message(_controller_rpc(stranger, method_name, session_id="solo")) == _NOT_OWNED
        assert _error_message(_controller_rpc(owner, "browser.controller.heartbeat", session_id="solo")) == _NO_CONTROLLER
    finally:
        server._sessions.pop("solo", None)


def test_browser_controller_registers_and_detaches_inside_a_fanout(monkeypatch):
    """End to end: registration on a mirrored session, then a clean detach."""
    from gateway import browser_control_broker

    monkeypatch.setattr(
        "gateway.browser_control_broker.browser_control_enabled", lambda: True
    )
    owner, watcher, stranger = (
        _controller_client("owner"),
        _controller_client("watcher"),
        _controller_client("stranger"),
    )
    session = _session(transport=FanoutTransport(owner, watcher), profile="default")
    server._sessions["sid"] = session
    registration = {}
    try:
        registration = _controller_rpc(
            owner,
            "browser.controller.register",
            session_id="sid",
            controller_id="controller-fixture",
            browser_profile_id="browser-profile-fixture",
            capabilities=["controller.noop"],
            protocol_version=browser_control_broker.BROWSER_CONTROL_PROTOCOL_VERSION,
        )
        assert registration.get("error") is None, registration
        assert registration["result"]["scope"]["session_id"] == "sid"

        # The registering peer owns the controller; the fan-out peer that did
        # not register clears the session gate and is refused by the broker.
        assert _controller_rpc(owner, "browser.controller.heartbeat", session_id="sid")["result"] == {"ok": True}
        assert _error_message(_controller_rpc(watcher, "browser.controller.heartbeat", session_id="sid")) == "controller is not owned by this transport"
        assert _error_message(_controller_rpc(stranger, "browser.controller.heartbeat", session_id="sid")) == _NOT_OWNED

        detached = _controller_rpc(owner, "browser.controller.detach", session_id="sid")
        assert detached.get("error") is None, detached
    finally:
        if registration.get("result"):
            broker = browser_control_broker.get_browser_control_broker()
            scope = broker.scope_for_session(
                session_id="sid",
                principal_id=registration["result"]["scope"]["principal_id"],
                transport_family=registration["result"]["scope"]["transport_family"],
            )
            if scope is not None:
                broker.detach(scope, notify_controller=False)
        server._sessions.pop("sid", None)


# ── event delivery ─────────────────────────────────────────────────────────


def test_two_attached_clients_both_receive_a_session_event(monkeypatch):
    """(a) The headline behaviour: one emit, two clients."""
    a, b = _FakeClient("a"), _FakeClient("b")
    session = _session(transport=a)
    server._attach_session_transport(session, b)
    server._sessions["sid"] = session
    try:
        server._emit("message.delta", "sid", {"text": "hi"})
    finally:
        server._sessions.pop("sid", None)

    assert a.types() == ["message.delta"]
    assert b.types() == ["message.delta"]
    assert a.frames[0] == b.frames[0]


def test_a_single_client_session_writes_through_the_bare_transport(monkeypatch):
    """(j) Control: nothing about the one-client path changed."""
    a = _FakeClient("a")
    server._sessions["sid"] = _session(transport=a)
    try:
        assert server._sessions["sid"]["transport"] is a
        server._emit("message.delta", "sid", {"text": "hi"})
    finally:
        server._sessions.pop("sid", None)

    # The replay contract stamps a monotonic ``seq`` on every WS event frame.
    # Strip it: this control test is about ROUTING, not numbering.
    frames = [
        {**f, "params": {k: v for k, v in f["params"].items() if k != "seq"}}
        for f in a.frames
    ]
    assert frames == [
        {
            "jsonrpc": "2.0",
            "method": "event",
            "params": {
                "type": "message.delta",
                "session_id": "sid",
                "payload": {"text": "hi"},
            },
        }
    ]


def test_both_attached_clients_receive_the_terminal_message_complete():
    """The frame that ENDS a turn fans out too, not just the streaming deltas.

    ``message.complete`` is how a client knows the turn is over. A mirrored
    session that delivered deltas but dropped the terminal frame would leave the
    second client rendering a turn that never finishes.
    """
    a, b = _FakeClient("a"), _FakeClient("b")
    session = _session(transport=a)
    server._attach_session_transport(session, b)
    server._sessions["sid"] = session
    try:
        server._emit("message.delta", "sid", {"text": "par"})
        server._emit("message.complete", "sid", {"text": "part", "status": "ok"})
    finally:
        server._sessions.pop("sid", None)

    assert a.types() == ["message.delta", "message.complete"]
    assert b.types() == ["message.delta", "message.complete"]
    assert a.frames[-1] == b.frames[-1]
    assert a.frames[-1]["params"]["payload"] == {"text": "part", "status": "ok"}

    # Control: the one-client path delivers the same terminal frame through the
    # bare transport, with no fan-out in the slot.
    solo = _FakeClient("solo")
    server._sessions["solo-sid"] = _session(transport=solo)
    try:
        assert server._sessions["solo-sid"]["transport"] is solo
        server._emit("message.complete", "solo-sid", {"text": "part", "status": "ok"})
    finally:
        server._sessions.pop("solo-sid", None)

    assert solo.types() == ["message.complete"]
    assert solo.frames[-1]["params"]["payload"] == {"text": "part", "status": "ok"}


# ── prompt.submit / queued drain ───────────────────────────────────────────


def test_second_clients_submit_attaches_and_the_first_keeps_streaming(monkeypatch):
    """(c) A second client's prompt.submit must not cut the first one out."""
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *a, **k: None)

    watcher, submitter = _FakeClient("watcher"), _FakeClient("submitter")
    session = _session(transport=watcher, agent=types.SimpleNamespace())
    server._sessions["sid"] = session
    token = server.bind_transport(submitter)
    try:
        server._methods["prompt.submit"]("r1", {"session_id": "sid", "text": "hello"})
        server._emit("message.delta", "sid", {"text": "answer"})
    finally:
        server.reset_transport(token)
        server._sessions.pop("sid", None)

    assert isinstance(session["transport"], FanoutTransport)
    assert watcher.types() == ["message.delta"]
    assert submitter.types() == ["message.delta"]


def test_queued_prompt_drain_keeps_both_clients_attached(monkeypatch):
    """(d) The hole every competing patch missed: the drain used to rebind.

    Client A is streaming; client B submits mid-turn, so B's prompt is queued
    with B's transport pinned to it. When the drain fires it must ATTACH B —
    rebinding pinned the whole drained turn to B and silenced A.
    """
    dispatched = []
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda rid, sid, _session, text, **kw: dispatched.append((rid, text)),
    )

    a, b = _FakeClient("a"), _FakeClient("b")
    session = _session(transport=a)
    server._enqueue_prompt(session, "from B", b)
    server._sessions["sid"] = session
    try:
        assert server._drain_queued_prompt("drain", "sid", session) is True
        server._emit("message.delta", "sid", {"text": "drained answer"})
    finally:
        server._sessions.pop("sid", None)

    assert dispatched == [("drain", "from B")]
    assert isinstance(session["transport"], FanoutTransport)
    assert a.types() == ["message.delta"]
    assert b.types() == ["message.delta"]


def test_queued_prompt_drain_skips_a_queuer_that_disconnected(monkeypatch):
    """B goes away while its prompt waits: the prompt runs, the dead pin does not.

    Attaching a transport whose client already left would pin a dead peer into
    the slot until the first failed write prunes it. A keeps its stream and
    stays the only attached client.
    """
    dispatched = []
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda rid, sid, _session, text, **kw: dispatched.append((rid, text)),
    )

    a, b = _FakeClient("a"), _FakeClient("b")
    session = _session(transport=a)
    server._enqueue_prompt(session, "from B", b)
    b._closed = True  # what _transport_is_dead reads: B's socket went away
    server._sessions["sid"] = session
    try:
        assert server._drain_queued_prompt("drain", "sid", session) is True
        server._emit("message.delta", "sid", {"text": "drained answer"})
    finally:
        server._sessions.pop("sid", None)

    assert dispatched == [("drain", "from B")]  # drain semantics unchanged
    assert session["transport"] is a
    assert server._session_transport_contains(session, b) is False
    assert a.types() == ["message.delta"]
    assert b.types() == []


def test_queued_prompt_drain_still_rebinds_a_single_client_session(monkeypatch):
    """(j) Control: with one client the drain lands on the queuer's transport."""
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **k: None)

    b = _FakeClient("b")
    session = _session(transport=server._detached_ws_transport)
    server._enqueue_prompt(session, "from B", b)

    assert server._drain_queued_prompt("drain", "sid", session) is True
    assert session["transport"] is b


# ── disconnect ─────────────────────────────────────────────────────────────


def test_watcher_disconnect_leaves_the_other_client_streaming():
    """(e) One client leaving must not park or reap a session someone is reading."""
    a, watcher = _FakeClient("a"), _FakeClient("watcher")
    session = _session(transport=a)
    server._attach_session_transport(session, watcher)
    server._sessions["sid"] = session
    try:
        assert server._close_sessions_for_transport(watcher) == (0, 0)
        assert session["transport"] is a
        assert server._ws_session_is_orphaned(session) is False
        server._emit("message.delta", "sid", {"text": "still here"})
    finally:
        server._sessions.pop("sid", None)

    assert a.types() == ["message.delta"]
    assert watcher.types() == []


def test_last_client_disconnect_parks_exactly_as_before(monkeypatch):
    """(f) Once the last client goes, today's park + grace-reap path runs."""
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)

    a, watcher = _FakeClient("a"), _FakeClient("watcher")
    session = _session(transport=a)
    server._attach_session_transport(session, watcher)
    server._sessions["sid"] = session
    try:
        assert server._close_sessions_for_transport(watcher) == (0, 0)
        assert server._close_sessions_for_transport(a) == (0, 1)
        assert session["transport"] is server._detached_ws_transport
        assert server._ws_session_is_orphaned(session) is True
    finally:
        server._sessions.pop("sid", None)


def test_last_client_disconnect_reaps_a_close_on_disconnect_session(monkeypatch):
    """(f) close_on_disconnect still fires — but only for the LAST client."""
    # The disconnect path claims the session with _pop_session_by_id and finishes
    # the close through _teardown_popped_session, so that is what this stubs; the
    # sid is asserted through the pop rather than through the call arguments.
    closed = []

    def _fake_teardown(session, *, end_reason: str = "tui_close") -> bool:
        closed.append((session, end_reason))
        return True

    monkeypatch.setattr(server, "_teardown_popped_session", _fake_teardown)

    a, watcher = _FakeClient("a"), _FakeClient("watcher")
    session = _session(transport=a, close_on_disconnect=True)
    server._attach_session_transport(session, watcher)
    server._sessions["sid"] = session
    try:
        assert server._close_sessions_for_transport(watcher) == (0, 0)
        assert closed == []
        assert server._close_sessions_for_transport(
            a, end_reason="ws_disconnect"
        ) == (1, 0)
        assert "sid" not in server._sessions
    finally:
        server._sessions.pop("sid", None)

    assert closed == [(session, "ws_disconnect")]


def test_orphan_check_spares_a_session_that_still_has_a_fanout_peer():
    # _ws_session_is_orphaned stayed slot-identity based on main: a fan-out is
    # never the drop sentinel, and the disconnect path only parks the sentinel
    # once the last peer is gone, so these hold without a liveness rewrite.
    a, watcher = _FakeClient("a"), _FakeClient("watcher")
    session = _session(transport=a)
    server._attach_session_transport(session, watcher)

    assert server._ws_session_is_orphaned(session) is False

    # Losing one of two clients collapses the slot but keeps the session live.
    assert server._detach_session_transport(session, a) is True
    assert server._ws_session_is_orphaned(session) is False

    # Losing the last one reports "no client left"; the caller then parks the
    # drop sentinel, which is what makes the session orphaned.
    assert server._detach_session_transport(session, watcher) is False
    session["transport"] = server._detached_ws_transport
    assert server._ws_session_is_orphaned(session) is True


# ---------------------------------------------------------------------------
# Concurrent fan-out delivery
#
# A SERIAL ``FanoutTransport.write`` — each peer written in turn — would let a
# client whose write parks for the full ``tui_gateway.ws._WS_WRITE_TIMEOUT_S``
# (10s, the non-streaming path's ``fut.result`` wait for a stalled event loop)
# hold the same frame back from every healthy client behind it in the list.
# That is the property these tests pin, so the writes run concurrently, with two
# cases deliberately kept inline: a lone peer (single-client sessions must not
# change at all) and a caller that is already on an event loop (handing that
# write to a pool thread would make the pool thread block on the loop it just
# left, freezing both).
#
# Extra imports for this block: ``asyncio`` below, plus a function-local
# ``tui_gateway.transport`` in the deadline test (it monkeypatches a module
# constant). Everything else — ``threading``, ``FanoutTransport``,
# ``_FakeClient`` — comes from the top of this file.
# ---------------------------------------------------------------------------

import asyncio


class _SlowPeer:
    """A peer whose write parks until the test releases it.

    Models the real stall: a non-streaming ``WSTransport.write`` blocks on
    ``fut.result(timeout=_WS_WRITE_TIMEOUT_S)`` while the owning event loop is
    busy, so the emitting thread sits inside that one peer's write for seconds.
    """

    def __init__(self, name: str = "slow") -> None:
        self.name = name
        self.frames: list[dict] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed = False

    def write(self, obj: dict) -> bool:
        self.entered.set()
        # Bounded: a regression must fail the assertion below, never hang the
        # suite waiting for a release that the failing path never reaches.
        self.release.wait(timeout=5.0)
        self.frames.append(obj)
        return True

    def close(self) -> None:
        self.closed = True


class _SignallingPeer:
    """A healthy peer that announces each frame the moment it lands."""

    def __init__(self, name: str = "healthy") -> None:
        self.name = name
        self.frames: list[dict] = []
        self.got_frame = threading.Event()
        self.closed = False

    def write(self, obj: dict) -> bool:
        self.frames.append(obj)
        self.got_frame.set()
        return True

    def close(self) -> None:
        self.closed = True


class _ThreadRecordingPeer:
    """A healthy peer that records which thread each of its writes ran on."""

    def __init__(self, name: str = "peer") -> None:
        self.name = name
        self.frames: list[dict] = []
        self.threads: list[threading.Thread] = []
        self.closed = False

    def write(self, obj: dict) -> bool:
        self.threads.append(threading.current_thread())
        self.frames.append(obj)
        return True

    def close(self) -> None:
        self.closed = True


def _healthy_peer_is_not_held_behind(slow_first: bool) -> None:
    """One wedged peer, one healthy peer, one ``write`` from a worker thread.

    The healthy peer must hold the frame while the wedged peer is still parked
    inside its own write. Asserted for both list orders so the fix cannot be a
    special case that only ever hurries the first (or the last) peer along.
    """
    slow = _SlowPeer()
    healthy = _SignallingPeer()
    peers = (slow, healthy) if slow_first else (healthy, slow)
    fanout = FanoutTransport(*peers)
    frame = {"jsonrpc": "2.0", "method": "event", "params": {"type": "message.complete"}}

    returned = threading.Event()
    result: dict = {}

    def emit() -> None:
        try:
            result["ok"] = fanout.write(frame)
        finally:
            returned.set()

    # Emit from a worker thread, which is where the gateway's event writes come
    # from: ``handle_ws`` runs ``server.dispatch`` on ``asyncio.to_thread``.
    worker = threading.Thread(target=emit, name="fanout-emitter", daemon=True)
    worker.start()
    try:
        assert slow.entered.wait(timeout=2.0), "the wedged peer never got the frame"
        assert healthy.got_frame.wait(timeout=2.0), (
            "the healthy peer did not get the frame while a wedged peer held its "
            "write open — the fan-out is still serial"
        )
        assert healthy.frames == [frame]
        assert slow.frames == []
    finally:
        slow.release.set()
        assert returned.wait(timeout=5.0), "the fan-out write never returned"
        worker.join(timeout=5.0)

    # The fan-out still collects: the wedged peer's frame landed before the
    # call returned, and both peers stay attached.
    assert slow.frames == [frame]
    assert result["ok"] is True
    assert fanout.transports() == [*peers]


def test_a_wedged_peer_does_not_hold_the_frame_from_a_healthy_peer_behind_it():
    _healthy_peer_is_not_held_behind(slow_first=True)


def test_a_wedged_peer_does_not_hold_the_frame_from_a_healthy_peer_ahead_of_it():
    _healthy_peer_is_not_held_behind(slow_first=False)


def test_a_write_from_inside_an_event_loop_stays_on_the_loop_thread():
    """The deadlock the concurrent path must never introduce.

    ``WSTransport.write`` fires and forgets when it can see it is running on
    its own loop, and BLOCKS on ``fut.result`` when it cannot. Hand a
    loop-thread write to a pool thread and that pool thread takes the blocking
    path, waiting on a loop that is itself waiting on the pool — the whole
    gateway stalls for the write timeout. So an on-loop caller stays inline.
    """
    a, b = _ThreadRecordingPeer("a"), _ThreadRecordingPeer("b")
    fanout = FanoutTransport(a, b)
    frame = {"params": {"type": "message.complete"}}
    outcome: dict = {}

    async def emit() -> None:
        outcome["loop_thread"] = threading.current_thread()
        outcome["ok"] = fanout.write(frame)

    # Driven from a side thread with a bounded join so a real deadlock fails
    # this test instead of hanging the run.
    runner = threading.Thread(target=lambda: asyncio.run(emit()), daemon=True)
    runner.start()
    runner.join(timeout=5.0)
    assert not runner.is_alive(), "fan-out write from the event loop did not return"

    assert outcome["ok"] is True
    assert a.frames == [frame] and b.frames == [frame]
    assert a.threads == [outcome["loop_thread"]]
    assert b.threads == [outcome["loop_thread"]]


def test_a_lone_peer_is_written_on_the_calling_thread():
    """Single-client sessions keep the exact behaviour they had before."""
    only = _ThreadRecordingPeer("only")
    fanout = FanoutTransport(only)

    frame = {"params": {"type": "message.complete"}}
    assert fanout.write(frame) is True

    assert only.frames == [frame]
    assert only.threads == [threading.current_thread()]


def test_the_concurrent_path_prunes_dead_peers_and_still_delivers():
    """Pruning is unchanged: ``False`` and a raise both detach the peer."""
    gone = _FakeClient("gone", ok=False)
    wedged = _FakeClient("wedged", boom=True)
    healthy = _FakeClient("healthy")
    fanout = FanoutTransport(gone, wedged, healthy)

    frame = {"params": {"type": "message.complete"}}
    assert fanout.write(frame) is True

    assert healthy.frames == [frame]
    assert fanout.transports() == [healthy]


def test_the_concurrent_path_reports_peer_gone_when_every_peer_is_dead():
    gone = _FakeClient("gone", ok=False)
    wedged = _FakeClient("wedged", boom=True)
    fanout = FanoutTransport(gone, wedged)

    assert fanout.write({"params": {"type": "message.complete"}}) is False
    assert fanout.transports() == []


def test_consecutive_frames_reach_every_peer_in_order():
    """The fan-out collects before returning, so frame A is on every peer
    before frame B is dispatched to any of them."""
    a, b = _ThreadRecordingPeer("a"), _ThreadRecordingPeer("b")
    fanout = FanoutTransport(a, b)
    first = {"params": {"type": "message.delta", "text": "1"}}
    second = {"params": {"type": "message.complete"}}

    assert fanout.write(first) is True
    # Collect-before-return: nothing is still in flight when write() answers.
    assert a.frames == [first] and b.frames == [first]

    assert fanout.write(second) is True
    assert a.frames == [first, second]
    assert b.frames == [first, second]


def test_a_peer_that_misses_the_deadline_is_kept_and_counts_as_delivered(monkeypatch):
    """Slow is not dead.

    A peer still writing when the fan-out deadline expires stays attached and
    the frame counts as in flight — which is what ``WSTransport.write`` itself
    reports when its own write times out. Here it is the ONLY live peer, so the
    ``True`` return rests entirely on it.
    """
    from tui_gateway import transport as transport_module

    monkeypatch.setattr(transport_module, "_FANOUT_WRITE_DEADLINE_S", 0.05)

    slow = _SlowPeer()
    gone = _FakeClient("gone", ok=False)
    fanout = FanoutTransport(slow, gone)
    frame = {"params": {"type": "message.complete"}}

    assert fanout.write(frame) is True
    # The dead peer is pruned; the slow one is not.
    assert fanout.transports() == [slow]
    assert slow.frames == []

    slow.release.set()
