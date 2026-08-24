"""Peer transport tests for hosted-room member turns."""

from __future__ import annotations

from typing import Any

from gateway.hosted_room_driver import TaskIdentity
from tui_gateway.hosted_room_driver import HostedRoomBinding, ROOM_SESSION_SOURCE
from tui_gateway.hosted_room_peer_transport import (
    FailoverHostedRoomPeerClient,
    PeerHostedRoomTransport,
    PeerMemberRoute,
    RoomLinkCandidate,
)
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


BINDING = HostedRoomBinding("room-1", "gateway-home", 2)
ROUTE = PeerMemberRoute(
    home_install_id="install-home",
    member_id="member-reviewer",
    target_install_id="install-peer",
    target_profile="reviewer",
    capability_digest="a" * 64,
    execution_policy_digest="b" * 64,
    cancellation_scope_id="cancel-1",
    trace_id="trace-1",
    grant="signed-room-grant",
)


class FakePeerClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.session = {"session_id": "group-session"}
        self.messages = []
        self.active = False
        self.task_id = None

    def prepare(self, **kwargs):
        self.calls.append(("prepare", kwargs))
        return (
            self.session
            if kwargs["create"] or kwargs.get("expected_session_id")
            else None
        )

    def dispatch(self, **kwargs):
        self.calls.append(("dispatch", kwargs))
        dispatch = kwargs["dispatch"]
        self.active = True
        self.task_id = dispatch["task_id"]
        return {"status": "accepted", "task_id": self.task_id}

    def history(self, **kwargs):
        self.calls.append(("history", kwargs))
        return list(self.messages)

    def status(self, **kwargs):
        self.calls.append(("status", kwargs))
        return {"active": self.active, "task_id": self.task_id}

    def stop(self, **kwargs):
        self.calls.append(("stop", kwargs))
        self.active = False
        return {"status": "cancelled", "task_id": self.task_id}


class FailingPeerClient(FakePeerClient):
    def __init__(self, *, method, retryable=True, not_admitted=False):
        super().__init__()
        self.method = method
        self.error = PeerRunsHTTPError(
            f"{method} failed",
            retryable=retryable,
            ambiguous=method == "dispatch" and not not_admitted,
            not_admitted=not_admitted,
        )

    def prepare(self, **kwargs):
        if self.method == "prepare":
            raise self.error
        return super().prepare(**kwargs)

    def dispatch(self, **kwargs):
        if self.method == "dispatch":
            self.calls.append(("dispatch", kwargs))
            raise self.error
        return super().dispatch(**kwargs)

    def status(self, **kwargs):
        if self.method == "status":
            raise self.error
        return super().status(**kwargs)


def _transport(client=None, *, source_event_seq=1):
    return PeerHostedRoomTransport(
        binding=BINDING,
        route=ROUTE,
        client=client or FakePeerClient(),
        source_event_seq=source_event_seq,
    )


def test_peer_transport_prepares_group_session_not_canonical_bot_chat():
    client = FakePeerClient()
    transport = _transport(client)
    assert (
        transport.resolve_exact(
            profile="reviewer",
            title="Group: room-1",
            source=ROOM_SESSION_SOURCE,
        )
        is None
    )
    created = transport.create(
        profile="reviewer",
        title="Group: room-1",
        source=ROOM_SESSION_SOURCE,
    )
    assert created["session_id"] == "group-session"
    prepare = [params for method, params in client.calls if method == "prepare"]
    assert all(params["room_id"] == "room-1" for params in prepare)
    assert all(params["source"] == "bot_room" for params in prepare)


def test_peer_transport_dispatches_full_fenced_coordinates_and_exact_stop():
    client = FakePeerClient()
    transport = _transport(client)
    transport.create(
        profile="reviewer",
        title="Group: room-1",
        source=ROOM_SESSION_SOURCE,
    )
    terminal = []
    task = TaskIdentity("room-1", "task-1", "thread-1", "turn-1")
    result = transport.submit(
        profile="reviewer",
        session_id="group-session",
        prompt="Review this change.",
        source=ROOM_SESSION_SOURCE,
        task=task,
        execution_generation=3,
        on_terminal=terminal.append,
    )
    assert result["status"] == "accepted"
    dispatch = next(params for method, params in client.calls if method == "dispatch")
    assert dispatch["dispatch"]["authority_epoch"] == 2
    assert dispatch["dispatch"]["execution_generation"] == 3
    assert dispatch["dispatch"]["target_profile"] == "reviewer"
    assert dispatch["dispatch"]["capability_digest"] == "a" * 64
    assert terminal == []

    assert (
        transport.interrupt(
            profile="reviewer",
            session_id="group-session",
            source=ROOM_SESSION_SOURCE,
            expected_task_id="other-task",
        )
        is None
    )
    stopped = transport.interrupt(
        profile="reviewer",
        session_id="group-session",
        source=ROOM_SESSION_SOURCE,
        expected_task_id="task-1",
    )
    assert stopped["status"] == "cancelled"
    assert len([call for call in client.calls if call[0] == "stop"]) == 1


def test_peer_transport_carries_each_turns_real_source_event_sequence():
    observed = []
    for index, source_event_seq in enumerate((7, 42), start=1):
        client = FakePeerClient()
        transport = _transport(client, source_event_seq=source_event_seq)
        transport.create(
            profile="reviewer",
            title="Group: room-1",
            source=ROOM_SESSION_SOURCE,
        )
        transport.submit(
            profile="reviewer",
            session_id="group-session",
            prompt=f"Turn {index}",
            source=ROOM_SESSION_SOURCE,
            task=TaskIdentity("room-1", f"task-{index}", "thread-1", f"turn-{index}"),
            execution_generation=1,
            on_terminal=lambda _receipt: None,
        )
        dispatch = next(
            params for method, params in client.calls if method == "dispatch"
        )
        observed.append(dispatch["dispatch"]["source_event_seq"])
    assert observed == [7, 42]


def test_peer_transport_rejects_profile_source_and_room_title_mismatch():
    transport = _transport()
    for kwargs in (
        {"profile": "other", "title": "Group: room-1", "source": "bot_room"},
        {"profile": "reviewer", "title": "Bot Chat", "source": "bot_room"},
        {"profile": "reviewer", "title": "Group: room-1", "source": "cli"},
    ):
        try:
            transport.resolve_exact(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"mismatch was accepted: {kwargs}")


def test_roomlink_falls_back_to_relay_on_retryable_prepare_failure():
    direct = FailingPeerClient(method="prepare")
    relay = FakePeerClient()
    client = FailoverHostedRoomPeerClient([
        RoomLinkCandidate("direct", "direct", "install-peer", direct),
        RoomLinkCandidate("relay", "relay", "install-peer", relay),
    ])

    session = client.prepare(
        room_id="room-1",
        profile="reviewer",
        source="bot_room",
        grant="grant",
        create=True,
    )

    assert session["session_id"] == "group-session"
    assert client.active_link.name == "relay"


def test_roomlink_never_falls_back_after_ambiguous_direct_failure():
    direct = FailingPeerClient(method="dispatch")
    relay = FakePeerClient()
    client = FailoverHostedRoomPeerClient([
        RoomLinkCandidate("direct", "direct", "install-peer", direct),
        RoomLinkCandidate("relay", "relay", "install-peer", relay),
    ])
    dispatch = {"task_id": "task-1", "execution_generation": 1}

    try:
        client.dispatch(dispatch=dispatch, grant="grant")
    except PeerRunsHTTPError as exc:
        assert exc.ambiguous is True
    else:
        raise AssertionError("ambiguous dispatch was automatically replayed")

    assert direct.calls[0][1]["dispatch"] is dispatch
    assert relay.calls == []
    assert client.active_link.name == "direct"


def test_roomlink_falls_back_after_proven_not_admitted_direct_failure():
    direct = FailingPeerClient(method="dispatch", not_admitted=True)
    relay = FakePeerClient()
    client = FailoverHostedRoomPeerClient([
        RoomLinkCandidate("direct", "direct", "install-peer", direct),
        RoomLinkCandidate("relay", "relay", "install-peer", relay),
    ])
    dispatch = {"task_id": "task-1", "execution_generation": 1}

    result = client.dispatch(dispatch=dispatch, grant="grant")

    assert result["status"] == "accepted"
    assert direct.calls[0][1]["dispatch"] is dispatch
    assert relay.calls[0][1]["dispatch"] is dispatch
    assert client.active_link.name == "relay"


def test_roomlink_never_falls_back_after_nonretryable_rejection():
    rejected = FailingPeerClient(method="prepare", retryable=False)
    relay = FakePeerClient()
    client = FailoverHostedRoomPeerClient([
        RoomLinkCandidate("direct", "direct", "install-peer", rejected),
        RoomLinkCandidate("relay", "relay", "install-peer", relay),
    ])

    try:
        client.prepare(
            room_id="room-1",
            profile="reviewer",
            source="bot_room",
            grant="grant",
            create=True,
        )
    except PeerRunsHTTPError:
        pass
    else:
        raise AssertionError("nonretryable rejection was silently bypassed")
    assert relay.calls == []


def test_roomlink_reprobes_and_upgrades_back_to_primary_after_cooldown():
    now = [0.0]
    direct = FailingPeerClient(method="prepare")
    relay = FakePeerClient()
    client = FailoverHostedRoomPeerClient(
        [
            RoomLinkCandidate("direct", "direct", "install-peer", direct),
            RoomLinkCandidate("relay", "relay", "install-peer", relay),
        ],
        reprobe_interval_seconds=30,
        clock=lambda: now[0],
    )
    client.prepare(
        room_id="room-1",
        profile="reviewer",
        source="bot_room",
        grant="grant",
        create=True,
    )
    assert client.active_link.name == "relay"

    direct.method = "none"
    now[0] = 10
    client.prepare(
        room_id="room-1",
        profile="reviewer",
        source="bot_room",
        grant="grant",
        create=True,
    )
    assert client.active_link.name == "relay"

    now[0] = 31
    client.prepare(
        room_id="room-1",
        profile="reviewer",
        source="bot_room",
        grant="grant",
        create=True,
    )
    assert client.active_link.name == "direct"
