"""Preflight evidence must never delete a successor's accepted remote input."""

from contextlib import nullcontext

import pytest

from gateway.hosted_room_driver import TaskIdentity
from tests.tui_gateway.test_hosted_room_peer_transport import (
    BINDING,
    ROUTE,
    FailingPeerClient,
)
from tui_gateway.hosted_room_driver import HostedRoomRuntime, ROOM_SESSION_SOURCE
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
from tui_gateway.hosted_room_peer_transport import PeerHostedRoomTransport


@pytest.fixture
def staged():
    client = FailingPeerClient(method="dispatch", not_admitted=False)
    client.error.dispatch_not_attempted = True
    task = TaskIdentity("room-1", "task-files", "thread-1", "turn-files")
    transport = PeerHostedRoomTransport(
        binding=BINDING,
        route=ROUTE,
        client=client,
        task_id=task.task_id,
        execution_generation=3,
    )
    common = dict(
        profile="reviewer",
        session_id="group-session",
        source=ROOM_SESSION_SOURCE,
        execution_generation=3,
    )
    transport.create(
        profile="reviewer", title="Group: room-1", source=ROOM_SESSION_SOURCE
    )
    transport.begin_attachment_staging(**common)
    transport.stage_attachment(
        **common,
        attachment={
            "attachment_id": "att_11111111111111111111111111111111",
            "kind": "file",
            "name": "brief.txt",
            "size": 5,
            "mime": "text/plain",
        },
        data=b"brief",
    )
    with pytest.raises(PeerRunsHTTPError):
        transport.submit(
            **common, prompt="Review the file", task=task, on_terminal=lambda *_: None
        )
    assert not any(method == "discard_attachments" for method, _ in client.calls)
    return transport, client, task, common


@pytest.mark.parametrize("proven", [True, False])
def test_driver_finalization_drops_local_bytes_without_retiring_remote_input(
    tmp_path, staged, proven
):
    transport, client, _, common = staged
    runtime = HostedRoomRuntime(
        db_path=tmp_path / "state.db",
        rooms=[],
        rpc=transport,
        turn_lock=lambda _profile: nullcontext(),
    )
    runtime._finish_attachment_staging_after_error(
        transport=transport,
        profile=common["profile"],
        session_id=common["session_id"],
        execution_generation=3,
        submit_attempted=True,
        not_admitted=proven,
    )
    discarded = [
        data for method, data in client.calls if method == "discard_attachments"
    ]
    assert discarded == []
    assert transport._attachment_attempt is None
    assert transport._pending_attachments == []


def test_wrong_generation_or_profile_cannot_retire_the_current_batch(staged):
    transport, client, _, common = staged
    transport.rollback_attachment_staging(**{**common, "execution_generation": 4})
    with pytest.raises(ValueError):
        transport.rollback_attachment_staging(**{**common, "profile": "another"})
    assert not any(method == "discard_attachments" for method, _ in client.calls)


def test_another_task_cannot_retire_the_staged_dispatch(staged):
    transport, client, _, common = staged
    transport.task_id = "another-task"
    transport.rollback_attachment_staging(**common)
    assert not any(method == "discard_attachments" for method, _ in client.calls)


def test_duplicate_rollback_does_not_send_another_discard(staged):
    transport, client, _, common = staged
    transport.rollback_attachment_staging(**common)
    transport.rollback_attachment_staging(**common)
    assert not any(method == "discard_attachments" for method, _ in client.calls)
