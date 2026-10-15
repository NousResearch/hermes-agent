"""The driver must name the member on every submit it hands to a transport.

`HostedRoomServerRPC.submit` requires `member_id` as a keyword-only argument
(`tui_gateway/hosted_room_server_rpc.py`): one in-process RPC instance serves
every member of the gateway. A driver that calls `submit` without it raises
`TypeError`, marks the attempt ambiguous and leaves the task `running`. The
shared `FakeSessionRPC` defaults `member_id` to `""`, which is how a submit
that never names the member stayed green here.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from gateway import hosted_room_driver as state
from gateway import hosted_rooms
from tests.tui_gateway.test_hosted_room_driver_runtime import (
    BINDING,
    FakeSessionRPC,
    PROFILE,
    ROOM_ID,
    _identity,
    _runtime,
    _wait_for,
)


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "state.db"
    hosted_rooms.create_room(
        path,
        room_id=ROOM_ID,
        name="Release room",
        members=[{"profile": PROFILE, "handle": PROFILE}],
        authority_gateway_id=BINDING.gateway_id,
        now=time.time(),
    )
    return path


class MemberIdRequiringRPC(FakeSessionRPC):
    """`FakeSessionRPC` behind the real in-process RPC's required keyword."""

    def __init__(self) -> None:
        super().__init__()
        self.member_ids: list[str] = []

    def submit(  # type: ignore[override]  # the real RPC has no default for member_id
        self,
        *,
        profile: str,
        session_id: str,
        prompt: str,
        source: str,
        task,
        execution_generation: int,
        on_terminal,
        member_id: str,
        attachments=None,
    ):
        self.member_ids.append(member_id)
        return super().submit(
            profile=profile,
            session_id=session_id,
            prompt=prompt,
            source=source,
            task=task,
            execution_generation=execution_generation,
            on_terminal=on_terminal,
        )


def test_submit_carries_the_target_member_id_the_transport_requires(db: Path):
    identity = _identity()
    state.admit_task(
        db,
        identity,
        payload={
            "target_profile": PROFILE,
            "target_member_id": "member-ops",
            "prompt": "Inspect the release candidate.",
            "source_event_seq": 1,
        },
        clock=time.time,
    )
    rpc = MemberIdRequiringRPC()
    runtime = _runtime(db, rpc)

    runtime.start()
    _wait_for(lambda: state.get_task(db, identity)["status"] == "settled")
    assert runtime.stop(timeout=5.0)

    assert rpc.member_ids == ["member-ops"]
    assert state.get_task(db, identity)["result"]["text"] == "Finished once."


def test_driver_submit_settles_through_the_real_in_process_rpc(db: Path):
    from tests.tui_gateway.test_hosted_room_server_rpc import _server
    from tui_gateway.hosted_room_server_rpc import HostedRoomServerRPC

    server, calls = _server()

    def prompt_submit(rid, params):
        calls.append(("prompt.submit", params))
        hosted = params["_hosted_task"]
        # The real handler settles from its background turn, after submit returns.
        threading.Timer(0.05, params["_hosted_terminal_callback"], args=[{
            "status": "settled", "settlement_id": "reply-1", "message_id": "reply-1",
            "text": "Finished once.", "task_id": hosted["task_id"],
            "execution_generation": hosted["execution_generation"],
        }]).start()
        return {"id": rid, "result": {"status": "streaming"}}

    server._methods["prompt.submit"] = prompt_submit
    identity = _identity()
    state.admit_task(
        db,
        identity,
        payload={
            "target_profile": PROFILE,
            "target_member_id": "member-ops",
            "prompt": "Inspect the release candidate.",
            "source_event_seq": 1,
        },
        clock=time.time,
    )
    runtime = _runtime(db, HostedRoomServerRPC(server))

    runtime.start()
    try:
        _wait_for(lambda: state.get_task(db, identity)["status"] == "settled")
    finally:
        assert runtime.stop(timeout=5.0)

    submits = [params for method, params in calls if method == "prompt.submit"]
    assert [params["_hosted_task"]["member_id"] for params in submits] == ["member-ops"]
    assert state.get_task(db, identity)["result"]["text"] == "Finished once."
