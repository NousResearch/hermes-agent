"""Terminal receipts carry a turn's shared-file metadata exactly, or not at all."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from gateway import hosted_room_driver as state
from gateway import hosted_rooms
from gateway.hosted_room_artifacts import terminal_artifact_manifest
from tui_gateway.hosted_room_driver import (
    HostedRoomBinding,
    HostedRoomRuntime,
    _bounded_terminal_result,
    _find_terminal_receipt,
)

BINDING = HostedRoomBinding(room_id="room-1", gateway_id="gateway-a", authority_epoch=1)
IDENTITY = state.TaskIdentity(room_id="room-1", task_id="dtask:abc", thread_id="thread-1", turn_id="turn-1")
SCOPE = {"room_id": "room-1", "task_id": "dtask:abc", "execution_generation": 1, "member_id": "ops",
         "target_profile": "ops", "home_install_id": "gateway-a", "target_install_id": "gateway-a",
         "authority_gateway_id": "gateway-a", "authority_epoch": 1}
ITEM = {"artifact_id": "rart_" + "a" * 32, "kind": "file", "name": "plan.md", "size": 5,
        "mime": "text/markdown", "sha256": "b" * 64}


def _output():
    return {"artifacts": terminal_artifact_manifest([ITEM]), "artifact_scope": dict(SCOPE)}


def test_a_bounded_receipt_keeps_exact_output_metadata():
    result = _bounded_terminal_result({"message_id": "m", "text": "done", **_output()})
    assert result == {"message_id": "m", "text": "done", **_output()}
    assert "artifacts" not in _bounded_terminal_result({"message_id": "m", "text": "done"})


@pytest.mark.parametrize("damage", ["digest", "scope", "fields"])
def test_damaged_output_metadata_is_dropped_so_the_turn_still_settles(damage):
    output = _output()
    if damage == "digest":
        output["artifacts"]["items"][0]["name"] = "other.md"
    elif damage == "scope":
        output["artifact_scope"]["authority_epoch"] = 0
    else:
        output["artifacts"]["extra"] = True
    result = _bounded_terminal_result({"message_id": "m", "text": "done", **output})
    assert result == {"message_id": "m", "text": "done"}


def test_history_receipts_carry_the_output_of_their_exact_attempt():
    history = [
        {"role": "assistant", "status": "settled", "task_id": "dtask:abc", "execution_generation": 1,
         "message_id": "admission-1", "content": "done", **_output()},
        {"role": "assistant", "status": "settled", "task_id": "dtask:abc", "execution_generation": 2,
         "message_id": "admission-2", "content": "newer", **_output()},
    ]
    receipt = _find_terminal_receipt(history, IDENTITY, 1)
    assert receipt.result["artifacts"] == _output()["artifacts"]
    assert receipt.result["artifact_scope"] == SCOPE


def test_a_late_receipt_for_an_attempt_that_cannot_publish_is_handed_to_retirement(tmp_path: Path):
    db = tmp_path / "state.db"
    hosted_rooms.create_room(db, room_id="room-1", name="Room", members=[{"profile": "ops", "handle": "ops"}],
                             authority_gateway_id=BINDING.gateway_id, now=time.time())
    state.admit_task(db, IDENTITY, payload={"target_profile": "ops", "prompt": "go", "source_event_seq": 1},
                     clock=time.time)
    runtime = HostedRoomRuntime(db_path=db, rooms=[BINDING], rpc=object(), turn_lock=lambda profile: None)
    retired = []
    runtime.retire_stale_output = lambda binding, task, generation, result: retired.append(
        (task["status"], generation, result["artifact_scope"]))
    lease = runtime._ensure_lease(BINDING)
    attempt = state.start_task(db, IDENTITY, lease, expected_cancel_generation=0, clock=time.time)
    stopping = state.begin_task_cancel(db, IDENTITY, cancel_id="stop", expected_cancel_generation=0, clock=time.time)
    state.complete_task_cancel(db, IDENTITY, cancel_id="stop", expected_cancel_generation=stopping["cancel_generation"],
                               clock=time.time)
    runtime._on_terminal(BINDING, attempt, {"status": "settled", "settlement_id": "late", "text": "late",
                                            **_output()})
    assert retired == [("cancelled", 1, SCOPE)]
    runtime._on_terminal(BINDING, attempt, {"status": "settled", "settlement_id": "late-text", "text": "late"})
    assert len(retired) == 1  # a text-only late receipt has nothing to retire
