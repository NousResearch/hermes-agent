"""Artifact publication coordinates only enter a genuine internal room submit."""

import pytest

from tui_gateway import server


def proof():
    return {
        "room_id": "room-1", "task_id": "task-1", "thread_id": "thread-1",
        "turn_id": "turn-1", "execution_generation": 1, "member_id": "member-1",
        "target_profile": "reviewer", "home_install_id": "home",
        "target_install_id": "target", "authority_gateway_id": "home",
        "authority_epoch": 1,
    }


def test_complete_internal_artifact_proof_is_accepted():
    assert server._hosted_submit_error(1, {"source": "bot_room"}, proof(), lambda value: None) is None


@pytest.mark.parametrize("field,value", [
    ("authority_epoch", True), ("authority_epoch", 0),
    ("execution_generation", False), ("execution_generation", -1),
    ("member_id", ""), ("target_profile", None), ("home_install_id", 4),
])
def test_invalid_coordinates_do_not_admit_an_artifact_scope(field, value):
    result = server._hosted_submit_error(
        1, {"source": "bot_room"}, {**proof(), field: value}, lambda value: None,
    )
    assert result["error"]["code"] == 4120


@pytest.mark.parametrize("change", ["missing", "extra", "no_callback", "non_room"])
def test_scope_requires_exact_fields_callback_and_room_session(change):
    task = proof()
    if change == "missing":
        task.pop("member_id")
    if change == "extra":
        task["untrusted"] = "value"
    result = server._hosted_submit_error(
        1, {"source": "cli" if change == "non_room" else "bot_room"}, task,
        None if change == "no_callback" else lambda value: None,
    )
    assert result["error"]["code"] == 4120
