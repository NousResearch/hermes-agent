"""task_commit admission coverage for Core-owned completion-gate identity."""

from __future__ import annotations

import json

import pytest

from hermes_cli.goals import GoalManager
from tools.task_commit_tool import task_commit


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))


def _call(operation: str, session_id: str = "s", **kwargs):
    if operation in {"create", "replace"}:
        kwargs.setdefault("outcome", "artifact exists")
        kwargs.setdefault("verification", "targeted tests pass")
    return json.loads(task_commit(operation=operation, session_id=session_id, **kwargs))


def test_create_persists_completion_gate():
    result = _call(
        "create",
        objective="ship verified work",
        outcome="artifact exists",
        verification="targeted tests pass",
        completion_gate="root-repair",
    )

    assert result["success"] is True
    assert result["goal"]["completion_gate"] == "root-repair"
    assert GoalManager("s").state.completion_gate == "root-repair"


def test_amend_preserves_gate_when_omitted_and_can_change_it():
    _call("create", objective="first", completion_gate="root-repair")

    preserved = _call("amend")
    assert preserved["goal"]["completion_gate"] == "root-repair"

    changed = _call("amend", completion_gate="other.gate")
    assert changed["goal"]["completion_gate"] == "other.gate"


def test_replace_without_gate_clears_previous_opt_in():
    _call("create", objective="first", completion_gate="root-repair")

    replaced = _call("replace", objective="second")

    assert replaced["goal"]["completion_gate"] == ""


def test_invalid_completion_gate_is_rejected_without_state_mutation():
    result = _call("create", objective="first", completion_gate="Bad Gate!")

    assert result["success"] is False
    assert "completion_gate must match" in result["error"]
    assert GoalManager("s").state is None
