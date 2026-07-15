import copy
import itertools
import json

import pytest

from hermes_cli import goals
from tools import approval, canonical_brain_tool
from tools.todo_tool import TODO_SCHEMA, TodoStore, todo_tool


CONTROL_ORDER = (
    "canonical_checkpoint",
    "plan_approval",
    "goal_outcome",
    "goal_contract",
    "delivery_outcome",
)

CONTROL_PAYLOADS = {
    "canonical_checkpoint": {
        "case_id": "case:atomicity",
        "summary": "Exact current snapshot",
        "source_refs": {"platform": "discord", "message_id": "message-1"},
        "plan": {
            "plan_id": "plan:atomicity",
            "revision": 1,
            "objective": "Prove one control side effect per call",
            "state": "active",
            "success_criteria": [
                {"id": "criterion:atomic", "content": "No partial controls"}
            ],
            "current_step_id": "step-1",
            "step_dependencies": {},
            "resume_cursor": {
                "summary": "Run the isolated control call",
                "next_step_id": "step-1",
            },
        },
        "idempotency_key": "todo-control-atomicity",
    },
    "plan_approval": {
        "plan_id": "plan:atomicity",
        "plan_revision": 1,
        "exact_commands": ["git status"],
        "ttl_seconds": 60,
        "max_uses_per_command": 1,
    },
    "goal_outcome": {"status": "continue", "reason": "Work remains"},
    "goal_contract": {
        "outcome": "Finish the approved task",
        "verification": "A receipt exists",
    },
    "delivery_outcome": {"action": "deliver", "reason": "Report is ready"},
}


def _install_control_spies(monkeypatch):
    effects = []

    monkeypatch.setattr(
        canonical_brain_tool,
        "canonical_event_append_tool",
        lambda **_kwargs: effects.append("canonical_checkpoint") or "{}",
    )
    monkeypatch.setattr(
        approval,
        "grant_plan_capability",
        lambda **_kwargs: effects.append("plan_approval") or {"state": "granted"},
    )

    class _GoalManager:
        def __init__(self, _session_id):
            pass

        def record_model_outcome(self, *_args, **_kwargs):
            effects.append("goal_outcome")
            return True

        def record_model_contract(self, *_args, **_kwargs):
            effects.append("goal_contract")
            return True

    monkeypatch.setattr(goals, "GoalManager", _GoalManager)

    def _record_delivery(_directive):
        effects.append("delivery_outcome")
        return {"recorded": True}

    return effects, _record_delivery


def _invoke(store, controls, delivery_recorder):
    return json.loads(
        todo_tool(
            store=store,
            originating_turn_id="turn-1",
            goal_generation_id="goal-generation-1",
            delivery_outcome_recorder=delivery_recorder,
            **{
                name: copy.deepcopy(CONTROL_PAYLOADS[name])
                for name in controls
            },
        )
    )


def test_schema_instructs_the_model_to_use_isolated_control_calls():
    description = TODO_SCHEMA["description"]
    checkpoint = TODO_SCHEMA["parameters"]["properties"][
        "canonical_checkpoint"
    ]["description"]

    assert "exactly one control field per todo call" in description
    assert "Never combine todos with a control field except canonical_checkpoint" in description
    assert "in the SAME call" in description
    assert "already bound plan" in checkpoint
    assert "same call" in checkpoint


@pytest.mark.parametrize(
    ("first", "second"),
    itertools.combinations(CONTROL_ORDER, 2),
)
def test_every_control_pair_is_rejected_before_any_side_effect(
    monkeypatch,
    first,
    second,
):
    effects, delivery_recorder = _install_control_spies(monkeypatch)
    store = TodoStore()
    original = [{"id": "old", "content": "Keep", "status": "pending"}]
    store.write(original)

    result = _invoke(store, {first, second}, delivery_recorder)

    assert "separate todo tool calls" in result["error"]
    assert result["requested_control_fields"] == [
        name for name in CONTROL_ORDER if name in {first, second}
    ]
    assert result["todo_update_applied"] is False
    assert result["control_side_effect_applied"] is False
    assert effects == []
    assert store.read() == original


@pytest.mark.parametrize(
    "control_name",
    [name for name in CONTROL_ORDER if name != "canonical_checkpoint"],
)
def test_todos_and_each_control_are_rejected_before_any_write(
    monkeypatch,
    control_name,
):
    effects, delivery_recorder = _install_control_spies(monkeypatch)
    store = TodoStore()
    original = [{"id": "old", "content": "Keep", "status": "pending"}]
    store.write(original)

    result = json.loads(
        todo_tool(
            todos=[{"id": "new", "content": "Do", "status": "in_progress"}],
            store=store,
            originating_turn_id="turn-1",
            goal_generation_id="goal-generation-1",
            delivery_outcome_recorder=delivery_recorder,
            **{control_name: copy.deepcopy(CONTROL_PAYLOADS[control_name])},
        )
    )

    assert "only canonical_checkpoint may accompany todos" in result["error"]
    assert result["requested_control_fields"] == [control_name]
    assert result["todo_update_applied"] is False
    assert result["control_side_effect_applied"] is False
    assert effects == []
    assert store.read() == original


def test_todos_and_checkpoint_form_one_readback_verified_update(monkeypatch):
    effects = []
    monkeypatch.setattr(
        canonical_brain_tool,
        "check_canonical_brain_requirements",
        lambda: True,
    )

    def _append(**kwargs):
        effects.append(kwargs)
        return json.dumps(
            {
                "success": True,
                "status": "CANONICAL_EVENT_APPEND_PASS",
                "event_id": "11111111-1111-4111-8111-111111111111",
                "event_type": "task.plan.updated",
                "case_id": "case:atomicity",
                "idempotency_key": "todo-control-atomicity",
                "canonical_content_sha256": "a" * 64,
                "readback_verified": True,
                "inserted": True,
                "deduped": False,
            }
        )

    monkeypatch.setattr(
        canonical_brain_tool,
        "canonical_event_append_tool",
        _append,
    )
    store = TodoStore()
    original = [{"id": "old", "content": "Keep", "status": "pending"}]
    store.write(original)
    candidate = [{"id": "new", "content": "Do", "status": "in_progress"}]
    checkpoint = copy.deepcopy(CONTROL_PAYLOADS["canonical_checkpoint"])
    checkpoint["plan"]["current_step_id"] = "new"

    result = json.loads(
        todo_tool(
            todos=candidate,
            store=store,
            canonical_checkpoint=checkpoint,
        )
    )

    assert "error" not in result
    assert len(effects) == 1
    assert effects[0]["payload"]["plan"]["steps"][0]["id"] == "new"
    assert store.read() == candidate
    assert result["canonical_sync"]["state"] == "clean"


def test_later_goal_failure_cannot_leave_an_earlier_plan_capability(monkeypatch):
    effects = []
    monkeypatch.setattr(
        approval,
        "grant_plan_capability",
        lambda **_kwargs: effects.append("plan_approval") or {"state": "granted"},
    )

    class _FailingGoalManager:
        def __init__(self, _session_id):
            pass

        def record_model_outcome(self, *_args, **_kwargs):
            raise RuntimeError("downstream goal failure")

    monkeypatch.setattr(goals, "GoalManager", _FailingGoalManager)

    result = _invoke(
        TodoStore(),
        {"plan_approval", "goal_outcome"},
        lambda _directive: {"recorded": True},
    )

    assert "separate todo tool calls" in result["error"]
    assert effects == []

def test_later_goal_contract_failure_cannot_leave_an_earlier_goal_outcome(
    monkeypatch,
):
    effects = []

    class _PartiallyFailingGoalManager:
        def __init__(self, _session_id):
            pass

        def record_model_outcome(self, *_args, **_kwargs):
            effects.append("goal_outcome")
            return True

        def record_model_contract(self, *_args, **_kwargs):
            raise RuntimeError("downstream contract failure")

    monkeypatch.setattr(goals, "GoalManager", _PartiallyFailingGoalManager)

    result = _invoke(
        TodoStore(),
        {"goal_outcome", "goal_contract"},
        lambda _directive: {"recorded": True},
    )

    assert "separate todo tool calls" in result["error"]
    assert effects == []


def test_downstream_todo_write_failure_cannot_follow_a_delivery_side_effect(
    monkeypatch,
):
    effects = []
    store = TodoStore()
    store.write([{"id": "old", "content": "Keep", "status": "pending"}])

    def _fail_if_reached(*_args, **_kwargs):
        raise RuntimeError("downstream todo write failure")

    monkeypatch.setattr(store, "write", _fail_if_reached)

    result = json.loads(
        todo_tool(
            todos=[{"id": "new", "content": "Do", "status": "in_progress"}],
            store=store,
            delivery_outcome=copy.deepcopy(CONTROL_PAYLOADS["delivery_outcome"]),
            delivery_outcome_recorder=lambda _directive: effects.append(
                "delivery_outcome"
            ),
        )
    )

    assert "separate todo tool calls" in result["error"]
    assert effects == []
