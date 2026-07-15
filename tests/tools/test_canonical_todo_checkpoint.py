import json

import pytest

from tools import canonical_brain_tool
from tools.todo_tool import (
    TODO_SCHEMA,
    TodoStore,
    TodoStoreFencedError,
    todo_tool,
)


def _checkpoint(**plan_overrides):
    plan = {
        "plan_id": "plan:workspace-1",
        "revision": 1,
        "objective": "Finish the exact approved task",
        "state": "active",
        "success_criteria": [
            {"id": "criterion:verified", "content": "Verification receipt exists"}
        ],
        "current_step_id": "step-2",
        "step_dependencies": {"step-2": ["step-1"]},
        "resume_cursor": {
            "summary": "Run the remaining verification",
            "next_step_id": "step-2",
        },
        "attempts": [],
        "decisions": [],
        "artifacts": [],
    }
    plan.update(plan_overrides)
    return {
        "case_id": "case:workspace-1",
        "summary": "Workspace checkpoint",
        "source_refs": {"platform": "discord", "message_id": "message-1"},
        "plan": plan,
        "idempotency_key": "workspace-checkpoint-1",
    }


def _verified_receipt(*, inserted=False, deduped=True):
    return {
        "success": True,
        "status": "CANONICAL_EVENT_APPEND_PASS",
        "event_id": "11111111-1111-4111-8111-111111111111",
        "event_type": "task.plan.updated",
        "case_id": "case:workspace-1",
        "idempotency_key": "workspace-checkpoint-1",
        "canonical_content_sha256": "a" * 64,
        "readback_verified": True,
        "inserted": inserted,
        "deduped": deduped,
    }


def test_schema_exposes_current_snapshot_checkpoint_without_model_authored_steps():
    schema = TODO_SCHEMA["parameters"]["properties"]["canonical_checkpoint"]

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["case_id", "summary", "source_refs", "plan"]
    assert "already bound plan" in schema["description"]
    assert "same call" in schema["description"]
    assert "steps" not in schema["properties"]["plan"]["properties"]
    assert "step_dependencies" in schema["properties"]["plan"]["properties"]


def test_checkpoint_attaches_current_exact_todos_and_requires_readback(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        canonical_brain_tool,
        "check_canonical_brain_requirements",
        lambda: True,
    )

    def _append(**kwargs):
        captured.update(kwargs)
        return json.dumps(
            {
                "success": True,
                "status": "CANONICAL_EVENT_APPEND_PASS",
                "event_id": "11111111-1111-4111-8111-111111111111",
                "event_type": "task.plan.updated",
                "case_id": "case:workspace-1",
                "idempotency_key": "workspace-checkpoint-1",
                "canonical_content_sha256": "a" * 64,
                "readback_verified": True,
                "inserted": True,
                "deduped": False,
            }
        )

    monkeypatch.setattr(canonical_brain_tool, "canonical_event_append_tool", _append)
    store = TodoStore()
    todos = [
        {"id": "step-1", "content": "Inspect", "status": "completed"},
        {"id": "step-2", "content": "Verify", "status": "in_progress"},
    ]
    todo_update = json.loads(todo_tool(todos=todos, store=store))
    assert todo_update["todos"] == todos
    result = json.loads(
        todo_tool(
            store=store,
            canonical_checkpoint=_checkpoint(),
        )
    )

    assert captured["event_type"] == "task.plan.updated"
    assert captured["payload"]["plan"]["steps"] == [
        {
            "id": "step-1",
            "content": "Inspect",
            "status": "completed",
            "depends_on": [],
        },
        {
            "id": "step-2",
            "content": "Verify",
            "status": "in_progress",
            "depends_on": ["step-1"],
        },
    ]
    assert result["todos"] == store.read()
    assert result["canonical_checkpoint"]["readback_verified"] is True
    assert result["canonical_checkpoint"]["event_id"].startswith("11111111")
    assert len(result["canonical_checkpoint"]["workspace_todos_sha256"]) == 64


def test_uncertain_checkpoint_retries_exact_idempotent_append_for_current_snapshot(
    monkeypatch,
):
    monkeypatch.setattr(
        canonical_brain_tool,
        "check_canonical_brain_requirements",
        lambda: True,
    )
    calls = []

    def _append(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return json.dumps(
                {
                    "success": False,
                    "status": "CANONICAL_EVENT_APPEND_READBACK_FAILED",
                    "readback_verified": False,
                    "write_may_have_occurred": True,
                    "inserted": True,
                }
            )
        return json.dumps(_verified_receipt())

    monkeypatch.setattr(
        canonical_brain_tool,
        "canonical_event_append_tool",
        _append,
    )
    store = TodoStore()
    original = [
        {"id": "old-step", "content": "Preserved", "status": "in_progress"}
    ]
    store.write(original)
    candidate = [
        {"id": "step-1", "content": "Inspect", "status": "completed"},
        {"id": "step-2", "content": "Verify", "status": "in_progress"},
    ]
    todo_tool(todos=candidate, store=store)

    result = json.loads(
        todo_tool(
            store=store,
            canonical_checkpoint=_checkpoint(),
        )
    )

    assert "error" not in result
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert result["canonical_checkpoint"]["canonical_reconciliation_attempts"] == 1
    assert result["canonical_checkpoint"]["deduped"] is True
    assert store.read() == result["todos"]
    assert store.read() != original


def test_missing_checkpoint_key_is_derived_once_and_reused_for_readback_retry(
    monkeypatch,
):
    monkeypatch.setattr(
        canonical_brain_tool,
        "check_canonical_brain_requirements",
        lambda: True,
    )
    calls = []

    def _append(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return json.dumps(
                {
                    "success": False,
                    "status": "CANONICAL_EVENT_APPEND_READBACK_FAILED",
                    "readback_verified": False,
                    "write_may_have_occurred": True,
                    "inserted": True,
                }
            )
        receipt = _verified_receipt()
        receipt["idempotency_key"] = kwargs["idempotency_key"]
        return json.dumps(receipt)

    monkeypatch.setattr(
        canonical_brain_tool,
        "canonical_event_append_tool",
        _append,
    )
    checkpoint = _checkpoint()
    checkpoint.pop("idempotency_key")
    store = TodoStore()
    todo_tool(
        todos=[
            {"id": "step-1", "content": "Inspect", "status": "completed"},
            {"id": "step-2", "content": "Verify", "status": "in_progress"},
        ],
        store=store,
    )

    result = json.loads(
        todo_tool(
            store=store,
            canonical_checkpoint=checkpoint,
        )
    )

    assert "error" not in result
    assert len(calls) == 2
    assert calls[0]["idempotency_key"].startswith("todo-checkpoint:")
    assert calls[0]["idempotency_key"] == calls[1]["idempotency_key"]


def test_repeated_uncertainty_fences_stale_local_state_until_exact_reconciliation(
    monkeypatch,
):
    monkeypatch.setattr(
        canonical_brain_tool,
        "check_canonical_brain_requirements",
        lambda: True,
    )
    calls = []

    def _append(**kwargs):
        calls.append(kwargs)
        if len(calls) <= 2:
            return json.dumps(
                {
                    "success": False,
                    "status": "CANONICAL_EVENT_APPEND_READBACK_FAILED",
                    "readback_verified": False,
                    "write_may_have_occurred": True,
                    "inserted": True,
                }
            )
        return json.dumps(_verified_receipt())

    monkeypatch.setattr(
        canonical_brain_tool,
        "canonical_event_append_tool",
        _append,
    )
    store = TodoStore()
    store.write([{"id": "old", "content": "Stale", "status": "in_progress"}])
    candidate = [
        {"id": "step-1", "content": "Inspect", "status": "completed"},
        {"id": "step-2", "content": "Verify", "status": "in_progress"},
    ]
    todo_tool(todos=candidate, store=store)

    failed = json.loads(
        todo_tool(
            store=store,
            canonical_checkpoint=_checkpoint(),
        )
    )

    assert failed["canonical_write_may_have_occurred"] is True
    assert failed["canonical_fence"]["status"] == (
        "canonical_reconciliation_required"
    )
    assert failed["todos"] == []
    assert failed["canonical_fence_state"]["items"] == candidate
    assert store.has_items() is True
    assert store.has_active_items() is True
    assert "Stale" not in (store.format_for_injection() or "")
    with pytest.raises(TodoStoreFencedError):
        store.read()
    with pytest.raises(TodoStoreFencedError):
        store.write(candidate)

    mismatched = json.loads(
        todo_tool(
            store=store,
            canonical_checkpoint=_checkpoint(objective="different"),
        )
    )
    assert mismatched["canonical_fence"]["status"] == (
        "canonical_reconciliation_required"
    )
    assert len(calls) == 2

    reconciled = json.loads(
        todo_tool(store=store, canonical_checkpoint=_checkpoint())
    )
    assert reconciled["canonical_reconciliation"]["resolved"] is True
    assert reconciled["canonical_checkpoint"]["readback_verified"] is True
    assert store.read() == candidate
    assert len(calls) == 3


def test_uncertain_fence_survives_fresh_agent_history_hydration(monkeypatch):
    from unittest.mock import patch

    from run_agent import AIAgent

    monkeypatch.setattr(
        canonical_brain_tool,
        "check_canonical_brain_requirements",
        lambda: True,
    )
    calls = []

    def _append(**kwargs):
        calls.append(kwargs)
        if len(calls) <= 2:
            return json.dumps(
                {
                    "success": False,
                    "status": "CANONICAL_EVENT_APPEND_READBACK_FAILED",
                    "readback_verified": False,
                    "write_may_have_occurred": True,
                    "inserted": True,
                }
            )
        return json.dumps(_verified_receipt())

    monkeypatch.setattr(
        canonical_brain_tool,
        "canonical_event_append_tool",
        _append,
    )
    candidate = [
        {"id": "step-1", "content": "Inspect", "status": "completed"},
        {"id": "step-2", "content": "Verify", "status": "in_progress"},
    ]
    original_store = TodoStore()
    original_store.write(
        [{"id": "old", "content": "Stale", "status": "in_progress"}]
    )
    todo_tool(todos=candidate, store=original_store)
    failed_json = todo_tool(
        store=original_store,
        canonical_checkpoint=_checkpoint(),
    )

    history = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "old-call",
                    "type": "function",
                    "function": {"name": "todo", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "content": json.dumps(
                {
                    "todos": [
                        {
                            "id": "old",
                            "content": "Stale",
                            "status": "in_progress",
                        }
                    ]
                }
            ),
            "tool_call_id": "old-call",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "fence-call",
                    "type": "function",
                    "function": {"name": "todo", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "content": failed_json,
            "tool_call_id": "fence-call",
        },
    ]
    fresh_agent = object.__new__(AIAgent)
    fresh_agent._todo_store = TodoStore()
    fresh_agent.session_id = "fresh-session"
    fresh_agent.quiet_mode = True
    with patch("run_agent._set_interrupt"):
        fresh_agent._hydrate_todo_store(history)

    assert fresh_agent._todo_store.has_items() is True
    assert fresh_agent._todo_store.has_active_items() is True
    with pytest.raises(TodoStoreFencedError):
        fresh_agent._todo_store.read()

    reconciled = json.loads(
        todo_tool(
            store=fresh_agent._todo_store,
            canonical_checkpoint=_checkpoint(),
        )
    )
    assert reconciled["canonical_reconciliation"]["resolved"] is True
    assert fresh_agent._todo_store.read() == candidate
    assert len(calls) == 3


def test_invalid_fence_history_tombstone_never_restores_older_snapshot(
    monkeypatch,
):
    from unittest.mock import patch

    from run_agent import AIAgent

    monkeypatch.setattr(
        canonical_brain_tool,
        "check_canonical_brain_requirements",
        lambda: True,
    )
    monkeypatch.setattr(
        canonical_brain_tool,
        "canonical_event_append_tool",
        lambda **_kwargs: json.dumps(
            {
                "success": False,
                "status": "CANONICAL_EVENT_APPEND_READBACK_FAILED",
                "readback_verified": False,
                "write_may_have_occurred": True,
                "inserted": True,
            }
        ),
    )
    store = TodoStore()
    todo_tool(
        todos=[
            {"id": "new", "content": "Candidate", "status": "in_progress"}
        ],
        store=store,
    )
    failed = json.loads(
        todo_tool(
            store=store,
            canonical_checkpoint=_checkpoint(
                current_step_id="new",
                step_dependencies={},
                resume_cursor={"summary": "Candidate", "next_step_id": "new"},
            ),
        )
    )
    failed["canonical_fence_state"]["checkpoint_sha256"] = "0" * 64
    history = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "old-call",
                    "type": "function",
                    "function": {"name": "todo", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "content": json.dumps(
                {
                    "todos": [
                        {"id": "old", "content": "Stale", "status": "pending"}
                    ]
                }
            ),
            "tool_call_id": "old-call",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "fence-call",
                    "type": "function",
                    "function": {"name": "todo", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "content": json.dumps(failed),
            "tool_call_id": "fence-call",
        },
    ]
    fresh_agent = object.__new__(AIAgent)
    fresh_agent._todo_store = TodoStore()
    fresh_agent.session_id = "fresh-session"
    fresh_agent.quiet_mode = True
    with patch("run_agent._set_interrupt"):
        fresh_agent._hydrate_todo_store(history)

    assert fresh_agent._todo_store.read() == []


def test_fenced_reconciliation_reuses_first_turn_observed_source_refs(monkeypatch):
    from gateway.session_context import clear_session_vars, set_session_vars

    monkeypatch.setattr(
        canonical_brain_tool,
        "check_canonical_brain_requirements",
        lambda: True,
    )
    calls = []

    def _append(**kwargs):
        calls.append(kwargs)
        if len(calls) <= 2:
            return json.dumps(
                {
                    "success": False,
                    "status": "CANONICAL_EVENT_APPEND_READBACK_FAILED",
                    "readback_verified": False,
                    "write_may_have_occurred": True,
                    "inserted": True,
                }
            )
        return json.dumps(_verified_receipt())

    monkeypatch.setattr(
        canonical_brain_tool,
        "canonical_event_append_tool",
        _append,
    )
    checkpoint = _checkpoint()
    checkpoint["source_refs"] = {"platform": "discord"}
    candidate = [
        {"id": "step-1", "content": "Inspect", "status": "completed"},
        {"id": "step-2", "content": "Verify", "status": "in_progress"},
    ]
    store = TodoStore()
    todo_tool(todos=candidate, store=store)

    first_turn = set_session_vars(
        platform="discord",
        user_id="owner",
        session_key="discord:channel:thread",
        message_id="message-first",
    )
    try:
        failed = json.loads(
            todo_tool(
                store=store,
                canonical_checkpoint=checkpoint,
            )
        )
    finally:
        clear_session_vars(first_turn)
    assert failed["canonical_write_may_have_occurred"] is True

    second_turn = set_session_vars(
        platform="discord",
        user_id="owner",
        session_key="discord:channel:thread",
        message_id="message-second",
    )
    try:
        reconciled = json.loads(
            todo_tool(store=store, canonical_checkpoint=checkpoint)
        )
    finally:
        clear_session_vars(second_turn)

    assert reconciled["canonical_reconciliation"]["resolved"] is True
    assert [call["source_refs"]["message_id"] for call in calls] == [
        "message-first",
        "message-first",
        "message-first",
    ]
    assert len({call["idempotency_key"] for call in calls}) == 1


def test_proven_no_write_checkpoint_failure_preserves_current_local_snapshot(
    monkeypatch,
):
    monkeypatch.setattr(
        canonical_brain_tool,
        "check_canonical_brain_requirements",
        lambda: True,
    )
    monkeypatch.setattr(
        canonical_brain_tool,
        "canonical_event_append_tool",
        lambda **_kwargs: json.dumps(
            {
                "success": False,
                "status": "CANONICAL_EVENT_APPEND_BLOCKED",
                "readback_verified": False,
                "write_may_have_occurred": False,
                "inserted": False,
            }
        ),
    )
    store = TodoStore()
    current = [{"id": "new", "content": "New", "status": "pending"}]
    store.write(current)

    result = json.loads(
        todo_tool(
            store=store,
            canonical_checkpoint=_checkpoint(
                current_step_id="new",
                step_dependencies={},
                resume_cursor={"summary": "New", "next_step_id": "new"},
            ),
        )
    )

    assert result["todo_update_applied"] is False
    assert result["todo_preserved"] is True
    assert result["canonical_write_may_have_occurred"] is False
    assert store.canonical_fence_receipt() is None
    assert store.read() == current


def test_todos_and_delivery_control_are_rejected_before_dispatch():
    store = TodoStore()
    original = [{"id": "old", "content": "Keep", "status": "in_progress"}]
    store.write(original)

    recorded = []
    result = json.loads(
        todo_tool(
            todos=[{"id": "new", "content": "New", "status": "pending"}],
            store=store,
            delivery_outcome={"action": "deliver", "reason": "done"},
            delivery_outcome_recorder=lambda directive: recorded.append(directive),
        )
    )

    assert "separate todo tool calls" in result["error"]
    assert result["todo_update_applied"] is False
    assert result["control_side_effect_applied"] is False
    assert recorded == []
    assert store.read() == original


def test_todos_and_plan_approval_are_rejected_before_dispatch(monkeypatch):
    from tools import approval

    called = []

    def _grant(**kwargs):
        called.append(kwargs)
        return {"granted": True}

    monkeypatch.setattr(
        approval,
        "grant_plan_capability",
        _grant,
    )
    store = TodoStore()
    original = [{"id": "old", "content": "Keep", "status": "in_progress"}]
    store.write(original)

    result = json.loads(
        todo_tool(
            todos=[{"id": "new", "content": "New", "status": "pending"}],
            store=store,
            plan_approval={
                "plan_id": "plan:workspace-1",
                "plan_revision": 1,
                "exact_commands": ["git status"],
                "ttl_seconds": 60,
                "max_uses_per_command": 1,
            },
        )
    )

    assert "separate todo tool calls" in result["error"]
    assert result["todo_update_applied"] is False
    assert result["control_side_effect_applied"] is False
    assert called == []
    assert store.read() == original


def test_checkpoint_rejects_dependency_keys_outside_current_todo_snapshot(monkeypatch):
    monkeypatch.setattr(
        canonical_brain_tool,
        "check_canonical_brain_requirements",
        lambda: True,
    )
    store = TodoStore()
    current = [{"id": "step-2", "content": "Verify", "status": "in_progress"}]
    store.write(current)

    result = json.loads(
        todo_tool(
            store=store,
            canonical_checkpoint=_checkpoint(
                current_step_id="step-2",
                step_dependencies={"missing-step": []},
            ),
        )
    )

    assert "unknown todo ids:missing-step" in result["error"]
    assert store.read() == current


def test_checkpoint_rejects_other_todo_side_effects_before_any_write(monkeypatch):
    called = []
    monkeypatch.setattr(
        canonical_brain_tool,
        "canonical_event_append_tool",
        lambda **kwargs: called.append(kwargs),
    )
    store = TodoStore()
    original = [{"id": "old", "content": "Keep", "status": "pending"}]
    store.write(original)

    result = json.loads(
        todo_tool(
            todos=[{"id": "new", "content": "Do", "status": "in_progress"}],
            store=store,
            canonical_checkpoint=_checkpoint(
                current_step_id="new",
                step_dependencies={},
                resume_cursor={"summary": "Do", "next_step_id": "new"},
            ),
            goal_outcome={"status": "continue", "reason": "working"},
        )
    )

    assert "separate todo tool calls" in result["error"]
    assert result["control_side_effect_applied"] is False
    assert store.read() == original
    assert called == []


def _install_verified_checkpoint_backend(monkeypatch, calls):
    monkeypatch.setattr(
        canonical_brain_tool,
        "check_canonical_brain_requirements",
        lambda: True,
    )

    def _append(**kwargs):
        calls.append(kwargs)
        receipt = _verified_receipt(inserted=True, deduped=False)
        receipt["case_id"] = kwargs["case_id"]
        receipt["idempotency_key"] = kwargs["idempotency_key"]
        receipt["event_id"] = (
            f"{len(calls):08d}-1111-4111-8111-111111111111"
        )
        return json.dumps(receipt)

    monkeypatch.setattr(
        canonical_brain_tool,
        "canonical_event_append_tool",
        _append,
    )


def test_verified_active_checkpoint_binds_and_forces_atomic_later_updates(
    monkeypatch,
):
    calls = []
    _install_verified_checkpoint_backend(monkeypatch, calls)
    store = TodoStore()
    original = [
        {"id": "step-1", "content": "Inspect", "status": "completed"},
        {"id": "step-2", "content": "Verify", "status": "in_progress"},
    ]
    store.write(original)

    first = json.loads(
        todo_tool(store=store, canonical_checkpoint=_checkpoint())
    )
    binding = first["canonical_binding_state"]

    assert binding["case_id"] == "case:workspace-1"
    assert binding["plan_id"] == "plan:workspace-1"
    assert binding["plan_revision"] == 1
    assert binding["plan_state"] == "active"
    assert binding["plan_event_id"].startswith("00000001")
    assert first["canonical_sync"]["state"] == "clean"
    assert first["canonical_sync"]["durable_completion_verified"] is True

    candidate = [
        {"id": "step-1", "content": "Inspect", "status": "completed"},
        {"id": "step-2", "content": "Verify", "status": "completed"},
    ]
    rejected = json.loads(todo_tool(todos=candidate, store=store))
    assert rejected["canonical_checkpoint_required"] is True
    assert rejected["todo_update_applied"] is False
    assert store.read() == original
    with pytest.raises(TodoStoreFencedError):
        store.write(candidate)


def test_bound_todos_and_checkpoint_advance_canonical_then_local_atomically(
    monkeypatch,
):
    calls = []
    _install_verified_checkpoint_backend(monkeypatch, calls)
    store = TodoStore()
    original = [
        {"id": "step-1", "content": "Inspect", "status": "completed"},
        {"id": "step-2", "content": "Verify", "status": "in_progress"},
    ]
    store.write(original)
    json.loads(todo_tool(store=store, canonical_checkpoint=_checkpoint()))

    candidate = [
        {"id": "step-1", "content": "Inspect", "status": "completed"},
        {"id": "step-2", "content": "Verify", "status": "completed"},
    ]
    checkpoint = _checkpoint(
        revision=2,
        current_step_id="step-2",
        resume_cursor={"summary": "Close verification", "next_step_id": "step-2"},
    )
    checkpoint["idempotency_key"] = "workspace-checkpoint-2"
    advanced = json.loads(
        todo_tool(
            todos=candidate,
            store=store,
            canonical_checkpoint=checkpoint,
        )
    )

    assert "error" not in advanced
    assert len(calls) == 2
    assert calls[1]["payload"]["plan"]["steps"] == [
        {**candidate[0], "depends_on": []},
        {**candidate[1], "depends_on": ["step-1"]},
    ]
    assert store.read() == candidate
    assert advanced["canonical_binding_state"]["plan_revision"] == 2
    assert advanced["canonical_sync"]["state"] == "clean"
    assert (
        advanced["canonical_checkpoint"]["todo_items_sha256"]
        == advanced["canonical_binding_state"]["todo_items_sha256"]
    )


def test_verified_terminal_checkpoint_releases_active_binding(monkeypatch):
    calls = []
    _install_verified_checkpoint_backend(monkeypatch, calls)
    store = TodoStore()
    active = [{"id": "step-1", "content": "Finish", "status": "in_progress"}]
    store.write(active)
    initial = _checkpoint(
        current_step_id="step-1",
        step_dependencies={},
        resume_cursor={"summary": "Finish", "next_step_id": "step-1"},
    )
    json.loads(todo_tool(store=store, canonical_checkpoint=initial))

    completed = [{"id": "step-1", "content": "Finish", "status": "completed"}]
    terminal = _checkpoint(
        revision=2,
        state="completed",
        current_step_id="step-1",
        step_dependencies={},
        resume_cursor={"summary": "Complete", "next_step_id": ""},
    )
    terminal["idempotency_key"] = "workspace-checkpoint-terminal"
    result = json.loads(
        todo_tool(
            todos=completed,
            store=store,
            canonical_checkpoint=terminal,
        )
    )

    assert "canonical_binding_state" not in result
    assert "canonical_sync" not in result
    assert store.canonical_binding_state() is None
    assert store.read() == completed
    assert json.loads(
        todo_tool(
            todos=[
                {"id": "new-plan", "content": "New draft", "status": "pending"}
            ],
            store=store,
        )
    )["todos"][0]["id"] == "new-plan"


def test_verified_cancelled_checkpoint_releases_only_terminal_workspace(
    monkeypatch,
):
    from unittest.mock import patch

    from run_agent import AIAgent

    calls = []
    _install_verified_checkpoint_backend(monkeypatch, calls)
    store = TodoStore()
    active = [{"id": "step-1", "content": "Stop safely", "status": "in_progress"}]
    store.write(active)
    initial = _checkpoint(
        current_step_id="step-1",
        step_dependencies={},
        resume_cursor={"summary": "Stop safely", "next_step_id": "step-1"},
    )
    json.loads(todo_tool(store=store, canonical_checkpoint=initial))

    cancelled = [{"id": "step-1", "content": "Stop safely", "status": "cancelled"}]
    terminal = _checkpoint(
        revision=2,
        state="cancelled",
        current_step_id="",
        step_dependencies={},
        resume_cursor={"summary": "Cancelled by the model", "next_step_id": ""},
    )
    terminal["idempotency_key"] = "workspace-checkpoint-cancelled"
    result_json = todo_tool(
        todos=cancelled,
        store=store,
        canonical_checkpoint=terminal,
    )
    result = json.loads(result_json)

    assert "error" not in result
    assert "canonical_binding_state" not in result
    assert "canonical_sync" not in result
    assert store.canonical_binding_state() is None
    assert store.read() == cancelled
    assert store.has_active_items() is False

    history = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "cancelled-call",
                    "type": "function",
                    "function": {"name": "todo", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "content": result_json,
            "tool_call_id": "cancelled-call",
        },
    ]
    fresh_agent = object.__new__(AIAgent)
    fresh_agent._todo_store = TodoStore()
    fresh_agent.session_id = "fresh-cancelled-session"
    fresh_agent.quiet_mode = True
    with patch("run_agent._set_interrupt"):
        fresh_agent._hydrate_todo_store(history)

    assert fresh_agent._todo_store.read() == cancelled
    assert fresh_agent._todo_store.canonical_binding_state() is None
    assert fresh_agent._todo_store.has_active_items() is False


def test_plan_transition_content_conflict_blocks_as_canonical_truth_diverged(
    monkeypatch,
):
    from unittest.mock import patch

    from run_agent import AIAgent

    calls = []
    _install_verified_checkpoint_backend(monkeypatch, calls)
    store = TodoStore()
    original = [{"id": "step-1", "content": "Follow truth", "status": "in_progress"}]
    store.write(original)
    initial = _checkpoint(
        current_step_id="step-1",
        step_dependencies={},
        resume_cursor={"summary": "Follow truth", "next_step_id": "step-1"},
    )
    json.loads(todo_tool(store=store, canonical_checkpoint=initial))
    original_binding = store.canonical_binding_state()

    conflict_calls = []

    def _conflict(**kwargs):
        conflict_calls.append(kwargs)
        return json.dumps(
            {
                "success": False,
                "status": "CANONICAL_EVENT_APPEND_IDEMPOTENCY_CONFLICT",
                "error": "idempotency key already exists with different canonical content",
                "readback_verified": False,
                "write_may_have_occurred": False,
                "inserted": False,
                "deduped": True,
            }
        )

    monkeypatch.setattr(
        canonical_brain_tool,
        "canonical_event_append_tool",
        _conflict,
    )
    candidate = [{"id": "step-1", "content": "Follow truth", "status": "completed"}]
    conflicting_checkpoint = _checkpoint(
        revision=2,
        state="completed",
        current_step_id="",
        step_dependencies={},
        resume_cursor={"summary": "Candidate complete", "next_step_id": ""},
    )
    conflicting_checkpoint["idempotency_key"] = "workspace-checkpoint-conflict"
    blocked_json = todo_tool(
        todos=candidate,
        store=store,
        canonical_checkpoint=conflicting_checkpoint,
    )
    blocked = json.loads(blocked_json)

    assert len(conflict_calls) == 1
    assert blocked["canonical_sync_blocked"] is True
    assert blocked["canonical_sync_error_code"] == "canonical_truth_diverged"
    assert blocked["canonical_write_may_have_occurred"] is False
    assert blocked["canonical_sync"] == {
        "state": "sync_blocked",
        "error_code": "canonical_truth_diverged",
        "current_todos_sha256": blocked["canonical_sync"]["current_todos_sha256"],
        "canonical_retry_required": True,
        "durable_completion_verified": False,
    }
    assert store.read() == original
    assert store.canonical_binding_state() == original_binding
    assert store.has_active_items() is False
    injected = store.format_for_injection() or ""
    assert "truth diverged" in injected
    assert "different content" in injected
    assert "Writer was proven unavailable" not in injected

    history = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "conflict-call",
                    "type": "function",
                    "function": {"name": "todo", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "content": blocked_json,
            "tool_call_id": "conflict-call",
        },
    ]
    fresh_agent = object.__new__(AIAgent)
    fresh_agent._todo_store = TodoStore()
    fresh_agent.session_id = "fresh-conflict-session"
    fresh_agent.quiet_mode = True
    with patch("run_agent._set_interrupt"):
        fresh_agent._hydrate_todo_store(history)

    restored = fresh_agent._todo_store
    assert restored.read() == original
    assert restored.canonical_binding_state() == original_binding
    assert restored.canonical_sync_receipt()["error_code"] == (
        "canonical_truth_diverged"
    )
    assert restored.has_active_items() is False

    _install_verified_checkpoint_backend(monkeypatch, calls)
    authoritative = [
        {"id": "step-1", "content": "Follow authoritative truth", "status": "in_progress"}
    ]
    corrected = _checkpoint(
        revision=3,
        current_step_id="step-1",
        step_dependencies={},
        resume_cursor={
            "summary": "Continue from authoritative revision",
            "next_step_id": "step-1",
        },
    )
    corrected["idempotency_key"] = "workspace-checkpoint-corrected"
    recovered = json.loads(
        todo_tool(
            todos=authoritative,
            store=restored,
            canonical_checkpoint=corrected,
        )
    )

    assert "error" not in recovered
    assert restored.read() == authoritative
    assert restored.canonical_sync_blocked_state() is None
    assert restored.canonical_sync_receipt()["state"] == "clean"


def test_conflict_status_with_uncertain_write_envelope_retries_then_fences(
    monkeypatch,
):
    monkeypatch.setattr(
        canonical_brain_tool,
        "check_canonical_brain_requirements",
        lambda: True,
    )
    calls = []

    def _uncertain_conflict(**kwargs):
        calls.append(kwargs)
        return json.dumps(
            {
                "success": False,
                "status": "CANONICAL_EVENT_APPEND_IDEMPOTENCY_CONFLICT",
                "error": "malformed conflict envelope",
                "readback_verified": False,
                "write_may_have_occurred": True,
                "inserted": True,
                "deduped": False,
            }
        )

    monkeypatch.setattr(
        canonical_brain_tool,
        "canonical_event_append_tool",
        _uncertain_conflict,
    )
    store = TodoStore()
    candidate = [{"id": "step-1", "content": "Keep safe", "status": "in_progress"}]
    store.write(candidate)
    checkpoint = _checkpoint(
        current_step_id="step-1",
        step_dependencies={},
        resume_cursor={"summary": "Keep safe", "next_step_id": "step-1"},
    )

    result = json.loads(
        todo_tool(store=store, canonical_checkpoint=checkpoint)
    )

    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert result["canonical_write_may_have_occurred"] is True
    assert result["canonical_fence"]["status"] == (
        "canonical_reconciliation_required"
    )
    assert "canonical_sync_blocked_state" not in result
    with pytest.raises(TodoStoreFencedError):
        store.read()


def test_exact_reconciliation_conflict_replaces_retry_fence_with_truth_block(
    monkeypatch,
):
    monkeypatch.setattr(
        canonical_brain_tool,
        "check_canonical_brain_requirements",
        lambda: True,
    )
    calls = []

    def _append(**kwargs):
        calls.append(kwargs)
        if len(calls) <= 2:
            return json.dumps(
                {
                    "success": False,
                    "status": "CANONICAL_EVENT_APPEND_READBACK_FAILED",
                    "readback_verified": False,
                    "write_may_have_occurred": True,
                    "inserted": True,
                }
            )
        return json.dumps(
            {
                "success": False,
                "status": "CANONICAL_EVENT_APPEND_IDEMPOTENCY_CONFLICT",
                "error": "idempotency key already exists with different canonical content",
                "readback_verified": False,
                "write_may_have_occurred": False,
                "inserted": False,
                "deduped": True,
            }
        )

    monkeypatch.setattr(
        canonical_brain_tool,
        "canonical_event_append_tool",
        _append,
    )
    store = TodoStore()
    candidate = [{"id": "step-1", "content": "Candidate", "status": "in_progress"}]
    store.write(candidate)
    checkpoint = _checkpoint(
        current_step_id="step-1",
        step_dependencies={},
        resume_cursor={"summary": "Candidate", "next_step_id": "step-1"},
    )

    first = json.loads(
        todo_tool(store=store, canonical_checkpoint=checkpoint)
    )
    assert first["canonical_fence"]["status"] == (
        "canonical_reconciliation_required"
    )

    reconciled = json.loads(
        todo_tool(store=store, canonical_checkpoint=checkpoint)
    )

    assert len(calls) == 3
    assert reconciled["canonical_sync_error_code"] == "canonical_truth_diverged"
    assert reconciled["canonical_sync"]["state"] == "sync_blocked"
    assert reconciled["canonical_sync"]["error_code"] == (
        "canonical_truth_diverged"
    )
    assert reconciled["todos"] == []
    assert "canonical_fence" not in reconciled
    assert store.read() == []
    assert store.has_active_items() is False


def test_writer_unavailable_preserves_binding_and_candidate_until_exact_retry(
    monkeypatch,
):
    calls = []
    _install_verified_checkpoint_backend(monkeypatch, calls)
    store = TodoStore()
    original = [{"id": "step-1", "content": "Finish", "status": "in_progress"}]
    store.write(original)
    first_checkpoint = _checkpoint(
        current_step_id="step-1",
        step_dependencies={},
        resume_cursor={"summary": "Finish", "next_step_id": "step-1"},
    )
    json.loads(todo_tool(store=store, canonical_checkpoint=first_checkpoint))
    original_binding = store.canonical_binding_state()

    monkeypatch.setattr(
        canonical_brain_tool,
        "check_canonical_brain_requirements",
        lambda: False,
    )
    candidate = [{"id": "step-1", "content": "Finish", "status": "completed"}]
    retry_checkpoint = _checkpoint(
        revision=2,
        state="completed",
        current_step_id="step-1",
        step_dependencies={},
        resume_cursor={"summary": "Complete", "next_step_id": ""},
    )
    retry_checkpoint["idempotency_key"] = "workspace-checkpoint-retry"
    blocked = json.loads(
        todo_tool(
            todos=candidate,
            store=store,
            canonical_checkpoint=retry_checkpoint,
        )
    )

    assert blocked["canonical_sync_blocked"] is True
    assert blocked["canonical_sync"]["state"] == "sync_blocked"
    assert store.read() == original
    assert store.canonical_binding_state() == original_binding
    assert store.has_active_items() is False

    monkeypatch.setattr(
        canonical_brain_tool,
        "check_canonical_brain_requirements",
        lambda: True,
    )
    malformed = json.loads(
        todo_tool(
            store=store,
            canonical_checkpoint={"case_id": "case:workspace-1"},
        )
    )
    assert "error" in malformed
    assert store.canonical_sync_blocked_state() is not None
    assert store.read() == original

    _install_verified_checkpoint_backend(monkeypatch, calls)
    recovered = json.loads(
        todo_tool(
            todos=candidate,
            store=store,
            canonical_checkpoint=retry_checkpoint,
        )
    )
    assert "error" not in recovered
    assert store.read() == candidate
    assert store.canonical_sync_blocked_state() is None
    assert store.canonical_binding_state() is None


def test_bound_atomic_uncertainty_fences_then_installs_exact_candidate(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        canonical_brain_tool,
        "check_canonical_brain_requirements",
        lambda: True,
    )

    def _append(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1 or len(calls) == 4:
            receipt = _verified_receipt(inserted=True, deduped=False)
            receipt["idempotency_key"] = kwargs["idempotency_key"]
            receipt["event_id"] = (
                f"{len(calls):08d}-1111-4111-8111-111111111111"
            )
            return json.dumps(receipt)
        return json.dumps(
            {
                "success": False,
                "status": "CANONICAL_EVENT_APPEND_READBACK_FAILED",
                "readback_verified": False,
                "write_may_have_occurred": True,
                "inserted": True,
            }
        )

    monkeypatch.setattr(
        canonical_brain_tool,
        "canonical_event_append_tool",
        _append,
    )
    store = TodoStore()
    original = [{"id": "step-1", "content": "Finish", "status": "in_progress"}]
    store.write(original)
    first_checkpoint = _checkpoint(
        current_step_id="step-1",
        step_dependencies={},
        resume_cursor={"summary": "Finish", "next_step_id": "step-1"},
    )
    json.loads(todo_tool(store=store, canonical_checkpoint=first_checkpoint))

    candidate = [{"id": "step-1", "content": "Finish", "status": "completed"}]
    terminal = _checkpoint(
        revision=2,
        state="completed",
        current_step_id="step-1",
        step_dependencies={},
        resume_cursor={"summary": "Complete", "next_step_id": ""},
    )
    terminal["idempotency_key"] = "workspace-checkpoint-uncertain"
    failed = json.loads(
        todo_tool(
            todos=candidate,
            store=store,
            canonical_checkpoint=terminal,
        )
    )

    assert failed["canonical_fence"]["status"] == (
        "canonical_reconciliation_required"
    )
    assert failed["canonical_fence_state"]["items"] == candidate
    with pytest.raises(TodoStoreFencedError):
        store.read()

    reconciled = json.loads(
        todo_tool(store=store, canonical_checkpoint=terminal)
    )
    assert reconciled["canonical_reconciliation"]["resolved"] is True
    assert store.read() == candidate
    assert store.canonical_binding_state() is None
    assert calls[1] == calls[2] == calls[3]


def test_real_writer_timeout_path_never_degrades_to_proven_no_write(
    monkeypatch,
):
    from gateway.canonical_writer_client import CanonicalWriterClientError
    from gateway.canonical_writer_protocol import ErrorCode

    monkeypatch.setattr(
        canonical_brain_tool,
        "check_canonical_brain_requirements",
        lambda: True,
    )
    calls = []

    def _timeout(*_args, **_kwargs):
        calls.append(True)
        raise CanonicalWriterClientError(
            ErrorCode.TIMEOUT,
            "Canonical writer request timed out.",
            retryable=True,
            write_may_have_occurred=True,
        )

    monkeypatch.setattr(
        canonical_brain_tool,
        "_writer_proxy_result",
        _timeout,
    )
    store = TodoStore()
    original = [{"id": "step-1", "content": "Keep", "status": "pending"}]
    store.write(original)
    candidate = [{"id": "step-1", "content": "Keep", "status": "in_progress"}]
    checkpoint = _checkpoint(
        state="active",
        current_step_id="step-1",
        step_dependencies={},
        resume_cursor={"summary": "Keep", "next_step_id": "step-1"},
    )

    result = json.loads(
        todo_tool(
            todos=candidate,
            store=store,
            canonical_checkpoint=checkpoint,
        )
    )

    assert len(calls) == 2
    assert result["canonical_write_may_have_occurred"] is True
    assert result["canonical_fence"]["status"] == (
        "canonical_reconciliation_required"
    )
    assert result["canonical_fence_state"]["items"] == candidate
    with pytest.raises(TodoStoreFencedError):
        store.read()


def test_real_writer_connect_failure_becomes_typed_sync_block(monkeypatch):
    from gateway.canonical_writer_client import CanonicalWriterClientError
    from gateway.canonical_writer_protocol import ErrorCode

    monkeypatch.setattr(
        canonical_brain_tool,
        "check_canonical_brain_requirements",
        lambda: True,
    )

    def _connect_failed(*_args, **_kwargs):
        raise CanonicalWriterClientError(
            ErrorCode.TRANSPORT_ERROR,
            "Canonical writer transport failed.",
            retryable=True,
            write_may_have_occurred=False,
        )

    monkeypatch.setattr(
        canonical_brain_tool,
        "_writer_proxy_result",
        _connect_failed,
    )
    store = TodoStore()
    original = [{"id": "step-1", "content": "Keep", "status": "pending"}]
    store.write(original)
    candidate = [{"id": "step-1", "content": "Keep", "status": "in_progress"}]
    checkpoint = _checkpoint(
        current_step_id="step-1",
        step_dependencies={},
        resume_cursor={"summary": "Keep", "next_step_id": "step-1"},
    )

    result = json.loads(
        todo_tool(
            todos=candidate,
            store=store,
            canonical_checkpoint=checkpoint,
        )
    )

    assert result["canonical_sync_blocked"] is True
    assert result["canonical_writer_error_code"] == "transport_error"
    assert result["canonical_sync"]["state"] == "sync_blocked"
    assert result["todo_update_applied"] is False
    assert store.read() == original
    assert store.has_active_items() is False


def test_missing_writer_certainty_is_unknown_and_fences_after_exact_retry(
    monkeypatch,
):
    monkeypatch.setattr(
        canonical_brain_tool,
        "check_canonical_brain_requirements",
        lambda: True,
    )
    calls = []

    def _legacy_error(**_kwargs):
        calls.append(True)
        return json.dumps(
            {
                "error": "transport failed without certainty",
                "status": "CANONICAL_EVENT_APPEND_FAIL",
            }
        )

    monkeypatch.setattr(
        canonical_brain_tool,
        "canonical_event_append_tool",
        _legacy_error,
    )
    store = TodoStore()
    candidate = [{"id": "step-1", "content": "Keep", "status": "in_progress"}]
    store.write(candidate)
    checkpoint = _checkpoint(
        current_step_id="step-1",
        step_dependencies={},
        resume_cursor={"summary": "Keep", "next_step_id": "step-1"},
    )

    result = json.loads(
        todo_tool(store=store, canonical_checkpoint=checkpoint)
    )

    assert len(calls) == 2
    assert result["canonical_fence"]["status"] == (
        "canonical_reconciliation_required"
    )
    with pytest.raises(TodoStoreFencedError):
        store.read()
