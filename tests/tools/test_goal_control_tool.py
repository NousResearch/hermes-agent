"""Model-callable goal control at the real registry dispatch seam."""

import asyncio
import json
import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import goals
from model_tools import get_tool_definitions, handle_function_call


def _call(action: str, *, session_id: str, **kwargs):
    raw = handle_function_call(
        "goal_control",
        {"action": action, **kwargs},
        session_id=session_id,
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    )
    return json.loads(raw)


def _home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    goals._DB_BOOTSTRAP_INFLIGHT.clear()
    goals._GOAL_GENERATIONS.clear()
    return home


def test_readback_receipt_is_stable_for_a_revision_and_rotates_on_mutation(
    tmp_path, monkeypatch
):
    _home(tmp_path, monkeypatch)

    created = _call(
        "set",
        session_id="session-current",
        condition="Run checks",
        max_turns=3,
    )
    status = _call("status", session_id="session-current")
    paused = _call("pause", session_id="session-current")

    assert created["goal_readback"]["receipt_id"] == (
        "goal_control:session-current:revision:1"
    )
    assert status["goal_readback"] == created["goal_readback"]
    assert paused["goal_readback"]["revision"] == 2
    assert paused["goal_readback"]["receipt_id"] == (
        "goal_control:session-current:revision:2"
    )


def test_acceptance_evidence_update_omit_clear_and_replace_semantics(
    tmp_path, monkeypatch
):
    _home(tmp_path, monkeypatch)
    original = [
        {"kind": "file_exists", "locator": "report.json", "assertion": "exists"}
    ]
    replacement = [
        {"kind": "test_result", "locator": "goal tests", "assertion": "passes"}
    ]

    created = _call(
        "set",
        session_id="session-current",
        condition="Finish when report.json exists",
        acceptance_evidence=original,
    )
    preserved = _call(
        "update",
        session_id="session-current",
        condition="Still finish when report.json exists",
    )
    cleared = _call(
        "update",
        session_id="session-current",
        condition="Continue without structured checks",
        acceptance_evidence=[],
    )
    replaced = _call(
        "update",
        session_id="session-current",
        condition="Finish when goal tests passes",
        acceptance_evidence=replacement,
    )

    assert created["goal_readback"]["acceptance_evidence"] == original
    assert preserved["goal_readback"]["acceptance_evidence"] == original
    assert cleared["goal_readback"]["acceptance_evidence"] == []
    assert replaced["goal_readback"]["acceptance_evidence"] == replacement
    assert goals.load_goal_authoritative("session-current").acceptance_evidence == (
        replacement
    )


def test_update_rejects_preserved_evidence_missing_from_new_condition(
    tmp_path, monkeypatch
):
    _home(tmp_path, monkeypatch)
    _call(
        "set",
        session_id="session-current",
        condition="Finish when report.json exists",
        acceptance_evidence=[
            {"kind": "file_exists", "locator": "report.json", "assertion": "exists"}
        ],
    )
    before = goals.load_goal_authoritative("session-current").to_json()

    result = _call(
        "update",
        session_id="session-current",
        condition="Finish the report",
    )

    assert result["success"] is False
    assert result["error"]["code"] == "invalid_acceptance_evidence"
    assert goals.load_goal_authoritative("session-current").to_json() == before


@pytest.mark.parametrize(
    "evidence",
    [
        None,
        [{"kind": "invented", "locator": "report.json", "assertion": "exists"}],
        [{"kind": "file_exists", "locator": "report.json", "assertion": "passes"}],
        [{"kind": "file_exists", "locator": "", "assertion": "exists"}],
        [{"kind": "file_exists", "locator": "report.json", "assertion": "exists", "extra": "no"}],
    ],
)
def test_set_rejects_malformed_or_unsupported_acceptance_evidence(
    tmp_path, monkeypatch, evidence
):
    _home(tmp_path, monkeypatch)

    result = _call(
        "set",
        session_id="session-current",
        condition="Finish when report.json exists",
        acceptance_evidence=evidence,
    )

    assert result["success"] is False
    assert result["error"]["code"] == "invalid_acceptance_evidence"
    assert goals.load_goal("session-current") is None


def test_evidence_locator_and_assertion_must_be_named_in_condition(
    tmp_path, monkeypatch
):
    _home(tmp_path, monkeypatch)

    result = _call(
        "set",
        session_id="session-current",
        condition="Finish the run",
        acceptance_evidence=[
            {"kind": "file_exists", "locator": "report.json", "assertion": "exists"}
        ],
    )

    assert result["success"] is False
    assert result["error"]["code"] == "invalid_acceptance_evidence"
    assert goals.load_goal("session-current") is None


def test_set_returns_authoritative_persisted_current_session_state(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)

    result = _call(
        "set",
        session_id="session-current",
        condition="Run the focused checks",
        max_turns=7,
    )

    assert result["success"] is True
    assert result["session_id"] == "session-current"
    assert result["state"] == {
        "exists": True,
        "active": True,
        "paused": False,
        "status": "active",
        "condition": "Run the focused checks",
        "turns_used": 0,
        "max_turns": 7,
        "revision": 1,
        "stop_reason": None,
        "error_reason": None,
    }
    persisted = goals.load_goal("session-current")
    assert persisted is not None
    assert persisted.goal == "Run the focused checks"
    assert persisted.max_turns == 7


def test_goal_control_is_exposed_on_the_core_model_surface(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)

    definitions = get_tool_definitions(
        enabled_toolsets=["hermes-cli"],
        quiet_mode=True,
    )

    names = {item["function"]["name"] for item in definitions}
    assert "goal_control" in names


def test_goal_control_can_be_enabled_as_a_narrow_toolset(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)

    definitions = get_tool_definitions(
        enabled_toolsets=["goal"],
        quiet_mode=True,
        skip_tool_search_assembly=True,
    )

    assert {item["function"]["name"] for item in definitions} == {"goal_control"}


def test_set_uses_configured_goal_budget_when_omitted(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    (home / "config.yaml").write_text("goals:\n  max_turns: 11\n", encoding="utf-8")

    result = _call("set", session_id="session-current", condition="Configured budget")

    assert result["success"] is True
    assert result["state"]["max_turns"] == 11


def test_set_rejects_budget_above_configured_limit(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    (home / "config.yaml").write_text("goals:\n  max_turns: 5\n", encoding="utf-8")

    result = _call(
        "set",
        session_id="session-current",
        condition="Too much work",
        max_turns=6,
    )

    assert result["success"] is False
    assert result["error"]["code"] == "invalid_budget"
    assert goals.load_goal("session-current") is None


@pytest.mark.parametrize("existing_status", ["active", "paused", "done", "cleared"])
def test_set_does_not_replace_existing_goal(tmp_path, monkeypatch, existing_status):
    _home(tmp_path, monkeypatch)
    original = goals.GoalState(
        goal="User-owned goal",
        status=existing_status,
        turns_used=3,
        max_turns=8,
    )
    goals.save_goal("session-current", original)

    result = _call(
        "set",
        session_id="session-current",
        condition="Replacement",
        max_turns=5,
    )

    assert result["success"] is False
    assert result["error"]["code"] == "invalid_transition"
    assert goals.load_goal("session-current").to_json() == original.to_json()


def test_status_requires_calling_session_identity():
    result = _call("status", session_id="")

    assert result == {
        "success": False,
        "session_id": None,
        "error": {
            "code": "missing_session_identity",
            "message": "calling session identity is required",
        },
    }


def test_cross_session_argument_is_rejected_without_mutation(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    raw = handle_function_call(
        "goal_control",
        {"action": "set", "condition": "wrong target", "session_id": "session-other"},
        session_id="session-current",
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    )

    result = json.loads(raw)
    assert result["success"] is False
    assert result["error"]["code"] == "cross_session_forbidden"
    assert goals.load_goal("session-current") is None
    assert goals.load_goal("session-other") is None


def test_pause_resume_and_clear_return_persisted_state(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    _call("set", session_id="session-current", condition="Keep going", max_turns=4)

    paused = _call("pause", session_id="session-current")
    assert paused["success"] is True
    assert paused["state"]["paused"] is True
    assert paused["state"]["stop_reason"] == "model-paused"
    assert _call("pause", session_id="session-current")["error"]["code"] == (
        "invalid_transition"
    )

    resumed = _call("resume", session_id="session-current")
    assert resumed["success"] is True
    assert resumed["state"]["active"] is True
    assert resumed["state"]["turns_used"] == 0
    assert _call("resume", session_id="session-current")["error"]["code"] == (
        "invalid_transition"
    )

    for _ in range(2):
        cleared = _call("clear", session_id="session-current")
        assert cleared["success"] is True
        assert cleared["state"]["status"] == "cleared"
        assert cleared["state"]["active"] is False

    persisted = goals.load_goal("session-current")
    assert persisted is not None
    assert persisted.status == "cleared"


@pytest.mark.parametrize(
    ("lifecycle_status", "paused_reason", "last_verdict", "last_reason"),
    [
        ("paused", "operator-paused", None, "earlier reason"),
        ("active", None, "continue", "keep working"),
    ],
)
def test_clear_persists_fresh_terminal_reason(
    tmp_path,
    monkeypatch,
    lifecycle_status,
    paused_reason,
    last_verdict,
    last_reason,
):
    _home(tmp_path, monkeypatch)
    goals.save_goal(
        "session-current",
        goals.GoalState(
            goal="Goal with history",
            status=lifecycle_status,
            turns_used=3,
            max_turns=8,
            paused_reason=paused_reason,
            last_verdict=last_verdict,
            last_reason=last_reason,
            consecutive_transport_failures=1,
        ),
    )

    cleared = _call("clear", session_id="session-current")

    assert cleared["success"] is True
    assert cleared["state"]["status"] == "cleared"
    assert cleared["state"]["stop_reason"] == "model-cleared"
    assert cleared["state"]["error_reason"] is None
    persisted = goals.load_goal("session-current")
    assert persisted is not None
    assert persisted.paused_reason is None
    assert persisted.last_reason == "model-cleared"


def test_resume_preserves_consumed_budget(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    goals.save_goal(
        "session-current",
        goals.GoalState(
            goal="Paused work",
            status="paused",
            turns_used=3,
            max_turns=8,
            paused_reason="model-paused",
        ),
    )

    resumed = _call("resume", session_id="session-current")

    assert resumed["success"] is True
    assert resumed["state"]["active"] is True
    assert resumed["state"]["turns_used"] == 3
    assert resumed["state"]["max_turns"] == 8


def test_model_pause_updates_cached_cli_lifecycle_manager(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    from cli import HermesCLI

    session_id = "session-current"
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = session_id
    cli.agent = MagicMock(session_id=session_id)
    cli._pending_input = queue.Queue()
    cli._last_turn_interrupted = False
    cli.conversation_history = [
        {"role": "assistant", "content": "work completed this turn"}
    ]
    cli._goal_manager = goals.GoalManager(session_id=session_id, default_max_turns=4)
    cli._goal_manager.set("Keep going")

    paused = _call("pause", session_id=session_id)
    assert paused["success"] is True

    with patch("hermes_cli.goals.judge_goal") as judge:
        cli._maybe_continue_goal_after_turn()

    judge.assert_not_called()
    assert cli._pending_input.empty()
    assert cli._goal_manager.state.status == "paused"
    assert goals.load_goal(session_id).status == "paused"


def test_resume_does_not_reactivate_cleared_goal(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    _call("set", session_id="session-current", condition="Terminal goal")
    _call("clear", session_id="session-current")

    resumed = _call("resume", session_id="session-current")

    assert resumed["success"] is False
    assert resumed["error"]["code"] == "invalid_transition"
    assert goals.load_goal("session-current").status == "cleared"


@pytest.mark.parametrize("terminal_status", ["cleared", "done"])
def test_pause_does_not_reactivate_terminal_goal(
    tmp_path, monkeypatch, terminal_status
):
    _home(tmp_path, monkeypatch)
    goals.save_goal(
        "session-current",
        goals.GoalState(goal="Terminal goal", status=terminal_status),
    )

    paused = _call("pause", session_id="session-current")

    assert paused["success"] is False
    assert paused["error"]["code"] == "invalid_transition"
    assert goals.load_goal("session-current").status == terminal_status


def test_failed_cli_refresh_does_not_execute_stale_goal(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    from cli import HermesCLI

    session_id = "session-current"
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = session_id
    cli.agent = MagicMock(session_id=session_id)
    cli._pending_input = queue.Queue()
    cli._last_turn_interrupted = False
    cli.conversation_history = [
        {"role": "assistant", "content": "work completed this turn"}
    ]
    cli._goal_manager = goals.GoalManager(session_id=session_id, default_max_turns=4)
    cli._goal_manager.set("Keep going")

    paused = _call("pause", session_id=session_id)
    assert paused["success"] is True

    def _failed_readback(_session_id):
        raise RuntimeError("storage temporarily unavailable")

    monkeypatch.setattr(goals, "load_goal_authoritative", _failed_readback)
    with patch(
        "hermes_cli.goals.judge_goal",
        return_value=("continue", "more work", False, None, False),
    ) as judge:
        cli._maybe_continue_goal_after_turn()

    judge.assert_not_called()
    assert cli._pending_input.empty()
    assert goals.load_goal(session_id).status == "paused"


def test_update_replaces_condition_and_decreases_budget(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    _call("set", session_id="session-current", condition="First", max_turns=9)

    updated = _call(
        "update",
        session_id="session-current",
        condition="Second",
        max_turns=2,
    )

    assert updated["success"] is True
    assert updated["state"]["condition"] == "Second"
    assert updated["state"]["max_turns"] == 2


def test_update_rejects_budget_increase(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    original = goals.GoalState(
        goal="Bounded goal",
        status="active",
        turns_used=1,
        max_turns=2,
    )
    goals.save_goal("session-current", original)

    updated = _call(
        "update",
        session_id="session-current",
        condition="Still bounded",
        max_turns=3,
    )

    assert updated["success"] is False
    assert updated["error"]["code"] == "invalid_budget"
    assert goals.load_goal("session-current").to_json() == original.to_json()


@pytest.mark.parametrize(
    ("lifecycle_status", "paused_reason"),
    [("active", None), ("paused", "operator-paused")],
)
def test_update_preserves_lifecycle_progress_and_omitted_budget(
    tmp_path, monkeypatch, lifecycle_status, paused_reason
):
    _home(tmp_path, monkeypatch)
    goals.save_goal(
        "session-current",
        goals.GoalState(
            goal="First",
            status=lifecycle_status,
            turns_used=3,
            max_turns=8,
            paused_reason=paused_reason,
        ),
    )

    updated = _call(
        "update",
        session_id="session-current",
        condition="Second",
    )

    assert updated["success"] is True
    assert updated["state"]["condition"] == "Second"
    assert updated["state"]["status"] == lifecycle_status
    assert updated["state"]["turns_used"] == 3
    assert updated["state"]["max_turns"] == 8
    assert updated["state"]["stop_reason"] == paused_reason


@pytest.mark.parametrize("terminal_status", ["cleared", "done"])
def test_update_does_not_reactivate_terminal_goal(
    tmp_path, monkeypatch, terminal_status
):
    _home(tmp_path, monkeypatch)
    original = goals.GoalState(
        goal="Terminal goal",
        status=terminal_status,
        turns_used=3,
        max_turns=8,
    )
    goals.save_goal("session-current", original)

    updated = _call(
        "update",
        session_id="session-current",
        condition="Replacement",
        max_turns=9,
    )

    assert updated["success"] is False
    assert updated["error"]["code"] == "invalid_transition"
    assert goals.load_goal("session-current").to_json() == original.to_json()


def test_status_returns_stop_and_error_reasons_from_persisted_state(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    state = goals.GoalState(
        goal="Repair judge routing",
        status="paused",
        turns_used=3,
        max_turns=8,
        paused_reason="judge unavailable",
        last_reason="transport timeout",
        consecutive_transport_failures=2,
    )
    goals.save_goal("session-current", state)

    result = _call("status", session_id="session-current")

    assert result["success"] is True
    assert result["state"]["stop_reason"] == "judge unavailable"
    assert result["state"]["error_reason"] == "transport timeout"


def test_pause_does_not_resurrect_concurrently_cleared_goal(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    original = goals.GoalState(goal="Active work", status="active", max_turns=8)
    cleared = goals.GoalState(
        goal="Active work",
        status="cleared",
        max_turns=8,
        last_reason="operator-cleared",
    )
    goals.save_goal("session-current", original)
    db = goals._get_session_db()
    real_get_meta = db.get_meta
    injected = False

    def _get_then_clear(key):
        nonlocal injected
        raw = real_get_meta(key)
        if not injected:
            injected = True
            db.set_meta(key, cleared.to_json())
        return raw

    monkeypatch.setattr(db, "get_meta", _get_then_clear)

    result = _call("pause", session_id="session-current")

    assert result["success"] is False
    assert result["error"]["code"] == "transition_conflict"
    assert goals.load_goal("session-current").to_json() == cleared.to_json()


def test_set_does_not_overwrite_concurrently_created_goal(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    winner = goals.GoalState(
        goal="Concurrent winner",
        status="active",
        max_turns=6,
    )
    db = goals._get_session_db()
    real_get_meta = db.get_meta
    injected = False

    def _get_then_set(key):
        nonlocal injected
        raw = real_get_meta(key)
        if not injected:
            injected = True
            db.set_meta(key, winner.to_json())
        return raw

    monkeypatch.setattr(db, "get_meta", _get_then_set)

    result = _call(
        "set",
        session_id="session-current",
        condition="Stale setter",
        max_turns=5,
    )

    assert result["success"] is False
    assert result["error"]["code"] == "mutation_conflict"
    assert goals.load_goal("session-current").to_json() == winner.to_json()


def test_failed_persistence_never_returns_success(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)

    class _DroppedWriteDB:
        def get_meta(self, _key):
            return None

        def compare_and_set_meta(self, _key, _expected, _value):
            return True

    monkeypatch.setattr(goals, "_get_session_db", lambda: _DroppedWriteDB())

    result = _call("set", session_id="session-current", condition="Must persist")

    assert result["success"] is False
    assert result["error"]["code"] == "mutation_conflict"


def test_dropped_budget_update_is_not_mistaken_for_success(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    raw = goals.GoalState(
        goal="Same condition",
        status="active",
        max_turns=9,
    ).to_json()

    class _DroppedUpdateDB:
        def get_meta(self, _key):
            return raw

        def compare_and_set_meta(self, _key, _expected, _value):
            return True

    monkeypatch.setattr(goals, "_get_session_db", lambda: _DroppedUpdateDB())

    result = _call(
        "update",
        session_id="session-current",
        condition="Same condition",
        max_turns=2,
    )

    assert result["success"] is False
    assert result["error"]["code"] == "mutation_conflict"


@pytest.mark.parametrize("action", ["set", "update", "pause", "resume", "clear"])
def test_acknowledged_mutation_requires_exact_persisted_readback(
    tmp_path, monkeypatch, action
):
    _home(tmp_path, monkeypatch)
    session_id = "session-current"
    if action in {"update", "pause", "clear", "resume"}:
        _call("set", session_id=session_id, condition="Original", max_turns=9)
    if action == "resume":
        _call("pause", session_id=session_id)

    db = goals._get_session_db()
    real_cas = db.compare_and_set_meta
    drifted = False

    def commit_then_drift(key, expected, replacement):
        nonlocal drifted
        committed = real_cas(key, expected, replacement)
        if committed and not drifted:
            drifted = True
            independent = goals.GoalState.from_json(replacement)
            independent.revision += 1
            independent.last_reason = "independent post-write drift"
            assert real_cas(key, replacement, independent.to_json()) is True
        return committed

    monkeypatch.setattr(db, "compare_and_set_meta", commit_then_drift)
    kwargs = {"condition": "Updated", "max_turns": 2} if action in {"set", "update"} else {}
    result = _call(action, session_id=session_id, **kwargs)

    assert result["success"] is False
    expected_code = (
        "transition_conflict" if action in {"pause", "resume"} else "mutation_conflict"
    )
    assert result["error"]["code"] == expected_code
    persisted = goals.load_goal_authoritative(session_id)
    assert persisted.last_reason == "independent post-write drift"


@pytest.mark.parametrize("action", ["set", "update", "pause", "resume", "clear"])
def test_committed_mutation_with_unavailable_readback_is_retry_unsafe(
    tmp_path, monkeypatch, action
):
    _home(tmp_path, monkeypatch)
    session_id = "session-current"
    if action in {"update", "pause", "clear", "resume"}:
        _call("set", session_id=session_id, condition="Original", max_turns=9)
    if action == "resume":
        _call("pause", session_id=session_id)

    db = goals._get_session_db()
    real_cas = db.compare_and_set_meta
    real_get_meta = db.get_meta
    committed = False

    def commit_then_hide_readback(key, expected, replacement):
        nonlocal committed
        committed = real_cas(key, expected, replacement)
        return committed

    def fail_only_after_commit(key):
        if committed and key == goals._meta_key(session_id):
            raise OSError("private post-commit detail")
        return real_get_meta(key)

    monkeypatch.setattr(db, "compare_and_set_meta", commit_then_hide_readback)
    monkeypatch.setattr(db, "get_meta", fail_only_after_commit)
    kwargs = (
        {"condition": "Updated", "max_turns": 2}
        if action in {"set", "update"}
        else {}
    )
    result = _call(action, session_id=session_id, **kwargs)

    assert committed is True
    assert result["success"] is False
    assert result["error"]["code"] == "mutation_outcome_unknown"
    assert "may have committed" in result["error"]["message"]
    assert "do not retry blindly" in result["error"]["message"].lower()
    assert "private post-commit detail" not in str(result)
    assert real_get_meta(goals._meta_key(session_id)) is not None


def test_cas_exception_after_commit_is_reported_as_retry_unsafe(
    tmp_path, monkeypatch
):
    _home(tmp_path, monkeypatch)
    db = goals._get_session_db()
    assert db is not None
    real_cas = db.compare_and_set_meta

    def commit_then_raise(key, expected, replacement):
        assert real_cas(key, expected, replacement) is True
        raise OSError("private commit acknowledgement detail")

    monkeypatch.setattr(db, "compare_and_set_meta", commit_then_raise)

    result = _call(
        "set", session_id="session-current", condition="durable cas commit"
    )

    assert result["success"] is False
    assert result["error"]["code"] == "mutation_outcome_unknown"
    assert "may have committed" in result["error"]["message"].lower()
    assert "do not retry blindly" in result["error"]["message"].lower()
    assert "private commit acknowledgement detail" not in str(result)
    persisted = goals.GoalState.from_json(db.get_meta("goal:session-current"))
    assert persisted is not None
    assert persisted.goal == "durable cas commit"


def test_update_cas_conflict_does_not_resurrect_concurrently_cleared_goal(
    tmp_path, monkeypatch
):
    _home(tmp_path, monkeypatch)
    session_id = "session-current"
    _call("set", session_id=session_id, condition="Original", max_turns=9)
    db = goals._get_session_db()
    real_cas = db.compare_and_set_meta
    competed = False

    def clear_before_first_candidate(key, expected, replacement):
        nonlocal competed
        if not competed:
            competed = True
            current = goals.GoalState.from_json(db.get_meta(key))
            current.status = "cleared"
            current.revision += 1
            current.last_reason = "independent clear"
            assert real_cas(key, expected, current.to_json()) is True
        return real_cas(key, expected, replacement)

    monkeypatch.setattr(db, "compare_and_set_meta", clear_before_first_candidate)
    result = _call(
        "update",
        session_id=session_id,
        condition="Must not resurrect",
        max_turns=2,
    )

    assert result["success"] is False
    assert result["error"]["code"] == "mutation_conflict"
    persisted = goals.load_goal_authoritative(session_id)
    assert persisted.status == "cleared"
    assert persisted.last_reason == "independent clear"


@pytest.mark.parametrize("action", ["pause", "resume"])
def test_pause_resume_is_bound_to_the_initial_authoritative_snapshot(
    tmp_path, monkeypatch, action
):
    _home(tmp_path, monkeypatch)
    session_id = "session-current"
    _call("set", session_id=session_id, condition="Original", max_turns=9)
    if action == "resume":
        _call("pause", session_id=session_id)

    original_transition = getattr(goals.GoalManager, action)

    def replace_before_transition(self, *args, **kwargs):
        replacement = goals.GoalManager(session_id)
        replacement.set("Independent replacement", max_turns=7)
        if action == "resume":
            replacement.pause("independent pause")
        return original_transition(self, *args, **kwargs)

    monkeypatch.setattr(goals.GoalManager, action, replace_before_transition)
    result = _call(action, session_id=session_id)

    assert result["success"] is False
    assert result["error"]["code"] == "transition_conflict"
    persisted = goals.load_goal_authoritative(session_id)
    assert persisted.goal == "Independent replacement"
    assert persisted.status == ("paused" if action == "resume" else "active")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("goal", "   "),
        ("status", "unknown"),
        ("revision", -1),
        ("turns_used", -1),
        ("max_turns", 0),
        ("consecutive_transport_failures", -1),
        ("consecutive_parse_failures", -1),
    ],
)
def test_semantically_invalid_persisted_goal_is_rejected(
    tmp_path, monkeypatch, field, value
):
    _home(tmp_path, monkeypatch)
    session_id = "session-current"
    payload = json.loads(goals.GoalState(goal="Valid goal").to_json())
    payload[field] = value
    goals._get_session_db().set_meta(
        goals._meta_key(session_id), json.dumps(payload)
    )

    with pytest.raises(goals.GoalPersistenceError):
        goals.load_goal_authoritative(session_id)
    result = _call("status", session_id=session_id)
    assert result["success"] is False
    assert result["error"]["code"] == "persistence_unavailable"


def test_slow_initialization_fails_closed_instead_of_claiming_success(
    tmp_path, monkeypatch
):
    _home(tmp_path, monkeypatch)
    import hermes_state

    initialized = threading.Event()
    real_session_db = hermes_state.SessionDB

    class _SlowSessionDB(real_session_db):
        def __init__(self, *args, **kwargs):
            time.sleep(0.1)
            super().__init__(*args, **kwargs)
            initialized.set()

    monkeypatch.setattr(hermes_state, "SessionDB", _SlowSessionDB)
    monkeypatch.setattr(goals, "_DB_BOOTSTRAP_INIT_WAIT_S", 0.01)

    async def _on_runtime_loop():
        return _call("set", session_id="session-current", condition="Slow write")

    result = asyncio.run(_on_runtime_loop())

    assert result["success"] is False
    assert result["error"]["code"] == "persistence_unavailable"
    assert initialized.wait(10), "background SessionDB initialization did not finish"
    assert goals.load_goal("session-current") is None
