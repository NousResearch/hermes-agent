from __future__ import annotations

import sys
from types import SimpleNamespace

from hermes_cli import profile_activity_ledger as ledger
from tools import delegate_tool


def test_delegation_start_event_is_metadata_only_and_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(ledger, "is_enabled", lambda cfg=None: True)
    parent = SimpleNamespace(session_id="parent-session", _current_turn_id="turn-1", profile="kensei")
    child = SimpleNamespace(session_id="child-session", model="test-model", provider="test-provider")

    delegate_tool._record_delegation_event(
        "delegation.started",
        parent_agent=parent,
        child=child,
        child_subagent_id="sa-1",
        child_role="leaf",
        parent_subagent_id=None,
    )
    delegate_tool._record_delegation_event(
        "delegation.started",
        parent_agent=parent,
        child=child,
        child_subagent_id="sa-1",
        child_role="leaf",
        parent_subagent_id=None,
    )

    events = ledger.query_events(event_types=["delegation.started"])
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload == {
        "child_role": "leaf",
        "child_session_id": "child-session",
        "child_subagent_id": "sa-1",
        "model": "test-model",
        "parent_session_id": "parent-session",
        "parent_subagent_id": None,
        "parent_turn_id": "turn-1",
        "provider": "test-provider",
        "status": "started",
    }


def test_delegation_finish_event_excludes_summary_and_tool_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(ledger, "is_enabled", lambda cfg=None: True)
    parent = SimpleNamespace(session_id="parent-session", _current_turn_id="turn-1", profile="kensei")
    child = SimpleNamespace(
        session_id="child-session",
        _subagent_id="sa-1",
        _delegate_role="leaf",
        model="test-model",
        provider="test-provider",
    )
    result = {
        "task_index": 0,
        "status": "completed",
        "summary": "prompt=DO_NOT_PERSIST secret=sk-proj-DO_NOT_PERSIST",
        "exit_reason": "completed",
        "duration_seconds": 2.5,
        "api_calls": 3,
        "tokens": {"input": 11, "output": 22},
        "cost_usd": 0.125,
        "tool_trace": [
            {"tool": "write_file", "input_summary": {"content": "sk-proj-DO_NOT_PERSIST"}},
            {"tool": "terminal", "input_summary": {"command": "print(secret)"}},
        ],
        "_child_role": "leaf",
        "_child_cost_usd": 0.125,
    }

    delegate_tool._record_delegation_event(
        "delegation.finished",
        parent_agent=parent,
        child=child,
        child_subagent_id="sa-1",
        child_role="leaf",
        parent_subagent_id=None,
        result=result,
    )
    delegate_tool._record_delegation_event(
        "delegation.finished",
        parent_agent=parent,
        child=child,
        child_subagent_id="sa-1",
        child_role="leaf",
        parent_subagent_id=None,
        result=result,
    )

    events = ledger.query_events(event_types=["delegation.finished"])
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload == {
        "api_calls": 3,
        "child_role": "leaf",
        "child_session_id": "child-session",
        "child_subagent_id": "sa-1",
        "cost_usd": 0.125,
        "duration_ms": 2500,
        "exit_reason": "completed",
        "model": "test-model",
        "parent_session_id": "parent-session",
        "parent_subagent_id": None,
        "parent_turn_id": "turn-1",
        "provider": "test-provider",
        "status": "completed",
        "token_counts": {"input": 11, "output": 22},
        "tool_count": 2,
        "tool_names": ["terminal", "write_file"],
    }
    raw = str(events[0])
    assert "DO_NOT_PERSIST" not in raw
    assert "print(secret)" not in raw


def test_finalizer_records_finished_event_when_plugin_hook_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(ledger, "is_enabled", lambda cfg=None: True)
    parent = SimpleNamespace(
        session_id="parent-session",
        _current_turn_id="turn-1",
        profile="kensei",
        _memory_manager=None,
        session_estimated_cost_usd=0.0,
        session_cost_source="none",
        session_cost_status="unknown",
    )
    child = SimpleNamespace(session_id="child-session", _subagent_id="sa-1", _delegate_role="leaf")
    result = {
        "task_index": 0,
        "status": "interrupted",
        "summary": None,
        "exit_reason": "interrupted",
        "duration_seconds": 1,
        "api_calls": 0,
        "tokens": {"input": 0, "output": 0},
        "tool_trace": [],
        "_child_role": "leaf",
        "_child_cost_usd": 0,
    }
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", None)

    delegate_tool._finalize_child_results(
        [result],
        [{"goal": "redacted goal"}],
        [(0, {"goal": "redacted goal"}, child)],
        parent,
    )

    events = ledger.query_events(event_types=["delegation.finished"])
    assert len(events) == 1
    assert events[0]["payload"]["status"] == "interrupted"


def test_failed_delegation_event_is_recorded_without_error_text(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(ledger, "is_enabled", lambda cfg=None: True)
    parent = SimpleNamespace(session_id="parent-session", _current_turn_id="turn-1")
    child = SimpleNamespace(
        session_id="child-session",
        _subagent_id="sa-failed",
        _delegate_role="leaf",
        model="test-model",
        provider="test-provider",
    )
    delegate_tool._record_delegation_event(
        "delegation.finished",
        parent_agent=parent,
        child=child,
        child_subagent_id="sa-failed",
        child_role="leaf",
        result={
            "status": "failed",
            "exit_reason": "max_iterations",
            "duration_seconds": 3,
            "api_calls": 1,
            "tokens": {"input": 4, "output": 5},
            "tool_trace": [],
            "error": "secret prompt output must not be persisted",
        },
    )
    event = ledger.query_events(event_types=["delegation.finished"])[0]
    assert event["payload"]["status"] == "failed"
    assert "error" not in event["payload"]


def test_specialist_profile_is_separate_from_child_role(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(ledger, "is_enabled", lambda cfg=None: True)
    parent = SimpleNamespace(session_id="parent-session", _current_turn_id="turn-1")
    child = SimpleNamespace(
        session_id="child-session",
        _subagent_id="sa-profile",
        _delegate_role="leaf",
        model="test-model",
        provider="test-provider",
    )

    delegate_tool._record_delegation_event(
        "delegation.started",
        parent_agent=parent,
        child=child,
        child_subagent_id="sa-profile",
        child_role="leaf",
        child_profile="octacon",
    )

    event = ledger.query_events(event_types=["delegation.started"])[0]
    assert event["target_profile"] == "octacon"
    assert event["payload"]["child_profile"] == "octacon"
    assert event["payload"]["child_role"] == "leaf"
