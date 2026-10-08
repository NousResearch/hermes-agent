"""Profile observers receive generic child evidence and the parent's actual home."""
from types import SimpleNamespace

from tools.delegate_tool_results import _fire_subagent_stop_hooks


def test_stop_hook_carries_profile_goal_and_generic_evidence(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *args, **kwargs: calls.append((args, kwargs)))
    home = tmp_path / "profiles" / "research"
    parent = SimpleNamespace(
        session_id="parent", _current_turn_id="turn", _hermes_home="wrong-ambient-home",
        _session_db=SimpleNamespace(db_path=home / "state.db"),
    )
    child = SimpleNamespace(session_id="child")
    entry = {
        "task_index": 0, "status": "failed", "summary": "missing input", "error": "source unavailable",
        "api_calls": 2, "verified_input_sha256": "input-hash", "facts_sha256": "facts-hash",
        "_child_role": "leaf", "_child_cost_usd": 0.25, "duration_seconds": 1.5,
    }
    cost = _fire_subagent_stop_hooks([entry], {0: child}, parent, [{"goal": "verify the source"}])
    assert cost == 0.25
    assert len(calls) == 1
    args, fields = calls[0]
    assert args == ("subagent_stop",)
    assert fields["hermes_home"] == str(home)
    assert fields["child_goal"] == "verify the source"
    assert fields["child_session_id"] == "child"
    assert fields["child_status"] == "failed"
    assert fields["duration_ms"] == 1500
    assert fields["child_result_fields"] == {
        "api_calls": 2, "error": "source unavailable", "verified_input_sha256": "input-hash", "facts_sha256": "facts-hash",
    }
    assert "_child_role" not in entry and "_child_cost_usd" not in entry
