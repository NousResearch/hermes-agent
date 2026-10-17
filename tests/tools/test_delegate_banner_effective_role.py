"""The delegate_task completion banner must report the role children actually ran as.

`role` is depth-derived in `_build_child_agent` (the requested arg is wire-compat legacy,
ignored), so a depth-1 child under `delegation.max_spawn_depth: 2` runs as an orchestrator —
yet the batch-completed banner said `Role: leaf`, because `_dispatch_unit` stamped the
registry record with the requested role (#138032).
"""
from types import SimpleNamespace

from tools.delegate_tool_dispatch import _Batch, _dispatch_unit
from tools.process_registry_notifications import format_process_notification


def _batch(top_role: str, child_role=None) -> _Batch:
    # A bare object() carries no _delegate_role attribute: the fallback branch.
    child = SimpleNamespace(_delegate_role=child_role) if child_role else object()
    return _Batch(
        task_list=[{"goal": "spawn one nested subagent"}], children=[(0, {"goal": "g"}, child)],
        parent_agent=SimpleNamespace(), creds={"model": "m"}, context=None, top_role=top_role,
        max_children=1, live_deleg_id="deleg_t", live_writers=[], live_paths=[],
        origin_wake_sid="", origin_ui_session_id="", origin_owner_transport=None,
        origin_owner_session_record=None, origin_session_history_delivery=False, overall_start=0.0,
    )


def _capture_dispatch(monkeypatch):
    captured = {}

    def fake_dispatch(**kwargs):
        captured.update(kwargs)
        return {"status": "dispatched", "delegation_id": kwargs.get("delegation_id")}

    monkeypatch.setattr("tools.async_delegation.dispatch_async_delegation_batch", fake_dispatch)
    return captured


def test_dispatch_unit_stamps_the_record_with_the_effective_role(monkeypatch):
    captured = _capture_dispatch(monkeypatch)
    result = _dispatch_unit(_batch(top_role="leaf", child_role="orchestrator"), "deleg_t", None,
                            {"session_key": "sk"})
    assert result["status"] == "dispatched"
    assert captured["role"] == "orchestrator"  # not the requested "leaf"


def test_dispatch_unit_falls_back_to_the_requested_role_without_child_metadata(monkeypatch):
    captured = _capture_dispatch(monkeypatch)
    _dispatch_unit(_batch(top_role="leaf"), "deleg_t", None, {"session_key": "sk"})
    assert captured["role"] == "leaf"


def test_batch_banner_renders_the_effective_role():
    evt = {"type": "async_delegation", "delegation_id": "deleg_x", "is_batch": True,
           "goals": ["a"], "goal": "a", "results": [{"task_index": 0, "status": "completed", "summary": "ok"}],
           "role": "orchestrator", "model": "m", "dispatched_at": 1.0, "completed_at": 2.0,
           "total_duration_seconds": 1.0}
    assert "Role: orchestrator   Model: m" in format_process_notification(evt)
