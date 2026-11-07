"""Regression for #140262: summaries are not terminal failure causes."""

from types import SimpleNamespace


import pytest

from tools.delegate_tool_child_run import _ChildRun, _SchemaOutcome, _build_result_entry
from tools.delegate_tool_progress import _build_child_progress_callback


@pytest.mark.parametrize("surface", ["cli", "gateway"])
@pytest.mark.parametrize("error,reason", [
    ("Provider rejected the oversized request", None),
    (None, None),
    ("Raw provider detail", "rate_limit"),
])
def test_completion_notice_uses_cause_without_losing_summary(surface, error, reason):
    summary = "5/5 OK, 0 failures, EXIT=0."
    notices = []
    events = []
    if surface == "gateway":
        from gateway.run_turn_runner import TurnRunner
        from gateway.turn_context import TurnContext
        runner = TurnRunner(SimpleNamespace(), TurnContext(
            source=SimpleNamespace(platform="telegram"), user_config={},
            _run_still_current=lambda: True,
        ))
        async def deliver(source, text):
            notices.append(text)
        runner._runner._deliver_platform_notice = deliver
        import asyncio
        runner._schedule = lambda coro, message: asyncio.run(coro)
        def parent_callback(event, tool_name=None, preview=None, args=None, **kwargs):
            events.append(kwargs)
            runner._progress_subagent_notice(preview, kwargs)
        parent = SimpleNamespace(tool_progress_callback=parent_callback)
    else:
        parent = SimpleNamespace(
            platform="cli", _notification_config={},
            _delegate_spinner=SimpleNamespace(print_above=notices.append),
            tool_progress_callback=lambda event, tool_name=None, preview=None, args=None, **kwargs: events.append(kwargs),
        )
    callback = _build_child_progress_callback(0, "build artifacts", parent)
    child = SimpleNamespace()
    result = {"failed": True, "completed": False, "final_response": summary,
              "error": error, "failure_reason": reason}
    entry = _build_result_entry(child, result, 0, 12, _SchemaOutcome(None, None, [], 0))
    run = _ChildRun(child, parent, 0, "build artifacts", None, callback)
    run.emit_complete(result, entry, 12)

    assert len(notices) == 1
    assert summary not in notices[0]
    if reason:
        assert "rate" in notices[0].lower() or "limit" in notices[0].lower()
        assert error not in notices[0]
    elif error:
        assert error in notices[0]
    assert entry["summary"] == events[0]["summary"] == summary
    assert entry["status"] == events[0]["status"] == "failed"
    assert events[0]["error"] == error


@pytest.mark.parametrize("payload,expected", [
    ({"error": "actual cause", "summary": "work report"}, "actual cause"),
    ({"error": "", "summary": "work report"}, None),
    ({"summary": "legacy error detail"}, "legacy error detail"),
    ({}, "legacy preview"),
])
def test_progress_cause_preserves_legacy_fallback(payload, expected):
    notices = []
    parent = SimpleNamespace(platform="cli", _notification_config={},
        _delegate_spinner=SimpleNamespace(print_above=notices.append))
    callback = _build_child_progress_callback(0, "task", parent)
    callback("subagent.complete", preview="legacy preview", status="failed", **payload)
    assert len(notices) == 1
    assert "work report" not in notices[0]
    if expected:
        assert expected in notices[0]
    else:
        assert "legacy preview" not in notices[0]

    notices.clear()
    callback("subagent.complete", preview="success", status="completed", **payload)
    assert notices == []
