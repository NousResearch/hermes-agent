from types import ModuleType

from tui_gateway.subagent_progress import (
    build_subagent_progress_payload,
    install_server_overlay,
)


def test_completion_projection_preserves_logical_schema_failure_evidence():
    schema_errors = ["'city' is a required property"]

    payload = build_subagent_progress_payload(
        "subagent.complete",
        None,
        None,
        {
            "goal": "produce the address",
            "task_index": 2,
            "task_count": 3,
            "subagent_id": "child-2",
            "status": "completed",
            "outcome": "failed",
            "summary": "{}",
            "schema_valid": False,
            "schema_errors": schema_errors,
            "schema_retries": 1,
            "error": "Final answer does not satisfy the declared output_schema.",
            "error_authoritative": True,
            "exit_reason": "completed",
            "interrupted": False,
            "tool_error_count": 0,
        },
    )

    assert payload["status"] == "completed"
    assert payload["outcome"] == "failed"
    assert payload["schema_valid"] is False
    assert payload["schema_errors"] == schema_errors
    assert payload["schema_retries"] == 1
    assert payload["error_authoritative"] is True
    assert "output_schema" in str(payload["error"])
    assert payload["exit_reason"] == "completed"
    assert payload["interrupted"] is False


def test_projection_keeps_existing_tool_preview_and_rollup_contract():
    payload = build_subagent_progress_payload(
        "subagent.tool",
        "terminal",
        "pytest -q",
        {
            "goal": "test",
            "task_index": 0,
            "task_count": 1,
            "api_calls": "4",
            "files_read": ["a.py"],
            "files_written": ["b.py"],
            "toolsets": ["terminal"],
        },
    )

    assert payload == {
        "api_calls": 4,
        "files_read": ["a.py"],
        "files_written": ["b.py"],
        "goal": "test",
        "task_count": 1,
        "task_index": 0,
        "text": "pytest -q",
        "tool_name": "terminal",
        "tool_preview": "pytest -q",
        "toolsets": ["terminal"],
    }


def test_server_overlay_intercepts_only_subagent_events_and_is_idempotent():
    server = ModuleType("tui_gateway.server")
    calls: list[tuple] = []

    def original(*args, **kwargs):
        calls.append(("original", args, kwargs))
        return "legacy"

    setattr(server, "_on_tool_progress", original)
    setattr(server, "_tool_progress_enabled", lambda _sid: True)
    setattr(server, "_emit", lambda *args: calls.append(("emit", *args)))
    setattr(
        server,
        "_mirror_subagent_to_child",
        lambda *args: calls.append(("mirror", *args)),
    )

    install_server_overlay(server)
    installed = getattr(server, "_on_tool_progress")
    install_server_overlay(server)

    assert getattr(server, "_on_tool_progress") is installed
    assert installed("parent", "moa.aggregating", "model") == "legacy"
    installed(
        "parent",
        "subagent.complete",
        status="completed",
        outcome="failed",
        schema_valid=False,
        error_authoritative=True,
    )

    assert calls[0][0] == "original"
    emitted = calls[1]
    assert emitted[:3] == ("emit", "subagent.complete", "parent")
    assert emitted[3]["status"] == "completed"
    assert emitted[3]["outcome"] == "failed"
    assert emitted[3]["schema_valid"] is False
    assert emitted[3]["error_authoritative"] is True
    assert calls[2] == ("mirror", "subagent.complete", emitted[3])
