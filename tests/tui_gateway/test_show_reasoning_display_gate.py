"""display.show_reasoning false is answer-only on the live callback path.

The session flag, not reasoning_effort, decides whether reasoning/thinking
deltas and non-essential tool chrome leave the gateway.
"""

import json
from types import SimpleNamespace

from tui_gateway import server


def _capture(monkeypatch):
    events = []
    monkeypatch.setattr(
        server, "_emit", lambda event_type, sid, payload=None: events.append((event_type, sid, payload))
    )
    return events


def _session(monkeypatch, sid, *, show_reasoning, effort="high"):
    monkeypatch.setitem(
        server._sessions,
        sid,
        {
            "show_reasoning": show_reasoning,
            "tool_progress_mode": "all",
            "tool_started_at": {},
            "edit_snapshots": {},
            "agent": SimpleNamespace(reasoning_config={"enabled": True, "effort": effort}),
        },
    )


def test_hidden_reasoning_does_not_emit_reasoning_deltas(monkeypatch):
    events = _capture(monkeypatch)
    _session(monkeypatch, "hide-deltas", show_reasoning=False, effort="high")

    callbacks = server._agent_cbs("hide-deltas")
    callbacks["reasoning_callback"]("chain of thought from any provider")
    # thinking.delta is the wait/spinner line, not a reasoning block.
    callbacks["thinking_callback"]("⏳ waiting on the provider")

    assert [event[0] for event in events] == ["thinking.delta"]
    assert events[0][2]["text"] == "⏳ waiting on the provider"


def test_shown_reasoning_still_emits_reasoning_delta(monkeypatch):
    events = _capture(monkeypatch)
    _session(monkeypatch, "show-deltas", show_reasoning=True, effort="high")

    server._agent_cbs("show-deltas")["reasoning_callback"]("visible thought")

    assert [event[0] for event in events] == ["reasoning.delta"]
    assert events[0][2]["text"] == "visible thought"


def test_child_mirror_skips_reasoning_delta_when_hidden(monkeypatch):
    events = _capture(monkeypatch)
    _session(monkeypatch, "child-sid", show_reasoning=False)
    monkeypatch.setattr(
        server, "_find_live_session_by_key", lambda key: ("child-sid", {"agent": None, "show_reasoning": False})
    )

    server._mirror_subagent_to_child(
        "subagent.thinking", {"child_session_id": "child-key", "text": "delegated thought"}
    )

    assert not any(event[0] == "reasoning.delta" for event in events)


def test_hidden_reasoning_drops_completed_reasoning_block(monkeypatch):
    events = _capture(monkeypatch)
    _session(monkeypatch, "hide-available", show_reasoning=False, effort="medium")

    server._on_tool_progress("hide-available", "reasoning.available", "_thinking", "finished thought", None)

    assert events == []


def test_hidden_reasoning_suppresses_nonessential_tool_chrome_without_effort_none(monkeypatch):
    events = _capture(monkeypatch)
    _session(monkeypatch, "hide-tools", show_reasoning=False, effort="high")

    server._on_tool_start("hide-tools", "tool-read", "read_file", {"path": "README.md"})
    server._on_tool_complete("hide-tools", "tool-read", "read_file", {"path": "README.md"}, "contents")
    server._agent_cbs("hide-tools")["tool_gen_callback"]("terminal")

    clarify_args = {"question": "Pick one", "choices": ["A", "B"]}
    server._on_tool_start("hide-tools", "tool-clarify", "clarify", clarify_args)
    server._on_tool_complete(
        "hide-tools",
        "tool-clarify",
        "clarify",
        clarify_args,
        json.dumps({"question": "Pick one", "user_response": "A"}),
    )
    server._on_tool_complete(
        "hide-tools",
        "tool-fail",
        "terminal",
        {"command": "deploy"},
        json.dumps({"error": "disk full"}),
    )

    kinds = [event[0] for event in events]
    assert "tool.generating" not in kinds
    assert not any(event[2].get("name") == "read_file" for event in events)
    assert [event[0] for event in events if event[2].get("name") == "clarify"] == ["tool.start", "tool.complete"]
    failed = [event for event in events if event[2].get("tool_id") == "tool-fail"]
    assert [event[0] for event in failed] == ["tool.complete"]
    assert failed[0][2]["result"]["error"] == "disk full"


def test_hidden_reasoning_drops_moa_reference_chrome(monkeypatch):
    events = _capture(monkeypatch)
    _session(monkeypatch, "hide-moa", show_reasoning=False, effort="high")

    server._on_tool_progress("hide-moa", "moa.reference", "reference-a", "other model's thoughts", None)

    assert events == []


def test_hidden_reasoning_drops_moa_status_and_aggregating_chrome(monkeypatch):
    events = _capture(monkeypatch)
    _session(monkeypatch, "hide-moa-status", show_reasoning=False, effort="high")

    server._on_tool_progress(
        "hide-moa-status", "moa.progress", "aggregator-a", None, None, moa_refs_done=1, moa_refs_total=3
    )
    server._on_tool_progress(
        "hide-moa-status", "moa.phase", "aggregator-a", None, None, moa_phase="aggregator"
    )
    server._on_tool_progress("hide-moa-status", "moa.aggregating", "aggregator-a", None, None)

    assert events == []


def test_shown_reasoning_still_emits_moa_aggregating(monkeypatch):
    events = _capture(monkeypatch)
    _session(monkeypatch, "show-moa-status", show_reasoning=True, effort="high")

    server._on_tool_progress("show-moa-status", "moa.aggregating", "aggregator-a", None, None)

    assert [event[0] for event in events] == ["moa.aggregating"]
    assert events[0][2]["aggregator"] == "aggregator-a"


def test_hidden_reasoning_drops_subagent_thinking_text_on_parent(monkeypatch):
    events = _capture(monkeypatch)
    _session(monkeypatch, "hide-sub", show_reasoning=False)

    server._on_tool_progress(
        "hide-sub",
        "subagent.thinking",
        "tool",
        "the child's private chain of thought",
        None,
        child_session_id="child-key",
    )

    # The lifecycle frame still reaches the parent (the delegate card renders
    # progress), but the child's reasoning text must not ride along.
    assert [event[0] for event in events] == ["subagent.thinking"]
    assert "text" not in events[0][2]
    assert events[0][2]["child_session_id"] == "child-key"


def test_shown_reasoning_keeps_subagent_thinking_text(monkeypatch):
    events = _capture(monkeypatch)
    _session(monkeypatch, "show-sub", show_reasoning=True)

    server._on_tool_progress(
        "show-sub",
        "subagent.thinking",
        "tool",
        "the child's visible thought",
        None,
        child_session_id="child-key",
    )

    assert [event[0] for event in events] == ["subagent.thinking"]
    assert events[0][2]["text"] == "the child's visible thought"


def test_hidden_reasoning_shows_failed_terminal_exit_code(monkeypatch):
    events = _capture(monkeypatch)
    _session(monkeypatch, "hide-exit", show_reasoning=False, effort="high")

    server._on_tool_complete(
        "hide-exit",
        "tool-exit",
        "terminal",
        {"command": "deploy"},
        json.dumps({"output": "boom", "exit_code": 1, "error": None}),
    )

    failed = [event for event in events if event[2].get("tool_id") == "tool-exit"]
    assert [event[0] for event in failed] == ["tool.complete"]
    assert failed[0][2]["result"]["exit_code"] == 1


def test_hidden_reasoning_hides_successful_terminal_exit(monkeypatch):
    events = _capture(monkeypatch)
    _session(monkeypatch, "hide-exit-ok", show_reasoning=False, effort="high")

    server._on_tool_complete(
        "hide-exit-ok",
        "tool-exit-ok",
        "terminal",
        {"command": "deploy"},
        json.dumps({"output": "ok", "exit_code": 0, "error": None}),
    )

    assert not any(event[2].get("tool_id") == "tool-exit-ok" for event in events)


def test_tool_result_needs_user_treats_nonzero_exit_code_as_failure():
    assert server._tool_result_needs_user(json.dumps({"output": "boom", "exit_code": 1, "error": None})) is True
    assert server._tool_result_needs_user(json.dumps({"output": "ok", "exit_code": 0, "error": None})) is False
    # A boolean exit_code is not an exit status; True must not read as failure-by-1.
    assert server._tool_result_needs_user(json.dumps({"exit_code": True})) is False
    assert server._tool_result_needs_user(json.dumps({"exit_code": "1"})) is False
    assert server._tool_result_needs_user(json.dumps({"success": False})) is True
    assert server._tool_result_needs_user(json.dumps({"ok": False, "output": "denied"})) is True
    assert server._tool_result_needs_user(json.dumps({"error": "disk full"})) is True
    assert server._tool_result_needs_user("not json") is False
