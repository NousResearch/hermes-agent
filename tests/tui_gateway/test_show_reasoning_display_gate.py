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
