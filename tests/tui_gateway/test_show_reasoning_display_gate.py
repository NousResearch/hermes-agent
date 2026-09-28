"""Reasoning display and the tool feed are independent switches on the live callback path.

``display.show_reasoning`` (the session flag, not reasoning_effort) gates
reasoning/thinking deltas. The tool feed's chrome — tool.start/complete,
tool.generating, child-mirror rows — follows ``display.tool_progress``; the
answer-only default that ``show_reasoning: false`` sets still hides chrome,
unless the user stated a feed preference themselves (``display.tool_progress``
written in config, or HERMES_TUI_TOOL_PROGRESS).
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from tui_gateway import server

# The real predicate, captured before the autouse fixture swaps it out.
_REAL_TOOL_PROGRESS_EXPLICIT = server._tool_progress_explicit


@pytest.fixture(autouse=True)
def _no_explicit_feed(monkeypatch):
    """Pin the answer-only default. Unless a test overrides this, a real config
    on the dev box carrying ``display.tool_progress`` must not leak in and flip
    the suppression cases below."""
    monkeypatch.setattr(server, "_tool_progress_explicit", lambda: False)


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
        server, "_find_live_session_by_key", lambda key, _home: ("child-sid", {"agent": None, "show_reasoning": False})
    )

    server._mirror_subagent_to_child(
        "subagent.thinking", {"child_session_id": "child-key", "text": "delegated thought"}, None
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


def test_explicit_feed_keeps_process_chrome_when_reasoning_is_hidden(monkeypatch):
    """The requested combination: execution flow without thinking.

    A user-stated display.tool_progress (config key or HERMES_TUI_TOOL_PROGRESS)
    is the feed's own switch — it overrides the answer-only hiding that
    display.show_reasoning false otherwise implies, matching the classic CLI,
    where the feed has always followed tool_progress alone.
    """
    events = _capture(monkeypatch)
    _session(monkeypatch, "flow-only", show_reasoning=False, effort="high")
    monkeypatch.setattr(server, "_tool_progress_explicit", lambda: True)

    server._on_tool_start("flow-only", "tool-read", "read_file", {"path": "README.md"})
    server._on_tool_complete("flow-only", "tool-read", "read_file", {"path": "README.md"}, "contents")
    server._agent_cbs("flow-only")["tool_gen_callback"]("terminal")

    assert [event[0] for event in events if event[2].get("tool_id") == "tool-read"] == [
        "tool.start",
        "tool.complete",
    ]
    assert "tool.generating" in [event[0] for event in events]


def test_tool_progress_off_silences_the_feed_even_with_reasoning_on(monkeypatch):
    events = _capture(monkeypatch)
    _session(monkeypatch, "feed-off", show_reasoning=True, effort="high")
    monkeypatch.setitem(server._sessions["feed-off"], "tool_progress_mode", "off")

    server._on_tool_start("feed-off", "tool-read", "read_file", {"path": "README.md"})
    server._on_tool_complete("feed-off", "tool-read", "read_file", {"path": "README.md"}, "contents")
    server._on_tool_complete(
        "feed-off",
        "tool-fail",
        "terminal",
        {"command": "deploy"},
        json.dumps({"error": "disk full"}),
    )

    assert not any(event[2].get("name") == "read_file" for event in events)
    failed = [event for event in events if event[2].get("tool_id") == "tool-fail"]
    assert [event[0] for event in failed] == ["tool.complete"]


def test_tool_progress_explicit_reads_config_key_and_env(monkeypatch):
    monkeypatch.setattr(server, "_tool_progress_explicit", _REAL_TOOL_PROGRESS_EXPLICIT)
    monkeypatch.delenv("HERMES_TUI_TOOL_PROGRESS", raising=False)

    monkeypatch.setattr(server, "_display_cfg", lambda: {})
    assert server._tool_progress_explicit() is False

    monkeypatch.setattr(server, "_display_cfg", lambda: {"tool_progress": "off"})
    assert server._tool_progress_explicit() is True

    # A non-mode env value is not a stated preference; a valid one is.
    monkeypatch.setattr(server, "_display_cfg", lambda: {})
    monkeypatch.setenv("HERMES_TUI_TOOL_PROGRESS", "bogus")
    assert server._tool_progress_explicit() is False
    monkeypatch.setenv("HERMES_TUI_TOOL_PROGRESS", "verbose")
    assert server._tool_progress_explicit() is True


def test_hidden_reasoning_drops_moa_reference_chrome(monkeypatch):
    events = _capture(monkeypatch)
    _session(monkeypatch, "hide-moa", show_reasoning=False, effort="high")

    server._on_tool_progress("hide-moa", "moa.reference", "reference-a", "other model's thoughts", None)

    assert events == []


def test_hidden_reasoning_drops_moa_status_lines_but_keeps_aggregating(monkeypatch):
    events = _capture(monkeypatch)
    _session(monkeypatch, "hide-moa-status", show_reasoning=False, effort="high")

    server._on_tool_progress(
        "hide-moa-status", "moa.progress", "aggregator-a", None, None, moa_refs_done=1, moa_refs_total=3
    )
    server._on_tool_progress(
        "hide-moa-status", "moa.phase", "aggregator-a", None, None, moa_phase="aggregator"
    )
    server._on_tool_progress("hide-moa-status", "moa.aggregating", "aggregator-a", None, None)

    # progress/phase write lines into the reasoning disclosure (Desktop) and the
    # activity log (TUI). aggregating is a bare busy-state transition, no content.
    assert [event[0] for event in events] == ["moa.aggregating"]


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


def test_hidden_reasoning_keeps_card_tool_lifecycle(monkeypatch):
    """Card tools are the turn's deliverable, not chrome.

    Desktop's answer-only gate (`message-parts.tsx`) keeps card tools visible via
    `isCardTool` (tool-render-class.ts): clarify, delegate_task, image_generate,
    manage_catalog, manage_connections. The gateway must emit their full
    lifecycle under answer-only mode too — no event means no part means nothing
    for Desktop to keep. A suppressed `tool.start` would also orphan a failed
    call's `tool.complete`.
    """
    events = _capture(monkeypatch)
    _session(monkeypatch, "hide-cards", show_reasoning=False, effort="high")

    cases = {
        "tool-image": ("image_generate", {"prompt": "a cat"}),
        "tool-catalog": ("manage_catalog", {"action": "list"}),
        "tool-delegate": ("delegate_task", {"goal": "do it"}),
    }
    for tool_id, (name, args) in cases.items():
        server._on_tool_start("hide-cards", tool_id, name, args)
        server._on_tool_complete("hide-cards", tool_id, name, args, json.dumps({"success": True}))

    for tool_id in cases:
        lifecycle = [event[0] for event in events if event[2].get("tool_id") == tool_id]
        assert lifecycle == ["tool.start", "tool.complete"], tool_id


def test_hidden_reasoning_keeps_file_edit_lifecycle_pair(monkeypatch):
    """A file edit is a diff card (`isFileEditTool`): answer-only keeps it visible.

    The complete-side guard admits edits through `inline_diff`; without the same
    membership on the start side, the diff arrived with no row and a lifecycle
    consumer got a `tool.complete` it never saw a `tool.start` for.
    """
    events = _capture(monkeypatch)
    _session(monkeypatch, "hide-edit", show_reasoning=False, effort="high")

    def fake_edit_diff(tool_name, result, *, function_args=None, snapshot=None, print_fn=None):
        if print_fn is not None:
            print_fn("— edited src/a.py —")
        return True

    monkeypatch.setattr("agent.display.render_edit_diff_with_delta", fake_edit_diff)

    cases = {
        "tool-edit": ("edit_file", {"path": "src/a.py", "old_string": "a", "new_string": "b"}),
        "tool-patch": ("patch", {"path": "src/a.py", "patches": []}),
        "tool-write": ("write_file", {"path": "src/a.py", "content": "b"}),
    }
    for tool_id, (name, args) in cases.items():
        server._on_tool_start("hide-edit", tool_id, name, args)
        server._on_tool_complete("hide-edit", tool_id, name, args, json.dumps({"success": True}))

    for tool_id in cases:
        lifecycle = [event[0] for event in events if event[2].get("tool_id") == tool_id]
        assert lifecycle == ["tool.start", "tool.complete"], tool_id


def test_gateway_lifecycle_set_covers_desktop_card_tools():
    """`isCardTool` (tool-render-class.ts) and the gateway lifecycle set must not drift.

    Every name the desktop classifies as a card needs its full lifecycle under
    answer-only, or its `tool.complete` arrives orphaned.
    """
    ts_path = Path(__file__).resolve().parents[2] / "apps" / "desktop" / "src" / "lib" / "tool-render-class.ts"
    source = ts_path.read_text(encoding="utf-8")

    def set_literal(var: str) -> set[str]:
        match = re.search(rf"const {var} = new Set\(\[(.*?)\]\)", source, re.DOTALL)
        assert match, f"{var} not found in {ts_path.name}"
        return set(re.findall(r"'([^']+)'", match.group(1)))

    # CONNECTION_CARD_KEY is the run-splitter's alias for a manage_connections
    # part, not a tool name the gateway ever sees.
    desktop_cards = (
        set_literal("CARD_TOOL_NAMES")
        | set_literal("FILE_EDIT_TOOL_NAMES")
        | {"manage_connections"}
    )
    assert desktop_cards <= set(server._TOOL_LIFECYCLE_UI_TOOLS)


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
