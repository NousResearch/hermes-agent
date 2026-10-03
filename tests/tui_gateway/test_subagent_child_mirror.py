"""Tests for the gateway's child-session live mirror.

A delegated child runs synchronously inside the parent's turn; its activity
reaches the gateway only as relayed ``subagent.*`` events on the PARENT sid
(tagged with ``child_session_id``). When a UI resumes the child's own session
(desktop open-in-new-window), ``_mirror_subagent_to_child`` translates those
relayed events into native stream events on the CHILD's live sid so the window
shows a real midstream turn instead of sitting silent until persistence.
"""

from __future__ import annotations

import json

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server():
    # Mocks are scoped to the initial import only (see
    # tests/tui_gateway/test_protocol.py for the rationale).
    with patch.dict(
        "sys.modules",
        {
            "hermes_constants": MagicMock(
                get_hermes_home=MagicMock(return_value="/tmp/hermes_test_child_mirror")
            ),
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        },
    ):
        import importlib

        mod = importlib.import_module("tui_gateway.server")

    yield mod
    mod._sessions.clear()
    __import__("tui_gateway.server_requests", fromlist=["x"]).reset_for_tests()
    mod._child_mirrors.clear()
    mod._active_child_runs.clear()


@pytest.fixture()
def emits(server, monkeypatch):
    captured: list = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: captured.append((event, sid, payload)),
    )
    monkeypatch.setattr(server, "_tool_progress_enabled", lambda sid: True)
    return captured


def _relay(server, event_type, **payload):
    """Drive _on_tool_progress the way the delegate relay does; the parent record
    exists like any live turn's does (launch profile: no profile_home)."""
    server._sessions.setdefault("parent-sid", {"session_key": "parent"})
    server._on_tool_progress(
        "parent-sid",
        event_type,
        payload.pop("tool_name", None),
        payload.pop("preview", None),
        None,
        goal="research X",
        task_count=1,
        task_index=0,
        **payload,
    )


def test_no_live_child_session_no_mirror(server, emits):
    _relay(server, "subagent.tool", tool_name="terminal", child_session_id="child-1")

    # Only the parent-sid relay event — nothing mirrored, no state retained.
    assert [(e, s) for e, s, _ in emits] == [("subagent.tool", "parent-sid")]
    assert server._child_mirrors == {}


def test_live_child_session_gets_native_stream(server, emits):
    # A window resumed the child session: live sid differs from the stored key.
    server._sessions["live-1"] = {"session_key": "child-1", "agent": None}

    _relay(server, "subagent.tool", tool_name="terminal", preview="ls", child_session_id="child-1")
    _relay(server, "subagent.thinking", preview="hmm", child_session_id="child-1")
    _relay(server, "subagent.tool", tool_name="read_file", child_session_id="child-1")
    _relay(
        server,
        "subagent.complete",
        child_session_id="child-1",
        status="completed",
        summary="done deal",
    )

    child = [(e, p) for e, s, p in emits if s == "live-1"]

    # Synthetic turn: start → tool → reasoning → tool rotation → close + summary.
    assert [e for e, _ in child] == [
        "message.start",
        "tool.start",
        "reasoning.delta",
        "tool.complete",
        "tool.start",
        "tool.complete",
        "message.complete",
    ]
    first_tool = child[1][1]
    assert first_tool["name"] == "terminal"
    assert first_tool["tool_id"].startswith("submirror:child-1:")
    assert child[2][1] == {"text": "hmm"}
    # The rotated-out tool closes with the same id it opened with.
    assert child[3][1]["tool_id"] == first_tool["tool_id"]
    assert child[6][1] == {"text": "done deal"}

    # Parent relay is untouched alongside the mirror.
    assert [e for e, s, _ in emits if s == "parent-sid"] == [
        "subagent.tool",
        "subagent.thinking",
        "subagent.tool",
        "subagent.complete",
    ]
    # Completion clears mirror state.
    assert server._child_mirrors == {}


def test_window_closed_midrun_drops_state_then_fresh_turn_on_reopen(server, emits):
    server._sessions["live-1"] = {"session_key": "child-1", "agent": None}
    _relay(server, "subagent.tool", tool_name="terminal", child_session_id="child-1")
    assert (None, "child-1") in server._child_mirrors

    # Window closes → live session gone → state dropped on the next event.
    server._sessions.clear()
    _relay(server, "subagent.tool", tool_name="read_file", child_session_id="child-1")
    assert server._child_mirrors == {}

    # Reopen under a new live sid → a fresh synthetic turn starts.
    emits.clear()
    server._sessions["live-2"] = {"session_key": "child-1", "agent": None}
    _relay(server, "subagent.tool", tool_name="web_search", child_session_id="child-1")
    assert [(e, s) for e, s, _ in emits if s == "live-2"] == [
        ("message.start", "live-2"),
        ("tool.start", "live-2"),
    ]


def test_upgraded_child_session_not_mirrored(server, emits):
    """A watch window upgraded to a full session (agent built) owns a real
    native stream — mirroring on top would interleave two turns on one sid."""
    server._sessions["live-1"] = {"session_key": "child-1", "agent": object()}

    _relay(server, "subagent.tool", tool_name="terminal", child_session_id="child-1")

    assert [(e, s) for e, s, _ in emits] == [("subagent.tool", "parent-sid")]
    assert server._child_mirrors == {}
    # Liveness registry still updates — it serves resume, not the mirror.
    assert (None, "child-1") in server._active_child_runs


def test_stale_child_run_not_reported_active(server, emits):
    """A leaked registry entry (lost completion event) must age out instead of
    pinning running=true on every future lazy resume of that child."""
    server._active_child_runs[(None, "child-1")] = 0.0  # epoch — ancient

    assert server._child_run_active("child-1", None) is False

    _relay(server, "subagent.tool", tool_name="terminal", child_session_id="child-1")
    assert server._child_run_active("child-1", None) is True


def test_prompt_submit_rejected_while_child_run_active(server, emits):
    """Typing into a watch window mid-run must not build a second agent racing
    the in-flight child on the same stored session — busy error instead."""
    import threading

    server._sessions["live-1"] = {
        "agent": None,
        "history_lock": threading.Lock(),
        "lazy": True,
        "running": False,
        "session_key": "child-1",
    }
    _relay(server, "subagent.tool", tool_name="terminal", child_session_id="child-1")

    result = server._methods["prompt.submit"]("rid-1", {"session_id": "live-1", "text": "hi"})
    assert result["error"]["code"] == 4009

    # Run completes → the same submit upgrades into a real conversation
    # (passes the guard; fails later only because this test stubs no agent).
    _relay(server, "subagent.complete", child_session_id="child-1", status="completed", summary="ok")
    assert server._child_run_active("child-1", None) is False


def test_active_child_runs_registry_tracks_liveness(server, emits):
    """Every relayed event marks the child as in flight (even with no window
    open), and completion clears it — lazy watch resumes read this registry to
    report running=true while the child is silent inside a long tool call."""
    _relay(server, "subagent.start", preview="go", child_session_id="child-1")
    assert (None, "child-1") in server._active_child_runs

    _relay(server, "subagent.tool", tool_name="terminal", child_session_id="child-1")
    assert (None, "child-1") in server._active_child_runs

    _relay(server, "subagent.complete", child_session_id="child-1", status="completed", summary="ok")
    assert (None, "child-1") not in server._active_child_runs


def test_start_mirrors_as_immediate_header_line(server, emits):
    server._sessions["live-1"] = {"session_key": "child-1", "agent": None}

    # subagent.start emits a one-time header (the goal) so a freshly opened
    # window shows context immediately. subagent.progress (batched tool-name
    # rollups) no longer pollutes the message body — tools mirror natively via
    # tool.start and the reply streams via subagent.text.
    _relay(server, "subagent.start", preview="starting child branch", child_session_id="child-1")
    _relay(server, "subagent.progress", preview="step 1/3", child_session_id="child-1")

    child = [(e, p) for e, s, p in emits if s == "live-1"]
    assert child == [
        ("message.start", None),
        ("message.delta", {"text": "starting child branch\n"}),
    ]


def test_text_mirrors_as_message_delta(server, emits):
    """The child's streamed reply (subagent.text) becomes a native
    message.delta on the live child sid — the watch window streams it as the
    agent 'talking', the piece that was previously missing entirely."""
    server._sessions["live-1"] = {"session_key": "child-1", "agent": None}

    _relay(server, "subagent.text", preview="Here is ", child_session_id="child-1")
    _relay(server, "subagent.text", preview="the answer.", child_session_id="child-1")

    child = [(e, p) for e, s, p in emits if s == "live-1"]
    assert child == [
        ("message.start", None),
        ("message.delta", {"text": "Here is "}),
        ("message.delta", {"text": "the answer."}),
    ]




@pytest.mark.parametrize("results", [
    ('{"output":"actual stdout","exit_code":0}', '{"output":"failed stderr","exit_code":2}'),
    ("", "null"),
])
def test_real_child_relay_mirrors_parallel_results_with_durable_ids(server, emits, results):
    from tools.delegate_tool_progress import _build_child_progress_callback

    parent = MagicMock()
    parent._delegate_spinner = None
    parent.tool_progress_callback = lambda *args, **kw: server._on_tool_progress("parent-sid", *args, **kw)
    server._sessions["parent-sid"] = {"session_key": "parent"}
    server._sessions["live-1"] = {"session_key": "child-1", "agent": None}
    relay = _build_child_progress_callback(0, "inspect", parent, session_ref={"session_id": "child-1"})

    relay("tool.started", "terminal", "echo a", {"command": "echo a"}, tool_call_id="call-a")
    relay("tool.started", "terminal", "echo b", {"command": "echo b"}, tool_call_id="call-b")
    # Completing B first must never close A or attach B's output to A.
    relay("tool.completed", "terminal", result=results[1], tool_call_id="call-b", duration=2)
    relay("tool.completed", "terminal", result=results[0], tool_call_id="call-a", duration=3)
    relay("subagent.complete", status="completed", summary="done")

    child_tools = [(event, payload) for event, sid, payload in emits if sid == "live-1" and event.startswith("tool.")]
    assert [event for event, _ in child_tools] == ["tool.start", "tool.start", "tool.complete", "tool.complete"]
    assert [payload["tool_id"] for _, payload in child_tools] == ["call-a", "call-b", "call-b", "call-a"]
    from tui_gateway.contracts.events import ToolCompletePayload

    for event, payload in child_tools:
        if event == "tool.complete":
            ToolCompletePayload.model_validate(payload)

    def decode(value):
        try:
            return json.loads(value)
        except ValueError:
            return value
    assert child_tools[2][1]["result"] == decode(results[1])
    assert child_tools[3][1]["result"] == decode(results[0])
    assert child_tools[3][1]["args"] == {"command": "echo a"}
    assert child_tools[3][1]["duration_s"] == 3
    # The parent receives activity only, never this private child output.
    assert all("result" not in (payload or {}) for _, sid, payload in emits if sid == "parent-sid")


def test_child_mirror_does_not_fabricate_success_for_lost_completion(server, emits):
    server._sessions["live-1"] = {"session_key": "child-1", "agent": None}
    _relay(server, "subagent.tool", tool_name="terminal", child_session_id="child-1", tool_call_id="missing")
    _relay(server, "subagent.complete", child_session_id="child-1", status="completed", summary="done")
    completions = [payload for event, sid, payload in emits if event == "tool.complete" and sid == "live-1"]
    assert len(completions) == 1
    assert "result" not in completions[0]


def test_connector_child_output_is_redacted_without_parent_owner_mismatch(server, emits):
    parent = {"session_key": "parent"}
    server._sessions["parent-sid"] = parent
    server._sessions["live-1"] = {"session_key": "child-1", "agent": None}
    token = server._current_runtime_session_record.set(parent)
    try:
        _relay(server, "subagent.tool", tool_name="manage_connections", child_session_id="child-1", tool_call_id="connector")
        _relay(server, "subagent.tool_complete", tool_name="manage_connections", child_session_id="child-1",
               tool_call_id="connector", result={"api_key": "private-key", "status": "connected"})
    finally:
        server._current_runtime_session_record.reset(token)
    completion = next(payload for event, sid, payload in emits if event == "tool.complete" and sid == "live-1")
    assert completion["result"] == {"api_key": "[REDACTED]", "status": "connected"}


def test_native_child_failure_signal_survives_actual_bridge_and_mirror(server, emits):
    from types import SimpleNamespace
    from agent.codex_runtime import make_codex_app_server_event_bridge
    from tools.delegate_tool_progress import _build_child_progress_callback
    from tui_gateway.contracts.events import ToolCompletePayload

    parent = SimpleNamespace(_delegate_spinner=None,
        tool_progress_callback=lambda *args, **kw: server._on_tool_progress("parent-sid", *args, **kw))
    server._sessions["parent-sid"] = {"session_key": "parent"}
    server._sessions["live-1"] = {"session_key": "child-1", "agent": None}
    relay = _build_child_progress_callback(0, "inspect", parent, session_ref={"session_id": "child-1"})
    agent = SimpleNamespace(tool_progress_callback=relay)
    bridge = make_codex_app_server_event_bridge(agent)
    item = {"type": "dynamicToolCall", "id": "rejected", "tool": "custom_tool", "arguments": {}}
    bridge({"method": "item/started", "params": {"item": item}})
    content = [{"type": "text", "text": "Operation was declined"}]
    bridge({"method": "item/completed", "params": {"item": {**item, "success": False, "contentItems": content}}})

    completion = next(payload for event, sid, payload in emits if event == "tool.complete" and sid == "live-1")
    ToolCompletePayload.model_validate(completion)
    assert completion["result"] == content
    assert completion["error"] is True
