"""Tests for delegate_task(action='stream') — typed-event peek of a running child."""

import json
import weakref

from tools.delegate_tool import (
    _handle_control_action,
    _register_subagent,
)


class _StubChild:
    """Weakref-able stand-in for a live child AIAgent with a transcript file."""

    def __init__(self, parent=None, accept_steer: bool = True, transcript_path: str | None = None):
        self.steered: list[str] = []
        self.accept_steer = accept_steer
        self._live_transcript_path = transcript_path
        if parent is not None:
            self._delegate_parent_ref = weakref.ref(parent)


class _StubParent:
    pass


def _register(sid: str, child, **extra) -> None:
    record = {
        "subagent_id": sid,
        "parent_id": None,
        "depth": 0,
        "goal": "test goal",
        "model": "test-model",
        "started_at": 1000.0,
        "status": "running",
        "tool_count": 0,
        "agent": child,
    }
    record.update(extra)
    _register_subagent(record)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_stream_requires_subagent_id():
    payload = json.loads(_handle_control_action("stream", None, None, _StubParent()))
    assert "error" in payload


def test_stream_with_blank_subagent_id_errors():
    payload = json.loads(_handle_control_action("stream", "", None, _StubParent()))
    assert "error" in payload


def test_stream_unknown_subagent_id_errors():
    payload = json.loads(_handle_control_action("stream", "no-such-child", None, _StubParent()))
    assert "error" in payload


def test_stream_child_with_no_transcript_path():
    parent = _StubParent()
    child = _StubChild(parent=parent, transcript_path=None)
    _register("c1", child)
    payload = json.loads(_handle_control_action("stream", "c1", None, parent))
    assert payload["action"] == "stream"
    assert payload["status"] == "no_transcript"


# ---------------------------------------------------------------------------
# Real transcript parsing
# ---------------------------------------------------------------------------


def _write_transcript(tmp_path, lines: list[str]) -> str:
    p = tmp_path / "task-0.log"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


def test_stream_returns_typed_events_with_summary(tmp_path):
    parent = _StubParent()
    path = _write_transcript(tmp_path, [
        "14:22:08 start    | goal",
        "14:22:08 user     | kickoff",
        "14:22:09 assistant| planning",
        "14:22:10 tool     | -> terminal({\"command\": \"ls\"})",
        "14:22:11 result   | terminal ok 0.42s: done",
        "14:22:12 tool     | -> web_extract({\"urls\": []})",
        "14:22:13 result   | web_extract ERROR 1.20s: 404",
    ])
    child = _StubChild(parent=parent, transcript_path=path)
    _register("c2", child)

    payload = json.loads(_handle_control_action("stream", "c2", None, parent))
    assert payload["action"] == "stream"
    assert payload["subagent_id"] == "c2"
    assert payload["transcript_path"] == path
    assert payload["event_count"] == len(payload["events"])
    # 2 tool_start + 2 tool_result = 4 tool events in default (all-kinds) mode.
    assert payload["event_count"] == 7
    assert payload["summary"]["tool_call_count"] == 2
    assert payload["summary"]["tool_error_count"] == 1


def test_stream_filters_by_kinds(tmp_path):
    parent = _StubParent()
    path = _write_transcript(tmp_path, [
        "14:22:10 tool     | -> terminal({\"command\": \"ls\"})",
        "14:22:11 result   | terminal ok 0.10s: ok",
        "14:22:12 tool     | -> web_extract({\"urls\": []})",
        "14:22:13 result   | web_extract ERROR 1.20s: 404",
    ])
    child = _StubChild(parent=parent, transcript_path=path)
    _register("c3", child)

    payload = json.loads(_handle_control_action(
        "stream", "c3", None, parent,
        args={"kinds": ["tool_result"]},
    ))
    kinds = [e["kind"] for e in payload["events"]]
    assert all(k == "tool_result" for k in kinds)
    assert payload["event_count"] == 2


def test_stream_filters_by_tool_name(tmp_path):
    parent = _StubParent()
    path = _write_transcript(tmp_path, [
        "14:22:10 tool     | -> terminal({\"command\": \"ls\"})",
        "14:22:11 result   | terminal ok 0.10s: ok",
        "14:22:12 tool     | -> web_extract({\"urls\": []})",
        "14:22:13 result   | web_extract ERROR 1.20s: 404",
    ])
    child = _StubChild(parent=parent, transcript_path=path)
    _register("c4", child)

    payload = json.loads(_handle_control_action(
        "stream", "c4", None, parent,
        args={"tool_name": "terminal"},
    ))
    names = {e.get("tool_name") for e in payload["events"]}
    assert names == {"terminal"}


def test_stream_errors_only(tmp_path):
    parent = _StubParent()
    path = _write_transcript(tmp_path, [
        "14:22:10 tool     | -> terminal({\"command\": \"ls\"})",
        "14:22:11 result   | terminal ok 0.10s: ok",
        "14:22:12 tool     | -> web_extract({\"urls\": []})",
        "14:22:13 result   | web_extract ERROR 1.20s: 404",
    ])
    child = _StubChild(parent=parent, transcript_path=path)
    _register("c5", child)

    payload = json.loads(_handle_control_action(
        "stream", "c5", None, parent,
        args={"errors_only": True},
    ))
    assert payload["event_count"] == 1
    assert payload["events"][0]["is_error"] is True
    assert payload["events"][0]["tool_name"] == "web_extract"


def test_stream_caps_oversized_lines_param(tmp_path):
    parent = _StubParent()
    path = _write_transcript(tmp_path, ["x"] * 5)
    child = _StubChild(parent=parent, transcript_path=path)
    _register("c6", child)

    payload = json.loads(_handle_control_action(
        "stream", "c6", None, parent,
        lines=99999,   # way over the 1000 cap
    ))
    assert payload["lines_requested"] == 1000
    # But the file only had 5 lines.
    assert payload["lines_returned"] == 5
    assert payload["event_count"] == 5


def test_stream_reports_missing_transcript_file(tmp_path):
    parent = _StubParent()
    missing = str(tmp_path / "does-not-exist.log")
    child = _StubChild(parent=parent, transcript_path=missing)
    _register("c7", child)

    payload = json.loads(_handle_control_action("stream", "c7", None, parent))
    assert payload["status"] == "transcript_missing"
    assert payload["transcript_path"] == missing
