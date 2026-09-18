"""Tests for ``desktop_preview`` action=screenshot — the preview pane's camera."""

import json
from types import SimpleNamespace

from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
from tools import screenshot_preview_tool as sp
from tools.registry import registry


def test_stays_an_action_of_desktop_preview():
    """The diet holds: the camera is an action, not a twelfth desktop tool."""
    import tools.preview_tool  # noqa: F401 — registers desktop_preview

    assert registry.get_entry("screenshot_preview") is None
    entry = registry.get_entry("desktop_preview")
    assert entry is not None
    assert "screenshot" in entry.schema["parameters"]["properties"]["action"]["enum"]


def test_requires_callback():
    """Outside the desktop GUI there is no bridge — a clear error, no hang."""
    result = json.loads(sp.screenshot_preview_tool(callback=None))
    assert "desktop" in result["error"]


def test_empty_answer_means_nothing_open():
    result = json.loads(sp.screenshot_preview_tool(callback=lambda: ""))
    assert "error" in result


def test_passes_json_through():
    payload = {"success": True, "path": "/tmp/preview_1.png", "width": 1200, "height": 800,
               "kind": "url", "title": "HN", "host": "news.ycombinator.com"}
    result = json.loads(sp.screenshot_preview_tool(callback=lambda: json.dumps(payload)))
    assert result == payload


def test_callback_failure_is_reported():
    def _boom():
        raise RuntimeError("preview guest is gone")

    result = json.loads(sp.screenshot_preview_tool(callback=_boom))
    assert "preview guest is gone" in result["error"]
    assert result.get("success") is not True


def test_inline_dispatch_uses_the_agent_callback():
    """action=screenshot must reach the GUI callback the gateway registers; the
    registry handler cannot see it."""
    agent = SimpleNamespace(screenshot_preview_callback=lambda: json.dumps({"success": True, "path": "/tmp/p.png"}))
    ctx = InlineToolContext(effective_task_id="task-1", tool_call_id="call-1")
    result = json.loads(INLINE_TOOL_EXECUTORS["desktop_preview"](agent, {"action": "screenshot"}, ctx))
    assert result == {"success": True, "path": "/tmp/p.png"}


def test_inline_dispatch_without_a_callback_teaches():
    agent = SimpleNamespace()
    ctx = InlineToolContext(effective_task_id="task-1", tool_call_id="call-1")
    result = json.loads(INLINE_TOOL_EXECUTORS["desktop_preview"](agent, {"action": "screenshot"}, ctx))
    assert "desktop" in result["error"]
