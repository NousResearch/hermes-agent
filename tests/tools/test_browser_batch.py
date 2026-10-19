"""browser_batch on the built-in browser (the lane without an extension controller).

The steps run one after another through the same handlers as the single tools and the result has the
extension's shape (per-step entries, ``completed``/``remaining``), so both lanes share one schema.
A step the built-in browser cannot run is reported on that step; nothing is dropped silently.
"""

import json

import pytest

from tools import browser_tool
from tools import browser_tool_eval_policy as bt_eval_policy
from tools import browser_tool_session as bt_session
from tools.registry import registry


@pytest.fixture
def browser(monkeypatch):
    """The built-in browser with a fake command runner; returns the list of ``(command, args)`` it ran."""
    commands = []

    def run(task_id, command, args, **_kwargs):
        commands.append((command, list(args)))
        if command == "snapshot":
            return {"success": True, "data": {"snapshot": "- button [ref=e1]", "refs": {"e1": {}}}}
        if command == "click" and args == ["@e404"]:
            return {"success": False, "error": "Element @e404 not found"}
        return {"success": True, "data": {}}

    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_last_session_key", lambda task_id: task_id)
    monkeypatch.setattr(bt_eval_policy, "_eval_ssrf_guard_active", lambda task_id: False)
    monkeypatch.setattr(bt_session, "_run_browser_command", run)
    return commands


def _run(actions, **extra):
    """Through the registry handler, as the agent calls it (feature off: the built-in browser runs)."""
    return json.loads(registry.get_entry("browser_batch").handler({"actions": actions, **extra}, task_id="task-1"))


def test_steps_run_in_order_through_the_single_tool_handlers(browser):
    out = _run([
        {"action": "click", "ref": "@e1"},
        {"action": "type", "ref": "@e1", "text": "hello"},
        {"action": "press", "key": "Enter"},
        {"action": "scroll", "direction": "down"},
    ])

    assert [command for command, _ in browser] == ["click", "fill", "press", "scroll", "snapshot"]
    assert browser[1] == ("fill", ["@e1", "hello"])
    assert out["success"] is True and out["status"] == "completed"
    assert [(s["index"], s["action"], s["status"]) for s in out["steps"]] == [
        (0, "click", "ok"), (1, "type", "ok"), (2, "press", "ok"), (3, "scroll", "ok")]
    assert (out["completed"], out["remaining"]) == (4, 0)
    assert out["snapshot"] == "- button [ref=e1]" and out["element_count"] == 1


def test_the_default_stops_at_the_first_failing_step(browser):
    out = _run([{"action": "click", "ref": "@e1"}, {"action": "click", "ref": "@e404"}, {"action": "press", "key": "Tab"}])

    assert [command for command, _ in browser] == ["click", "click", "snapshot"]
    assert out["success"] is False and out["status"] == "stopped" and out["stopped_reason"] == "command_error"
    assert out["steps"][1] == {"index": 1, "action": "click", "error": "command_error", "message": "Element @e404 not found"}
    assert (out["completed"], out["remaining"]) == (1, 1)
    assert len(out["steps"]) == 2


def test_stop_on_error_false_carries_on_past_a_failing_step(browser):
    out = _run([{"action": "click", "ref": "@e404"}, {"action": "press", "key": "Tab"}], stop_on_error=False)

    assert [command for command, _ in browser] == ["click", "press", "snapshot"]
    assert out["status"] == "completed_with_errors" and out["success"] is False
    assert "stopped_reason" not in out
    assert (out["completed"], out["remaining"]) == (1, 0)
    assert "error" in out["steps"][0] and out["steps"][1]["status"] == "ok"


def test_navigate_and_back_steps_use_their_handlers(browser, monkeypatch):
    opened = []
    monkeypatch.setattr(browser_tool, "browser_navigate",
                        lambda url, task_id=None: opened.append(url) or json.dumps({"success": True, "snapshot": "big"}))

    out = _run([{"action": "navigate", "url": "https://example.test"}, {"action": "back"}])

    assert opened == ["https://example.test"]
    assert ("back", []) in browser
    assert out["status"] == "completed"
    assert out["snapshot"] == "- button [ref=e1]"  # one closing snapshot, not the navigate step's


def test_a_step_the_built_in_browser_cannot_run_is_reported_not_skipped(browser):
    out = _run([{"action": "hover", "ref": "@e1"}, {"action": "press", "key": "Tab"}], stop_on_error=False)

    assert out["steps"][0]["error"] == "unsupported_action"
    assert "cannot run hover steps" in out["steps"][0]["message"] and "Extension" in out["steps"][0]["message"]
    assert out["steps"][1]["status"] == "ok"
    assert [command for command, _ in browser] == ["press", "snapshot"]


def test_a_step_argument_the_built_in_browser_does_not_support_is_reported_not_dropped(browser):
    out = _run([{"action": "click", "x": 10, "y": 20}], stop_on_error=False)

    assert out["steps"][0]["error"] == "unsupported_argument"
    assert "x, y" in out["steps"][0]["message"]
    assert browser == []  # nothing ran, and no snapshot is taken for a batch that did nothing


def test_an_invalid_batch_is_refused_before_any_step_runs(browser):
    out = _run([{"action": "back"}] * 21)

    assert out["success"] is False and "at most 20" in out["error"]
    assert browser == []
