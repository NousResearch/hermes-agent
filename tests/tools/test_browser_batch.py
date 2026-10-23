"""browser_batch on the built-in browser (the lane without an extension controller).

Every browser session gets browser_batch, and on the built-in browser it is also how the model fills, selects,
hovers, drags or scrolls an element into view. The steps run one after another through the same handlers as the
single tools, and the result has the extension's shape (per-step entries, ``completed``/``remaining``), so both
lanes share the step fields and the result. A step the built-in browser cannot run is reported on that step, and
what a single tool would have said about the browser (an engine fallback, a bot-protection page) is kept on the
step; nothing is dropped silently.
"""

import json

import pytest

from agent.display import _detect_tool_failure
from agent.tool_guardrails import classify_tool_failure
from tools import browser_tool
from tools import browser_tool_eval_policy as bt_eval_policy
from tools import browser_tool_session as bt_session
from tools.registry import registry


class _Ran(list):
    """The ``(command, args)`` the fake runner saw, and the task each ran under (``task_ids``)."""

    def __init__(self):
        super().__init__()
        self.task_ids = []


@pytest.fixture
def browser(monkeypatch):
    """The built-in browser with a fake command runner; returns the list of ``(command, args)`` it ran."""
    commands = _Ran()

    def run(task_id, command, args, **_kwargs):
        commands.append((command, list(args)))
        commands.task_ids.append(task_id)
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
    assert "error" not in out
    assert set(browser.task_ids) == {"task-1"}  # every step and the closing snapshot, in the caller's session


def test_the_default_stops_at_the_first_failing_step(browser):
    out = _run([{"action": "click", "ref": "@e1"}, {"action": "click", "ref": "@e404"}, {"action": "press", "key": "Tab"}])

    assert [command for command, _ in browser] == ["click", "click", "snapshot"]
    assert out["success"] is False and out["status"] == "stopped" and out["stopped_reason"] == "command_error"
    assert out["steps"][1] == {"index": 1, "action": "click", "error": "command_error", "message": "Element @e404 not found"}
    assert (out["completed"], out["remaining"]) == (1, 1)
    assert len(out["steps"]) == 2
    assert out["error"] == "Step 2 (click) failed: Element @e404 not found The batch stopped there."
    # the CLI's [error] tag and the loop guardrails see it as a failed call, as they do for browser_click
    assert _detect_tool_failure("browser_batch", json.dumps(out))[0] is True
    assert classify_tool_failure("browser_batch", json.dumps(out))[0] is True


@pytest.mark.parametrize("stop_on_error", [None, True])
def test_a_null_stop_on_error_is_the_default_and_stops(browser, stop_on_error):
    """Models send null for optional arguments; null must not turn "stop at the first failure" off, or the steps
    after a failed click would type and press Enter wherever the focus happens to be."""
    out = _run([{"action": "click", "ref": "@e404"}, {"action": "type", "ref": "@e1", "text": "x"},
                {"action": "press", "key": "Enter"}], stop_on_error=stop_on_error)

    assert out["status"] == "stopped" and [command for command, _ in browser] == ["click"]


def test_stop_on_error_false_carries_on_past_a_failing_step(browser):
    out = _run([{"action": "click", "ref": "@e404"}, {"action": "press", "key": "Tab"}], stop_on_error=False)

    assert [command for command, _ in browser] == ["click", "press", "snapshot"]
    assert out["status"] == "completed_with_errors" and out["success"] is False
    assert "stopped_reason" not in out
    assert out["error"] == "Step 1 (click) failed: Element @e404 not found The other steps ran."
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


def test_every_step_action_runs_on_the_built_in_browser(browser):
    out = _run([
        {"action": "fill", "ref": "@e1", "value": "hello"},
        {"action": "type", "ref": "@e1", "text": "!", "clear": False},
        {"action": "select", "ref": "@e2", "value": "Two"},
        {"action": "hover", "ref": "@e1"},
        {"action": "drag", "ref": "@e1", "to_ref": "@e2"},
        {"action": "scroll_to", "ref": "@e2"},
        {"action": "scroll", "direction": "right", "amount": 300},
        {"action": "press", "key": "a", "modifiers": ["Control"]},
        {"action": "click", "x": 10, "y": 20},
    ])

    assert out["status"] == "completed", out
    assert [command for command, _ in browser] == [
        "fill", "focus", "press", "keyboard", "select", "hover", "drag", "scrollintoview", "scroll", "press", "mouse",
        "mouse", "mouse", "snapshot"]
    assert browser[1:4] == [("focus", ["@e1"]), ("press", ["Control+End"]), ("keyboard", ["type", "!"])]
    assert ("scroll", ["right", "300"]) in browser and ("press", ["Control+a"]) in browser
    assert browser[10:13] == [("mouse", ["move", "10", "20"]), ("mouse", ["down"]), ("mouse", ["up"])]
    assert set(browser.task_ids) == {"task-1"}


def test_a_step_action_named_like_its_tool_runs_that_tool(browser):
    out = _run([{"action": "browser_click", "ref": "@e1"}])

    assert out["status"] == "completed" and browser[0] == ("click", ["@e1"])
    assert out["steps"] == [{"index": 0, "action": "click", "status": "ok"}]


def test_a_batch_whose_first_step_fails_reads_no_page(browser):
    out = _run([{"action": "click", "ref": "@e404"}, {"action": "press", "key": "Tab"}])

    assert [command for command, _ in browser] == ["click"]
    assert not {"snapshot", "element_count", "page_error"} & set(out)


def test_a_page_that_cannot_be_read_after_the_batch_is_reported(browser, monkeypatch):
    monkeypatch.setattr(browser_tool, "browser_snapshot",
                        lambda **kw: json.dumps({"success": False, "error": "Browser session closed"}))

    out = _run([{"action": "click", "ref": "@e1"}])

    assert out["status"] == "completed" and out["page_error"] == "Browser session closed"
    assert "snapshot" not in out


def test_a_step_whose_tool_answers_something_unreadable_is_a_failed_step(browser, monkeypatch):
    monkeypatch.setattr(browser_tool, "browser_hover", lambda ref, task_id=None: "not json")

    out = _run([{"action": "hover", "ref": "@e1"}])

    assert out["steps"][0] == {"index": 0, "action": "hover", "error": "command_error",
                               "message": "browser_hover returned an unreadable result."}


def test_what_a_step_says_about_the_browser_is_kept_on_that_step(browser, monkeypatch):
    """A Lightpanda click that fell back to a throwaway Chrome session may not have reached the real page, and a
    navigation can land on a bot-protection page: the single tools say so, and so must the batch."""
    fallback = {"fallback_warning": "Ran in a temporary Chrome session.", "browser_engine": "chrome",
                "browser_engine_fallback": "lightpanda"}
    monkeypatch.setattr(browser_tool, "browser_navigate", lambda url, task_id=None: json.dumps({
        "success": True, "url": "https://example.test/landed", "title": "Just a moment...",
        "bot_detection_warning": "Possible bot detection.", "snapshot": "big"}))
    monkeypatch.setattr(browser_tool, "browser_click",
                        lambda ref="", task_id=None, x=None, y=None: json.dumps({"success": True, **fallback}))
    monkeypatch.setattr(browser_tool, "browser_snapshot", lambda **kw: json.dumps({
        "success": True, "snapshot": "- page", "element_count": 1, **fallback}))

    out = _run([{"action": "navigate", "url": "https://example.test"}, {"action": "click", "ref": "@e1"}])

    assert out["steps"][0] == {"index": 0, "action": "navigate", "status": "ok", "url": "https://example.test/landed",
                               "title": "Just a moment...", "bot_detection_warning": "Possible bot detection."}
    assert out["steps"][1] == {"index": 1, "action": "click", "status": "ok", **fallback}
    assert {key: out[key] for key in fallback} == fallback  # the closing snapshot's own fallback
    assert "snapshot" not in out["steps"][0]  # the page is read once, at the end


def test_a_step_the_active_browser_backend_cannot_do_is_reported_on_that_step(browser, monkeypatch):
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: True)
    monkeypatch.setattr(browser_tool, "_camofox", lambda name, *args: json.dumps({"success": True}))

    out = _run([{"action": "hover", "ref": "@e1"}, {"action": "press", "key": "Tab"}], stop_on_error=False)

    assert out["steps"][0]["error"] == "unsupported_in_this_browser"
    message = out["steps"][0]["message"]
    assert message.startswith("The hover step is not available in this browser: the Camofox backend has no hover")
    # a remedy the session can act on: it may have no extension to turn on
    assert "Hermes Control" not in message and "Extension" not in message and "Click the element instead" in message
    assert out["steps"][1]["status"] == "ok"
    assert browser == []


def test_an_invalid_batch_is_refused_before_any_step_runs(browser):
    out = _run([{"action": "back"}] * 21)

    assert out["success"] is False and "at most 20" in out["error"]
    assert browser == []
