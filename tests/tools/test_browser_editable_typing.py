"""Regression for #80602: clear rich editors through native input events."""

import json

import pytest

from tools import browser_tool as bt
from tools import browser_tool_session as bs
from tools.registry import registry


def _call(monkeypatch, text, *, editable=True, platform="Linux x86_64", failure=None):
    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(bt, "_blocked_private_page_action", lambda *args: None)
    calls = []

    def command(task, name, args, **kwargs):
        calls.append((task, name, args))
        if name == failure:
            return {"success": False, "error": "driver failed", "code": "human_has_control"}
        if name == "eval":
            return {"success": True, "data": {"result": json.dumps({"editable": editable, "platform": platform})}}
        return {"success": True}

    monkeypatch.setattr(bs, "_run_browser_command", command)
    result = registry.dispatch("browser_type", {"ref": "e1", "text": text}, task_id="editor-test")
    return json.loads(result), calls


def test_native_inputs_retain_the_driver_fill_path(monkeypatch):
    result, calls = _call(monkeypatch, "new text", editable=False)
    assert result["success"]
    assert [(name, args) for _, name, args in calls][-1] == ("fill", ["@e1", "new text"])
    assert not any(name in {"click", "keyboard", "press"} for _, name, _ in calls)


@pytest.mark.parametrize("platform,key", [("Linux x86_64", "Control+a"), ("Win32", "Control+a"), ("MacIntel", "Meta+a")])
def test_editor_replacement_uses_the_browser_platform_and_preserves_blank_lines(monkeypatch, platform, key):
    result, calls = _call(monkeypatch, "first\r\n\r\nlast\n", platform=platform)
    assert result["success"]
    assert all(task == "editor-test" for task, _, _ in calls)
    assert [(name, args) for _, name, args in calls][2:] == [
        ("click", ["@e1"]), ("press", [key]), ("press", ["Backspace"]),
        ("keyboard", ["inserttext", "first"]), ("press", ["Enter"]),
        ("press", ["Enter"]), ("keyboard", ["inserttext", "last"]), ("press", ["Enter"]),
    ]


def test_empty_text_clears_the_editor(monkeypatch):
    result, calls = _call(monkeypatch, "")
    assert result["success"]
    assert calls[-1][1:] == ("press", ["Backspace"])
    assert not any(name == "keyboard" for _, name, _ in calls)


@pytest.mark.parametrize("failure", ["focus", "eval", "click", "press", "keyboard"])
def test_failed_step_stops_input_and_reports_failure(monkeypatch, failure):
    result, calls = _call(monkeypatch, "new\ncontent", failure=failure)
    assert not result["success"]
    assert calls[-1][1] == failure


def test_private_page_guard_precedes_any_input_probe(monkeypatch):
    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(bt, "_blocked_private_page_action", lambda *args: json.dumps({"success": False, "error": "blocked"}))
    monkeypatch.setattr(bs, "_run_browser_command", lambda *args, **kwargs: pytest.fail("guard must run first"))
    result = registry.dispatch("browser_type", {"ref": "e1", "text": "text"}, task_id="guard-test")
    assert not json.loads(result)["success"]


def test_native_select_chooses_an_option_and_reports_a_miss(monkeypatch):
    # agent-browser `fill` on a <select> reports success and changes nothing; the option is
    # chosen in-page instead, and a no-match outcome must surface as a failure.
    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(bt, "_blocked_private_page_action", lambda *args: None)
    calls = []

    def command(task, name, args, **kwargs):
        calls.append((name, args))
        if name == "eval" and "activeElement.isContentEditable" in args[0]:
            return {"success": True, "data": {"result": json.dumps({"editable": False, "select": True})}}
        if name == "eval":
            outcome = {"ok": True} if '"Skill"' in args[0] else {"error": 'no option matches "Nope"; options: Skill'}
            return {"success": True, "data": {"result": json.dumps(outcome)}}
        return {"success": True}

    monkeypatch.setattr(bs, "_run_browser_command", command)
    ok = json.loads(registry.dispatch("browser_type", {"ref": "e2", "text": "Skill"}, task_id="select-test"))
    miss = json.loads(registry.dispatch("browser_type", {"ref": "e2", "text": "Nope"}, task_id="select-test"))
    assert ok["success"] is True
    assert miss["success"] is False and "options: Skill" in miss["error"]
    assert not any(name == "fill" for name, _ in calls)
