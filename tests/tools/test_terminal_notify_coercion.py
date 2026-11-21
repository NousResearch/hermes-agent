"""terminal accepts string-typed notify / notify_on_complete / watch_patterns.

DeepSeek-class models send ``"true"`` for a boolean and a JSON-encoded string for a list. The
generic schema coercer skips these three (``notify`` is a type-less anyOf; the other two are
unadvertised aliases), so the handler used to refuse them and the model retried without notify.
The job then ran silently and never woke the agent.
"""

import json

import pytest

import tools.terminal_tool as terminal_module
from tools.terminal_tool import _coerce_notify_args, _handle_terminal


@pytest.fixture
def spawned(monkeypatch):
    """Capture the keyword arguments _handle_terminal hands to terminal_tool."""
    calls = []

    def fake_terminal_tool(**kwargs):
        calls.append(kwargs)
        return json.dumps({"output": "ok", "exit_code": 0})

    monkeypatch.setattr(terminal_module, "terminal_tool", fake_terminal_tool)
    return calls


def _background(**extra):
    return {"command": "sleep 1", "background": True, **extra}


class TestNotifyBooleanStrings:
    @pytest.mark.parametrize("raw", ["true", "True", "TRUE", " true "])
    def test_notify_string_true_notifies_on_complete(self, spawned, raw):
        result = json.loads(_handle_terminal(_background(notify=raw), task_id="t1"))
        assert "error" not in result
        assert spawned[0]["notify_on_complete"] is True
        assert spawned[0]["watch_patterns"] is None

    @pytest.mark.parametrize("raw", ["false", "False", "FALSE"])
    def test_notify_string_false_is_off(self, spawned, raw):
        _handle_terminal(_background(notify=raw), task_id="t1")
        assert spawned[0]["notify_on_complete"] is False

    @pytest.mark.parametrize("raw", ["true", "TRUE", "True"])
    def test_notify_on_complete_string_true(self, spawned, raw):
        result = json.loads(_handle_terminal(_background(notify_on_complete=raw), task_id="t1"))
        assert "error" not in result
        assert spawned[0]["notify_on_complete"] is True

    @pytest.mark.parametrize("raw", ["false", "False"])
    def test_notify_on_complete_string_false_is_not_truthy(self, spawned, raw):
        _handle_terminal(_background(notify_on_complete=raw), task_id="t1")
        assert spawned[0]["notify_on_complete"] is False

    def test_real_booleans_unchanged(self, spawned):
        _handle_terminal(_background(notify=True), task_id="t1")
        _handle_terminal(_background(notify_on_complete=True), task_id="t1")
        assert [c["notify_on_complete"] for c in spawned] == [True, True]


class TestNotifyListStrings:
    def test_notify_json_encoded_list_becomes_watch_patterns(self, spawned):
        result = json.loads(_handle_terminal(_background(notify='["ERROR", "ready"]'), task_id="t1"))
        assert "error" not in result
        assert spawned[0]["watch_patterns"] == ["ERROR", "ready"]
        assert spawned[0]["notify_on_complete"] is False

    def test_watch_patterns_json_encoded_list(self, spawned):
        _handle_terminal(_background(watch_patterns='["ERROR"]'), task_id="t1")
        assert spawned[0]["watch_patterns"] == ["ERROR"]

    def test_watch_patterns_bare_string_is_one_pattern(self, spawned):
        _handle_terminal(_background(watch_patterns="ERROR"), task_id="t1")
        assert spawned[0]["watch_patterns"] == ["ERROR"]

    def test_native_list_unchanged(self, spawned):
        _handle_terminal(_background(notify=["ERROR"]), task_id="t1")
        assert spawned[0]["watch_patterns"] == ["ERROR"]


class TestStillRefused:
    @pytest.mark.parametrize("raw", ["maybe", "ERROR"])
    def test_unrecognised_notify_string_is_refused(self, spawned, raw):
        result = json.loads(_handle_terminal(_background(notify=raw), task_id="t1"))
        assert "notify must be true/false" in result["error"]
        assert spawned == []

    def test_malformed_json_list_is_refused(self, spawned):
        result = json.loads(_handle_terminal(_background(notify='["ERROR"'), task_id="t1"))
        assert "could not be parsed" in result["error"]
        assert spawned == []

    def test_unrecognised_notify_on_complete_string_is_refused(self, spawned):
        result = json.loads(_handle_terminal(_background(notify_on_complete="soon"), task_id="t1"))
        assert "notify_on_complete must be true or false" in result["error"]
        assert spawned == []


class TestForeground:
    def test_string_false_on_foreground_is_not_a_notify_request(self, spawned):
        result = json.loads(_handle_terminal({"command": "echo hi", "notify": "false"}, task_id="t1"))
        assert "error" not in result

    def test_string_true_on_foreground_is_refused_like_real_true(self, spawned):
        result = json.loads(_handle_terminal({"command": "echo hi", "notify": "true"}, task_id="t1"))
        assert "background=true" in result["error"]
        assert spawned == []


class TestHelper:
    def test_null_words_clear_the_value(self):
        assert _coerce_notify_args("null", "none", "") == (None, False, None, None)

    def test_none_passes_through(self):
        assert _coerce_notify_args(None, False, None) == (None, False, None, None)
