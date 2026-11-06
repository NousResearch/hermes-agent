"""terminal's ``notify`` (``anyOf: [boolean, array<string>]``) survives string-coercing bridges.

Gateway surfaces (Discord, desktop) delivered ``notify="true"`` and the dispatch
wrapper refused it, while the unadvertised ``notify_on_complete=true`` worked:
the schema-guided coercion layer skipped every property without a top-level
``type``. Calls go through ``model_tools.handle_function_call`` (real dispatch).
"""

import json

import pytest

import model_tools
from tools import terminal_tool as tt


@pytest.fixture
def captured(monkeypatch):
    calls = {}

    def fake_terminal_tool(**kwargs):
        calls.update(kwargs)
        return json.dumps({"output": "Background process started", "session_id": "proc_x", "exit_code": 0})

    monkeypatch.setattr(tt, "terminal_tool", fake_terminal_tool)
    return calls


def _dispatch(**args):
    return json.loads(model_tools.handle_function_call("terminal", {"command": "make test", **args}, task_id="t1"))


@pytest.mark.parametrize("notify", [True, "true", " True "])
def test_notify_true_arms_completion(captured, notify):
    result = _dispatch(background=True, notify=notify)
    assert not result.get("error"), result
    assert captured["notify_on_complete"] is True and captured["watch_patterns"] is None


@pytest.mark.parametrize("notify", [False, "false"])
def test_notify_false_stays_silent(captured, notify):
    result = _dispatch(background=True, notify=notify)
    assert not result.get("error"), result
    assert captured["notify_on_complete"] is False and captured["watch_patterns"] is None


@pytest.mark.parametrize("notify", [["startup complete"], '["startup complete"]'])
def test_notify_list_arms_watch_patterns(captured, notify):
    result = _dispatch(background=True, notify=notify)
    assert not result.get("error"), result
    assert captured["watch_patterns"] == ["startup complete"] and captured["notify_on_complete"] is False


def test_non_string_patterns_are_dropped(captured):
    result = _dispatch(background=True, notify=["ready", 7])
    assert not result.get("error"), result
    assert captured["watch_patterns"] == ["ready"]


@pytest.mark.parametrize(("notify", "shown"), [("yess", "'yess'"), ("[1, 2]", "[1, 2]"), ({"on": 1}, "{'on': 1}")])
def test_unusable_notify_is_refused_naming_the_value(captured, notify, shown):
    result = _dispatch(background=True, notify=notify)
    assert shown in result["error"]
    assert not captured


def test_stringified_heartbeat_and_booleans_are_typed(captured):
    result = _dispatch(background="true", pty="false", persist_on_release="false", heartbeat="300")
    assert not result.get("error"), result
    assert captured["background"] is True and captured["pty"] is False
    assert captured["persist_on_release"] is False
    assert captured["heartbeat"] == 300 and captured["notify_on_complete"] is True
