"""Regression tests for #157759 — string-typed boolean args on ``terminal``.

Models routinely emit the Python spellings ``"True"`` / ``"False"`` for the
schema's boolean parameters.  Two layers have to repair that:

* the schema-guided dispatch coercion (``tools/arg_coercion``) had no scalar
  branch for an ``anyOf`` union such as ``notify: boolean | array<string>``,
  so ``notify="True"`` reached the handler as a string and was refused;
* the handler is also reachable from callers that never run that layer
  (ACP/relay forwarding, approval-batch preparation), where a string flag was
  read for its truthiness — so ``background="False"`` started a *background*
  job and ``persist_on_release="False"`` was treated as True.
"""

import json

import pytest

from tools import terminal_tool as tt


@pytest.fixture()
def captured(monkeypatch):
    """Capture the kwargs ``_handle_terminal`` hands to ``terminal_tool``."""
    calls = {}

    def fake_terminal_tool(**kwargs):
        calls.update(kwargs)
        return json.dumps({"output": "ok", "exit_code": 0})

    monkeypatch.setattr(tt, "terminal_tool", fake_terminal_tool)
    return calls


class TestStringBooleanFlags:
    def test_background_false_string_is_not_background(self, captured):
        raw = tt._handle_terminal({"command": "echo hi", "background": "False"}, task_id="t1")
        assert "error" not in json.loads(raw)
        assert captured["background"] is False

    def test_background_true_string_starts_background(self, captured):
        tt._handle_terminal({"command": "echo hi", "background": "True"}, task_id="t1")
        assert captured["background"] is True

    def test_notify_true_string_is_accepted(self, captured):
        raw = tt._handle_terminal(
            {"command": "echo hi", "background": "True", "notify": "True"}, task_id="t1"
        )
        assert "error" not in json.loads(raw), raw
        assert captured["notify_on_complete"] is True

    def test_notify_false_string_is_a_no_notify_request(self, captured):
        raw = tt._handle_terminal(
            {"command": "echo hi", "background": "True", "notify": "False"}, task_id="t1"
        )
        assert "error" not in json.loads(raw), raw
        assert captured["notify_on_complete"] is False

    def test_persist_on_release_false_string_is_not_persisted(self, captured):
        tt._handle_terminal(
            {"command": "echo hi", "background": "True", "persist_on_release": "False"},
            task_id="t1",
        )
        assert captured["persist_on_release"] is False

    def test_pty_true_string_is_accepted(self, captured):
        tt._handle_terminal(
            {"command": "echo hi", "background": "True", "pty": "True"}, task_id="t1"
        )
        assert captured["pty"] is True

    def test_pty_false_string_on_foreground_call_is_not_refused(self, captured):
        raw = tt._handle_terminal({"command": "echo hi", "pty": "False"}, task_id="t1")
        assert "error" not in json.loads(raw), raw
        assert captured["background"] is False

    def test_notify_list_still_selects_pattern_watch(self, captured):
        tt._handle_terminal(
            {"command": "echo hi", "background": "True", "notify": ["ERR"]}, task_id="t1"
        )
        assert captured["watch_patterns"] == ["ERR"]
        assert captured["notify_on_complete"] is False

    def test_real_bools_are_unchanged(self, captured):
        tt._handle_terminal(
            {"command": "echo hi", "background": True, "notify": True, "pty": False},
            task_id="t1",
        )
        assert captured["background"] is True
        assert captured["notify_on_complete"] is True
        assert captured["pty"] is False

    def test_unparseable_notify_string_still_refused(self, captured):
        raw = tt._handle_terminal(
            {"command": "echo hi", "background": "True", "notify": "maybe"}, task_id="t1"
        )
        assert "error" in json.loads(raw)


class TestDeadlineFloorReadsStringBackground:
    def test_background_false_string_is_not_a_background_call(self):
        assert tt._terminal_deadline_floor({"command": "x", "background": "False"}) is not None

    def test_background_true_string_is_a_background_call(self):
        assert tt._terminal_deadline_floor({"command": "x", "background": "True"}) is None
