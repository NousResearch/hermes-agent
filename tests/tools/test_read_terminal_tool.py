"""Tests for the GUI-surface ``read_terminal`` tool."""

import json

from tools import read_terminal_tool as rt
from tests.tools import desktop_read_redaction as drr


def test_requires_callback():
    result = json.loads(rt.read_terminal_tool(callback=None))
    assert "desktop" in result["error"]


def test_passes_json_through():
    payload = {"total_lines": 3, "start": 0, "text": "hello"}
    assert json.loads(rt.read_terminal_tool(callback=lambda **_: json.dumps(payload))) == payload


def test_output_is_redacted():
    drr.assert_redacted(rt.read_terminal_tool(callback=lambda **_: json.dumps(drr.PAYLOAD)))


def test_redaction_failure_does_not_leak(monkeypatch):
    drr.assert_fail_closed(monkeypatch, lambda cb: rt.read_terminal_tool(callback=cb))


def test_non_string_leaves_kept():
    drr.assert_non_string_leaves_kept(lambda cb: rt.read_terminal_tool(callback=cb))
