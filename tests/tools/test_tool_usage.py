"""Tests for the opt-in per-tool usage tracker (tools/tool_usage.py)."""

from __future__ import annotations

import json
import os
import tempfile
import shutil

import pytest


@pytest.fixture
def hermes_home(monkeypatch):
    d = tempfile.mkdtemp(prefix="hermes_tool_usage_test_")
    home = os.path.join(d, ".hermes")
    os.makedirs(home)
    monkeypatch.setenv("HERMES_HOME", home)
    yield home
    shutil.rmtree(d, ignore_errors=True)


def _set_analytics_on(hermes_home, enabled=True):
    import hermes_cli.config as cfg
    c = cfg.load_config()
    c.setdefault("tools", {})["analytics"] = enabled
    cfg.save_config(c)


def _record_calls(hermes_home, count=5, tool="web_search"):
    from tools.tool_usage import record_call
    for i in range(count):
        record_call(tool, result='{"ok": true}', session_id="sess-1")


def test_disabled_by_default(hermes_home):
    from tools.tool_usage import is_enabled
    assert is_enabled() is False


def test_enabled_after_config(hermes_home):
    _set_analytics_on(hermes_home, True)
    from tools.tool_usage import is_enabled
    assert is_enabled() is True


def test_record_call_stores_data(hermes_home):
    _set_analytics_on(hermes_home, True)
    _record_calls(hermes_home, 3, "terminal")
    from tools.tool_usage import tool_summary
    summary = tool_summary()
    assert summary["total_calls"] == 3
    assert len(summary["tools"]) == 1
    assert summary["tools"][0]["tool_name"] == "terminal"
    assert summary["tools"][0]["total"] == 3


def test_tool_summary_multiple_tools(hermes_home):
    _set_analytics_on(hermes_home, True)
    _record_calls(hermes_home, 4, "web_search")
    _record_calls(hermes_home, 2, "read_file")
    from tools.tool_usage import tool_summary
    summary = tool_summary()
    assert summary["total_calls"] == 6
    assert len(summary["tools"]) == 2


def test_cost_summary(hermes_home):
    _set_analytics_on(hermes_home, True)
    _record_calls(hermes_home, 2, "terminal")
    from tools.tool_usage import cost_summary
    costs = cost_summary()
    assert costs["total_tokens"] > 0


def test_suggest_prune_empty_when_no_data(hermes_home):
    from tools.tool_usage import suggest_prune
    assert suggest_prune() == []


def test_analytics_report_empty(hermes_home):
    _set_analytics_on(hermes_home, True)
    from tools.tool_usage import analytics_report
    report = analytics_report()
    assert "No tool usage data" in report


def test_analytics_report_with_data(hermes_home):
    _set_analytics_on(hermes_home, True)
    _record_calls(hermes_home, 5, "web_search")
    from tools.tool_usage import analytics_report
    report = analytics_report()
    assert "web_search" in report
    assert "5 calls" in report
    assert "100%" in report


def test_record_call_no_effect_when_disabled(hermes_home):
    from tools.tool_usage import record_call, tool_summary
    record_call("terminal", result='{"ok": true}')
    summary = tool_summary()
    assert summary["total_calls"] == 0


def test_record_call_detects_failure_from_result(hermes_home):
    _set_analytics_on(hermes_home, True)
    from tools.tool_usage import record_call, tool_summary
    record_call("web_search", result='{"error": "rate limited"}', session_id="sess-1")
    record_call("browser", result="[TOOL_ERROR] crashed", session_id="sess-1")
    record_call("read_file", result='{"ok": true}', session_id="sess-1")
    summary = tool_summary()
    assert summary["total_calls"] == 3
    web = next(t for t in summary["tools"] if t["tool_name"] == "web_search")
    browser = next(t for t in summary["tools"] if t["tool_name"] == "browser")
    read = next(t for t in summary["tools"] if t["tool_name"] == "read_file")
    assert web["successes"] == 0
    assert browser["successes"] == 0
    assert read["successes"] == 1