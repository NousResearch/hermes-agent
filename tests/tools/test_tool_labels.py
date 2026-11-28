"""Deferred-tool progress uses the active display language."""

import pytest

from tools.tool_labels import labels_for_call


@pytest.mark.parametrize(
    ("language", "tool", "args", "expected_text"),
    [
        ("zh", "tool_search", {"queries": ["日历"]}, "正在搜索工具 · 日历"),
        ("zh", "tool_search", {"queries": []}, "正在搜索工具"),
        ("zh", "tool_describe", {"names": "calendar"}, "正在读取工具详情 · 1 个工具"),
        ("zh", "tool_describe", {"names": ["calendar", "tasks"]}, "正在读取工具详情 · 2 个工具"),
        ("en", "tool_describe", {"names": "calendar"}, "Reading tool details · 1 tool"),
        ("en", "tool_describe", {"names": ["calendar", "tasks"]}, "Reading tool details · 2 tools"),
        (None, "tool_describe", {"names": []}, "Reading tool details · 0 tools"),
        ("de", "tool_search", {"queries": "Kalender"}, "Werkzeuge werden gesucht · Kalender"),
        ("de", "tool_describe", {"names": ["calendar", "tasks"]}, "Werkzeugdetails lesen · 2 Werkzeuge"),
    ],
)
def test_bridge_tool_labels_use_the_display_language(monkeypatch, language, tool, args, expected_text):
    if language is None:
        monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    else:
        monkeypatch.setenv("HERMES_LANGUAGE", language)
    [label] = labels_for_call(tool, args)
    assert label.as_payload()["text"] == expected_text
    assert label.as_payload()["name"] == tool
