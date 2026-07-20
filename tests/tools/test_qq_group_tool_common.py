"""Compatibility tests for QQ group tool helper facade."""

from tools.qq_group_tool_common import resolve_delivery_target, resolve_group_target


def test_qq_group_tool_common_keeps_qq_prefixed_group_shorthand():
    assert resolve_delivery_target("group:123456") == "qq_napcat:group:123456"


def test_qq_group_tool_common_resolves_current_group_from_session(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "qq_napcat")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "group")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "987654321")

    assert resolve_group_target(None) == "987654321"

