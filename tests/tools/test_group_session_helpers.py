"""Tests for generic group/session helper functions."""

from tools.group_session_helpers import (
    current_chat_delivery_target,
    current_user_dm_delivery_target,
    resolve_delivery_target,
)


def test_current_chat_delivery_target_handles_qq_group(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "qq_napcat")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "group")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "987654321")

    assert current_chat_delivery_target() == "qq_napcat:group:987654321"


def test_current_chat_delivery_target_handles_weixin_group(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "weixin")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "group")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "project@chatroom")

    assert current_chat_delivery_target() == "weixin:project@chatroom"


def test_current_user_dm_delivery_target_handles_qq(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "qq_napcat")
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "179033731")

    assert current_user_dm_delivery_target() == "qq_napcat:dm:179033731"


def test_current_user_dm_delivery_target_handles_weixin(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "weixin")
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "wxid_admin")

    assert current_user_dm_delivery_target() == "weixin:wxid_admin"


def test_resolve_delivery_target_uses_current_chat_in_active_session(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "weixin")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "group")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "project@chatroom")

    assert resolve_delivery_target("current_chat") == "weixin:project@chatroom"

