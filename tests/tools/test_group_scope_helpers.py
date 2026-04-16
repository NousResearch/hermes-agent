"""Tests for generic group scope resolution helpers."""

from tools.group_scope_helpers import (
    current_group_chat_id,
    current_group_scope_key,
    resolve_group_chat_id,
)


def test_current_group_scope_key_handles_qq_group(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "qq_napcat")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "group")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "987654321")

    assert current_group_scope_key() == "qq_napcat:987654321"


def test_current_group_scope_key_handles_weixin_group(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "weixin")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "group")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "project@chatroom")

    assert current_group_scope_key() == "weixin:project@chatroom"


def test_resolve_group_chat_id_uses_explicit_target():
    assert (
        resolve_group_chat_id(
            "group:123456",
            expected_platform="qq_napcat",
            explicit_resolver=lambda value: "123456" if value == "group:123456" else "",
        )
        == "123456"
    )


def test_resolve_group_chat_id_uses_current_group_when_target_missing(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "qq_napcat")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "group")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "987654321")

    assert (
        resolve_group_chat_id(
            None,
            expected_platform="qq_napcat",
            explicit_resolver=lambda value: value,
        )
        == "987654321"
    )


def test_resolve_group_chat_id_uses_home_chat_when_configured():
    assert (
        resolve_group_chat_id(
            None,
            expected_platform="qq_napcat",
            explicit_resolver=lambda value: value,
            home_chat_id="group:888999",
        )
        == "group:888999"
    )

