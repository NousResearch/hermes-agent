"""Tests for tools/qq_group_moderation_tool.py."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform


def _run_async_immediately(coro):
    return asyncio.run(coro)


def _make_qq_napcat_config():
    platform = getattr(Platform, "QQ_NAPCAT")
    qq_cfg = SimpleNamespace(
        enabled=True,
        token=None,
        api_key=None,
        extra={
            "ws_url": "ws://127.0.0.1:3001",
            "admin_users": ["179033731"],
        },
    )
    return SimpleNamespace(platforms={platform: qq_cfg}), qq_cfg


def test_mute_group_member_calls_member_info_then_set_group_ban(monkeypatch):
    from tools.qq_group_moderation_tool import qq_group_moderation_tool

    config, qq_cfg = _make_qq_napcat_config()
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "qq_napcat")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "group")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "987654321")

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_moderation_tool.request_dangerous_action_approval",
             return_value={"approved": True, "message": None},
         ) as approval_mock, \
         patch(
             "tools.qq_group_moderation_tool._qq_napcat_call",
             new=AsyncMock(side_effect=[
                 ({"user_id": 123456, "role": "member", "nickname": "刷屏怪"}, None),
                 ({"user_id": 999001}, None),
                 ({}, None),
             ]),
         ) as call_mock:
        result = json.loads(
            qq_group_moderation_tool(
                {
                    "action": "mute_user",
                    "user_id": "123456",
                    "duration_seconds": 600,
                    "reason": "连续刷屏",
                }
            )
        )

    assert result["success"] is True
    assert result["action"] == "mute_user"
    assert result["group_id"] == "987654321"
    assert result["user_id"] == "123456"
    assert result["duration_seconds"] == 600
    approval_mock.assert_called_once()
    assert call_mock.await_args_list[0].args == (
        qq_cfg.extra,
        "get_group_member_info",
        {"group_id": 987654321, "user_id": 123456, "no_cache": True},
    )
    assert call_mock.await_args_list[1].args == (
        qq_cfg.extra,
        "get_login_info",
        {},
    )
    assert call_mock.await_args_list[2].args == (
        qq_cfg.extra,
        "set_group_ban",
        {"group_id": 987654321, "user_id": 123456, "duration": 600},
    )


def test_kick_rejects_group_owner_before_executing(monkeypatch):
    from tools.qq_group_moderation_tool import qq_group_moderation_tool

    config, qq_cfg = _make_qq_napcat_config()
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "qq_napcat")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "group")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "987654321")

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_moderation_tool.request_dangerous_action_approval",
             return_value={"approved": True, "message": None},
         ), \
         patch(
             "tools.qq_group_moderation_tool._qq_napcat_call",
             new=AsyncMock(return_value=({"user_id": 123456, "role": "owner", "nickname": "群主"}, None)),
         ) as call_mock:
        result = json.loads(
            qq_group_moderation_tool(
                {
                    "action": "kick_user",
                    "user_id": "123456",
                    "reason": "测试踢人",
                }
            )
        )

    assert "owner" in result["error"].lower() or "群主" in result["error"]
    assert call_mock.await_count == 1
    assert call_mock.await_args.args == (
        qq_cfg.extra,
        "get_group_member_info",
        {"group_id": 987654321, "user_id": 123456, "no_cache": True},
    )


def test_mute_group_member_resolves_user_query_by_member_name(monkeypatch):
    from tools.qq_group_moderation_tool import qq_group_moderation_tool

    config, qq_cfg = _make_qq_napcat_config()
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "qq_napcat")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "group")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "987654321")

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_moderation_tool.request_dangerous_action_approval",
             return_value={"approved": True, "message": None},
         ) as approval_mock, \
         patch(
             "tools.qq_group_moderation_tool._qq_napcat_call",
             new=AsyncMock(side_effect=[
                 ([{"user_id": 123456, "card": "小美", "nickname": "Alice"}], None),
                 ({"user_id": 123456, "role": "member", "card": "小美", "nickname": "Alice"}, None),
                 ({"user_id": 999001}, None),
                 ({}, None),
             ]),
         ) as call_mock:
        result = json.loads(
            qq_group_moderation_tool(
                {
                    "action": "mute_user",
                    "user_query": "小美",
                    "duration_seconds": 600,
                    "reason": "连续刷屏",
                }
            )
        )

    assert result["success"] is True
    assert result["user_id"] == "123456"
    approval_mock.assert_called_once()
    assert call_mock.await_args_list[0].args == (
        qq_cfg.extra,
        "get_group_member_list",
        {"group_id": 987654321},
    )


def test_user_query_rejects_ambiguous_member_name(monkeypatch):
    from tools.qq_group_moderation_tool import qq_group_moderation_tool

    config, qq_cfg = _make_qq_napcat_config()
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "qq_napcat")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "group")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "987654321")

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_moderation_tool._qq_napcat_call",
             new=AsyncMock(return_value=([
                 {"user_id": 123456, "card": "小美", "nickname": "Alice"},
                 {"user_id": 654321, "card": "小美", "nickname": "Alicia"},
             ], None)),
         ) as call_mock:
        result = json.loads(
            qq_group_moderation_tool(
                {
                    "action": "mute_user",
                    "user_query": "小美",
                    "duration_seconds": 600,
                    "reason": "连续刷屏",
                }
            )
        )

    assert "多个" in result["error"]
    assert "123456" in result["error"]
    assert "654321" in result["error"]
    assert call_mock.await_count == 1


def test_requires_group_target_when_not_in_qq_group_session(monkeypatch):
    from tools.qq_group_moderation_tool import qq_group_moderation_tool

    config, _qq_cfg = _make_qq_napcat_config()
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "qq_napcat")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "dm")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "179033731")

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False):
        result = json.loads(
            qq_group_moderation_tool(
                {
                    "action": "mute_user",
                    "user_id": "123456",
                    "duration_seconds": 600,
                    "reason": "连续刷屏",
                }
            )
        )

    assert "group" in result["error"].lower()
    assert "target" in result["error"].lower()


def test_returns_error_when_action_is_not_approved(monkeypatch):
    from tools.qq_group_moderation_tool import qq_group_moderation_tool

    config, _qq_cfg = _make_qq_napcat_config()
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "qq_napcat")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "group")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "987654321")

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_moderation_tool.request_dangerous_action_approval",
             return_value={
                 "approved": False,
                 "message": "这事我得先取得董事长的授权。",
             },
         ), \
         patch(
             "tools.qq_group_moderation_tool._qq_napcat_call",
             new=AsyncMock(side_effect=[
                 ({"user_id": 123456, "role": "member", "nickname": "刷屏怪"}, None),
                 ({"user_id": 999001}, None),
             ]),
         ):
        result = json.loads(
            qq_group_moderation_tool(
                {
                    "action": "kick_user",
                    "user_id": "123456",
                    "target": "group:987654321",
                    "reason": "测试踢人",
                }
            )
        )

    assert "董事长" in result["error"]
    assert "授权" in result["error"]


def test_protected_user_ids_include_gateway_and_session_admin_env(monkeypatch):
    from tools.qq_group_moderation_tool import _protected_user_ids

    monkeypatch.setenv("GATEWAY_ADMIN_USERS", "90001,90002")
    monkeypatch.setenv("QQ_NAPCAT_ADMIN_USERS", "90003")
    monkeypatch.setenv("HERMES_SESSION_ADMIN_USER_IDS", "90004,90005")

    protected = _protected_user_ids({"admin_users": ["179033731"], "protected_users": ["123456"]})

    assert protected == {"179033731", "123456", "90001", "90002", "90003", "90004", "90005"}


def test_kick_rejects_session_admin_user_from_env(monkeypatch):
    from tools.qq_group_moderation_tool import qq_group_moderation_tool

    config, qq_cfg = _make_qq_napcat_config()
    qq_cfg.extra["admin_users"] = []
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "qq_napcat")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "group")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "987654321")
    monkeypatch.setenv("HERMES_SESSION_ADMIN_USER_IDS", "179033731")

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_moderation_tool.request_dangerous_action_approval",
             return_value={"approved": True, "message": None},
         ) as approval_mock, \
         patch(
             "tools.qq_group_moderation_tool._qq_napcat_call",
             new=AsyncMock(return_value=({"user_id": 179033731, "role": "member", "nickname": "董事长"}, None)),
         ) as call_mock:
        result = json.loads(
            qq_group_moderation_tool(
                {
                    "action": "kick_user",
                    "user_id": "179033731",
                    "target": "group:987654321",
                    "reason": "测试踢人",
                }
            )
        )

    assert "protected" in result["error"].lower()
    approval_mock.assert_not_called()
    assert call_mock.await_count == 1


def test_mute_refuses_when_bot_identity_cannot_be_verified(monkeypatch):
    from tools.qq_group_moderation_tool import qq_group_moderation_tool

    config, qq_cfg = _make_qq_napcat_config()
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "qq_napcat")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "group")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "987654321")

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_moderation_tool.request_dangerous_action_approval",
             return_value={"approved": True, "message": None},
         ) as approval_mock, \
         patch(
             "tools.qq_group_moderation_tool._qq_napcat_call",
             new=AsyncMock(side_effect=[
                 ({"user_id": 123456, "role": "member", "nickname": "刷屏怪"}, None),
                 (None, {"success": False, "error": "napcat get_login_info failed"}),
             ]),
         ) as call_mock:
        result = json.loads(
            qq_group_moderation_tool(
                {
                    "action": "mute_user",
                    "user_id": "123456",
                    "duration_seconds": 600,
                    "reason": "连续刷屏",
                }
            )
        )

    assert "bot" in result["error"].lower()
    approval_mock.assert_not_called()
    assert call_mock.await_count == 2
