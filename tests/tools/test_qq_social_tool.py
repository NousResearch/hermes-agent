"""Tests for tools/qq_social_tool.py."""

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.config import Platform
from gateway.qq_social_policy import get_social_policy
from gateway.qq_social_requests import record_social_request_event
from tools.qq_social_tool import qq_social_tool


def _run_async_immediately(coro):
    return asyncio.run(coro)


def _make_qq_napcat_config():
    platform = getattr(Platform, "QQ_NAPCAT")
    qq_cfg = SimpleNamespace(
        enabled=True,
        token=None,
        api_key=None,
        extra={"ws_url": "ws://127.0.0.1:3001"},
    )
    return SimpleNamespace(platforms={platform: qq_cfg}), qq_cfg


def test_list_requests_returns_pending_entries():
    record_social_request_event(
        {
            "post_type": "request",
            "request_type": "group",
            "sub_type": "add",
            "group_id": 987654321,
            "user_id": 179033731,
            "comment": "想让机器人进群",
            "flag": "group-flag-2",
            "time": 1713012345,
        }
    )
    config, _qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch.dict(
             os.environ,
             {
                 "HERMES_SESSION_PLATFORM": "qq_napcat",
                 "HERMES_SESSION_CHAT_TYPE": "dm",
                 "HERMES_SESSION_CHAT_ID": "179033731",
                 "HERMES_SESSION_USER_ID": "179033731",
                 "HERMES_SESSION_IS_ADMIN": "true",
             },
             clear=False,
         ):
        result = json.loads(
            qq_social_tool(
                {
                    "action": "list_requests",
                    "status": "pending",
                }
            )
        )

    assert result["success"] is True
    assert result["requests"][0]["request_key"] == "group:group-flag-2"


def test_list_requests_returns_filters_and_state_summary():
    record_social_request_event(
        {
            "post_type": "request",
            "request_type": "friend",
            "user_id": 456789,
            "comment": "加个好友",
            "flag": "friend-flag-state-1",
            "time": 1713012345,
        }
    )
    record_social_request_event(
        {
            "post_type": "request",
            "request_type": "group",
            "sub_type": "invite",
            "group_id": 987654321,
            "user_id": 179033731,
            "comment": "来群里聊项目",
            "flag": "group-flag-state-1",
            "time": 1713012401,
        }
    )
    config, _qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch.dict(
             os.environ,
             {
                 "HERMES_SESSION_PLATFORM": "qq_napcat",
                 "HERMES_SESSION_CHAT_TYPE": "dm",
                 "HERMES_SESSION_CHAT_ID": "179033731",
                 "HERMES_SESSION_USER_ID": "179033731",
                 "HERMES_SESSION_IS_ADMIN": "true",
             },
             clear=False,
         ):
        result = json.loads(
            qq_social_tool(
                {
                    "action": "list_requests",
                    "status": "pending",
                    "request_type": "friend",
                    "limit": 5,
                }
            )
        )

    assert result["success"] is True
    assert result["filters"] == {"status": "pending", "request_type": "friend", "limit": 5}
    assert result["summary"]["total"] == 1
    assert result["summary"]["actionable"] == 1
    assert result["summary"]["by_type"] == {"friend": 1, "group": 0}
    assert result["requests"][0]["request_key"] == "friend:friend-flag-state-1"
    assert result["requests"][0]["request_state"]["available_actions"] == ["approve_request", "reject_request"]


def test_approve_group_request_calls_api_and_marks_request_approved():
    record_social_request_event(
        {
            "post_type": "request",
            "request_type": "group",
            "sub_type": "invite",
            "group_id": 987654321,
            "user_id": 179033731,
            "comment": "请进群",
            "flag": "group-flag-3",
            "time": 1713012345,
        }
    )
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_social_tool._qq_napcat_call",
             new=AsyncMock(return_value=({"ok": True}, None)),
         ) as call_mock, \
         patch.dict(
             os.environ,
             {
                 "HERMES_SESSION_PLATFORM": "qq_napcat",
                 "HERMES_SESSION_CHAT_TYPE": "dm",
                 "HERMES_SESSION_CHAT_ID": "179033731",
                 "HERMES_SESSION_USER_ID": "179033731",
                 "HERMES_SESSION_IS_ADMIN": "true",
             },
             clear=False,
         ):
        result = json.loads(
            qq_social_tool(
                {
                    "action": "approve_request",
                    "request_key": "group:group-flag-3",
                }
            )
        )

    assert result["success"] is True
    assert result["request"]["status"] == "approved"
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "set_group_add_request",
        {"flag": "group-flag-3", "sub_type": "invite", "approve": True},
    )


def test_get_user_profile_calls_get_stranger_info():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_social_tool._qq_napcat_call",
             new=AsyncMock(
                 return_value=(
                     {"user_id": 456789, "nickname": "Alice", "sex": "female", "age": 22},
                     None,
                 )
             ),
         ) as call_mock, \
         patch.dict(
             os.environ,
             {
                 "HERMES_SESSION_PLATFORM": "qq_napcat",
                 "HERMES_SESSION_CHAT_TYPE": "dm",
                 "HERMES_SESSION_CHAT_ID": "179033731",
                 "HERMES_SESSION_USER_ID": "179033731",
                 "HERMES_SESSION_IS_ADMIN": "true",
             },
             clear=False,
         ):
        result = json.loads(
            qq_social_tool(
                {
                    "action": "get_user_profile",
                    "user_id": "456789",
                }
            )
        )

    assert result["success"] is True
    assert result["profile"]["user_id"] == "456789"
    assert result["profile"]["nickname"] == "Alice"
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "get_stranger_info",
        {"user_id": 456789, "no_cache": False},
    )


def test_set_social_policy_updates_auto_handling_flags():
    config, _qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch.dict(
             os.environ,
             {
                 "HERMES_SESSION_PLATFORM": "qq_napcat",
                 "HERMES_SESSION_CHAT_TYPE": "dm",
                 "HERMES_SESSION_CHAT_ID": "179033731",
                 "HERMES_SESSION_USER_ID": "179033731",
                 "HERMES_SESSION_USER_NAME": "發發發",
                 "HERMES_SESSION_IS_ADMIN": "true",
             },
             clear=False,
         ):
        result = json.loads(
            qq_social_tool(
                {
                    "action": "set_social_policy",
                    "auto_approve_friend_requests": True,
                    "auto_approve_group_add_requests": True,
                    "auto_approve_group_invites": True,
                    "notify_target": "current_user_dm",
                    "notes": "董事长已批准自动处理社交请求",
                }
            )
        )

    assert result["success"] is True
    policy = result["policy"]
    assert policy["auto_approve_friend_requests"] is True
    assert policy["auto_approve_group_add_requests"] is True
    assert policy["auto_approve_group_invites"] is True
    assert policy["notify_target"] == "qq_napcat:dm:179033731"
    assert policy["notes"] == "董事长已批准自动处理社交请求"


def test_get_social_policy_returns_saved_policy():
    config, _qq_cfg = _make_qq_napcat_config()
    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch.dict(
             os.environ,
             {
                 "HERMES_SESSION_PLATFORM": "qq_napcat",
                 "HERMES_SESSION_CHAT_TYPE": "dm",
                 "HERMES_SESSION_CHAT_ID": "179033731",
                 "HERMES_SESSION_USER_ID": "179033731",
                 "HERMES_SESSION_USER_NAME": "發發發",
                 "HERMES_SESSION_IS_ADMIN": "true",
             },
             clear=False,
         ):
        qq_social_tool(
            {
                "action": "set_social_policy",
                "auto_approve_friend_requests": True,
                "notify_target": "current_user_dm",
            }
        )
        result = json.loads(qq_social_tool({"action": "get_social_policy"}))

    assert result["success"] is True
    assert result["policy"]["auto_approve_friend_requests"] is True
    assert result["policy"]["notify_target"] == "qq_napcat:dm:179033731"
    assert result["policy_state"]["auto_approval_enabled"] is True
    assert result["policy_state"]["enabled_scopes"] == ["friend_requests"]
    assert result["policy_state"]["notify_configured"] is True


def test_clear_social_policy_resets_defaults():
    config, _qq_cfg = _make_qq_napcat_config()
    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch.dict(
             os.environ,
             {
                 "HERMES_SESSION_PLATFORM": "qq_napcat",
                 "HERMES_SESSION_CHAT_TYPE": "dm",
                 "HERMES_SESSION_CHAT_ID": "179033731",
                 "HERMES_SESSION_USER_ID": "179033731",
                 "HERMES_SESSION_USER_NAME": "發發發",
                 "HERMES_SESSION_IS_ADMIN": "true",
             },
             clear=False,
         ):
        qq_social_tool(
            {
                "action": "set_social_policy",
                "auto_approve_friend_requests": True,
                "auto_approve_group_invites": True,
                "notify_target": "current_user_dm",
            }
        )
        result = json.loads(qq_social_tool({"action": "clear_social_policy"}))

    assert result["success"] is True
    assert result["policy"] == get_social_policy()
    assert result["policy"]["auto_approve_friend_requests"] is False
    assert result["policy"]["auto_approve_group_add_requests"] is False
    assert result["policy"]["auto_approve_group_invites"] is False
    assert result["policy"]["notify_target"] is None


def test_reject_friend_request_calls_api_and_marks_request_rejected():
    record_social_request_event(
        {
            "post_type": "request",
            "request_type": "friend",
            "user_id": 456789,
            "comment": "加一下",
            "flag": "friend-flag-2",
            "time": 1713012345,
        }
    )
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_social_tool._qq_napcat_call",
             new=AsyncMock(return_value=({"ok": True}, None)),
         ) as call_mock, \
         patch.dict(
             os.environ,
             {
                 "HERMES_SESSION_PLATFORM": "qq_napcat",
                 "HERMES_SESSION_CHAT_TYPE": "dm",
                 "HERMES_SESSION_CHAT_ID": "179033731",
                 "HERMES_SESSION_USER_ID": "179033731",
                 "HERMES_SESSION_IS_ADMIN": "true",
             },
             clear=False,
         ):
        result = json.loads(
            qq_social_tool(
                {
                    "action": "reject_request",
                    "request_key": "friend:friend-flag-2",
                    "message": "先不加",
                }
            )
        )

    assert result["success"] is True
    assert result["request"]["status"] == "rejected"
    assert result["request"]["decision_note"] == "先不加"
    assert result["request"]["request_state"]["handled_via"] == "manual_tool"
    assert result["request"]["request_state"]["handled_automatically"] is False
    call_mock.assert_awaited_once_with(
        qq_cfg.extra,
        "set_friend_add_request",
        {"flag": "friend-flag-2", "approve": False},
    )


def test_cannot_rehandle_non_pending_request():
    record_social_request_event(
        {
            "post_type": "request",
            "request_type": "group",
            "sub_type": "add",
            "group_id": 987654321,
            "user_id": 179033731,
            "comment": "再进一次",
            "flag": "group-flag-4",
            "time": 1713012345,
        }
    )
    config, _qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_social_tool._qq_napcat_call",
             new=AsyncMock(return_value=({"ok": True}, None)),
         ), \
         patch.dict(
             os.environ,
             {
                 "HERMES_SESSION_PLATFORM": "qq_napcat",
                 "HERMES_SESSION_CHAT_TYPE": "dm",
                 "HERMES_SESSION_CHAT_ID": "179033731",
                 "HERMES_SESSION_USER_ID": "179033731",
                 "HERMES_SESSION_IS_ADMIN": "true",
             },
             clear=False,
         ):
        first = json.loads(
            qq_social_tool(
                {
                    "action": "approve_request",
                    "request_key": "group:group-flag-4",
                }
            )
        )
        second = json.loads(
            qq_social_tool(
                {
                    "action": "reject_request",
                    "request_key": "group:group-flag-4",
                }
            )
        )

    assert first["success"] is True
    assert second["error"] == "QQ social request 'group:group-flag-4' has already been handled."
