"""Tests for tools/weixin_control_tool.py."""

import json
from unittest.mock import patch

from tools.weixin_control_tool import weixin_control_tool


def test_routes_send_action_to_send_message_tool():
    with patch(
        "tools.weixin_control_tool.send_message_tool",
        return_value=json.dumps({"success": True, "tool": "send"}),
    ) as send_mock:
        result = json.loads(
            weixin_control_tool(
                {
                    "action": "send_message",
                    "target": "weixin:project@chatroom",
                    "message": "开工",
                }
            )
        )

    assert result["tool"] == "send"
    send_mock.assert_called_once_with(
        {
            "action": "send",
            "target": "weixin:project@chatroom",
            "message": "开工",
        }
    )


def test_routes_group_policy_alias_to_specialized_tool():
    with patch(
        "tools.weixin_control_tool.weixin_group_policy_tool",
        return_value=json.dumps({"success": True, "tool": "policy"}),
    ) as policy_mock:
        result = json.loads(
            weixin_control_tool(
                {
                    "action": "no_reply",
                    "target": "project@chatroom",
                }
            )
        )

    assert result["tool"] == "policy"
    policy_mock.assert_called_once_with(
        {
            "action": "set_policy",
            "target": "project@chatroom",
            "mode": "collect_only",
            "archive_enabled": True,
        }
    )


def test_routes_report_now_alias_to_archive_tool():
    with patch(
        "tools.weixin_control_tool.weixin_group_archive_tool",
        return_value=json.dumps({"success": True, "tool": "archive"}),
    ) as archive_mock:
        result = json.loads(
            weixin_control_tool(
                {
                    "action": "report_now",
                    "target": "project@chatroom",
                    "delivery_target": "current_chat",
                }
            )
        )

    assert result["tool"] == "archive"
    archive_mock.assert_called_once_with(
        {
            "action": "deliver_report",
            "target": "project@chatroom",
            "delivery_target": "current_chat",
        }
    )


def test_routes_employee_route_action_to_employee_route_tool():
    with patch(
        "tools.weixin_control_tool.employee_route_tool",
        return_value=json.dumps({"success": True, "tool": "employee-route"}),
    ) as route_mock:
        result = json.loads(
            weixin_control_tool(
                {
                    "action": "clear_employee_route",
                    "worker_name": "阿旺",
                }
            )
        )

    assert result["tool"] == "employee-route"
    route_mock.assert_called_once_with(
        {
            "action": "clear_route",
            "platform": "weixin",
            "worker_name": "阿旺",
        }
    )


def test_returns_structured_not_capable_for_group_moderation_action():
    result = json.loads(
        weixin_control_tool(
            {
                "action": "mute_user",
                "target": "project@chatroom",
                "user_query": "广告哥",
                "reason": "广告",
            }
        )
    )

    assert result == {
        "success": False,
        "platform": "weixin",
        "action": "mute_user",
        "capability": "not_capable",
        "target": "project@chatroom",
        "detail": "微信群暂不支持禁言/踢人。",
    }


def test_normalizes_group_moderation_alias_to_structured_not_capable_result():
    result = json.loads(
        weixin_control_tool(
            {
                "action": "kick",
                "target": "project@chatroom",
                "user_query": "广告哥",
                "reason": "广告",
            }
        )
    )

    assert result["action"] == "kick_user"
    assert result["capability"] == "not_capable"
    assert result["target"] == "project@chatroom"
