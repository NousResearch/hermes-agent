"""Tests for tools/messaging_control_tool.py."""

import json
from unittest.mock import patch

from tools.messaging_control_tool import messaging_control_tool


def test_routes_to_qq_control_when_platform_is_explicit():
    with patch(
        "tools.messaging_control_tool.qq_control_tool",
        return_value=json.dumps({"success": True, "platform": "qq"}),
    ) as qq_mock:
        result = json.loads(
            messaging_control_tool(
                {
                    "platform": "qq",
                    "action": "set_policy",
                    "target": "group:987654321",
                    "mode": "collect_only",
                }
            )
        )

    assert result["platform"] == "qq"
    qq_mock.assert_called_once_with(
        {
            "action": "set_policy",
            "target": "group:987654321",
            "mode": "collect_only",
        }
    )


def test_infers_qq_platform_from_target():
    with patch(
        "tools.messaging_control_tool.qq_control_tool",
        return_value=json.dumps({"success": True, "platform": "qq"}),
    ) as qq_mock:
        result = json.loads(
            messaging_control_tool(
                {
                    "action": "send_message",
                    "target": "123456789",
                    "message": "开工",
                }
            )
        )

    assert result["platform"] == "qq"
    qq_mock.assert_called_once_with(
        {
            "action": "send_message",
            "target": "123456789",
            "message": "开工",
        }
    )


def test_routes_to_weixin_control_when_platform_alias_is_explicit():
    with patch(
        "tools.messaging_control_tool.weixin_control_tool",
        return_value=json.dumps({"success": True, "platform": "weixin"}),
    ) as weixin_mock:
        result = json.loads(
            messaging_control_tool(
                {
                    "platform": "wechat",
                    "action": "set_policy",
                    "target": "project@chatroom",
                    "mode": "collect_only",
                }
            )
        )

    assert result["platform"] == "weixin"
    weixin_mock.assert_called_once_with(
        {
            "action": "set_policy",
            "target": "project@chatroom",
            "mode": "collect_only",
        }
    )


def test_infers_weixin_platform_from_target():
    with patch(
        "tools.messaging_control_tool.weixin_control_tool",
        return_value=json.dumps({"success": True, "platform": "weixin"}),
    ) as weixin_mock:
        result = json.loads(
            messaging_control_tool(
                {
                    "action": "report_now",
                    "target": "project@chatroom",
                }
            )
        )

    assert result["platform"] == "weixin"
    weixin_mock.assert_called_once_with(
        {
            "action": "report_now",
            "target": "project@chatroom",
        }
    )


def test_infers_qq_platform_for_qq_only_action_without_target():
    with patch(
        "tools.messaging_control_tool.qq_control_tool",
        return_value=json.dumps({"success": True, "platform": "qq"}),
    ) as qq_mock:
        result = json.loads(messaging_control_tool({"action": "list_requests"}))

    assert result["platform"] == "qq"
    qq_mock.assert_called_once_with({"action": "list_requests"})


def test_employee_route_action_requires_explicit_platform_when_ambiguous():
    result = json.loads(
        messaging_control_tool(
            {
                "action": "set_employee_route",
                "worker_name": "铁柱",
            }
        )
    )

    assert result["success"] is False
    assert "platform" in result["error"].lower()
    assert "employee-route" in result["error"].lower()


def test_returns_error_when_action_is_missing():
    result = json.loads(messaging_control_tool({"target": "group:123"}))
    assert result["error"] == "'action' is required."
