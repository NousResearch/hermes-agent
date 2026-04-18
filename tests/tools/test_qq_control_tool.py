"""Tests for tools/qq_control_tool.py."""

import json
from unittest.mock import patch

from tools.qq_control_tool import qq_control_tool


def test_routes_send_action_to_send_message_tool():
    with patch(
        "tools.qq_control_tool.send_message_tool",
        return_value=json.dumps({"success": True, "tool": "send"}),
    ) as send_mock:
        result = json.loads(
            qq_control_tool(
                {
                    "action": "send_message",
                    "target": "qq_napcat:group:987654321",
                    "message": "绿帽哥！",
                }
            )
        )

    assert result["tool"] == "send"
    send_mock.assert_called_once_with(
        {
            "action": "send",
            "target": "qq_napcat:group:987654321",
            "message": "绿帽哥！",
        }
    )


def test_routes_social_action_to_qq_social_tool():
    with patch(
        "tools.qq_control_tool.qq_social_tool",
        return_value=json.dumps({"success": True, "tool": "social"}),
    ) as social_mock:
        result = json.loads(
            qq_control_tool(
                {
                    "action": "list_requests",
                    "status": "pending",
                }
            )
        )

    assert result["tool"] == "social"
    social_mock.assert_called_once_with({"action": "list_requests", "status": "pending"})


def test_routes_social_policy_action_to_qq_social_tool():
    with patch(
        "tools.qq_control_tool.qq_social_tool",
        return_value=json.dumps({"success": True, "tool": "social"}),
    ) as social_mock:
        result = json.loads(
            qq_control_tool(
                {
                    "action": "set_social_policy",
                    "auto_approve_friend_requests": True,
                    "notify_target": "current_user_dm",
                }
            )
        )

    assert result["tool"] == "social"
    social_mock.assert_called_once_with(
        {
            "action": "set_social_policy",
            "auto_approve_friend_requests": True,
            "notify_target": "current_user_dm",
        }
    )


def test_routes_social_decision_action_to_qq_social_tool():
    with patch(
        "tools.qq_control_tool.qq_social_tool",
        return_value=json.dumps({"success": True, "tool": "social"}),
    ) as social_mock:
        result = json.loads(
            qq_control_tool(
                {
                    "action": "reject_request",
                    "request_key": "friend:friend-flag-2",
                    "message": "先不加",
                }
            )
        )

    assert result["tool"] == "social"
    social_mock.assert_called_once_with(
        {
            "action": "reject_request",
            "request_key": "friend:friend-flag-2",
            "message": "先不加",
        }
    )


def test_routes_intel_action_to_qq_intel_tool():
    with patch(
        "tools.qq_control_tool.qq_intel_tool",
        return_value=json.dumps({"success": True, "tool": "intel"}),
    ) as intel_mock:
        result = json.loads(
            qq_control_tool(
                {
                    "action": "hire_worker",
                    "worker_name": "钢镚",
                    "target_group": "group:987654321",
                }
            )
        )

    assert result["tool"] == "intel"
    intel_mock.assert_called_once_with(
        {
            "action": "hire_worker",
            "worker_name": "钢镚",
            "target_group": "group:987654321",
        }
    )


def test_routes_group_moderation_action_to_specialized_tool():
    with patch(
        "tools.qq_control_tool.qq_group_moderation_tool",
        return_value=json.dumps({"success": True, "tool": "moderation"}),
    ) as moderation_mock:
        result = json.loads(
            qq_control_tool(
                {
                    "action": "mute_user",
                    "target": "group:987654321",
                    "user_id": "123456",
                    "reason": "广告",
                    "duration_seconds": 600,
                }
            )
        )

    assert result["tool"] == "moderation"
    moderation_mock.assert_called_once_with(
        {
            "action": "mute_user",
            "target": "group:987654321",
            "user_id": "123456",
            "reason": "广告",
            "duration_seconds": 600,
        }
    )


def test_routes_group_moderation_alias_to_specialized_tool():
    with patch(
        "tools.qq_control_tool.qq_group_moderation_tool",
        return_value=json.dumps({"success": True, "tool": "moderation"}),
    ) as moderation_mock:
        result = json.loads(
            qq_control_tool(
                {
                    "action": "kick_member",
                    "target": "group:987654321",
                    "user_query": "广告哥",
                    "reason": "广告",
                }
            )
        )

    assert result["tool"] == "moderation"
    moderation_mock.assert_called_once_with(
        {
            "action": "kick_user",
            "target": "group:987654321",
            "user_query": "广告哥",
            "reason": "广告",
        }
    )


def test_routes_group_file_action_to_specialized_tool():
    with patch(
        "tools.qq_control_tool.qq_group_file_tool",
        return_value=json.dumps({"success": True, "tool": "group_file"}),
    ) as file_mock:
        result = json.loads(
            qq_control_tool(
                {
                    "action": "list_files",
                    "target": "group:987654321",
                    "folder_id": "/",
                }
            )
        )

    assert result["tool"] == "group_file"
    file_mock.assert_called_once_with(
        {
            "action": "list",
            "target": "group:987654321",
            "folder_id": "/",
        }
    )


def test_routes_group_policy_alias_to_specialized_tool():
    with patch(
        "tools.qq_control_tool.qq_group_policy_tool",
        return_value=json.dumps({"success": True, "tool": "policy"}),
    ) as policy_mock:
        result = json.loads(
            qq_control_tool(
                {
                    "action": "set_group_policy",
                    "target": "group:987654321",
                    "mode": "collect_only",
                    "archive_enabled": True,
                }
            )
        )

    assert result["tool"] == "policy"
    policy_mock.assert_called_once_with(
        {
            "action": "set_policy",
            "target": "group:987654321",
            "mode": "collect_only",
            "archive_enabled": True,
        }
    )


def test_routes_resume_chat_alias_to_group_policy_tool():
    with patch(
        "tools.qq_control_tool.qq_group_policy_tool",
        return_value=json.dumps({"success": True, "tool": "policy"}),
    ) as policy_mock:
        result = json.loads(
            qq_control_tool(
                {
                    "action": "resume_chat",
                    "target": "group:987654321",
                }
            )
        )

    assert result["tool"] == "policy"
    policy_mock.assert_called_once_with(
        {
            "action": "resume_chat",
            "target": "group:987654321",
        }
    )


def test_routes_no_reply_alias_to_group_policy_tool():
    with patch(
        "tools.qq_control_tool.qq_group_policy_tool",
        return_value=json.dumps({"success": True, "tool": "policy"}),
    ) as policy_mock:
        result = json.loads(
            qq_control_tool(
                {
                    "action": "no_reply",
                    "target": "group:987654321",
                }
            )
        )

    assert result["tool"] == "policy"
    policy_mock.assert_called_once_with(
        {
            "action": "set_policy",
            "target": "group:987654321",
            "mode": "collect_only",
            "archive_enabled": True,
        }
    )


def test_routes_report_now_alias_to_intel_tool_when_worker_name_is_present():
    with patch(
        "tools.qq_control_tool.qq_intel_tool",
        return_value=json.dumps({"success": True, "tool": "intel"}),
    ) as intel_mock:
        result = json.loads(
            qq_control_tool(
                {
                    "action": "report_now",
                    "worker_name": "钢镚",
                    "manual_report_target": "current_user_dm",
                }
            )
        )

    assert result["tool"] == "intel"
    intel_mock.assert_called_once_with(
        {
            "action": "run_report_now",
            "worker_name": "钢镚",
            "manual_report_target": "current_user_dm",
        }
    )


def test_routes_report_now_alias_to_group_archive_tool_when_group_target_is_present():
    with patch(
        "tools.qq_control_tool.qq_group_archive_tool",
        return_value=json.dumps({"success": True, "tool": "archive"}),
    ) as archive_mock:
        result = json.loads(
            qq_control_tool(
                {
                    "action": "report_now",
                    "target": "group:987654321",
                    "delivery_target": "current_chat",
                }
            )
        )

    assert result["tool"] == "archive"
    archive_mock.assert_called_once_with(
        {
            "action": "deliver_report",
            "target": "group:987654321",
            "delivery_target": "current_chat",
        }
    )


def test_routes_employee_route_action_to_employee_route_tool():
    with patch(
        "tools.qq_control_tool.employee_route_tool",
        return_value=json.dumps({"success": True, "tool": "employee-route"}),
    ) as route_mock:
        result = json.loads(
            qq_control_tool(
                {
                    "action": "set_employee_route",
                    "worker_name": "铁柱",
                    "preloaded_skills": ["frontend-design-pro"],
                    "match_modes": ["explicit", "heuristic"],
                    "action_terms": ["打磨"],
                }
            )
        )

    assert result["tool"] == "employee-route"
    route_mock.assert_called_once_with(
        {
            "action": "set_route",
            "platform": "qq_napcat",
            "worker_name": "铁柱",
            "preloaded_skills": ["frontend-design-pro"],
            "match_modes": ["explicit", "heuristic"],
            "action_terms": ["打磨"],
        }
    )


def test_get_policy_is_augmented_with_reporting_targets():
    with patch(
        "tools.qq_control_tool.qq_group_policy_tool",
        return_value=json.dumps(
            {
                "success": True,
                "policy": {
                    "group_id": "987654321",
                    "mode": "default",
                    "archive_enabled": False,
                    "daily_report_enabled": False,
                    "daily_report_target": None,
                    "manual_report_target": None,
                    "purge_raw_after_rollup": True,
                },
            }
        ),
    ) as policy_mock, patch(
        "tools.qq_control_tool.QqGroupArchiveStore.describe_group_reporting",
        return_value={
            "group_id": "987654321",
            "policy_targets": {
                "daily_report_target": None,
                "manual_report_target": None,
            },
            "worker_targets": {
                "daily_report_targets": ["qq_napcat:dm:179033731"],
                "manual_report_targets": ["qq_napcat:dm:179033731"],
                "notify_targets": ["qq_napcat:dm:179033731"],
            },
            "effective_targets": {
                "daily_report_targets": ["qq_napcat:dm:179033731"],
                "manual_report_targets": ["qq_napcat:dm:179033731"],
            },
            "overlay": {
                "active": True,
                "mode": "collect_only",
                "daily_report_enabled": True,
                "worker_names": ["钢镚"],
            },
        },
    ) as reporting_mock:
        result = json.loads(
            qq_control_tool(
                {
                    "action": "get_policy",
                    "target": "group:987654321",
                }
            )
        )

    policy_mock.assert_called_once_with({"action": "get_policy", "target": "group:987654321"})
    reporting_mock.assert_called_once_with(group_id="987654321")
    assert result["policy"]["reporting"]["overlay"]["active"] is True
    assert result["policy"]["effective_policy"]["mode"] == "collect_only"
    assert result["policy"]["effective_policy"]["daily_report_target"] == "qq_napcat:dm:179033731"
    assert result["policy"]["effective_policy"]["manual_report_targets"] == ["qq_napcat:dm:179033731"]


def test_list_policies_is_augmented_with_reporting_targets():
    with patch(
        "tools.qq_control_tool.qq_group_policy_tool",
        return_value=json.dumps(
            {
                "success": True,
                "policies": [
                    {
                        "group_id": "987654321",
                        "mode": "collect_only",
                        "archive_enabled": True,
                        "daily_report_enabled": True,
                        "daily_report_target": "qq_napcat:dm:179033731",
                        "manual_report_target": "qq_napcat:dm:179033731",
                        "purge_raw_after_rollup": True,
                    }
                ],
            }
        ),
    ), patch(
        "tools.qq_control_tool.QqGroupArchiveStore.describe_group_reporting",
        return_value={
            "group_id": "987654321",
            "policy_targets": {
                "daily_report_target": "qq_napcat:dm:179033731",
                "manual_report_target": "qq_napcat:dm:179033731",
            },
            "worker_targets": {
                "daily_report_targets": [],
                "manual_report_targets": [],
                "notify_targets": [],
            },
            "effective_targets": {
                "daily_report_targets": ["qq_napcat:dm:179033731"],
                "manual_report_targets": ["qq_napcat:dm:179033731"],
            },
            "overlay": {
                "active": False,
                "mode": "default",
                "daily_report_enabled": False,
                "worker_names": [],
            },
        },
    ):
        result = json.loads(qq_control_tool({"action": "list_policies"}))

    assert result["policies"][0]["reporting"]["policy_targets"]["daily_report_target"] == "qq_napcat:dm:179033731"
    assert result["policies"][0]["effective_policy"]["mode"] == "collect_only"
