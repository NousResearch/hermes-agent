"""Tests for tools/qq_group_policy_tool.py."""

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.config import Platform
from tools.qq_group_policy_tool import qq_group_policy_tool


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


def test_set_and_get_collect_only_policy():
    config, _qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False):
        result = json.loads(
            qq_group_policy_tool(
                {
                    "action": "set_policy",
                    "target": "group:987654321",
                    "mode": "collect_only",
                }
            )
        )
        fetched = json.loads(
            qq_group_policy_tool(
                {
                    "action": "get_policy",
                    "target": "group:987654321",
                }
            )
        )

    assert result["success"] is True
    assert result["policy"]["mode"] == "collect_only"
    assert result["policy"]["archive_enabled"] is True
    assert result["policy"]["daily_report_enabled"] is False
    assert fetched["success"] is True
    assert fetched["policy"]["mode"] == "collect_only"


def test_set_policy_can_explicitly_enable_daily_report():
    config, _qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False):
        result = json.loads(
            qq_group_policy_tool(
                {
                    "action": "set_policy",
                    "target": "group:987654321",
                    "mode": "collect_only",
                    "daily_report_enabled": True,
                }
            )
        )

    assert result["success"] is True
    assert result["policy"]["mode"] == "collect_only"
    assert result["policy"]["archive_enabled"] is True
    assert result["policy"]["daily_report_enabled"] is True


def test_set_policy_accepts_no_reply_shortcut_shape():
    config, _qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False):
        result = json.loads(
            qq_group_policy_tool(
                {
                    "action": "no_reply",
                    "target": "group:987654321",
                }
            )
        )

    assert result["success"] is True
    assert result["action"] == "set_policy"
    assert result["group_id"] == "987654321"
    assert result["policy"]["mode"] == "collect_only"
    assert result["policy"]["archive_enabled"] is True
    assert result["reply_behavior"] == "no_reply"


def test_resume_chat_shortcut_restores_default_chat_mode():
    config, _qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False):
        result = json.loads(
            qq_group_policy_tool(
                {
                    "action": "resume_chat",
                    "target": "group:987654321",
                }
            )
        )

    assert result["success"] is True
    assert result["action"] == "set_policy"
    assert result["policy"]["mode"] == "default"
    assert result["policy"]["archive_enabled"] is False
    assert result["policy"]["daily_report_enabled"] is False


def test_disable_group_shortcut_sets_disabled_mode():
    config, _qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False):
        result = json.loads(
            qq_group_policy_tool(
                {
                    "action": "disable_group",
                    "target": "group:987654321",
                }
            )
        )

    assert result["success"] is True
    assert result["action"] == "set_policy"
    assert result["policy"]["mode"] == "disabled"
    assert result["policy"]["archive_enabled"] is False
    assert result["policy"]["daily_report_enabled"] is False


def test_set_policy_can_toggle_daily_report_without_resending_mode():
    config, _qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False):
        json.loads(
            qq_group_policy_tool(
                {
                    "action": "set_policy",
                    "target": "group:987654321",
                    "mode": "collect_only",
                }
            )
        )
        result = json.loads(
            qq_group_policy_tool(
                {
                    "action": "set_policy",
                    "target": "group:987654321",
                    "daily_report_enabled": True,
                }
            )
        )

    assert result["success"] is True
    assert result["policy"]["mode"] == "collect_only"
    assert result["policy"]["daily_report_enabled"] is True


def test_set_policy_can_resolve_report_targets_from_current_session():
    config, _qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch.dict(
             os.environ,
             {
                 "HERMES_SESSION_PLATFORM": "qq_napcat",
                 "HERMES_SESSION_CHAT_TYPE": "group",
                 "HERMES_SESSION_CHAT_ID": "987654321",
                 "HERMES_SESSION_USER_ID": "179033731",
                 "HERMES_SESSION_IS_ADMIN": "true",
             },
             clear=False,
         ):
        result = json.loads(
            qq_group_policy_tool(
                {
                    "action": "set_policy",
                    "target": "group:987654321",
                    "mode": "collect_only",
                    "daily_report_enabled": True,
                    "daily_report_target": "current_user_dm",
                    "manual_report_target": "current_chat",
                }
            )
        )

    assert result["success"] is True
    assert result["policy"]["daily_report_target"] == "qq_napcat:dm:179033731"
    assert result["policy"]["manual_report_target"] == "qq_napcat:group:987654321"
    assert result["policy"]["archive_enabled"] is True


def test_get_policy_includes_reporting_summary_and_effective_targets():
    config, _qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch.dict(
             os.environ,
             {
                 "HERMES_SESSION_PLATFORM": "qq_napcat",
                 "HERMES_SESSION_CHAT_TYPE": "group",
                 "HERMES_SESSION_CHAT_ID": "987654321",
                 "HERMES_SESSION_USER_ID": "179033731",
                 "HERMES_SESSION_IS_ADMIN": "true",
             },
             clear=False,
         ):
        json.loads(
            qq_group_policy_tool(
                {
                    "action": "set_policy",
                    "target": "group:987654321",
                    "mode": "collect_only",
                    "daily_report_enabled": True,
                    "daily_report_target": "current_user_dm",
                    "manual_report_target": "current_chat",
                }
            )
        )
        fetched = json.loads(
            qq_group_policy_tool(
                {
                    "action": "get_policy",
                    "target": "group:987654321",
                }
            )
        )

    assert fetched["success"] is True
    assert fetched["policy"]["reporting"]["delivery_targets"]["daily_report_targets"] == [
        "qq_napcat:dm:179033731"
    ]
    assert fetched["policy"]["reporting"]["delivery_targets"]["manual_report_targets"] == [
        "qq_napcat:group:987654321"
    ]
    assert fetched["policy"]["effective_policy"]["daily_report_targets"] == [
        "qq_napcat:dm:179033731"
    ]
    assert fetched["policy"]["effective_policy"]["manual_report_targets"] == [
        "qq_napcat:group:987654321"
    ]
    assert fetched["policy_summary"]["collect_only"] is True
    assert fetched["policy_summary"]["replies_disabled"] is True
    assert fetched["reply_behavior"] == "no_reply"
    assert fetched["report_control"]["daily_report_enabled"] is True
    assert fetched["delivery_targets"]["manual_report_targets"] == [
        "qq_napcat:group:987654321"
    ]


def test_list_joined_groups_merges_saved_policy():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False):
        json.loads(
            qq_group_policy_tool(
                {
                    "action": "set_policy",
                    "target": "group:987654321",
                    "mode": "collect_only",
                }
            )
        )

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_group_policy_tool._qq_napcat_call",
             new=AsyncMock(
                 return_value=(
                     [{"group_id": 987654321, "group_name": "研发群", "member_count": 12}],
                     None,
                 )
             ),
         ) as call_mock:
        result = json.loads(
            qq_group_policy_tool(
                {
                    "action": "list_joined_groups",
                }
            )
        )

    assert result["success"] is True
    assert result["count"] == 1
    assert result["groups"][0]["group_id"] == "987654321"
    assert result["groups"][0]["policy"]["mode"] == "collect_only"
    call_mock.assert_awaited_once_with(qq_cfg.extra, "get_group_list", {})


def test_non_admin_gateway_session_cannot_mutate_policy():
    config, _qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch.dict(
             os.environ,
             {
                 "HERMES_SESSION_PLATFORM": "qq_napcat",
                 "HERMES_SESSION_IS_ADMIN": "false",
             },
             clear=False,
         ):
        result = json.loads(
            qq_group_policy_tool(
                {
                    "action": "set_policy",
                    "target": "group:987654321",
                    "mode": "disabled",
                }
            )
        )

    assert "error" in result
    assert "董事长" in result["error"]
