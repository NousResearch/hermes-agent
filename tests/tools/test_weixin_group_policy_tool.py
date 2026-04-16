"""Tests for tools/weixin_group_policy_tool.py."""

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

from gateway.config import Platform
from tools.weixin_group_policy_tool import weixin_group_policy_tool


def _make_weixin_config():
    platform = getattr(Platform, "WEIXIN")
    weixin_cfg = SimpleNamespace(
        enabled=True,
        token="test-token",
        api_key=None,
        extra={"account_id": "wxid_bot"},
    )
    return SimpleNamespace(platforms={platform: weixin_cfg}), weixin_cfg


def test_set_and_get_collect_only_policy():
    config, _weixin_cfg = _make_weixin_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch.dict(os.environ, {"HERMES_SESSION_IS_ADMIN": "true"}, clear=False):
        result = json.loads(
            weixin_group_policy_tool(
                {
                    "action": "set_policy",
                    "target": "project@chatroom",
                    "mode": "collect_only",
                }
            )
        )
        fetched = json.loads(
            weixin_group_policy_tool(
                {
                    "action": "get_policy",
                    "target": "project@chatroom",
                }
            )
        )

    assert result["success"] is True
    assert result["policy"]["mode"] == "collect_only"
    assert result["policy"]["archive_enabled"] is True
    assert fetched["success"] is True
    assert fetched["policy"]["chat_id"] == "project@chatroom"
    assert fetched["policy"]["reporting"]["report_control"]["reply_behavior"] == "no_reply"


def test_set_policy_can_resolve_report_targets_from_current_session():
    config, _weixin_cfg = _make_weixin_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch.dict(
             os.environ,
             {
                 "HERMES_SESSION_PLATFORM": "weixin",
                 "HERMES_SESSION_CHAT_TYPE": "group",
                 "HERMES_SESSION_CHAT_ID": "project@chatroom",
                 "HERMES_SESSION_USER_ID": "wxid_admin",
                 "HERMES_SESSION_IS_ADMIN": "true",
             },
             clear=False,
         ):
        result = json.loads(
            weixin_group_policy_tool(
                {
                    "action": "set_policy",
                    "target": "project@chatroom",
                    "mode": "collect_only",
                    "daily_report_enabled": True,
                    "daily_report_target": "current_user_dm",
                    "manual_report_target": "current_chat",
                }
            )
        )

    assert result["success"] is True
    assert result["policy"]["daily_report_target"] == "weixin:wxid_admin"
    assert result["policy"]["manual_report_target"] == "weixin:project@chatroom"
