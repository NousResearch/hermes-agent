"""Tests for tools/weixin_group_archive_tool.py."""

import json
import os
from unittest.mock import patch

from gateway.weixin_group_archive import WeixinGroupArchiveStore
from tools.weixin_group_archive_tool import weixin_group_archive_tool


def test_list_recent_returns_archived_weixin_messages():
    store = WeixinGroupArchiveStore()
    store.archive_inbound_message(
        chat_id="project@chatroom",
        message_id="msg-1",
        observed_at="2026-04-16T10:00:00+08:00",
        user_id="wxid_a",
        user_name="A",
        text="今天继续推进",
        media_types=[],
    )

    with patch("tools.interrupt.is_interrupted", return_value=False), patch.dict(
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
            weixin_group_archive_tool(
                {
                    "action": "list_recent",
                    "target": "project@chatroom",
                }
            )
        )

    assert result["success"] is True
    assert result["messages"][0]["chat_id"] == "project@chatroom"
    assert result["messages"][0]["text"] == "今天继续推进"


def test_deliver_report_sends_via_send_message_tool():
    store = WeixinGroupArchiveStore()
    store.archive_inbound_message(
        chat_id="project@chatroom",
        message_id="msg-1",
        observed_at="2026-04-16T10:00:00+08:00",
        user_id="wxid_a",
        user_name="A",
        text="今天继续推进",
        media_types=[],
    )

    with patch("tools.interrupt.is_interrupted", return_value=False), patch.dict(
        os.environ,
        {
            "HERMES_SESSION_PLATFORM": "weixin",
            "HERMES_SESSION_CHAT_TYPE": "group",
            "HERMES_SESSION_CHAT_ID": "project@chatroom",
            "HERMES_SESSION_USER_ID": "wxid_admin",
            "HERMES_SESSION_IS_ADMIN": "true",
        },
        clear=False,
    ), patch(
        "tools.weixin_group_archive_tool.send_message_tool",
        return_value=json.dumps({"success": True, "platform": "weixin"}),
    ) as send_mock:
        result = json.loads(
            weixin_group_archive_tool(
                {
                    "action": "deliver_report",
                    "target": "project@chatroom",
                    "report_date": "2026-04-16",
                    "delivery_target": "current_user_dm",
                }
            )
        )

    assert result["success"] is True
    assert result["delivery"]["target"] == "weixin:wxid_admin"
    send_mock.assert_called_once()


def test_deliver_report_rejects_non_success_send_results():
    store = WeixinGroupArchiveStore()
    store.archive_inbound_message(
        chat_id="project@chatroom",
        message_id="msg-2",
        observed_at="2026-04-16T10:30:00+08:00",
        user_id="wxid_a",
        user_name="A",
        text="这条发送会失败",
        media_types=[],
    )

    with patch("tools.interrupt.is_interrupted", return_value=False), patch.dict(
        os.environ,
        {
            "HERMES_SESSION_PLATFORM": "weixin",
            "HERMES_SESSION_CHAT_TYPE": "group",
            "HERMES_SESSION_CHAT_ID": "project@chatroom",
            "HERMES_SESSION_USER_ID": "wxid_admin",
            "HERMES_SESSION_IS_ADMIN": "true",
        },
        clear=False,
    ), patch(
        "tools.weixin_group_archive_tool.send_message_tool",
        return_value=json.dumps({}),
    ):
        result = json.loads(
            weixin_group_archive_tool(
                {
                    "action": "deliver_report",
                    "target": "project@chatroom",
                    "report_date": "2026-04-16",
                    "delivery_target": "current_user_dm",
                }
            )
        )

    assert result["success"] is False
    assert result["error"] == "微信群日报发送失败：工具未返回成功结果"
    assert result["delivery"]["state"]["attempt_count"] == 1
    assert result["delivery"]["state"]["delivered_at"] is None
    assert result["delivery"]["state"]["last_error"] == "微信群日报发送失败：工具未返回成功结果"
