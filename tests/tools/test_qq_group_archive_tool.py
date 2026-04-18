"""Tests for tools/qq_group_archive_tool.py."""

import json
from unittest.mock import patch
from datetime import datetime

from gateway.qq_group_archive import QqGroupArchiveStore
from gateway.qq_group_policies import set_group_policy
from tools.qq_group_archive_tool import qq_group_archive_tool

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]


def _group_payload(*, message_id, user_id, text, when, group_id=987654321, nickname="Alice", card=None):
    return {
        "post_type": "message",
        "message_type": "group",
        "message_id": message_id,
        "user_id": user_id,
        "group_id": group_id,
        "time": int(when.timestamp()),
        "raw_message": text,
        "message": [{"type": "text", "data": {"text": text}}],
        "sender": {"nickname": nickname, "card": card or nickname},
    }


def test_archive_tool_lists_recent_messages(monkeypatch):
    monkeypatch.setenv("HERMES_TIMEZONE", "Asia/Shanghai")
    from hermes_time import reset_cache

    reset_cache()
    store = QqGroupArchiveStore()
    store.archive_payload(
        _group_payload(
            message_id=501,
            user_id=456789,
            text="这是一条采集记录",
            when=datetime(2026, 4, 12, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        )
    )

    result = json.loads(
        qq_group_archive_tool(
            {
                "action": "list_recent",
                "target": "group:987654321",
                "limit": 5,
            }
        )
    )

    assert result["success"] is True
    assert len(result["messages"]) == 1
    assert result["messages"][0]["message_id"] == "501"
    reset_cache()


def test_archive_tool_rollup_daily_and_fetch_report(monkeypatch):
    monkeypatch.setenv("HERMES_TIMEZONE", "Asia/Shanghai")
    from hermes_time import reset_cache

    reset_cache()
    shanghai = ZoneInfo("Asia/Shanghai")
    store = QqGroupArchiveStore()
    store.archive_payload(
        _group_payload(
            message_id=601,
            user_id=456789,
            text="今天安排啥？",
            when=datetime(2026, 4, 12, 9, 0, tzinfo=shanghai),
            nickname="Alice",
            card="AliceCard",
        )
    )
    store.archive_payload(
        _group_payload(
            message_id=602,
            user_id=111222,
            text="看下这个链接 https://example.com/roadmap",
            when=datetime(2026, 4, 12, 10, 30, tzinfo=shanghai),
            nickname="Bob",
            card="BobCard",
        )
    )

    rolled = json.loads(
        qq_group_archive_tool(
            {
                "action": "rollup_daily",
                "target": "group:987654321",
                "report_date": "2026-04-12",
            }
        )
    )
    fetched = json.loads(
        qq_group_archive_tool(
            {
                "action": "get_report",
                "target": "group:987654321",
                "report_date": "2026-04-12",
            }
        )
    )

    assert rolled["success"] is True
    assert rolled["report"]["summary"]["total_messages"] == 2
    assert rolled["purged_raw_messages"] == 2
    assert fetched["success"] is True
    assert fetched["report"]["summary"]["links"] == ["https://example.com/roadmap"]
    reset_cache()


def test_archive_tool_deliver_report_uses_policy_manual_target(monkeypatch):
    monkeypatch.setenv("HERMES_TIMEZONE", "Asia/Shanghai")
    from hermes_time import reset_cache

    reset_cache()
    shanghai = ZoneInfo("Asia/Shanghai")
    store = QqGroupArchiveStore()
    set_group_policy(
        "987654321",
        mode="collect_only",
        manual_report_target="qq_napcat:dm:179033731",
        updated_by="test",
    )
    store.archive_payload(
        _group_payload(
            message_id=603,
            user_id=456789,
            text="今天进度不错",
            when=datetime(2026, 4, 13, 16, 30, tzinfo=shanghai),
            nickname="Alice",
            card="AliceCard",
        )
    )

    with patch(
        "tools.qq_group_archive_tool.send_message_tool",
        return_value=json.dumps({"success": True, "message_id": "88"}),
    ) as send_mock:
        delivered = json.loads(
            qq_group_archive_tool(
                {
                    "action": "deliver_report",
                    "target": "group:987654321",
                    "report_date": "2026-04-13",
                }
            )
        )

    assert delivered["success"] is True
    assert delivered["delivery"]["target"] == "qq_napcat:dm:179033731"
    assert delivered["report"]["report_date"] == "2026-04-13"
    send_mock.assert_called_once()
    send_args = send_mock.call_args.args[0]
    assert send_args["target"] == "qq_napcat:dm:179033731"
    assert "QQ 群快照" in send_args["message"]
    reset_cache()


def test_archive_tool_deliver_report_stays_available_after_rollup_purge(monkeypatch):
    monkeypatch.setenv("HERMES_TIMEZONE", "Asia/Shanghai")
    from hermes_time import reset_cache

    reset_cache()
    shanghai = ZoneInfo("Asia/Shanghai")
    store = QqGroupArchiveStore()
    set_group_policy(
        "987654321",
        mode="collect_only",
        daily_report_enabled=True,
        daily_report_target="qq_napcat:dm:179033731",
        manual_report_target="qq_napcat:dm:179033731",
        group_name="研发群",
        updated_by="test",
    )
    store.archive_payload(
        _group_payload(
            message_id=604,
            user_id=456789,
            text="昨天的群内情报",
            when=datetime(2026, 4, 12, 22, 0, tzinfo=shanghai),
            nickname="Alice",
            card="AliceCard",
        )
    )

    rolled = json.loads(
        qq_group_archive_tool(
            {
                "action": "rollup_daily",
                "target": "group:987654321",
                "report_date": "2026-04-12",
            }
        )
    )

    with patch(
        "tools.qq_group_archive_tool.send_message_tool",
        return_value=json.dumps({"success": True, "message_id": "89"}),
    ) as send_mock:
        delivered = json.loads(
            qq_group_archive_tool(
                {
                    "action": "deliver_report",
                    "target": "group:987654321",
                    "report_date": "2026-04-12",
                }
            )
        )

    assert rolled["success"] is True
    assert rolled["purged_raw_messages"] == 1
    assert delivered["success"] is True
    assert delivered["report"]["snapshot"] is False
    assert delivered["reporting"]["group_name"] == "研发群"
    assert delivered["reporting"]["delivery_targets"]["daily_report_targets"] == [
        "qq_napcat:dm:179033731"
    ]
    assert delivered["reporting"]["delivery_targets"]["manual_report_targets"] == [
        "qq_napcat:dm:179033731"
    ]
    assert delivered["report_control"]["daily_report_enabled"] is True
    assert delivered["delivery"]["state"]["delivery_key"] == "manual:qq_napcat:dm:179033731"
    assert delivered["delivery"]["state"]["attempt_count"] == 1
    assert delivered["delivery"]["state"]["delivered_at"] is not None
    send_mock.assert_called_once()
    reset_cache()


def test_archive_tool_deliver_report_rejects_non_success_send_results(monkeypatch):
    monkeypatch.setenv("HERMES_TIMEZONE", "Asia/Shanghai")
    from hermes_time import reset_cache

    reset_cache()
    shanghai = ZoneInfo("Asia/Shanghai")
    store = QqGroupArchiveStore()
    set_group_policy(
        "987654321",
        mode="collect_only",
        manual_report_target="qq_napcat:dm:179033731",
        updated_by="test",
    )
    store.archive_payload(
        _group_payload(
            message_id=605,
            user_id=456789,
            text="这条发送会失败",
            when=datetime(2026, 4, 13, 17, 0, tzinfo=shanghai),
            nickname="Alice",
            card="AliceCard",
        )
    )

    with patch(
        "tools.qq_group_archive_tool.send_message_tool",
        return_value=json.dumps({}),
    ):
        delivered = json.loads(
            qq_group_archive_tool(
                {
                    "action": "deliver_report",
                    "target": "group:987654321",
                    "report_date": "2026-04-13",
                }
            )
        )

    assert delivered["success"] is False
    assert delivered["error"] == "QQ 群日报发送失败：工具未返回成功结果"
    assert delivered["delivery"]["state"]["attempt_count"] == 1
    assert delivered["delivery"]["state"]["delivered_at"] is None
    assert delivered["delivery"]["state"]["last_error"] == "QQ 群日报发送失败：工具未返回成功结果"
    reset_cache()
