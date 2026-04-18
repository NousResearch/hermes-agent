"""Tests for tools/qq_intel_tool.py."""

import asyncio
import json
import os
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.config import Platform
from gateway.qq_group_archive import QqGroupArchiveStore
from tools.qq_intel_tool import qq_intel_tool

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]


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


def test_hire_worker_becomes_active_when_group_is_already_joined():
    config, qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_intel_tool._qq_napcat_call",
             new=AsyncMock(
                 return_value=(
                     [{"group_id": 987654321, "group_name": "研发群"}],
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
            qq_intel_tool(
                {
                    "action": "hire_worker",
                    "worker_name": "钢镚",
                    "target_group": "group:987654321",
                    "objective": "去刺探情报",
                }
            )
        )

    assert result["success"] is True
    assert result["worker"]["status"] == "active_collecting"
    assert result["worker"]["target_group_id"] == "987654321"
    assert result["worker"]["target_group_name"] == "研发群"
    assert result["worker"]["daily_report_target"] == "qq_napcat:dm:179033731"
    assert result["worker"]["manual_report_target"] == "qq_napcat:dm:179033731"
    assert result["worker"]["notify_target"] == "qq_napcat:dm:179033731"
    call_mock.assert_awaited_once_with(qq_cfg.extra, "get_group_list", {})


def test_hire_worker_from_group_session_defaults_reports_to_admin_dm():
    config, _qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_intel_tool._qq_napcat_call",
             new=AsyncMock(
                 return_value=(
                     [{"group_id": 987654321, "group_name": "研发群"}],
                     None,
                 )
             ),
         ), \
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
            qq_intel_tool(
                {
                    "action": "hire_worker",
                    "worker_name": "二狗",
                    "target_group": "group:987654321",
                    "objective": "潜伏收集情况",
                }
            )
        )

    assert result["success"] is True
    assert result["worker"]["daily_report_target"] == "qq_napcat:dm:179033731"
    assert result["worker"]["manual_report_target"] == "qq_napcat:dm:179033731"
    assert result["worker"]["notify_target"] == "qq_napcat:dm:179033731"


def test_resume_worker_reactivates_when_group_is_already_joined():
    config, _qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_intel_tool._qq_napcat_call",
             new=AsyncMock(
                 return_value=(
                     [{"group_id": 987654321, "group_name": "研发群"}],
                     None,
                 )
             ),
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
        hire = json.loads(
            qq_intel_tool(
                {
                    "action": "hire_worker",
                    "worker_name": "钢镚",
                    "target_group": "group:987654321",
                    "objective": "去刺探情报",
                }
            )
        )
        paused = json.loads(
            qq_intel_tool(
                {
                    "action": "pause_worker",
                    "worker_name": "钢镚",
                }
            )
        )
        resumed = json.loads(
            qq_intel_tool(
                {
                    "action": "resume_worker",
                    "worker_name": "钢镚",
                }
            )
        )

    assert hire["worker"]["status"] == "active_collecting"
    assert paused["worker"]["status"] == "paused"
    assert resumed["success"] is True
    assert resumed["worker"]["status"] == "active_collecting"


def test_pause_worker_returns_stable_shortcut_response_shape():
    config, _qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_intel_tool._qq_napcat_call",
             new=AsyncMock(
                 return_value=(
                     [{"group_id": 987654321, "group_name": "研发群"}],
                     None,
                 )
             ),
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
        json.loads(
            qq_intel_tool(
                {
                    "action": "hire_worker",
                    "worker_name": "钢镚",
                    "target_group": "group:987654321",
                    "objective": "去刺探情报",
                }
            )
        )
        paused = json.loads(
            qq_intel_tool(
                {
                    "action": "pause_worker",
                    "worker_name": "钢镚",
                }
            )
        )

    assert paused["success"] is True
    assert paused["action"] == "pause_worker"
    assert paused["worker_name"] == "钢镚"
    assert paused["status"] == "paused"
    assert paused["worker"]["status"] == "paused"


def test_run_report_now_delivers_to_worker_manual_target(monkeypatch):
    monkeypatch.setenv("HERMES_TIMEZONE", "Asia/Shanghai")
    from hermes_time import reset_cache

    reset_cache()
    shanghai = ZoneInfo("Asia/Shanghai")
    config, _qq_cfg = _make_qq_napcat_config()
    store = QqGroupArchiveStore()
    store.archive_payload(
        _group_payload(
            message_id=901,
            user_id=456789,
            text="今日情报一条",
            when=datetime(2026, 4, 13, 18, 0, tzinfo=shanghai),
        )
    )

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_intel_tool._qq_napcat_call",
             new=AsyncMock(
                 return_value=(
                     [{"group_id": 987654321, "group_name": "研发群"}],
                     None,
                 )
             ),
         ), \
         patch(
             "tools.qq_intel_tool.send_message_tool",
             return_value=json.dumps({"success": True, "message_id": "44"}),
         ) as send_mock, \
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
        hire = json.loads(
            qq_intel_tool(
                {
                    "action": "hire_worker",
                    "worker_name": "钢镚",
                    "target_group": "group:987654321",
                    "objective": "去刺探情报",
                }
            )
        )
        result = json.loads(
            qq_intel_tool(
                {
                    "action": "run_report_now",
                    "worker_name": "钢镚",
                    "report_date": "2026-04-13",
                }
            )
        )

    assert hire["success"] is True
    assert result["success"] is True
    assert result["worker"]["last_report_at"] is not None
    assert result["delivery"]["target"] == "qq_napcat:dm:179033731"
    send_args = send_mock.call_args.args[0]
    assert send_args["target"] == "qq_napcat:dm:179033731"
    assert "钢镚" in send_args["message"]
    reset_cache()


def test_report_now_alias_returns_stable_delivery_shape(monkeypatch):
    monkeypatch.setenv("HERMES_TIMEZONE", "Asia/Shanghai")
    from hermes_time import reset_cache

    reset_cache()
    shanghai = ZoneInfo("Asia/Shanghai")
    config, _qq_cfg = _make_qq_napcat_config()
    store = QqGroupArchiveStore()
    store.archive_payload(
        _group_payload(
            message_id=902,
            user_id=456789,
            text="再来一条情报",
            when=datetime(2026, 4, 13, 18, 30, tzinfo=shanghai),
        )
    )

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_intel_tool._qq_napcat_call",
             new=AsyncMock(
                 return_value=(
                     [{"group_id": 987654321, "group_name": "研发群"}],
                     None,
                 )
             ),
         ), \
         patch(
             "tools.qq_intel_tool.send_message_tool",
             return_value=json.dumps({"success": True, "message_id": "45"}),
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
        json.loads(
            qq_intel_tool(
                {
                    "action": "hire_worker",
                    "worker_name": "钢镚",
                    "target_group": "group:987654321",
                    "objective": "去刺探情报",
                }
            )
        )
        result = json.loads(
            qq_intel_tool(
                {
                    "action": "report_now",
                    "worker_name": "钢镚",
                    "report_date": "2026-04-13",
                }
            )
        )

    assert result["success"] is True
    assert result["action"] == "run_report_now"
    assert result["worker_name"] == "钢镚"
    assert result["delivery"]["target"] == "qq_napcat:dm:179033731"
    assert result["delivery"]["state"]["delivery_key"] == "intel:钢镚:qq_napcat:dm:179033731"
    assert result["delivery"]["state"]["attempt_count"] == 1
    reset_cache()


def test_run_report_now_rejects_non_success_send_results(monkeypatch):
    monkeypatch.setenv("HERMES_TIMEZONE", "Asia/Shanghai")
    from hermes_time import reset_cache

    reset_cache()
    shanghai = ZoneInfo("Asia/Shanghai")
    config, _qq_cfg = _make_qq_napcat_config()
    store = QqGroupArchiveStore()
    store.archive_payload(
        _group_payload(
            message_id=903,
            user_id=456789,
            text="失败情报一条",
            when=datetime(2026, 4, 13, 19, 0, tzinfo=shanghai),
        )
    )

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_intel_tool._qq_napcat_call",
             new=AsyncMock(
                 return_value=(
                     [{"group_id": 987654321, "group_name": "研发群"}],
                     None,
                 )
             ),
         ), \
         patch(
             "tools.qq_intel_tool.send_message_tool",
             return_value=json.dumps({}),
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
        json.loads(
            qq_intel_tool(
                {
                    "action": "hire_worker",
                    "worker_name": "钢镚",
                    "target_group": "group:987654321",
                    "objective": "去刺探情报",
                }
            )
        )
        failed = json.loads(
            qq_intel_tool(
                {
                    "action": "run_report_now",
                    "worker_name": "钢镚",
                    "report_date": "2026-04-13",
                }
            )
        )
        fetched = json.loads(
            qq_intel_tool(
                {
                    "action": "get_worker",
                    "worker_name": "钢镚",
                }
            )
        )

    assert failed["success"] is False
    assert failed["error"] == "情报汇报发送失败：工具未返回成功结果"
    assert failed["delivery"]["state"]["attempt_count"] == 1
    assert failed["delivery"]["state"]["delivered_at"] is None
    assert failed["delivery"]["state"]["last_error"] == "情报汇报发送失败：工具未返回成功结果"
    assert fetched["success"] is True
    assert fetched["worker"]["last_report_at"] is None
    reset_cache()


def test_get_worker_includes_reporting_summary_for_assigned_group():
    config, _qq_cfg = _make_qq_napcat_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch(
             "tools.qq_intel_tool._qq_napcat_call",
             new=AsyncMock(
                 return_value=(
                     [{"group_id": 987654321, "group_name": "研发群"}],
                     None,
                 )
             ),
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
        json.loads(
            qq_intel_tool(
                {
                    "action": "hire_worker",
                    "worker_name": "钢镚",
                    "target_group": "group:987654321",
                    "objective": "去刺探情报",
                    "manual_report_target": "qq_napcat:group:987654321",
                    "notify_target": "qq_napcat:dm:200000001",
                }
            )
        )
        fetched = json.loads(
            qq_intel_tool(
                {
                    "action": "get_worker",
                    "worker_name": "钢镚",
                }
            )
        )

    assert fetched["success"] is True
    assert fetched["worker"]["reporting"]["group_id"] == "987654321"
    assert fetched["worker"]["reporting"]["delivery_targets"]["daily_report_targets"] == [
        "qq_napcat:dm:179033731"
    ]
    assert fetched["worker"]["reporting"]["delivery_targets"]["manual_report_targets"] == [
        "qq_napcat:group:987654321"
    ]
    assert fetched["worker"]["reporting"]["delivery_targets"]["notify_targets"] == [
        "qq_napcat:dm:200000001"
    ]
    assert fetched["worker_summary"]["collecting"] is True
    assert fetched["worker_summary"]["manual_report_target"] == "qq_napcat:group:987654321"
