"""Weixin archive and daily report inspection tool."""

from __future__ import annotations

import json

from hermes_time import now as hermes_now
from gateway.weixin_group_archive import (
    WeixinGroupArchiveStore,
    format_group_report_for_delivery,
)
from gateway.weixin_group_policies import get_group_policy
from tools.group_manual_report_delivery import deliver_manual_group_report
from tools.registry import registry, tool_error
from tools.send_message_tool import _check_send_message, _error, send_message_tool
from tools.weixin_group_tool_common import (
    current_chat_delivery_target,
    require_admin_session,
    resolve_delivery_target,
    resolve_group_target,
    resolve_optional_group_target,
)


WEIXIN_GROUP_ARCHIVE_SCHEMA = {
    "name": "weixin_group_archive",
    "description": (
        "Inspect Weixin group raw archives and daily reports. "
        "Use this to list recent collected messages, search archived text, list reports, fetch a specific report, "
        "or manually trigger a daily rollup that purges raw rows after the report is stored."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list_recent", "search", "list_reports", "get_report", "rollup_daily", "rollup_due", "snapshot_report", "deliver_report"],
                "description": "Archive/report operation to perform.",
            },
            "target": {
                "type": "string",
                "description": (
                    "Weixin group target. Accepts 'group@chatroom', 'group:group@chatroom', or 'weixin:group@chatroom'. "
                    "If omitted, Hermes uses the current Weixin group session when the action needs a group."
                ),
            },
            "report_date": {
                "type": "string",
                "description": "Daily report date in YYYY-MM-DD format when action='get_report', 'rollup_daily', 'snapshot_report', or 'deliver_report'. Defaults to the current local day for snapshot/deliver.",
            },
            "query": {
                "type": "string",
                "description": "Text query when action='search'.",
            },
            "delivery_target": {
                "type": "string",
                "description": "Optional delivery target for action='deliver_report'. Use current_chat, current_user_dm, none, weixin:group@chatroom, or weixin:wxid_xxx. If omitted, Hermes uses the group's manual report target or the current chat.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of rows to return. Defaults to 20.",
            },
        },
        "required": ["action"],
    },
}


def weixin_group_archive_tool(args, **kw):
    del kw

    action = str(args.get("action") or "").strip().lower()
    if action not in {"list_recent", "search", "list_reports", "get_report", "rollup_daily", "rollup_due", "snapshot_report", "deliver_report"}:
        return tool_error(
            "Unsupported action. Use 'list_recent', 'search', 'list_reports', 'get_report', 'rollup_daily', 'rollup_due', 'snapshot_report', or 'deliver_report'."
        )

    from tools.interrupt import is_interrupted

    if is_interrupted():
        return tool_error("Interrupted")

    admin_error = require_admin_session("查看或维护微信群采集归档")
    if admin_error:
        return json.dumps(_error(admin_error), ensure_ascii=False)

    store = WeixinGroupArchiveStore()
    limit = _normalize_limit(args.get("limit"))

    try:
        if action == "rollup_due":
            return json.dumps(store.rollup_due_days(), ensure_ascii=False)

        if action == "list_reports":
            chat_id = resolve_optional_group_target(args.get("target"))
            payload = {
                "success": True,
                "reports": store.list_reports(chat_id=chat_id, limit=limit),
            }
            if chat_id:
                reporting = store.describe_group_reporting(chat_id=chat_id)
                payload["reporting"] = reporting
                payload["report_control"] = dict(reporting.get("report_control") or {})
            return json.dumps(payload, ensure_ascii=False)

        chat_id = resolve_group_target(args.get("target"))
        if action == "list_recent":
            reporting = store.describe_group_reporting(chat_id=chat_id)
            return json.dumps(
                {
                    "success": True,
                    "messages": store.list_recent_messages(chat_id=chat_id, limit=limit),
                    "reporting": reporting,
                    "report_control": dict(reporting.get("report_control") or {}),
                },
                ensure_ascii=False,
            )
        if action == "search":
            query = str(args.get("query") or "").strip()
            if not query:
                raise ValueError("'query' is required when action='search'.")
            reporting = store.describe_group_reporting(chat_id=chat_id)
            return json.dumps(
                {
                    "success": True,
                    "messages": store.search_messages(chat_id=chat_id, query=query, limit=limit),
                    "reporting": reporting,
                    "report_control": dict(reporting.get("report_control") or {}),
                },
                ensure_ascii=False,
            )

        report_date = _resolve_report_date(args.get("report_date"))

        if action == "rollup_daily":
            if not report_date:
                raise ValueError("'report_date' is required for this action.")
            reporting = store.describe_group_reporting(chat_id=chat_id)
            return json.dumps(
                {
                    **store.rollup_daily(chat_id=chat_id, report_date=report_date),
                    "reporting": reporting,
                    "report_control": dict(reporting.get("report_control") or {}),
                },
                ensure_ascii=False,
            )

        if action == "snapshot_report":
            reporting = store.describe_group_reporting(chat_id=chat_id)
            return json.dumps(
                {
                    "success": True,
                    "report": store.build_snapshot_report(chat_id=chat_id, report_date=report_date),
                    "reporting": reporting,
                    "report_control": dict(reporting.get("report_control") or {}),
                },
                ensure_ascii=False,
            )

        report = _load_report_for_delivery_or_fetch(
            store=store,
            chat_id=chat_id,
            report_date=report_date,
        )
        if action == "get_report":
            reporting = store.describe_group_reporting(chat_id=chat_id)
            return json.dumps(
                {
                    "success": True,
                    "report": report,
                    "reporting": reporting,
                    "report_control": dict(reporting.get("report_control") or {}),
                },
                ensure_ascii=False,
            )

        policy = get_group_policy(chat_id)
        reporting = store.describe_group_reporting(chat_id=chat_id)
        delivery_payload = deliver_manual_group_report(
            report=report,
            policy=policy,
            reporting=reporting,
            explicit_delivery_target=args.get("delivery_target"),
            resolve_delivery_target=resolve_delivery_target,
            current_chat_delivery_target=current_chat_delivery_target,
            format_report=format_group_report_for_delivery,
            send_message=send_message_tool,
            failure_prefix="微信群日报发送失败",
            record_delivery=store.record_report_delivery,
            record_delivery_kwargs={
                "chat_id": chat_id,
                "report_date": report["report_date"],
            },
        )
        return json.dumps(
            delivery_payload,
            ensure_ascii=False,
        )
    except ValueError as exc:
        return json.dumps(_error(str(exc)), ensure_ascii=False)
    except Exception as exc:
        return json.dumps(_error(f"Weixin group archive action failed: {exc}"), ensure_ascii=False)


def _normalize_limit(value) -> int:
    try:
        parsed = int(value or 20)
    except (TypeError, ValueError):
        raise ValueError("'limit' must be an integer.") from None
    return max(1, min(parsed, 200))


def _resolve_report_date(value) -> str:
    explicit = str(value or "").strip()
    if explicit:
        return explicit
    return hermes_now().date().isoformat()


def _load_report_for_delivery_or_fetch(*, store: WeixinGroupArchiveStore, chat_id: str, report_date: str) -> dict:
    if not report_date:
        raise ValueError("'report_date' is required for this action.")
    return store.build_snapshot_report(chat_id=chat_id, report_date=report_date)


registry.register(
    name="weixin_group_archive",
    toolset="messaging",
    schema=WEIXIN_GROUP_ARCHIVE_SCHEMA,
    handler=weixin_group_archive_tool,
    check_fn=_check_send_message,
    emoji="🗃️",
)
