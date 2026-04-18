"""Unified control-plane tool for QQ intel workers."""

from __future__ import annotations

import json

from gateway.config import Platform
from gateway.qq_group_archive import (
    QqGroupArchiveStore,
    format_group_report_for_delivery,
)
from gateway.qq_intel_assignments import (
    get_intel_worker,
    hire_intel_worker,
    list_intel_workers,
    reconcile_intel_workers,
    resume_intel_worker,
    set_intel_worker_status,
    summarize_intel_worker_assignment,
    update_intel_worker,
)
from hermes_time import now as hermes_now
from tool_result_validation import normalize_tool_failure_result
from tools.qq_group_tool_common import (
    current_chat_delivery_target,
    current_user_dm_delivery_target,
    require_admin_session,
    resolve_delivery_target,
    session_actor_label,
)
from tools.registry import registry, tool_error
from tools.send_message_tool import _check_send_message, _error, _qq_napcat_call, send_message_tool

_ACTION_ALIASES = {
    "pause": "pause_worker",
    "resume": "resume_worker",
    "stop": "stop_worker",
    "report_now": "run_report_now",
    "report": "run_report_now",
}


QQ_INTEL_TOOL_SCHEMA = {
    "name": "qq_intel_control",
    "description": (
        "Manage QQ intel workers such as 情报员钢镚 or 情报员二狗. "
        "Use this tool to hire a worker onto a QQ group intelligence task, inspect current worker state, "
        "pause/resume/stop missions, trigger an immediate report, or reconcile whether the bot has already entered the target group."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list_workers",
                    "get_worker",
                    "hire_worker",
                    "pause_worker",
                    "resume_worker",
                    "stop_worker",
                    "set_reporting",
                    "run_report_now",
                    "reconcile_workers",
                    "pause",
                    "resume",
                    "stop",
                    "report_now",
                ],
                "description": "Intel worker operation to perform.",
            },
            "worker_name": {
                "type": "string",
                "description": "Worker name such as 钢镚 or 二狗.",
            },
            "target_group": {
                "type": "string",
                "description": "QQ target group reference. Supports group:<id>, qq_napcat:group:<id>, raw numeric id, or a joined group name.",
            },
            "objective": {
                "type": "string",
                "description": "Mission objective, for example '去刺探情报'.",
            },
            "daily_report_enabled": {
                "type": "boolean",
                "description": "Whether this worker should receive automatic daily reports for the assigned group.",
            },
            "daily_report_target": {
                "type": "string",
                "description": "Automatic daily-report target. Use current_chat, current_user_dm, none, qq_napcat:group:<id>, or qq_napcat:dm:<id>.",
            },
            "manual_report_target": {
                "type": "string",
                "description": "Default target for immediate/manual reports. Use current_chat, current_user_dm, none, qq_napcat:group:<id>, or qq_napcat:dm:<id>.",
            },
            "notify_target": {
                "type": "string",
                "description": "Where mission status updates should be sent. Use current_chat, current_user_dm, none, qq_napcat:group:<id>, or qq_napcat:dm:<id>.",
            },
            "report_date": {
                "type": "string",
                "description": "Report date in YYYY-MM-DD format for run_report_now. Defaults to the current local day.",
            },
            "notes": {
                "type": "string",
                "description": "Optional operator notes for the worker.",
            },
        },
        "required": ["action"],
    },
}


def qq_intel_tool(args, **kw):
    del kw

    normalized_args = dict(args)
    action = _normalize_action(normalized_args.get("action"))
    normalized_args["action"] = action
    if action not in {
        "list_workers",
        "get_worker",
        "hire_worker",
        "pause_worker",
        "resume_worker",
        "stop_worker",
        "set_reporting",
        "run_report_now",
        "reconcile_workers",
    }:
        return tool_error(
            "Unsupported action. Use 'list_workers', 'get_worker', 'hire_worker', "
            "'pause_worker', 'resume_worker', 'stop_worker', 'set_reporting', "
            "'run_report_now', or 'reconcile_workers'."
        )

    from tools.interrupt import is_interrupted

    if is_interrupted():
        return tool_error("Interrupted")

    try:
        from gateway.config import load_gateway_config

        config = load_gateway_config()
    except Exception as exc:
        return json.dumps(_error(f"Failed to load gateway config: {exc}"), ensure_ascii=False)

    pconfig = config.platforms.get(Platform.QQ_NAPCAT)
    if not pconfig or not pconfig.enabled:
        return tool_error(
            "Platform 'qq_napcat' is not configured. Set up NapCat credentials in ~/.hermes/config.yaml or environment variables."
        )

    try:
        if action == "list_workers":
            return json.dumps(
                {
                    "success": True,
                    "action": action,
                    "workers": [_augment_worker_with_reporting(worker) for worker in list_intel_workers()],
                },
                ensure_ascii=False,
            )

        worker_name = _require_worker_name(normalized_args.get("worker_name"))
        if action == "get_worker":
            worker = get_intel_worker(worker_name)
            if worker is None:
                raise ValueError(f"Intel worker '{worker_name}' does not exist.")
            return json.dumps(_worker_result_payload(action=action, worker=_augment_worker_with_reporting(worker)), ensure_ascii=False)

        admin_error = require_admin_session("调度情报员任务")
        if admin_error:
            return json.dumps(_error(admin_error), ensure_ascii=False)

        if action == "hire_worker":
            target_group = str(normalized_args.get("target_group") or "").strip()
            if not target_group:
                raise ValueError("'target_group' is required when action='hire_worker'.")
            joined_groups = _run_async_fetch_joined_groups(pconfig.extra)
            worker = hire_intel_worker(
                worker_name=worker_name,
                target_group_ref=target_group,
                objective=str(normalized_args.get("objective") or "").strip() or None,
                daily_report_enabled=bool(normalized_args.get("daily_report_enabled", True)),
                daily_report_target=_resolve_daily_report_target(normalized_args.get("daily_report_target")),
                manual_report_target=_resolve_manual_report_target(normalized_args.get("manual_report_target")),
                notify_target=_resolve_notify_target(normalized_args.get("notify_target")),
                notes=str(normalized_args.get("notes") or "").strip() or None,
                updated_by=session_actor_label(),
                joined_groups=joined_groups,
            )
            return json.dumps(_worker_result_payload(action=action, worker=_augment_worker_with_reporting(worker)), ensure_ascii=False)

        if action == "pause_worker":
            worker = set_intel_worker_status(
                worker_name,
                status="paused",
                updated_by=session_actor_label(),
                last_error=None,
            )
            return json.dumps(_worker_result_payload(action=action, worker=_augment_worker_with_reporting(worker)), ensure_ascii=False)

        if action == "resume_worker":
            joined_groups = _run_async_fetch_joined_groups(pconfig.extra)
            worker = resume_intel_worker(
                worker_name,
                joined_groups=joined_groups,
                updated_by=session_actor_label(),
            )
            return json.dumps(_worker_result_payload(action=action, worker=_augment_worker_with_reporting(worker)), ensure_ascii=False)

        if action == "stop_worker":
            worker = set_intel_worker_status(
                worker_name,
                status="stopped",
                updated_by=session_actor_label(),
                last_error=None,
            )
            return json.dumps(_worker_result_payload(action=action, worker=_augment_worker_with_reporting(worker)), ensure_ascii=False)

        if action == "set_reporting":
            worker = update_intel_worker(
                worker_name,
                daily_report_enabled=normalized_args.get("daily_report_enabled"),
                daily_report_target=_resolve_optional_delivery_target(normalized_args.get("daily_report_target")),
                manual_report_target=_resolve_optional_delivery_target(normalized_args.get("manual_report_target")),
                notify_target=_resolve_optional_delivery_target(normalized_args.get("notify_target")),
                notes=str(normalized_args.get("notes") or "").strip() or None,
                updated_by=session_actor_label(),
            )
            return json.dumps(_worker_result_payload(action=action, worker=_augment_worker_with_reporting(worker)), ensure_ascii=False)

        if action == "reconcile_workers":
            joined_groups = _run_async_fetch_joined_groups(pconfig.extra)
            result = reconcile_intel_workers(joined_groups, updated_by=session_actor_label())
            return json.dumps(
                {
                    "success": True,
                    "action": action,
                    **result,
                    "workers": [_augment_worker_with_reporting(worker) for worker in list_intel_workers()],
                },
                ensure_ascii=False,
            )

        worker = get_intel_worker(worker_name)
        if worker is None:
            raise ValueError(f"Intel worker '{worker_name}' does not exist.")
        if not str(worker.get("target_group_id") or "").strip():
            raise ValueError(f"Intel worker '{worker_name}' has not entered the target group yet.")

        report_date = str(normalized_args.get("report_date") or "").strip() or hermes_now().date().isoformat()
        report = QqGroupArchiveStore().build_snapshot_report(
            group_id=str(worker["target_group_id"]),
            report_date=report_date,
        )
        delivery_target = _resolve_manual_report_target(normalized_args.get("manual_report_target")) or str(worker.get("manual_report_target") or "").strip() or current_chat_delivery_target()
        message = _format_worker_report_message(worker, report)
        send_result = json.loads(
            send_message_tool(
                {
                    "action": "send",
                    "target": delivery_target,
                    "message": message,
                }
            )
        )
        failed_delivery = normalize_tool_failure_result(
            send_result,
            failure_prefix="情报汇报发送失败",
        )
        archive_store = QqGroupArchiveStore()
        delivery_state = archive_store.record_report_delivery(
            group_id=str(worker["target_group_id"]),
            report_date=report["report_date"],
            delivery_key=_worker_delivery_key(worker["worker_name"], delivery_target),
            target=delivery_target,
            error=str((failed_delivery or {}).get("error") or "").strip() or None,
        )
        if failed_delivery:
            return json.dumps(
                {
                    **failed_delivery,
                    "action": action,
                    "worker_name": worker["worker_name"],
                    "report": report,
                    "delivery": {
                        "target": delivery_target,
                        "result": send_result,
                        "state": delivery_state,
                    },
                },
                ensure_ascii=False,
            )
        worker = update_intel_worker(
            worker_name,
            last_report_at=hermes_now().isoformat(),
            updated_by=session_actor_label(),
        )
        return json.dumps(
            {
                **_worker_result_payload(action=action, worker=_augment_worker_with_reporting(worker)),
                "report": report,
                "delivery": {
                    "target": delivery_target,
                    "result": send_result,
                    "state": delivery_state,
                },
            },
            ensure_ascii=False,
        )
    except ValueError as exc:
        return json.dumps(_error(str(exc)), ensure_ascii=False)
    except Exception as exc:
        return json.dumps(_error(f"QQ intel worker action failed: {exc}"), ensure_ascii=False)


def _normalize_action(value) -> str:
    action = str(value or "").strip().lower()
    return _ACTION_ALIASES.get(action, action)


def _worker_result_payload(*, action: str, worker: dict) -> dict:
    return {
        "success": True,
        "action": action,
        "worker_name": str(worker.get("worker_name") or "").strip(),
        "status": str(worker.get("status") or "").strip() or None,
        "worker_summary": dict(worker.get("worker_summary") or {}),
        "worker": worker,
    }


def _require_worker_name(value) -> str:
    worker_name = str(value or "").strip()
    if not worker_name:
        raise ValueError("'worker_name' is required.")
    return worker_name


def _run_async_fetch_joined_groups(extra: dict) -> list[dict]:
    from model_tools import _run_async

    result = _run_async(_fetch_joined_groups(extra))
    if result.get("error"):
        raise ValueError(str(result["error"]))
    return result.get("groups") or []


async def _fetch_joined_groups(extra: dict) -> dict:
    data, error = await _qq_napcat_call(extra, "get_group_list", {})
    if error:
        return error
    groups = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        group_id = str(item.get("group_id") or item.get("groupCode") or "").strip()
        if not group_id:
            continue
        groups.append(
            {
                "group_id": group_id,
                "group_name": str(item.get("group_name") or item.get("groupName") or group_id).strip(),
            }
        )
    return {"success": True, "groups": groups}


def _resolve_optional_delivery_target(value) -> str | None:
    return resolve_delivery_target(value)


def _resolve_daily_report_target(value) -> str:
    resolved = resolve_delivery_target(value)
    if resolved is not None:
        return resolved
    try:
        return current_user_dm_delivery_target()
    except Exception:
        return current_chat_delivery_target()


def _resolve_manual_report_target(value) -> str | None:
    resolved = resolve_delivery_target(value)
    if resolved is not None:
        return resolved
    try:
        return current_user_dm_delivery_target()
    except Exception:
        try:
            return current_chat_delivery_target()
        except Exception:
            return None


def _resolve_notify_target(value) -> str:
    resolved = resolve_delivery_target(value)
    if resolved is not None:
        return resolved
    try:
        return current_user_dm_delivery_target()
    except Exception:
        return current_chat_delivery_target()


def _format_worker_report_message(worker: dict, report: dict) -> str:
    title = f"情报员 {worker['worker_name']} 回报"
    base = format_group_report_for_delivery(
        report,
        group_name=worker.get("target_group_name"),
    )
    objective = str(worker.get("objective") or "").strip()
    if objective:
        return f"{title}\n任务：{objective}\n{base}"
    return f"{title}\n{base}"


def _augment_worker_with_reporting(worker: dict) -> dict:
    augmented = dict(worker)
    augmented["worker_summary"] = summarize_intel_worker_assignment(worker)
    group_id = str(worker.get("target_group_id") or "").strip()
    if not group_id:
        augmented["reporting"] = None
        return augmented
    augmented["reporting"] = QqGroupArchiveStore().describe_group_reporting(group_id=group_id)
    return augmented


def _worker_delivery_key(worker_name: str, target: str) -> str:
    return f"intel:{str(worker_name or '').strip()}:{str(target or '').strip()}"


registry.register(
    name="qq_intel_control",
    toolset="messaging",
    schema=QQ_INTEL_TOOL_SCHEMA,
    handler=qq_intel_tool,
    check_fn=_check_send_message,
    emoji="🕵️",
)
