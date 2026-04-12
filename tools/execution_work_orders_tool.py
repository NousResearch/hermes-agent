#!/usr/bin/env python3
"""Operational surface for H007 direct execution work orders."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from cron.jobs import create_job, list_jobs, parse_schedule, remove_job, update_job
from tools.delegate_tool import delegate_task
from tools.execution_work_orders import (
    cancel_execution_work_order,
    claim_next_due_execution_work_order,
    enqueue_execution_work_order,
    execution_work_order_counts,
    finish_execution_work_order,
    get_execution_work_order,
    query_execution_work_orders,
    reclaim_stale_execution_work_orders,
    resume_execution_work_order,
    retry_execution_work_order,
)
from tools.registry import registry, tool_error, tool_result


RUNNER_JOB_NAME = "Execution work-order runner"
RUNNER_PROMPT_MARKER = "H007_EXECUTION_WORK_ORDERS_RUNNER"
_DEFAULT_RUNNER_SCHEDULE = "every 5m"
_DEFAULT_RUN_LIMIT = 10
_DEFAULT_CLAIM_TTL_SECONDS = 900
_DEFAULT_RECLAIM_LIMIT = 50


EXECUTION_WORK_ORDERS_SCHEMA = {
    "name": "execution_work_orders",
    "description": (
        "Persist and execute durable H007 execution work orders. This surface is intentionally narrow: "
        "it schedules one-shot direct terminal work orders, tracks retries/resume/reclaim state, and runs due work "
        "orders through the official delegate_task path so normal execution receipts still exist.\n\n"
        "ACTIONS:\n"
        "- list: recent work orders\n"
        "- query: filtered work order lookup\n"
        "- enqueue: create a new one-shot direct terminal work order\n"
        "- run_due: execute queued/retry_scheduled work orders that are due now\n"
        "- reclaim_stale: move expired running work orders back to queued\n"
        "- retry: manually requeue a finished work order\n"
        "- resume: requeue a failed/cancelled/expired-running work order from the same durable spec\n"
        "- cancel: cancel a queued/retry_scheduled work order (or an expired running one)\n"
        "- runner_status: inspect the cron-backed runner job plus queue counts\n"
        "- install_runner: create/update the cron-backed runner job\n"
        "- remove_runner: remove the cron-backed runner job\n\n"
        "Important scope rule: work orders currently support only the direct_terminal_work_order execution lane."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list",
                    "query",
                    "enqueue",
                    "run_due",
                    "reclaim_stale",
                    "retry",
                    "resume",
                    "cancel",
                    "runner_status",
                    "install_runner",
                    "remove_runner",
                ],
                "description": "Which work-order action to perform. Default: list.",
            },
            "work_order_id": {
                "type": "string",
                "description": "Target work order ID for query/retry/resume/cancel.",
            },
            "goal": {
                "type": "string",
                "description": "For enqueue: human-readable goal/label for this work order.",
            },
            "context": {
                "type": "string",
                "description": "Optional free-form context saved with the work order.",
            },
            "command": {
                "type": "string",
                "description": "For enqueue: exact terminal command string for the direct work order.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Optional command timeout for enqueue.",
            },
            "workdir": {
                "type": "string",
                "description": "Optional explicit workdir for the direct command.",
            },
            "schedule": {
                "type": "string",
                "description": "For enqueue/install_runner: one-shot schedule for work orders, or cron schedule for the runner job.",
            },
            "delay_seconds": {
                "type": "number",
                "description": "For enqueue/retry/resume: relative delay before the work order becomes due.",
            },
            "max_attempts": {
                "type": "integer",
                "description": "For enqueue: total attempts before the work order ends as failed.",
            },
            "retry_delay_seconds": {
                "type": "number",
                "description": "For enqueue: delay before the next automatic retry after a failed attempt.",
            },
            "status": {
                "type": "string",
                "description": "Optional status filter for list/query.",
            },
            "limit": {
                "type": "integer",
                "description": "Max rows to return for list/query, max work orders to run, or max stale work orders to reclaim.",
            },
            "claim_ttl_seconds": {
                "type": "number",
                "description": "For run_due/install_runner: lease TTL before a running work order is considered reclaimable.",
            },
            "reclaim_limit": {
                "type": "integer",
                "description": "For install_runner: max expired running work orders to reclaim per pass.",
            },
            "model": {
                "type": "string",
                "description": "Optional per-job model override for the runner cron job.",
            },
            "provider": {
                "type": "string",
                "description": "Optional per-job provider override for the runner cron job.",
            },
            "base_url": {
                "type": "string",
                "description": "Optional per-job base URL override for the runner cron job.",
            },
        },
        "required": [],
    },
}


def check_execution_work_orders_requirements() -> bool:
    return True


def _normalize_optional_text(value: str | None, *, strip_trailing_slash: bool = False) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if strip_trailing_slash:
        text = text.rstrip("/")
    return text or None



def _parent_can_run_work_orders(parent_agent) -> bool:
    enabled = set(getattr(parent_agent, "enabled_toolsets", []) or [])
    return {"delegation", "terminal"}.issubset(enabled)



def _build_ephemeral_runner_parent():
    from hermes_cli.runtime_provider import resolve_runtime_provider, _get_model_config
    from run_agent import AIAgent

    model_cfg = _get_model_config()
    runtime = resolve_runtime_provider()
    model_name = str(model_cfg.get("default") or "").strip() or "gpt-5.4"
    return AIAgent(
        model=model_name,
        provider=runtime.get("provider"),
        base_url=runtime.get("base_url"),
        api_key=runtime.get("api_key"),
        api_mode=runtime.get("api_mode"),
        command=runtime.get("command"),
        args=list(runtime.get("args") or []),
        enabled_toolsets=["delegation", "terminal", "execution_work_orders", "execution_receipts"],
        quiet_mode=True,
        max_iterations=2,
        session_id=f"work-order-runner-{int(time.time())}",
        persist_session=False,
        skip_memory=True,
    )



def _direct_execution_envelope(work_order: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "command": work_order["command"],
    }
    if work_order.get("timeout_seconds") is not None:
        payload["timeout_seconds"] = int(work_order["timeout_seconds"])
    if work_order.get("workdir"):
        payload["workdir"] = work_order["workdir"]
    return {
        "execution_mode": "direct_terminal_work_order",
        "task_spec": str(work_order.get("goal") or "Execute the scheduled direct terminal work order."),
        "direct_terminal_work_order": payload,
        "artifact_schema": {"required": ["summary"], "optional": ["files_modified", "issues"]},
    }



def _runner_context_package(work_order: dict[str, Any]) -> dict[str, Any]:
    package: dict[str, Any] = {
        "scheduled_work_order_id": work_order["work_order_id"],
        "execution_path": "direct_terminal_work_order",
    }
    if work_order.get("parent_session_id"):
        package["parent_session_id"] = work_order["parent_session_id"]
    return package



def _normalize_runner_config(
    *,
    schedule: str | None,
    limit: int | None,
    reclaim_limit: int | None,
    claim_ttl_seconds: float | None,
    model: str | None,
    provider: str | None,
    base_url: str | None,
) -> dict[str, Any]:
    normalized_schedule = _normalize_optional_text(schedule) or _DEFAULT_RUNNER_SCHEDULE
    normalized_limit = max(1, int(limit if limit is not None else _DEFAULT_RUN_LIMIT))
    normalized_reclaim_limit = max(1, int(reclaim_limit if reclaim_limit is not None else _DEFAULT_RECLAIM_LIMIT))
    normalized_claim_ttl = float(claim_ttl_seconds if claim_ttl_seconds is not None else _DEFAULT_CLAIM_TTL_SECONDS)
    if normalized_claim_ttl <= 0:
        raise ValueError("claim_ttl_seconds must be > 0")
    return {
        "schedule": normalized_schedule,
        "run_limit": normalized_limit,
        "reclaim_limit": normalized_reclaim_limit,
        "claim_ttl_seconds": normalized_claim_ttl,
        "model": _normalize_optional_text(model),
        "provider": _normalize_optional_text(provider),
        "base_url": _normalize_optional_text(base_url, strip_trailing_slash=True),
    }



def _build_runner_prompt(config: dict[str, Any]) -> str:
    config_json = json.dumps(config, ensure_ascii=False, sort_keys=True)
    return (
        f"[{RUNNER_PROMPT_MARKER}]\n"
        f"{config_json}\n"
        f"[/{RUNNER_PROMPT_MARKER}]\n\n"
        "You are the scheduled runner for Hermes H007 execution work orders.\n"
        "Use only the execution_work_orders tool for this task.\n\n"
        "Required sequence:\n"
        f"1. Call execution_work_orders with action='reclaim_stale' and limit={config['reclaim_limit']}.\n"
        f"2. Call execution_work_orders with action='run_due', limit={config['run_limit']}, and claim_ttl_seconds={config['claim_ttl_seconds']}.\n\n"
        "Final response rules:\n"
        "- If reclaimed_count=0 and executed_count=0, respond with exactly [SILENT].\n"
        "- Otherwise return a compact JSON object with keys: reclaimed_count, executed_count, statuses.\n"
        "Do not ask clarifying questions. Do not use send_message."
    )



def _extract_runner_config(prompt: str | None) -> dict[str, Any] | None:
    text = str(prompt or "")
    pattern = rf"\[{RUNNER_PROMPT_MARKER}\]\s*(\{{.*?\}})\s*\[/{RUNNER_PROMPT_MARKER}\]"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(1))
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed



def _matching_runner_jobs() -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    matches: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for job in list_jobs(include_disabled=True):
        config = _extract_runner_config(job.get("prompt"))
        if config is not None:
            matches.append((job, config))
    matches.sort(key=lambda item: item[0].get("created_at") or "")
    return matches



def _format_runner_job(job: dict[str, Any], config: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "job_id": job.get("id"),
        "name": job.get("name"),
        "schedule": job.get("schedule_display"),
        "next_run_at": job.get("next_run_at"),
        "last_run_at": job.get("last_run_at"),
        "last_status": job.get("last_status"),
        "last_delivery_error": job.get("last_delivery_error"),
        "enabled": job.get("enabled", True),
        "state": job.get("state", "scheduled" if job.get("enabled", True) else "paused"),
        "deliver": job.get("deliver", "local"),
        "model": job.get("model"),
        "provider": job.get("provider"),
        "base_url": job.get("base_url"),
        "config": config,
    }



def _runner_status_payload() -> dict[str, Any]:
    matches = _matching_runner_jobs()
    jobs = [_format_runner_job(job, config) for job, config in matches]
    return {
        "installed": bool(jobs),
        "installed_count": len(jobs),
        "jobs": jobs,
        **execution_work_order_counts(),
    }



def _upsert_runner_job(config: dict[str, Any]) -> dict[str, Any]:
    parsed_schedule = parse_schedule(config["schedule"])
    prompt = _build_runner_prompt(config)
    matches = _matching_runner_jobs()

    removed_duplicate_job_ids: list[str] = []
    primary = matches[0][0] if matches else None
    duplicates = matches[1:] if len(matches) > 1 else []
    for duplicate, _ in duplicates:
        duplicate_id = duplicate.get("id")
        if duplicate_id and remove_job(str(duplicate_id)):
            removed_duplicate_job_ids.append(str(duplicate_id))

    if primary is None:
        job = create_job(
            prompt=prompt,
            schedule=config["schedule"],
            name=RUNNER_JOB_NAME,
            deliver="local",
            model=config["model"],
            provider=config["provider"],
            base_url=config["base_url"],
        )
        created = True
    else:
        job = update_job(
            str(primary["id"]),
            {
                "name": RUNNER_JOB_NAME,
                "prompt": prompt,
                "schedule": parsed_schedule,
                "schedule_display": parsed_schedule.get("display", config["schedule"]),
                "deliver": "local",
                "model": config["model"],
                "provider": config["provider"],
                "base_url": config["base_url"],
                "enabled": True,
                "state": "scheduled",
                "paused_at": None,
                "paused_reason": None,
            },
        )
        created = False

    if not job:
        raise RuntimeError("Failed to create or update the runner cron job")

    return {
        "installed": True,
        "created": created,
        "updated": not created,
        "removed_duplicate_job_ids": removed_duplicate_job_ids,
        "job": _format_runner_job(job, config),
    }



def _remove_runner_jobs() -> dict[str, Any]:
    deleted_job_ids: list[str] = []
    for job, _config in _matching_runner_jobs():
        job_id = str(job.get("id"))
        if job_id and remove_job(job_id):
            deleted_job_ids.append(job_id)
    return {
        "deleted_count": len(deleted_job_ids),
        "deleted_job_ids": deleted_job_ids,
    }



def _run_due(parent_agent, *, limit: int, claim_ttl_seconds: float) -> dict[str, Any]:
    ephemeral_parent = None
    if parent_agent is None:
        ephemeral_parent = _build_ephemeral_runner_parent()
        parent_agent = ephemeral_parent
    if not _parent_can_run_work_orders(parent_agent):
        raise ValueError("run_due requires the parent session to have delegation and terminal toolsets enabled")

    executed: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    effective_limit = max(1, int(limit))
    try:
        for _ in range(effective_limit):
            claimed = claim_next_due_execution_work_order(
                claim_owner=getattr(parent_agent, "session_id", None),
                claim_ttl_seconds=claim_ttl_seconds,
            )
            if not claimed:
                break
            claim_token = claimed.get("claim_token")
            try:
                payload = json.loads(
                    delegate_task(
                        goal=str(claimed.get("goal") or "Execute the scheduled direct terminal work order."),
                        context=claimed.get("context"),
                        toolsets=["terminal"],
                        max_iterations=2,
                        execution_envelope=_direct_execution_envelope(claimed),
                        context_package=_runner_context_package(claimed),
                        parent_agent=parent_agent,
                    )
                )
                if payload.get("error"):
                    result = {
                        "status": "failed",
                        "summary": None,
                        "error": str(payload["error"]),
                        "fallback_reason": "work_order_runner_error",
                        "exit_reason": "work_order_runner_error",
                        "duration_seconds": 0,
                        "api_calls": 0,
                    }
                else:
                    results = payload.get("results") or []
                    result = results[0] if results else {
                        "status": "failed",
                        "summary": None,
                        "error": "delegate_task returned no result entries",
                        "fallback_reason": "empty_delegate_result",
                        "exit_reason": "empty_delegate_result",
                        "duration_seconds": 0,
                        "api_calls": 0,
                    }
            except Exception as exc:
                result = {
                    "status": "failed",
                    "summary": None,
                    "error": str(exc),
                    "fallback_reason": "work_order_runner_exception",
                    "exit_reason": "work_order_runner_exception",
                    "duration_seconds": 0,
                    "api_calls": 0,
                }
            finalized = finish_execution_work_order(
                str(claimed["work_order_id"]),
                claim_token=claim_token,
                result=result,
            )
            executed.append(finalized)
            final_status = str(finalized.get("status") or "unknown")
            status_counts[final_status] = status_counts.get(final_status, 0) + 1
    finally:
        if ephemeral_parent is not None:
            try:
                ephemeral_parent.close()
            except Exception:
                pass
    return {
        "executed_count": len(executed),
        "work_orders": executed,
        "statuses": status_counts,
    }



def execution_work_orders_tool(
    *,
    action: str | None = None,
    work_order_id: str | None = None,
    goal: str | None = None,
    context: str | None = None,
    command: str | None = None,
    timeout_seconds: int | None = None,
    workdir: str | None = None,
    schedule: str | None = None,
    delay_seconds: float | None = None,
    max_attempts: int | None = None,
    retry_delay_seconds: float | None = None,
    status: str | None = None,
    limit: int | None = None,
    claim_ttl_seconds: float | None = None,
    reclaim_limit: int | None = None,
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    parent_agent=None,
) -> str:
    action = (action or "list").strip().lower()

    if action in {"list", "query"}:
        effective_limit = max(1, int(limit if limit is not None else 10))
        rows = query_execution_work_orders(
            limit=effective_limit,
            status=status,
            parent_session_id=None,
            work_order_id=work_order_id,
        )
        return tool_result({
            "action": action,
            "count": len(rows),
            "work_orders": rows,
            **execution_work_order_counts(),
        })

    if action == "enqueue":
        if not goal or not str(goal).strip():
            return tool_error("enqueue requires goal")
        if not command or not str(command).strip():
            return tool_error("enqueue requires command")
        try:
            row = enqueue_execution_work_order(
                goal=str(goal),
                command=str(command),
                context=context,
                timeout_seconds=timeout_seconds,
                workdir=workdir,
                schedule=schedule,
                delay_seconds=delay_seconds,
                max_attempts=int(max_attempts if max_attempts is not None else 1),
                retry_delay_seconds=float(retry_delay_seconds if retry_delay_seconds is not None else 0.0),
                parent_session_id=getattr(parent_agent, "session_id", None),
            )
        except Exception as exc:
            return tool_error(f"enqueue failed: {exc}")
        return tool_result({"action": action, "work_order": row})

    if action == "reclaim_stale":
        try:
            result = reclaim_stale_execution_work_orders(limit=max(1, int(limit if limit is not None else _DEFAULT_RECLAIM_LIMIT)))
        except Exception as exc:
            return tool_error(f"reclaim_stale failed: {exc}")
        return tool_result({"action": action, **result})

    if action == "run_due":
        try:
            result = _run_due(
                parent_agent,
                limit=max(1, int(limit if limit is not None else _DEFAULT_RUN_LIMIT)),
                claim_ttl_seconds=float(claim_ttl_seconds if claim_ttl_seconds is not None else _DEFAULT_CLAIM_TTL_SECONDS),
            )
        except Exception as exc:
            return tool_error(f"run_due failed: {exc}")
        return tool_result({"action": action, **result})

    if action == "retry":
        if not work_order_id:
            return tool_error("retry requires work_order_id")
        try:
            row = retry_execution_work_order(str(work_order_id), delay_seconds=float(delay_seconds or 0.0))
        except Exception as exc:
            return tool_error(f"retry failed: {exc}")
        return tool_result({"action": action, "work_order": row})

    if action == "resume":
        if not work_order_id:
            return tool_error("resume requires work_order_id")
        try:
            row = resume_execution_work_order(str(work_order_id), delay_seconds=float(delay_seconds or 0.0))
        except Exception as exc:
            return tool_error(f"resume failed: {exc}")
        return tool_result({"action": action, "work_order": row})

    if action == "cancel":
        if not work_order_id:
            return tool_error("cancel requires work_order_id")
        try:
            row = cancel_execution_work_order(str(work_order_id))
        except Exception as exc:
            return tool_error(f"cancel failed: {exc}")
        return tool_result({"action": action, "work_order": row})

    if action == "runner_status":
        return tool_result({"action": action, **_runner_status_payload()})

    if action == "install_runner":
        try:
            config = _normalize_runner_config(
                schedule=schedule,
                limit=limit,
                reclaim_limit=reclaim_limit,
                claim_ttl_seconds=claim_ttl_seconds,
                model=model,
                provider=provider,
                base_url=base_url,
            )
            result = _upsert_runner_job(config)
        except Exception as exc:
            return tool_error(f"install_runner failed: {exc}")
        return tool_result({"action": action, **result})

    if action == "remove_runner":
        return tool_result({"action": action, **_remove_runner_jobs()})

    return tool_error(f"Unknown action: {action}")


registry.register(
    name="execution_work_orders",
    toolset="execution_work_orders",
    schema=EXECUTION_WORK_ORDERS_SCHEMA,
    handler=lambda args, **kw: execution_work_orders_tool(
        action=args.get("action"),
        work_order_id=args.get("work_order_id"),
        goal=args.get("goal"),
        context=args.get("context"),
        command=args.get("command"),
        timeout_seconds=args.get("timeout_seconds"),
        workdir=args.get("workdir"),
        schedule=args.get("schedule"),
        delay_seconds=args.get("delay_seconds"),
        max_attempts=args.get("max_attempts"),
        retry_delay_seconds=args.get("retry_delay_seconds"),
        status=args.get("status"),
        limit=args.get("limit"),
        claim_ttl_seconds=args.get("claim_ttl_seconds"),
        reclaim_limit=args.get("reclaim_limit"),
        model=args.get("model"),
        provider=args.get("provider"),
        base_url=args.get("base_url"),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=check_execution_work_orders_requirements,
    emoji="🛫",
)
