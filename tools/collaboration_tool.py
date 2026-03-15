#!/usr/bin/env python3
"""Cross-agent collaboration tool for webhook-backed Hermes peers."""

import asyncio
import json
import threading
from typing import Any, Dict, Optional

from gateway.collaboration import create_collaboration_job, resolve_target_alias
from gateway.session import SessionSource, build_session_key
from tools.registry import registry


DEFAULT_TIMEOUT_SECONDS = 300


def _collaboration_config_enabled() -> bool:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        collaboration = cfg.get("collaboration", {}) if isinstance(cfg, dict) else {}
        if not isinstance(collaboration, dict):
            return False
        return bool(collaboration.get("enabled"))
    except Exception:
        return False


def check_collaboration_requirements() -> bool:
    return _collaboration_config_enabled()


COLLABORATION_TOOL_SCHEMA = {
    "name": "collaborate_with_agent",
    "description": (
        "Collaborate with another configured Hermes agent session alias such as a webhook/miniverse peer. "
        "Use this only for configured peer agents with their own session identity and return channel, "
        "not for ordinary in-process subagent work."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target_agent": {"type": "string", "description": "Configured collaboration target alias."},
            "task": {"type": "string", "description": "The task for the peer agent."},
            "context": {"type": "string", "description": "Optional extra context appended to the task."},
            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 900},
        },
        "required": ["target_agent", "task"],
    },
}


def collaborate_with_agent(
    *,
    target_agent: str,
    task: str,
    context: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
    parent_agent=None,
) -> str:
    if parent_agent is None:
        return json.dumps({"status": "error", "error": "collaborate_with_agent requires a parent agent context."})

    runner = getattr(parent_agent, "gateway_runner", None)
    requester_session_key = getattr(parent_agent, "gateway_session_key", None)
    source = getattr(parent_agent, "gateway_source", None)
    if runner is None or requester_session_key is None or source is None:
        return json.dumps({"status": "error", "error": "collaborate_with_agent is only supported for gateway-backed agent sessions."})

    if not check_collaboration_requirements():
        return json.dumps({"status": "error", "error": "collaboration is not enabled in config."})

    timeout_seconds = int(timeout_seconds or DEFAULT_TIMEOUT_SECONDS)
    combined_task = task.strip()
    if context and context.strip():
        combined_task += f"\n\nContext:\n{context.strip()}"

    try:
        resolved = resolve_target_alias(runner.config, target_agent, requester_session_key=requester_session_key)
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)})

    target_source = SessionSource(
        platform=source.platform,
        chat_id=str(resolved["chat_id"]),
        chat_type="dm",
        user_id=str(resolved.get("chat_id", target_agent)),
        user_name=str(resolved.get("display_name", target_agent)),
    )
    target_session_key = build_session_key(target_source)

    lineage = list(getattr(parent_agent, "_collaboration_lineage", []) or [])
    if target_session_key in lineage or target_session_key == requester_session_key:
        return json.dumps({"status": "error", "error": "Collaboration cycle detected for target session."})

    job = create_collaboration_job(
        store=runner.collaboration_store,
        requester_session_key=requester_session_key,
        target_session_key=target_session_key,
        target_agent=target_agent,
        task_text=combined_task,
        lineage=lineage + [requester_session_key],
    )

    handle = runner.register_collaboration_wait(job.job_id, requester_session_key)
    loop = getattr(runner, "_event_loop", None)
    if loop is None:
        runner.unregister_collaboration_wait(job.job_id)
        return json.dumps({"status": "error", "error": "Gateway event loop is not available for collaboration."})

    future = asyncio.run_coroutine_threadsafe(
        runner.route_collaboration_request(
            requester_session_key=requester_session_key,
            target_session_key=target_session_key,
            job_id=job.job_id,
            task_text=combined_task,
            lineage=job.lineage,
            requester_agent=getattr(source, "chat_id", requester_session_key),
            target_source=target_source.to_dict(),
        ),
        loop,
    )
    future.result(timeout=5)

    try:
        payload = handle["future"].result(timeout=timeout_seconds)
    except Exception as exc:
        runner.unregister_collaboration_wait(job.job_id)
        stored = runner.collaboration_store.get_job(job.job_id)
        if stored:
            stored.status = "failed"
            stored.error_reason = type(exc).__name__
            runner.collaboration_store.save_job(stored)
        return json.dumps({"status": "error", "error": f"Collaboration failed: {exc}"})
    finally:
        runner.unregister_collaboration_wait(job.job_id)

    if isinstance(payload, dict) and "result_text" in payload:
        try:
            result = json.loads(payload["result_text"])
        except Exception:
            result = {"status": "completed", "result": payload["result_text"]}
    else:
        result = {"status": "completed", "result": payload}

    return json.dumps(result)


registry.register(
    name="collaborate_with_agent",
    toolset="collaboration",
    schema=COLLABORATION_TOOL_SCHEMA,
    handler=lambda args, **kw: collaborate_with_agent(
        target_agent=args.get("target_agent", ""),
        task=args.get("task", ""),
        context=args.get("context"),
        timeout_seconds=args.get("timeout_seconds"),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=check_collaboration_requirements,
)
