"""Tool wrappers for the standard Hermes knowledge router."""

from __future__ import annotations

import json
from typing import Any

from hermes_cli.config import load_config
from knowledge.adapters.hindsight import HindsightMemoryAdapter
from knowledge.adapters.siyuan import SiYuanKnowledgeAdapter
from knowledge.policy import RoutePolicy
from knowledge.router import KnowledgeRouter, route_knowledge_write
from knowledge.types import KnowledgeWriteRequest
from tools.registry import registry, tool_error, tool_result


def _request_from_args(args: dict[str, Any]) -> KnowledgeWriteRequest:
    return KnowledgeWriteRequest(
        content_type=args.get("content_type", ""),
        title=args.get("title", "") or "",
        content=args.get("content", "") or args.get("markdown", "") or "",
        context=args.get("context", "") or "",
        tags=args.get("tags") or (),
        dry_run=bool(args.get("dry_run", False)),
        destination_override=args.get("destination_override") or None,
        idempotency_key=args.get("idempotency_key", "") or "",
        update_mode=args.get("update_mode", "") or "",
        duplicate_policy=args.get("duplicate_policy", "update_existing") or "update_existing",
        metadata=args.get("metadata") or {},
        notebook=args.get("notebook", "") or "",
        path=args.get("path", "") or "",
    )


def build_default_router() -> KnowledgeRouter:
    try:
        policy = RoutePolicy.from_config(load_config())
    except Exception:
        policy = RoutePolicy()
    return KnowledgeRouter(
        static_adapter=SiYuanKnowledgeAdapter(),
        dynamic_adapter=HindsightMemoryAdapter(),
        policy=policy,
    )


def handle_knowledge_route_decision(args: dict[str, Any], **_: Any) -> str:
    try:
        request = _request_from_args({**args, "dry_run": True})
        decision = route_knowledge_write(request)
        return tool_result(success=True, decision=decision.to_dict())
    except Exception as exc:
        return tool_error(str(exc), success=False)


def handle_knowledge_write(args: dict[str, Any], **_: Any) -> str:
    try:
        request = _request_from_args(args)
        decision = route_knowledge_write(request)
        # Dry-run/review/skip must not construct backend adapters; constructing
        # SiYuan/Hindsight may require credentials or network that are irrelevant
        # to classification.
        if request.dry_run or decision.destination in {"none", "review"}:
            return tool_result(
                success=True,
                dry_run=True,
                written=False,
                destination=str(decision.destination),
                action=str(decision.action),
                decision=decision.to_dict(),
            )
        return tool_result(build_default_router().write(request).to_dict())
    except Exception as exc:
        return tool_error(str(exc), success=False)


KNOWLEDGE_ROUTE_DECISION_SCHEMA = {
    "name": "knowledge_route_decision",
    "description": "Preferred dry-run classifier for durable knowledge routing. Use this when unsure whether content belongs in static knowledge, dynamic memory, skip, or review.",
    "parameters": {
        "type": "object",
        "properties": {
            "content_type": {"type": "string", "description": "runbook, architecture, user_preference, lesson, task_log, etc."},
            "title": {"type": "string", "description": "Document title when relevant."},
            "content": {"type": "string", "description": "Content to classify."},
            "destination_override": {"type": "string", "enum": ["static_knowledge", "dynamic_memory", "none", "review"], "description": "Optional explicit destination for dry-run classification."},
        },
        "required": ["content_type"],
    },
}

KNOWLEDGE_WRITE_SCHEMA = {
    "name": "knowledge_write",
    "description": (
        "Preferred high-level durable knowledge write router for Hermes. Static docs go to the configured static backend; "
        "dynamic preferences/facts/lessons go to the configured memory backend; ephemeral task logs are skipped. "
        "Use dry_run=true to inspect routing before writing."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content_type": {"type": "string", "description": "Known type: runbook, architecture, decision, postmortem, research, api_doc, user_preference, fact, lesson, task_log."},
            "title": {"type": "string", "description": "Required for static document writes."},
            "content": {"type": "string", "description": "Markdown/document text or memory content."},
            "notebook": {"type": "string", "description": "Optional static backend notebook override."},
            "path": {"type": "string", "description": "Optional static backend document path override."},
            "context": {"type": "string", "description": "Optional dynamic memory retain context."},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional dynamic memory tags."},
            "dry_run": {"type": "boolean", "description": "If true, only return the route decision."},
            "destination_override": {"type": "string", "enum": ["static_knowledge", "dynamic_memory", "none", "review"], "description": "Optional explicit destination override."},
            "idempotency_key": {"type": "string", "description": "Optional caller-supplied key for dedupe/update decisions."},
            "update_mode": {"type": "string", "description": "Optional backend-specific update mode."},
            "duplicate_policy": {"type": "string", "enum": ["update_existing", "create_new", "skip_existing", "review"], "description": "How to handle existing static knowledge matches."},
            "metadata": {"type": "object", "description": "Optional structured metadata."},
        },
        "required": ["content_type", "content"],
    },
}


def _check_knowledge_router_available() -> bool:
    return True


registry.register(
    name="knowledge_route_decision",
    toolset="knowledge",
    schema=KNOWLEDGE_ROUTE_DECISION_SCHEMA,
    handler=handle_knowledge_route_decision,
    check_fn=_check_knowledge_router_available,
    emoji="🧭",
)
registry.register(
    name="knowledge_write",
    toolset="knowledge",
    schema=KNOWLEDGE_WRITE_SCHEMA,
    handler=handle_knowledge_write,
    check_fn=_check_knowledge_router_available,
    emoji="🧠",
    max_result_size_chars=30_000,
)
