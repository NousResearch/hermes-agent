"""Tool wrapper for Hermes' structured self-evolution ledger."""

from __future__ import annotations

import json

from agent.self_evolution import (
    export_context,
    list_lessons,
    recall_lessons,
    record_lesson,
    resolve_lesson,
)
from tools.registry import registry


def check_requirements() -> bool:
    return True


SELF_EVOLUTION_SCHEMA = {
    "name": "self_evolution",
    "description": (
        "Record and recall structured lessons from Hermes' own mistakes so the "
        "agent can avoid repeating them. Use record after a verified mistake or "
        "workflow correction, recall before similar risky work, resolve when a "
        "lesson is obsolete, and export_context to get compact reminders."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["record", "recall", "list", "resolve", "export_context"],
                "description": "Operation to perform.",
            },
            "mistake": {
                "type": "string",
                "description": "For record: the mistaken pattern to avoid.",
            },
            "lesson": {
                "type": "string",
                "description": "For record: the corrected behavior to apply next time.",
            },
            "trigger": {
                "type": "string",
                "description": "Situation where this lesson should be recalled.",
            },
            "fix": {
                "type": "string",
                "description": "Concrete fix, command, check, or workflow.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Short retrieval tags like provider, tests, browser, memory.",
            },
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Brief evidence: user correction, failing test, error summary, or file path.",
            },
            "severity": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"],
                "description": "How damaging repetition would be.",
            },
            "confidence": {
                "type": "number",
                "description": "0.0-1.0 confidence that the lesson is durable.",
            },
            "source": {
                "type": "string",
                "description": "Origin such as background_review, user_correction, test_failure, manual.",
            },
            "query": {
                "type": "string",
                "description": "For recall/export_context: task or risk to search for.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum lessons to return.",
            },
            "lesson_id": {
                "type": "string",
                "description": "For resolve: lesson id to retire.",
            },
            "outcome": {
                "type": "string",
                "description": "For resolve: why the lesson is now obsolete or satisfied.",
            },
            "include_resolved": {
                "type": "boolean",
                "description": "For recall: include resolved lessons.",
            },
        },
        "required": ["action"],
    },
}


def self_evolution(args: dict, **_kw) -> str:
    action = str(args.get("action") or "").strip()
    if action == "record":
        result = record_lesson(
            mistake=args.get("mistake") or "",
            lesson=args.get("lesson") or "",
            trigger=args.get("trigger") or "",
            fix=args.get("fix") or "",
            tags=args.get("tags") or [],
            evidence=args.get("evidence") or [],
            severity=args.get("severity") or "medium",
            source=args.get("source") or "agent",
            confidence=args.get("confidence", 0.7),
        )
    elif action == "recall":
        result = recall_lessons(
            query=args.get("query") or "",
            tags=args.get("tags") or [],
            limit=args.get("limit") or 5,
            include_resolved=bool(args.get("include_resolved", False)),
        )
    elif action == "list":
        result = list_lessons(
            status=args.get("status") or "active",
            limit=args.get("limit") or 20,
        )
    elif action == "resolve":
        result = resolve_lesson(
            lesson_id=args.get("lesson_id") or "",
            outcome=args.get("outcome") or "",
        )
    elif action == "export_context":
        result = export_context(
            query=args.get("query") or "",
            tags=args.get("tags") or [],
            limit=args.get("limit") or 5,
        )
    else:
        result = {"success": False, "error": f"unknown action: {action}"}
    return json.dumps(result, ensure_ascii=False, indent=2)


registry.register(
    name="self_evolution",
    toolset="memory",
    schema=SELF_EVOLUTION_SCHEMA,
    handler=self_evolution,
    check_fn=check_requirements,
    emoji="",
)
