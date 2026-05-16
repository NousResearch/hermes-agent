#!/usr/bin/env python3
"""
ChiefAgent Tool — Hermes OS Brain Layer

Delegates complex tasks to Hermes OS ChiefAgent for intent parsing,
task decomposition, and Labor execution routing.
"""

from __future__ import annotations

import json
import sys
import os
import asyncio
from typing import Any

# --- Robust path setup ---
# Try multiple possible Hermes OS locations
_HERMES_OS_SEARCH_PATHS = [
    "/Volumes/LEGION/hermes",  # LEGION USB SSD (parent dir with hermes_os)
    "/Volumes/LEGION/.hermes",  # LEGION symlink
    "/Users/dongshenglu/hermes-os/src",  # MacBook local dev
]

for _path in _HERMES_OS_SEARCH_PATHS:
    if os.path.exists(_path):
        if _path not in sys.path:
            sys.path.insert(0, _path)
        # Also check if hermes_os is directly accessible
        _hermes_os_subpath = os.path.join(_path, "hermes_os")
        if os.path.isdir(_hermes_os_subpath) and _hermes_os_subpath not in sys.path:
            sys.path.insert(0, _hermes_os_subpath)
        break

# --- Import Hermes OS ---
HERMES_OS_AVAILABLE = False
ChiefAgent = None

try:
    from hermes_os.chief_agent import ChiefAgent, Intent
    HERMES_OS_AVAILABLE = True
except ImportError:
    try:
        from hermes_os.chief_agent import ChiefAgent
        HERMES_OS_AVAILABLE = True
        Intent = None  # Intent not critical
    except ImportError:
        HERMES_OS_AVAILABLE = False


def _run_async(coro):
    """Run async coroutine in sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def chief_agent_tool(message: str, action: str = "parse_intent", user_id: str = "default") -> str:
    """
    Invoke Hermes OS ChiefAgent for intent parsing or response synthesis.

    Args:
        message: The user message to process
        action: "parse_intent" (default) or "spring_breeze"
        user_id: User identifier for context tracking

    Returns:
        JSON string with processing result
    """
    if not HERMES_OS_AVAILABLE:
        return json.dumps({
            "error": "Hermes OS not available",
            "debug_paths": sys.path[:3],
        }, ensure_ascii=False)

    if not message or not message.strip():
        return json.dumps({"error": "message is required"}, ensure_ascii=False)

    chief = ChiefAgent()

    if action == "spring_breeze":
        result = _run_async(
            chief.get_spring_breeze_response(
                user_id=user_id,
                message=message,
                intent="unknown",
            )
        )
        return json.dumps({"success": True, "response": result}, ensure_ascii=False)

    # Default: parse_intent
    parsed = _run_async(
        chief.parse_intent(message, user_id=user_id)
    )
    return json.dumps({
        "success": True,
        "action": parsed.action.value,
        "confidence": parsed.confidence,
        "entities": parsed.entities,
        "suggested_next": parsed.suggested_next,
        "raw_text": parsed.raw_text,
    }, ensure_ascii=False, indent=2)


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

CHIEF_AGENT_SCHEMA = {
    "name": "chief_agent",
    "description": (
        "Delegate complex tasks to Hermes OS ChiefAgent brain layer.\n\n"
        "Use this tool when:\n"
        "- Task requires intent parsing and routing to specialized agents\n"
        "- User wants Hermes OS to handle the full orchestration\n"
        "- Complex tasks that need GoalTracker / TopicTracker context\n\n"
        "Returns structured intent analysis with confidence scores."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The user message to process",
            },
            "action": {
                "type": "string",
                "enum": ["parse_intent", "spring_breeze"],
                "description": "Action: parse_intent (default) or spring_breeze",
            },
            "user_id": {
                "type": "string",
                "description": "User identifier for context tracking",
            },
        },
        "required": ["message"],
    },
}


# --- Registry ---
def _chief_agent_handler(args, **kw):
    return chief_agent_tool(
        message=args.get("message", ""),
        action=args.get("action", "parse_intent"),
        user_id=args.get("user_id", "default"),
    )

# Registry.register must be at module level (bare Expr, not inside try/except)
from tools.registry import registry
registry.register(
    name="chief_agent",
    toolset="chief",
    schema=CHIEF_AGENT_SCHEMA,
    handler=_chief_agent_handler,
    check_fn=lambda: HERMES_OS_AVAILABLE,
    emoji="🧠",
)