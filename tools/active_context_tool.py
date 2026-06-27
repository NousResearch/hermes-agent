#!/usr/bin/env python3
"""
Active Context Tool — short-lived session task tracking.

Maintains exactly one "currently working on" string per session.
Auto-clears on session reset (/new, auto-reset). Prominently
surfaced in the system prompt so the agent always knows what
it was doing before a context compression or reset.

Difference from durable memory:
  - Auto-clears on session boundaries
  - Shorter TTL — hours, not forever
  - One field only — swapped atomically
  - Token-efficient — no scanning, just a structured string
"""

import json
import time
from typing import Optional

# Maximum length for the task description
_MAX_TASK_CHARS = 500

# Default TTL in seconds (4 hours)
_DEFAULT_TTL_SECONDS = 4 * 60 * 60


class ActiveContextStore:
    """In-memory store for the active task context. One per session."""

    def __init__(self, ttl_seconds: int = _DEFAULT_TTL_SECONDS):
        self._task: Optional[str] = None
        self._set_at: Optional[float] = None
        self._ttl = ttl_seconds

    def set(self, task: str) -> str:
        """Record the current task. Returns confirmation."""
        task = task.strip()[:_MAX_TASK_CHARS]
        if not task:
            return self.clear()
        self._task = task
        self._set_at = time.time()
        return json.dumps({"status": "set", "task": task})

    def get(self) -> str:
        """Return the current task, or empty if expired/cleared."""
        if self._task is None:
            return json.dumps({"status": "empty", "task": None})
        if self._set_at is not None and (time.time() - self._set_at) > self._ttl:
            expired = self._task
            self._task = None
            self._set_at = None
            return json.dumps({"status": "expired", "task": expired})
        return json.dumps({"status": "active", "task": self._task})

    def clear(self) -> str:
        """Clear the active context."""
        old = self._task
        self._task = None
        self._set_at = None
        if old:
            return json.dumps({"status": "cleared", "task": old})
        return json.dumps({"status": "empty", "task": None})

    def format_for_prompt(self) -> Optional[str]:
        """Return a prompt-injectable string, or None if no active task."""
        if self._task is None:
            return None
        if self._set_at is not None and (time.time() - self._set_at) > self._ttl:
            self._task = None
            self._set_at = None
            return None
        return f"[Active context: {self._task}]"

    def reset(self):
        """Clear on session reset."""
        self._task = None
        self._set_at = None


def active_context_tool(
    action: str = "get",
    task: Optional[str] = None,
    store: Optional[ActiveContextStore] = None,
) -> str:
    """Handle active_context tool calls.

    Args:
        action: 'set', 'get', or 'clear'
        task: The task description (required when action='set')
        store: The ActiveContextStore instance from the AIAgent.
    """
    if store is None:
        return json.dumps({"error": "ActiveContextStore not initialized"})

    if action == "set":
        if not task or not task.strip():
            return json.dumps({"error": "task is required when action='set'"})
        return store.set(task)
    elif action == "get":
        return store.get()
    elif action == "clear":
        return store.clear()
    else:
        return json.dumps({"error": f"Unknown action: {action}. Use 'set', 'get', or 'clear'."})


def check_active_context_requirements() -> bool:
    """No external requirements."""
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

ACTIVE_CONTEXT_SCHEMA = {
    "name": "active_context",
    "description": (
        "Track what you are currently working on in this session. "
        "Use this when switching between tasks or starting a complex multi-step "
        "task so you can recover context after auto-resets or /new.\n\n"
        "Actions:\n"
        "- set: record your current task (1-2 short sentences)\n"
        "- get: read the current task\n"
        "- clear: remove the current task\n\n"
        "The active context is automatically injected into your system prompt "
        "and auto-clears on session boundaries. Unlike durable memory, this is "
        "ephemeral and session-scoped."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["set", "get", "clear"],
                "description": (
                    "set: record current task. get: read current task. "
                    "clear: remove current task."
                ),
            },
            "task": {
                "type": "string",
                "description": (
                    "Short task description (1-2 sentences). "
                    "Required when action='set', ignored otherwise."
                ),
            },
        },
        "required": ["action"],
    },
}


# --- Registry ---
from tools.registry import registry

registry.register(
    name="active_context",
    toolset="memory",
    schema=ACTIVE_CONTEXT_SCHEMA,
    handler=lambda args, **kw: active_context_tool(
        action=args.get("action", "get"),
        task=args.get("task"),
        store=kw.get("store"),
    ),
    check_fn=check_active_context_requirements,
    emoji="🎯",
)
