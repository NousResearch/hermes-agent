"""Agent Messaging Tool — inter-agent communication.

Exposes ``send_agent_message`` to the LLM so agents can send messages to
other agents, including delegate_task children, cron jobs, and broadcast
to all agents in the current task group.

Part of the multi-agent collaboration protocol (Layer 1: AgentMailbox).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Literal, Optional

logger = logging.getLogger(__name__)


# ── Tool schema: send_agent_message ─────────────────────────────────────

_MESSAGE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "send_agent_message",
        "description": (
            "Send a message to another agent. Use this to communicate with "
            "delegate_task children, cron workers, or broadcast to all agents "
            "in the current task group. The recipient reads messages at the "
            "start of their next turn. Messages support TTL for auto-expiry."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": (
                        "Recipient agent ID. Format: 'session:<session_id>' for "
                        "chat/delegate agents, 'cron:<job_id>' for cron workers, "
                        "'*' to broadcast to all agents in the task group."
                    ),
                },
                "msg_type": {
                    "type": "string",
                    "enum": ["instruction", "result", "handoff", "ping", "broadcast"],
                    "description": "Message type: instruction (task), result (output), handoff (transfer work), ping (health check), broadcast (announcement).",
                },
                "payload": {
                    "type": "object",
                    "description": "JSON payload with type-specific content. For instructions: {task, context}. For results: {summary, artifact_paths}. For handoffs: {task_spec, completed, remaining}.",
                },
                "ttl": {
                    "type": "integer",
                    "description": "Time-to-live in seconds. 0 = permanent. Use for time-sensitive messages (e.g., 300 = 5 min).",
                },
            },
            "required": ["to", "msg_type", "payload"],
        },
    },
}


# ── Handler ────────────────────────────────────────────────────────────

def handle_send_agent_message(
    to: str,
    msg_type: Literal["instruction", "result", "handoff", "ping", "broadcast"],
    payload: Dict[str, Any],
    ttl: int = 0,
    _task_id: str = "",
    _agent_id: str = "",
) -> str:
    """Send a message to another agent's mailbox.

    The actual message persistence happens via the tool-execution middleware
    in model_tools.py, which intercepts successful send_agent_message calls
    and writes to the agent_messages table in state.db.
    """
    from_id = _agent_id or f"session:{_task_id}"

    # Validate required fields
    if not to or not msg_type:
        return json.dumps({"sent": False, "error": "to and msg_type are required"})

    if msg_type not in ("instruction", "result", "handoff", "ping", "broadcast"):
        return json.dumps({"sent": False, "error": f"Invalid msg_type: {msg_type}"})

    # Build the message envelope — actual DB write happens post-execution
    message_envelope = {
        "to": to,
        "from": from_id,
        "type": msg_type,
        "payload": payload,
        "ttl": ttl,
        "at": time.time(),
    }

    # Return acknowledgment — the middleware persists it
    return json.dumps({
        "sent": True,
        "to": to,
        "from": from_id,
        "type": msg_type,
        "payload_preview": str(payload)[:200],
        "at": time.time(),
        "note": "Message queued for delivery. Recipient reads on next turn.",
    })


# ── Registration ───────────────────────────────────────────────────────


# ── Tool schema: blackboard_write ──────────────────────────────────────

_BLACKBOARD_WRITE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "blackboard_write",
        "description": (
            "Write a key to the shared blackboard for your task group. "
            "Other agents in the same task group can read these entries. "
            "Use this to post intermediate artifacts, decisions, or results "
            "that sibling agents need to know about."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_group": {
                    "type": "string",
                    "description": "Task group identifier shared across collaborating agents.",
                },
                "key": {
                    "type": "string",
                    "description": "Artifact key (e.g., 'analysis_result', 'decision', 'file_map').",
                },
                "value": {
                    "type": "object",
                    "description": "Value to store (any JSON-serializable structure).",
                },
                "ttl": {
                    "type": "integer",
                    "description": "Time-to-live in seconds. 0 = permanent.",
                },
            },
            "required": ["task_group", "key", "value"],
        },
    },
}

# ── Tool schema: blackboard_read ──────────────────────────────────────

_BLACKBOARD_READ_SCHEMA = {
    "type": "function",
    "function": {
        "name": "blackboard_read",
        "description": (
            "Read from the shared blackboard. Pass a key to read one entry, "
            "or omit to read all entries for your task group."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_group": {
                    "type": "string",
                    "description": "Task group identifier.",
                },
                "key": {
                    "type": "string",
                    "description": "Specific key to read. Omit to read all.",
                },
            },
            "required": ["task_group"],
        },
    },
}

# ── Tool schema: agent_handoff ────────────────────────────────────────

_HANDOFF_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "agent_handoff",
        "description": (
            "Transfer a task (or subtask) to another agent. The recipient "
            "receives a structured handoff message with task specification, "
            "completed steps, remaining work, and a task checkpoint. Combine "
            "with send_agent_message to notify the recipient."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient agent ID (e.g., 'session:<id>', 'cron:<job_id>').",
                },
                "task_spec": {
                    "type": "object",
                    "description": "Task specification: {goal, context, constraints, expected_output}.",
                },
                "completed": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of completed steps the recipient should NOT repeat.",
                },
                "remaining": {
                    "type": "string",
                    "description": "Description of remaining work.",
                },
                "checkpoint": {
                    "type": "object",
                    "description": "Optional checkpoint data (files written, decisions made).",
                },
            },
            "required": ["to", "task_spec"],
        },
    },
}


# ── Handler: blackboard_write ─────────────────────────────────────────

def handle_blackboard_write(
    task_group: str,
    key: str,
    value: Any,
    ttl: int = 0,
    _task_id: str = "",
    _agent_id: str = "",
) -> str:
    """Write to the shared blackboard."""
    import json as _json
    import time as _time

    try:
        from hermes_state import SessionDB
        db = SessionDB()
        agent_id = _agent_id or f"session:{_task_id}"

        db.blackboard_set(
            task_group=task_group,
            key=key,
            value=value,
            updated_by=agent_id,
            ttl=ttl,
        )
        return _json.dumps({
            "written": True,
            "task_group": task_group,
            "key": key,
            "by": agent_id,
        })
    except Exception as exc:
        logger.exception("blackboard_write failed")
        return _json.dumps({"written": False, "error": str(exc)})


# ── Handler: blackboard_read ──────────────────────────────────────────

def handle_blackboard_read(
    task_group: str,
    key: str = "",
    _task_id: str = "",
    _agent_id: str = "",
) -> str:
    """Read from the shared blackboard."""
    import json as _json

    try:
        from hermes_state import SessionDB
        db = SessionDB()

        result = db.blackboard_get(
            task_group=task_group,
            key=key or None,
        )
        return _json.dumps({
            "task_group": task_group,
            "entries": result or {},
        })
    except Exception as exc:
        logger.exception("blackboard_read failed")
        return _json.dumps({"error": str(exc)})


# ── Handler: agent_handoff ────────────────────────────────────────────

def handle_agent_handoff(
    to: str,
    task_spec: Dict[str, Any],
    completed: list | None = None,
    remaining: str = "",
    checkpoint: dict | None = None,
    _task_id: str = "",
    _agent_id: str = "",
) -> str:
    """Transfer a task to another agent. Creates a task checkpoint for the recipient."""
    import json as _json
    import time as _time

    from_id = _agent_id or f"session:{_task_id}"

    try:
        from hermes_state import SessionDB
        db = SessionDB()

        # 1. Create task checkpoint for the recipient
        goal = task_spec.get("goal", task_spec.get("task", str(task_spec)[:200]))
        db.save_task_checkpoint(
            to.replace("session:", "").replace("cron:", ""),
            task_goal=goal,
            current_phase="receiving handoff",
            completed_tool_calls=completed or [],
            iteration_budget_remaining=task_spec.get("budget", 50),
        )

        # 2. Send handoff message via mailbox
        payload = {
            "task_spec": task_spec,
            "completed": completed or [],
            "remaining": remaining,
            "checkpoint": checkpoint or {},
            "handoff_at": _time.time(),
        }
        db.send_agent_message(
            to_id=to,
            from_id=from_id,
            msg_type="handoff",
            payload=payload,
        )

        return _json.dumps({
            "handoff": True,
            "from": from_id,
            "to": to,
            "goal": goal,
            "message_sent": True,
            "checkpoint_created": True,
        })
    except Exception as exc:
        logger.exception("agent_handoff failed")
        return _json.dumps({"handoff": False, "error": str(exc)})


# ── Registration ───────────────────────────────────────────────────────

def register(tools_registry):
    """Register all agent collaboration tools."""
    tools_registry.register_tool(
        name="send_agent_message",
        schema=_MESSAGE_TOOL_SCHEMA,
        handler=handle_send_agent_message,
        toolset="agent_collaboration",
    )
    tools_registry.register_tool(
        name="blackboard_write",
        schema=_BLACKBOARD_WRITE_SCHEMA,
        handler=handle_blackboard_write,
        toolset="agent_collaboration",
    )
    tools_registry.register_tool(
        name="blackboard_read",
        schema=_BLACKBOARD_READ_SCHEMA,
        handler=handle_blackboard_read,
        toolset="agent_collaboration",
    )
    tools_registry.register_tool(
        name="agent_handoff",
        schema=_HANDOFF_TOOL_SCHEMA,
        handler=handle_agent_handoff,
        toolset="agent_collaboration",
    )