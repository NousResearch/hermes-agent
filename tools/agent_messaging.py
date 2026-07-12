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


# ── Tool schema ────────────────────────────────────────────────────────

TOOL_SCHEMA = {
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

def register(tools_registry):
    """Register the agent messaging tool."""
    tools_registry.register_tool(
        name="send_agent_message",
        schema=TOOL_SCHEMA,
        handler=handle_send_agent_message,
        toolset="agent_messaging",
    )