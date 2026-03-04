#!/usr/bin/env python3
"""
AgentMail Tool — Agent-owned email inboxes.

Gives Hermes its own dedicated email address (e.g. hermes@agentmail.to)
so it can send, receive and manage email autonomously — separate from the
user's personal inbox.

Requires:
    pip install agentmail
    AGENTMAIL_API_KEY in ~/.hermes/.env
"""

import json
import logging
import os
from typing import Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy client
# ---------------------------------------------------------------------------

_client = None

def _get_client():
    global _client
    if _client is None:
        from agentmail import AgentMail
        api_key = os.getenv("AGENTMAIL_API_KEY")
        if not api_key:
            raise ValueError("AGENTMAIL_API_KEY environment variable not set")
        _client = AgentMail(api_key=api_key)
    return _client

# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def check_agentmail_requirements() -> tuple[bool, str]:
    try:
        import agentmail  # noqa: F401
    except ImportError:
        return False, "agentmail package not installed — run: pip install agentmail"
    if not os.getenv("AGENTMAIL_API_KEY"):
        return False, "AGENTMAIL_API_KEY not set in ~/.hermes/.env"
    return True, ""

# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def _handle_agentmail(
    action: str,
    username: Optional[str] = None,
    inbox_id: Optional[str] = None,
    to: Optional[str] = None,
    subject: Optional[str] = None,
    body: Optional[str] = None,
    message_id: Optional[str] = None,
    limit: int = 10,
) -> str:
    try:
        client = _get_client()

        # ── create_inbox ────────────────────────────────────────────────────
        if action == "create_inbox":
            kwargs = {}
            if username:
                kwargs["username"] = username
            inbox = client.inboxes.create(**kwargs)
            return json.dumps({
                "inbox_id": inbox.id,
                "email": inbox.email_address,
                "created_at": str(inbox.created_at),
            }, ensure_ascii=False)

        # ── list_inboxes ─────────────────────────────────────────────────────
        elif action == "list_inboxes":
            inboxes = client.inboxes.list()
            return json.dumps([
                {"inbox_id": i.id, "email": i.email_address}
                for i in (inboxes.inboxes or [])
            ], ensure_ascii=False)

        # ── send_email ───────────────────────────────────────────────────────
        elif action == "send_email":
            if not inbox_id:
                return "Error: inbox_id is required for send_email"
            if not to:
                return "Error: to is required for send_email"
            if not subject:
                return "Error: subject is required for send_email"
            if not body:
                return "Error: body is required for send_email"
            msg = client.inboxes.messages.send(
                inbox_id=inbox_id,
                to=[{"email": to}],
                subject=subject,
                text=body,
            )
            return json.dumps({
                "message_id": msg.id,
                "status": "sent",
                "to": to,
                "subject": subject,
            }, ensure_ascii=False)

        # ── list_messages ────────────────────────────────────────────────────
        elif action == "list_messages":
            if not inbox_id:
                return "Error: inbox_id is required for list_messages"
            msgs = client.inboxes.messages.list(inbox_id=inbox_id, limit=limit)
            return json.dumps([
                {
                    "message_id": m.id,
                    "from": m.from_address,
                    "subject": m.subject,
                    "received_at": str(m.received_at),
                    "snippet": (m.text or "")[:200],
                }
                for m in (msgs.messages or [])
            ], ensure_ascii=False)

        # ── get_message ──────────────────────────────────────────────────────
        elif action == "get_message":
            if not inbox_id:
                return "Error: inbox_id is required for get_message"
            if not message_id:
                return "Error: message_id is required for get_message"
            msg = client.inboxes.messages.get(inbox_id=inbox_id, message_id=message_id)
            return json.dumps({
                "message_id": msg.id,
                "from": msg.from_address,
                "subject": msg.subject,
                "body": msg.text or msg.html or "",
                "received_at": str(msg.received_at),
            }, ensure_ascii=False)

        # ── reply ────────────────────────────────────────────────────────────
        elif action == "reply":
            if not inbox_id:
                return "Error: inbox_id is required for reply"
            if not message_id:
                return "Error: message_id is required for reply"
            if not body:
                return "Error: body is required for reply"
            reply = client.inboxes.messages.reply(
                inbox_id=inbox_id,
                message_id=message_id,
                text=body,
            )
            return json.dumps({
                "message_id": reply.id,
                "status": "replied",
            }, ensure_ascii=False)

        # ── delete_inbox ─────────────────────────────────────────────────────
        elif action == "delete_inbox":
            if not inbox_id:
                return "Error: inbox_id is required for delete_inbox"
            client.inboxes.delete(inbox_id=inbox_id)
            return json.dumps({"status": "deleted", "inbox_id": inbox_id})

        else:
            return f"Error: unknown action '{action}'. Valid actions: create_inbox, list_inboxes, send_email, list_messages, get_message, reply, delete_inbox"

    except Exception as e:
        logger.error(f"AgentMail error: {e}")
        return f"Error: {e}"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

AGENTMAIL_SCHEMA = {
    "name": "agentmail",
    "description": (
        "Give Hermes its own dedicated email inbox via AgentMail. "
        "The agent gets a real email address (e.g. hermes@agentmail.to) and can "
        "send, receive, and reply to emails autonomously — independent of the user's personal inbox. "
        "Use this for outreach, automated workflows, sign-ups, or agent-to-human communication.\n\n"
        "Actions:\n"
        "  create_inbox  — Create a new agent-owned inbox (optional: username)\n"
        "  list_inboxes  — List all existing inboxes\n"
        "  send_email    — Send an email from an inbox (requires: inbox_id, to, subject, body)\n"
        "  list_messages — List received messages (requires: inbox_id)\n"
        "  get_message   — Get full message content (requires: inbox_id, message_id)\n"
        "  reply         — Reply to a message (requires: inbox_id, message_id, body)\n"
        "  delete_inbox  — Delete an inbox (requires: inbox_id)"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create_inbox", "list_inboxes", "send_email", "list_messages", "get_message", "reply", "delete_inbox"],
                "description": "The action to perform.",
            },
            "username": {
                "type": "string",
                "description": "Desired inbox username (e.g. 'hermes-agent' → hermes-agent@agentmail.to). Optional for create_inbox.",
            },
            "inbox_id": {
                "type": "string",
                "description": "The inbox ID to operate on.",
            },
            "to": {
                "type": "string",
                "description": "Recipient email address for send_email.",
            },
            "subject": {
                "type": "string",
                "description": "Email subject for send_email.",
            },
            "body": {
                "type": "string",
                "description": "Email body text for send_email or reply.",
            },
            "message_id": {
                "type": "string",
                "description": "Message ID for get_message or reply.",
            },
            "limit": {
                "type": "integer",
                "description": "Max messages to return for list_messages (default: 10).",
                "default": 10,
            },
        },
        "required": ["action"],
    },
}

# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

registry.register(
    name="agentmail",
    toolset="agentmail",
    schema=AGENTMAIL_SCHEMA,
    handler=_handle_agentmail,
    check_fn=check_agentmail_requirements,
    requires_env=["AGENTMAIL_API_KEY"],
)
