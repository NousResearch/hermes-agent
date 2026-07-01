"""CRWD handoff tool -- loop a human into the current Chatwoot conversation.

Registers a single LLM-callable tool ``crwd_handoff`` (gated on Chatwoot creds)
that posts an **internal private note** to the CRWD Coach's current Chatwoot
conversation so a human agent has context and can take over. The member-facing
"I'm looping in a human" message is just the agent's normal reply text -- this
tool only handles the internal notification.

Self-contained by design: it resolves the current conversation from the gateway
session context (``HERMES_SESSION_PLATFORM`` / ``HERMES_SESSION_CHAT_ID``) and
calls the Chatwoot messages API directly with ``private: true``. It does not
depend on the ``CHATWOOT_PRIVATE_NOTE_TRACE`` debug flag or on holding a
reference to the running adapter.

Connection comes from the same env the Chatwoot adapter uses:
``CHATWOOT_BASE_URL``, ``CHATWOOT_AGENT_TOKEN`` (falling back to
``CHATWOOT_TOKEN``), and ``CHATWOOT_ACCOUNT_ID`` (used when the session chat id
has no account prefix). If the note cannot be posted, the tool degrades
gracefully -- it never raises -- so the coach still delivers the warm handoff
message to the member.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

_TIMEOUT_S = 8
_MAX_SUMMARY = 1500


# --- Availability ---

def _agent_token() -> str:
    return (os.getenv("CHATWOOT_AGENT_TOKEN", "") or os.getenv("CHATWOOT_TOKEN", "")).strip()


def check_crwd_handoff_requirements() -> bool:
    """Available only in a Chatwoot deployment (base URL + a usable token)."""
    return bool(os.getenv("CHATWOOT_BASE_URL", "").strip() and _agent_token())


# --- Conversation resolution ---

def _resolve_conversation() -> Tuple[Optional[str], Optional[str]]:
    """Return ``(account_id, conversation_id)`` for the current session.

    Reads the gateway session context the same way cronjob/kanban tools do. The
    Chatwoot chat id is encoded as ``account:conversation`` (see the adapter's
    ``_format_chat_id``); a bare id falls back to ``CHATWOOT_ACCOUNT_ID``.
    """
    try:
        from gateway.session_context import get_session_env
    except Exception:  # pragma: no cover - gateway always present in prod
        return None, None

    platform = (get_session_env("HERMES_SESSION_PLATFORM", "") or "").strip().lower()
    if platform and platform != "chatwoot":
        return None, None

    chat_id = (get_session_env("HERMES_SESSION_CHAT_ID", "") or "").strip()
    if not chat_id:
        return None, None

    default_account = os.getenv("CHATWOOT_ACCOUNT_ID", "").strip()
    if ":" in chat_id:
        account, _, conversation = chat_id.partition(":")
        account = account.strip() or default_account
        conversation = conversation.strip()
    else:
        account, conversation = default_account, chat_id
    if not account or not conversation:
        return None, None
    return account, conversation


# --- Private-note post ---

def _post_private_note(account_id: str, conversation_id: str, content: str) -> Tuple[bool, str]:
    base_url = os.getenv("CHATWOOT_BASE_URL", "").strip().rstrip("/")
    token = _agent_token()
    url = f"{base_url}/api/v1/accounts/{account_id}/conversations/{conversation_id}/messages"
    body = json.dumps(
        {"content": content, "message_type": "outgoing", "private": True}
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "api_access_token": token},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            if 200 <= resp.status < 300:
                return True, ""
            return False, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except Exception as exc:  # network / URL / timeout
        return False, str(exc)


def _compose_note(reason: str, summary: str) -> str:
    reason = (reason or "").strip() or "handoff requested"
    summary = (summary or "").strip()[:_MAX_SUMMARY]
    lines = [f"🔔 CRWD Coach handoff — {reason}"]
    if summary:
        lines.append(summary)
    lines.append("A human agent should take over this conversation.")
    return "\n\n".join(lines)


# --- Handler ---

def crwd_handoff_tool(args: Dict[str, Any], **_kw: Any) -> str:
    if not check_crwd_handoff_requirements():
        # Not a Chatwoot session with creds — tell the agent to proceed with the
        # member-facing handoff message anyway (see the crwd-handoff skill).
        return json.dumps(
            {
                "_type": "crwd_handoff",
                "notified": False,
                "reason": "Chatwoot not configured; skip the note and still hand off to the member.",
                "error": None,
            },
            ensure_ascii=False,
        )

    account_id, conversation_id = _resolve_conversation()
    if not account_id or not conversation_id:
        return json.dumps(
            {
                "_type": "crwd_handoff",
                "notified": False,
                "reason": "No current Chatwoot conversation; skip the note and still hand off to the member.",
                "error": None,
            },
            ensure_ascii=False,
        )

    note = _compose_note(str(args.get("reason", "")), str(args.get("summary", "")))
    ok, err = _post_private_note(account_id, conversation_id, note)
    if not ok:
        # Never hard-fail: a note that can't post must not block the handoff.
        logger.warning("[crwd_handoff] private note failed (%s)", err)
        return json.dumps(
            {
                "_type": "crwd_handoff",
                "notified": False,
                "reason": "Internal note could not be posted; still hand off to the member warmly.",
                "error": None,
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "_type": "crwd_handoff",
            "notified": True,
            "reason": "Team notified via internal note. Send the member a warm handoff message, then stop replying.",
            "error": None,
        },
        ensure_ascii=False,
    )


# --- Schema ---

CRWD_HANDOFF_SCHEMA = {
    "name": "crwd_handoff",
    "description": (
        "Loop a human into the CURRENT CRWD conversation by posting an internal "
        "note for the team (frustration/anger, repeated unresolved issue, "
        "rejected submission, money/account dispute, or an out-of-scope-but-"
        "relevant question you can't safely answer). This posts the internal "
        "note only — you must still send the member a short, warm 'looping in a "
        "human' message yourself, then stop replying. Safe to call even outside "
        "Chatwoot: it no-ops the note and tells you to hand off anyway."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Short category of why (e.g. 'frustrated member', 'rejected submission', 'payment dispute').",
            },
            "summary": {
                "type": "string",
                "description": "One or two lines of context for the human agent: what the member needs and what you already tried.",
            },
        },
        "required": ["reason"],
    },
}


# --- Registration ---

registry.register(
    name="crwd_handoff",
    toolset="crwd",
    schema=CRWD_HANDOFF_SCHEMA,
    handler=crwd_handoff_tool,
    check_fn=check_crwd_handoff_requirements,
    requires_env=["CHATWOOT_BASE_URL"],
    emoji="🤝",
)
