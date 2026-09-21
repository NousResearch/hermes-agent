"""Bot Mode room post tool — ``room_post``.

A Bot Mode room is written by exactly ONE writer: the Desktop that hosts it. That
is what makes the room's watermark/epoch bookkeeping tractable, and it is why a
member bot cannot append to a room log itself — the log lives in the Desktop's
plugin storage and is mirrored into the hosting profile's ``ui_meta``, neither of
which a member's own backend can reach.

So a member asks, and the Desktop delivers. Calling this tool records the post in
the member's own session — the room mirrors a member's external writes out of
that transcript (#93813), and the call is read back from the tool row's NAME and
ARGUMENTS, which is what the gateway's `session.resume` projection actually
preserves (it drops tool result content). The Desktop that hosts the room appends
the post with the member's exact identity, and — when the post names someone —
drives exactly those members. Nothing about the room's single-writer rule changes.

Session scope: the tool is injected where the room can read it — the member's own
room session (`Group: …`), not the canonical Bot Chat. A post written in Bot Chat
would be authorized and then never delivered.

Containment, like ``message_agent``: the schema is injected ONLY into a managed
Bot Chat agent (see ``tools/bot_mode_probe.py``; never in the registry or any
toolset), and dispatch goes through ``agent/inline_tool_executors.py``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

ROOM_POST_TOOL_NAME = "room_post"

#: A room post is a chat message, not a report: long enough to say something real,
#: short enough that a room cannot be flooded. The Desktop enforces the same bound
#: when it harvests, so a post cut here is never silently longer there.
ROOM_POST_MAX_CHARS = 4000

#: Names a post may pull in. A room is a group of people, not a mailing list.
ROOM_POST_MAX_MENTIONS = 12


def room_post_tool_schema() -> dict:
    """OpenAI-format schema for ``room_post`` (injected, not registered)."""
    return {
        "type": "function",
        "function": {
            "name": ROOM_POST_TOOL_NAME,
            "description": (
                "Post a message into a Bot Mode group room you are a member of, "
                "WITHOUT being asked — the way a teammate drops a line in a group "
                "chat. Use it to announce something the room needs to know (a "
                "finished job, a blocker, a result that arrived late) instead of "
                "waiting for the user to type first. It is ASYNCHRONOUS: the "
                "Desktop that hosts the room delivers the post and it appears "
                "there; this call returns an acknowledgement, not the room's "
                "reply. Keep it short and conversational — a room is a "
                "conversation, not a status page. Mention a member (or 'user') in "
                "`mentions` only when you actually need them to answer: a post "
                "that mentions nobody is a line in the room, a post that mentions "
                "someone starts their turn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "room": {
                        "type": "string",
                        "description": (
                            "The room's name, exactly as it appears in your rooms "
                            "list. You can only post to rooms you are a member of; "
                            "the Desktop refuses any other name."
                        ),
                    },
                    "text": {
                        "type": "string",
                        "description": (
                            "What to say (max " f"{ROOM_POST_MAX_CHARS} chars). "
                            "Written the way you would say it out loud."
                        ),
                    },
                    "mentions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Member handles (or 'user') to pull into the "
                            "conversation. Omit for a plain announcement — nobody "
                            "answers a post that mentions no one."
                        ),
                    },
                },
                "required": ["room", "text"],
            },
        },
    }


def room_post_tool(room: str, text: str, mentions: Any = None, **_ignored: Any) -> str:
    """Validate one room post and return its acknowledgement.

    The post itself travels in this call: the Desktop harvests the call from the
    member's transcript, so there is nothing to write here beyond saying whether
    the request is well-formed enough to deliver.
    """
    name = str(room or "").strip()
    body = str(text or "").strip()

    if not name:
        return json.dumps({"success": False, "error": "room is required"})

    if not body:
        return json.dumps({"success": False, "error": "text is required"})

    if len(body) > ROOM_POST_MAX_CHARS:
        return json.dumps(
            {
                "success": False,
                "error": f"text is {len(body)} chars; the room accepts {ROOM_POST_MAX_CHARS}",
            }
        )

    wanted = mentions if isinstance(mentions, list) else []
    handles = [str(handle).strip() for handle in wanted if str(handle).strip()]

    if len(handles) > ROOM_POST_MAX_MENTIONS:
        return json.dumps(
            {
                "success": False,
                "error": f"at most {ROOM_POST_MAX_MENTIONS} mentions per post",
            }
        )

    return json.dumps(
        {
            "success": True,
            "queued": True,
            "room": name,
            "mentions": handles,
            "note": (
                "Queued for the Desktop that hosts the room; it delivers the post "
                "with your name on it and starts the mentioned members' turns. Do "
                "not wait for a reply — finish your turn."
            ),
        }
    )


#: A room drives a member in a hidden session of its own — `Group: <room>` or
#: `Group: <room> · <thread>` (see `ensureGroupChatSession`). That is the session
#: whose transcript the room mirrors, so it is the ONLY place a post can be
#: written and read back. The canonical Bot Chat is a different session: a post
#: recorded there would be authorized and then never delivered.
ROOM_SESSION_TITLE_PREFIX = "Group: "


def room_post_authorized(agent: Any) -> bool:
    """The ``room_post`` gate: a protocol-enabled agent on a managed Bot-Mode
    install, running in a session the room actually mirrors — a canonical Bot
    Chat OR one of the room's own member sessions.

    Session-stable, so re-evaluating it per tool-snapshot rebuild is prompt-cache
    safe. Never raises."""
    try:
        if not getattr(agent, "_bot_mode_protocol", True):
            return False

        from tools.bot_mode_dm import _agent_home, _session_title
        from tools.bot_mode_probe import BOT_CHAT_TITLE, is_bot_mode_managed

        title = str(_session_title(agent) or "")

        if not (title == BOT_CHAT_TITLE or title.startswith(ROOM_SESSION_TITLE_PREFIX)):
            return False

        return bool(is_bot_mode_managed(_agent_home(agent)))
    except Exception:  # pragma: no cover — must never break a turn
        logger.debug("room_post_authorized failed", exc_info=True)
        return False


def ensure_room_post_tool(agent: Any) -> bool:
    """Inject the ``room_post`` schema into a Bot Chat agent's tool list (once per
    turn, cached on the agent so the list is byte-identical across turns —
    prompt-cache safe). Never raises.

    Mirrors ``ensure_message_agent_tool``'s contract exactly, including the
    half that is easy to miss: success means BOTH halves hold. A tool-surface
    rebuild can keep the schema while ``valid_tool_names`` is republished
    without it, and an advertised-but-nondispatchable tool sends the model
    hunting for shellouts (#96105).
    """
    try:
        if not getattr(agent, "_bot_mode_protocol", True):
            return False

        tools = getattr(agent, "tools", None)
        present = bool(tools) and any(
            isinstance(tool, dict) and tool.get("function", {}).get("name") == ROOM_POST_TOOL_NAME
            for tool in tools
        )

        if not present:
            if not room_post_authorized(agent):
                return False

            if agent.tools is None:
                agent.tools = []

            agent.tools.append(room_post_tool_schema())

        valid = getattr(agent, "valid_tool_names", None)

        if isinstance(valid, set):
            valid.add(ROOM_POST_TOOL_NAME)

        return True
    except Exception:  # pragma: no cover — must never break a turn
        logger.debug("ensure_room_post_tool failed", exc_info=True)
        return False
