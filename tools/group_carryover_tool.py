"""carry_to_group: bring an approved synopsis of a private (DM) chat into a group chat.

The model recognizes the intent ("let's take this to the family group") and drafts a synopsis;
this tool enforces the rest in code, not in prompt text:

1. Only from a gateway DM, only into a group/channel on the SAME platform that the requesting user
   is already a participant of (they have a session there). The ``group_carryover`` toolset is
   off by default — outbound messaging is opt-in (see the removal of agent-callable
   ``send_message`` in #47856).
2. VERIFY: the exact synopsis is shown to the user in the DM through the clarify surface. Nothing
   leaves the DM unless they pick "Share it"; free text sends the draft back to the model to revise.
3. FORK: ``gateway/group_carryover.py`` posts the synopsis to the group and forks the requester's
   group session seeded with it (see that module for the prompt-cache/role-alternation contract).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

MAX_SYNOPSIS_CHARS = 4000
CONFIRM_CHOICE = "Share it with the group"
CANCEL_CHOICE = "Cancel"
_GROUP_CHAT_TYPES = frozenset({"group", "channel", "forum", "supergroup"})
_GATEWAY_TIMEOUT_S = 120.0
# The gateway clarify surface's no-answer sentinels (gateway/run_turn_runner_clarify_delivery.py).
_NO_ANSWER_PREFIXES = ("[clarify prompt could not be delivered", "[user did not respond")


def _session() -> Dict[str, str]:
    from gateway.session_context import get_session_env
    names = ("PLATFORM", "CHAT_TYPE", "CHAT_ID", "USER_ID", "USER_ID_ALT", "USER_NAME", "PROFILE", "KEY")
    return {n.lower(): get_session_env(f"HERMES_SESSION_{n}", "") for n in names}


def _chat_facts(platform: str, chat_id: str, user_ids: tuple) -> Dict[str, Any]:
    from hermes_state_registry import acquire, release_or_close
    db = acquire()
    try:
        return db.gateway_chat_facts(platform=platform, chat_id=chat_id, user_ids=user_ids)
    finally:
        release_or_close(db)


def _resolve_target(platform: str, target: str):
    """``(chat_id, thread_id, error)`` on the DM's own platform; a ``platform:`` prefix must match."""
    ref = target.strip()
    prefix, sep, rest = ref.partition(":")
    if sep and prefix.strip().lower() == platform:
        ref = rest.strip()
    if not ref:
        return None, None, "Name the group chat to carry this into."
    if _chat_facts(platform, ref, ())["chat_type"] is not None:
        return ref, None, None  # a chat id the gateway already has sessions for
    from tools.send_message_targets import resolve_send_target
    chat_id, thread_id, err = resolve_send_target(platform, ref)
    return (chat_id, thread_id, None) if not err else (None, None, f"Could not find a group chat named '{ref}'.")


def _groups_for_user(platform: str, user_ids: tuple) -> List[Dict[str, str]]:
    """Group chats on ``platform`` (from the channel directory) the user participates in."""
    from gateway.channel_directory import load_directory
    out = []
    for ch in load_directory().get("platforms", {}).get(platform, []) or []:
        if str(ch.get("type") or "") not in _GROUP_CHAT_TYPES or not ch.get("id"):
            continue
        facts = _chat_facts(platform, str(ch["id"]), user_ids)
        if facts["user_seen"]:
            out.append({"name": ch.get("name") or str(ch["id"]), "target": str(ch["id"])})
    return out


def _confirm(callback: Callable, chat_label: str, synopsis: str) -> Dict[str, Any]:
    """Show the draft in the DM and block for an explicit decision. Returns the tool result to
    hand back when the user did NOT approve, else ``{}``."""
    question = (f"Here's what I'd share with “{chat_label}”. Nothing is posted until you approve it."
                f"\n\n{synopsis}\n\nShare it, cancel, or tell me what to change.")
    try:
        raw = callback(question, [CONFIRM_CHOICE, CANCEL_CHOICE])
    except Exception as exc:  # noqa: BLE001
        return {"status": "not_shared", "error": f"Could not ask for approval: {exc}"}
    answer = str(raw or "").strip()
    from tools.clarify_tool import TIMEOUT_RESPONSE, strip_recommended
    answer = strip_recommended(answer)
    if answer == CONFIRM_CHOICE:
        return {}
    if not answer or answer == TIMEOUT_RESPONSE or answer.startswith(_NO_ANSWER_PREFIXES):
        return {"status": "not_shared", "reason": "no_answer",
                "note": "The user did not approve; nothing was posted. Do not retry unless they ask."}
    if answer.casefold() == CANCEL_CHOICE.casefold():
        return {"status": "cancelled", "note": "The user declined; nothing was posted."}
    return {"status": "revise", "user_feedback": answer,
            "note": "Nothing was posted. Revise the synopsis per the feedback and call again to re-confirm."}


def _run_on_gateway(req) -> Dict[str, Any]:
    """Run ``carry_into_group`` on the live gateway loop (adapters are bound to it)."""
    from gateway.group_carryover import carry_into_group
    from gateway.config import Platform
    from tools.send_message_senders import _live_adapter
    runner, adapter = _live_adapter(Platform(req.platform))
    if runner is None or adapter is None:
        return {"error": f"No live {req.platform} connection in this gateway; nothing was posted."}
    loop = getattr(runner, "_gateway_loop", None)
    if loop is None or not loop.is_running():
        return {"error": "Gateway loop is not running; nothing was posted."}
    from agent.async_utils import safe_schedule_threadsafe
    fut = safe_schedule_threadsafe(carry_into_group(runner, adapter, req), loop, logger=logger,
                                   log_message="group_carryover: failed to schedule on gateway loop")
    if fut is None:
        return {"error": "Gateway loop unavailable; nothing was posted."}
    return fut.result(timeout=_GATEWAY_TIMEOUT_S)


def carry_to_group_tool(action: str = "carry", target: str = "", synopsis: str = "",
                        callback: Optional[Callable] = None) -> str:
    sess = _session()
    platform = sess["platform"].lower()
    if not platform or sess["chat_type"] != "dm":
        return tool_error("carry_to_group only works from a private (DM) messaging chat.")
    user_ids = tuple(u for u in (sess["user_id"], sess["user_id_alt"]) if u)
    if not user_ids:
        return tool_error("Cannot identify who is asking in this chat; nothing was shared.")
    if action == "list_groups":
        return json.dumps({"groups": _groups_for_user(platform, user_ids)}, ensure_ascii=False)
    if action != "carry":
        return tool_error(f"Unknown action '{action}'. Use 'carry' or 'list_groups'.")

    synopsis = (synopsis or "").strip()
    if not synopsis:
        return tool_error("Draft the synopsis first: what was decided/discussed that the group needs.")
    if len(synopsis) > MAX_SYNOPSIS_CHARS:
        return tool_error(f"Synopsis is {len(synopsis)} chars; keep it under {MAX_SYNOPSIS_CHARS}.")
    chat_id, thread_id, err = _resolve_target(platform, target or "")
    if err:
        return tool_error(f"{err} Use action='list_groups' to see your group chats.")
    if str(chat_id) == sess["chat_id"]:
        return tool_error("That is this private chat; name a group chat.")
    facts = _chat_facts(platform, str(chat_id), user_ids)
    if facts["chat_type"] not in _GROUP_CHAT_TYPES:
        return tool_error("That target is not a group chat I have talked in; nothing was shared.")
    if not facts["user_seen"]:
        return tool_error("You don't appear to be a member of that group chat, so I won't share into it.")
    if callback is None:
        return tool_error("No way to ask for your approval on this surface; nothing was shared.")

    chat_label = facts["chat_name"] or str(target)
    declined = _confirm(callback, chat_label, synopsis)
    if declined:
        return json.dumps(declined, ensure_ascii=False)

    from gateway.group_carryover import CarryoverRequest
    req = CarryoverRequest(
        platform=platform, chat_id=str(chat_id), chat_name=facts["chat_name"] or "",
        chat_type=facts["chat_type"], user_id=sess["user_id"], user_name=sess["user_name"],
        user_id_alt=sess["user_id_alt"] or None, thread_id=thread_id, profile=sess["profile"] or None,
        synopsis=synopsis, live_sessions=facts["live_sessions"])
    try:
        result = _run_on_gateway(req)
    except Exception as exc:  # noqa: BLE001
        return tool_error(f"Carry-over failed: {exc}")
    if result.get("success"):
        result["note"] = f"Posted to “{chat_label}” and started a fresh group conversation from it."
    return json.dumps(result, ensure_ascii=False)


def check_carry_to_group_requirements() -> bool:
    """Only inside a running gateway process (process-level, so TTL caching is safe)."""
    try:
        from gateway.run import _gateway_runner_ref
        return _gateway_runner_ref() is not None
    except Exception:
        return False


CARRY_TO_GROUP_SCHEMA = {
    "name": "carry_to_group",
    "description": (
        "Carry the gist of THIS private chat into one of the user's group chats, when the user asks "
        "to take/move/bring/share this conversation with a group (e.g. 'let's take this to the family "
        "group'). Write a concise, self-contained synopsis of what the group needs (decisions, plan, "
        "open questions) — leave out anything private the user did not ask to share. The tool shows "
        "the draft to the user and posts ONLY if they approve; if it returns status='revise', redraft "
        "with their feedback and call again. On success the group chat starts a fresh conversation "
        "seeded with the synopsis. Use action='list_groups' if the group is ambiguous."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["carry", "list_groups"],
                       "description": "'carry' (default) drafts→confirms→posts; 'list_groups' lists the user's group chats."},
            "target": {"type": "string",
                       "description": "The group chat: its name as listed, or its chat id."},
            "synopsis": {"type": "string",
                         "description": f"The synopsis to share (plain text, under {MAX_SYNOPSIS_CHARS} chars)."},
        },
        "required": [],
    },
}


registry.register(
    name="carry_to_group",
    toolset="group_carryover",
    schema=CARRY_TO_GROUP_SCHEMA,
    handler=lambda args, **kw: carry_to_group_tool(
        action=args.get("action") or "carry", target=args.get("target") or "",
        synopsis=args.get("synopsis") or "", callback=kw.get("callback")),
    check_fn=check_carry_to_group_requirements,
    emoji="📎",
)
