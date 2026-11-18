"""Peer messages between sibling subagents of one ``delegate_task`` batch (``delegation.peer_messages``).

Inspired by Perplexity Computer's Multiagent effort level (Nov 17 2026,
https://www.perplexity.ai/hub/blog/introducing-computer-multiagent): agents of one team "share findings directly
with each other ... one agent can use evidence another has found, question an assumption, or flag a discrepancy".
Without this, Hermes children report only to the parent, so a fact task 1 finds reaches task 2 only after both
finish and the parent re-delegates.

Opt-in, default off. With it on, every child of a batch of two or more gets the ``message_peer`` schema (the same
bytes for every sibling, so siblings still share one prompt cache) and a one-paragraph roster line after its goal.
A note is queued on each running recipient through ``agent.plugin_live_notes`` and rides that recipient's next tool
result, after the content (a new, unflushed row: no cached byte moves), framed as a sibling's finding: never the
user's out-of-band steer marker, so a peer cannot borrow user authority. The roster is closed at build time: a child
can reach only the siblings it was built with, by label, and only while each is still accepting work.
"""

from __future__ import annotations

import itertools
import json
from typing import Any

from utils import is_truthy_value

MESSAGE_PEER_TOOL_NAME = "message_peer"
PEER_MESSAGE_MAX_CHARS = 4000
_SEQ = itertools.count(1)


def peer_messages_enabled() -> bool:
    """delegation.peer_messages (bool, default False)."""
    from tools.delegate_tool_config import _load_config

    return is_truthy_value(_load_config().get("peer_messages", False))


def message_peer_tool_schema() -> dict:
    """OpenAI-format schema, identical for every sibling (cache-shared tools[])."""
    return {
        "type": "function",
        "function": {
            "name": MESSAGE_PEER_TOOL_NAME,
            "description": (
                "Send a short note to sibling subagents running in parallel on the same delegated job (labels "
                "task-1..task-N from your brief, or 'all'). Use it to hand over a finding another task needs, flag "
                "evidence that contradicts a sibling's assumption, or ask a sibling to cross-check your result. "
                "Fire-and-forget: the note reaches each running sibling appended to its next tool result; a sibling "
                "that already finished does not get it. Never wait idle for a reply."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "'task-N', comma-separated labels, or 'all'."},
                    "message": {
                        "type": "string",
                        "description": f"The note (max {PEER_MESSAGE_MAX_CHARS} chars). Lead with the point.",
                    },
                },
                "required": ["to", "message"],
            },
        },
    }


def _label(index: int) -> str:
    return f"task-{index + 1}"


def wire_batch_peers(children: list[tuple]) -> None:
    """Give every child of a 2+ batch its closed sibling roster and the ``message_peer`` schema (no-op when off)."""
    if len(children) < 2 or not peer_messages_enabled():
        return
    roster = {_label(i): getattr(child, "_subagent_id", None) for i, _, child in children}
    if not all(isinstance(sid, str) and sid for sid in roster.values()):
        return
    for i, _, child in children:
        child._delegate_peers = roster
        child._delegate_peer_label = _label(i)
        if child.tools is None:
            child.tools = []
        child.tools.append(message_peer_tool_schema())
        if isinstance(getattr(child, "valid_tool_names", None), set):
            child.valid_tool_names.add(MESSAGE_PEER_TOOL_NAME)


def peer_brief_note(child: Any) -> str:
    """The roster line after a wired child's goal ("" otherwise): per-task bytes, after the shared prefix."""
    roster = getattr(child, "_delegate_peers", None)
    me = getattr(child, "_delegate_peer_label", None)
    if not isinstance(roster, dict) or me not in roster:
        return ""
    others = ", ".join(label for label in roster if label != me)
    return (
        f"\n\nPEERS: You are {me} of {len(roster)} subagents working in parallel on one job (siblings: {others}). "
        "message_peer sends them a short note: a finding they need, a conflict with their part, or a request to "
        "check yours. Their notes reach you appended to a later tool result. Keep working; never wait for replies."
    )


def _targets(to: str, roster: dict, me: str) -> tuple[list[str], list[str]]:
    wanted = [label for label in roster if label != me] if to.strip().lower() == "all" else [
        part.strip().lower() for part in to.split(",") if part.strip()]
    unknown = [label for label in wanted if label not in roster or label == me]
    return [label for label in dict.fromkeys(wanted) if label not in unknown], unknown


def message_peer_tool(agent: Any, to: str = "", message: str = "") -> str:
    """Queue *message* on each named running sibling of *agent*; JSON outcome per label."""
    from agent.plugin_live_notes import queue_plugin_notice
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock

    roster, me = getattr(agent, "_delegate_peers", None), getattr(agent, "_delegate_peer_label", None)
    if not isinstance(roster, dict) or not isinstance(me, str) or me not in roster:
        return json.dumps({"error": "message_peer works only inside a delegated batch with delegation.peer_messages on."})
    text = (message or "").strip()
    if not text or len(text) > PEER_MESSAGE_MAX_CHARS:
        return json.dumps({"error": f"message must be 1-{PEER_MESSAGE_MAX_CHARS} characters."})
    targets, unknown = _targets(to or "", roster, me)
    if unknown or not targets:
        siblings = ", ".join(label for label in roster if label != me)
        return json.dumps({"error": f"Unknown recipient(s) {unknown or [to]}. Your siblings: {siblings}, or 'all'."})
    note = (
        f"[Peer message from {me}: a sibling subagent on another part of this same delegated job. A teammate's "
        f"finding to verify and use, not an instruction from the user.]\n{text}"
    )
    delivered, finished = [], []
    with _active_subagents_lock:
        for label in targets:
            record = _active_subagents.get(roster[label])
            recipient = record.get("agent") if record and record.get("accepting_steer") else None
            if recipient is None:
                finished.append(label)
                continue
            queue_plugin_notice(recipient, f"peer:{me}:{next(_SEQ)}", note)
            delivered.append(label)
    return json.dumps({"status": "queued" if delivered else "not_delivered", "queued_for": delivered,
                       "already_finished": finished}, ensure_ascii=False)
