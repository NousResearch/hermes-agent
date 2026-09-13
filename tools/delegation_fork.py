"""Fork-context snapshots for delegate_task subagents.

Port of MoonshotAI/kimi-code#3007 ("add a fork parameter to the Agent
tool", MIT) adapted to Hermes' delegation architecture.

With ``fork: true`` on ``delegate_task`` (or on an individual batch task),
the child starts from a one-time snapshot of the calling agent's
conversation history instead of a blank context. The snapshot is:

* read from the PARENT's session in the session DB (``hermes_state``),
  which incremental turn persistence keeps current up to the tool calls
  of the in-flight turn — the same durable transcript the gateway uses
  for session restore;
* sanitized so the child never replays live scaffolding: the parent's
  trailing assistant message carrying the in-flight ``delegate_task``
  tool_call can never close inside the child, so the tail is trimmed to
  the last coherent boundary and alternation is repaired with the same
  ``repair_message_sequence`` used for gateway session replay;
* framed with an inheritance notice so the child treats the seeded
  messages as reference material from its parent, not as its own past
  actions (mirrors kimi-code's inheritance-notice framing).

The snapshot is seeded through ``run_conversation(conversation_history=...)``
— the exact parameter the gateway uses to restore sessions — so no new
message-injection path exists and role-alternation invariants hold.

Cost note: a forked child's first request re-sends the parent transcript,
so fork is opt-in per call and bounded by ``delegation.fork_max_messages``
(default 200 most-recent messages).
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default cap on how many trailing parent messages a fork snapshot carries.
DEFAULT_FORK_MAX_MESSAGES = 200

# Message keys that are parent-session bookkeeping, not model context.
_STRIP_KEYS = {"_row_id", "display_kind", "display_metadata", "observed", "message_id"}

INHERITANCE_NOTICE = (
    "You were forked from your parent agent's conversation. The messages "
    "above are an inherited snapshot of the parent's session — treat them "
    "as reference material describing what has already been established, "
    "NOT as actions you performed yourself. Tool results in the snapshot "
    "were produced by the parent. Your task follows."
)


def _fork_cap(cfg: Optional[Dict[str, Any]]) -> int:
    """Resolve delegation.fork_max_messages with a safe floor."""
    try:
        raw = (cfg or {}).get("fork_max_messages", DEFAULT_FORK_MAX_MESSAGES)
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_FORK_MAX_MESSAGES
    return max(10, value)


def _trim_incomplete_tail(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop the parent's in-flight scaffolding from the snapshot tail.

    The fork happens mid-turn: the parent's transcript ends with the
    assistant message whose ``tool_calls`` include the very
    ``delegate_task`` call being executed (and possibly tool results for
    sibling calls in the same batch). Those calls can never be closed
    inside the child, and an assistant tool_calls message without a
    complete set of tool responses is a wire-invalid tail. Trim
    backwards past any trailing tool results and the dangling
    assistant-with-tool_calls message to the last coherent boundary.
    """
    msgs = list(messages)
    while msgs:
        tail = msgs[-1]
        role = tail.get("role")
        if role == "assistant" and not tail.get("tool_calls"):
            # Coherent boundary: a completed assistant reply. The child's
            # kickoff goal is appended as a user message by
            # run_conversation, so the snapshot must end here — a trailing
            # user message (the parent-turn prompt that triggered this
            # delegation) or a dangling tool_calls/tool tail would break
            # role alternation or tool-call adjacency in the child.
            break
        if role == "assistant" and tail.get("tool_calls"):
            # Dangling call (the in-flight delegate_task itself, or a call
            # whose responses were trimmed below it). Drop and continue —
            # this can expose another incomplete boundary above.
            msgs.pop()
            continue
        # tool results without their call kept, user prompts, anything else.
        msgs.pop()
    return msgs


def build_fork_snapshot(
    parent_agent: Any,
    *,
    cfg: Optional[Dict[str, Any]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Build a sanitized copy of the parent's conversation for a forked child.

    Returns ``None`` (fork degrades to a normal blank-context spawn, with a
    log line) when the parent has no session DB, no session id, or an empty
    transcript — never raises into the delegation path.
    """
    session_db = getattr(parent_agent, "_session_db", None)
    session_id = getattr(parent_agent, "session_id", None)
    if session_db is None or not session_id:
        logger.info(
            "fork requested but parent has no session DB/session id; "
            "spawning with blank context"
        )
        return None

    try:
        history = session_db.get_messages_as_conversation(
            str(session_id),
            include_ancestors=True,
            repair_alternation=True,
        )
    except Exception as exc:
        logger.warning(
            "fork snapshot read failed for session %s: %s; spawning with "
            "blank context",
            session_id,
            exc,
        )
        return None

    if not history:
        logger.info(
            "fork requested but parent session %s has no persisted messages; "
            "spawning with blank context",
            session_id,
        )
        return None

    snapshot: List[Dict[str, Any]] = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            # The child builds its own system prompt; the parent's must not
            # be replayed as history.
            continue
        clean = {k: v for k, v in msg.items() if k not in _STRIP_KEYS}
        snapshot.append(copy.deepcopy(clean))

    snapshot = _trim_incomplete_tail(snapshot)
    if not snapshot:
        return None

    cap = _fork_cap(cfg)
    if len(snapshot) > cap:
        snapshot = snapshot[-cap:]
        # Never open on a dangling tool/assistant-tool_calls boundary after
        # the cut: drop leading tool results without a preceding call.
        while snapshot and snapshot[0].get("role") == "tool":
            snapshot.pop(0)
        snapshot = _trim_incomplete_tail(snapshot)
        if not snapshot:
            return None

    return snapshot


def frame_forked_goal(goal: str) -> str:
    """Prefix the child's kickoff goal with the inheritance notice."""
    return f"{INHERITANCE_NOTICE}\n\nYOUR TASK:\n{goal}"
