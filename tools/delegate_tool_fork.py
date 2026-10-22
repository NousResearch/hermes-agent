"""Fork-mode context for delegate_task children (``context_mode: "fork"``).

A forked child starts from the parent's conversation as it stood when the parent called delegate_task
(snapshot at launch, as in deepagents' fork mode): later parent messages never reach it.
"""

from __future__ import annotations

from typing import Any

_HANDED_OFF = (
    "Forked: this delegation was handed to the subagent now reading this conversation (you). "
    "Your task is in the next message."
)
_NOT_AVAILABLE = "Not available: this call was still running in the parent when it forked you."

_FORK_NOTICE = (
    "[Forked subagent] The messages above are your parent agent's conversation, copied when it delegated to you. "
    "Every request in them was addressed to the parent and the parent is handling it, including any steps planned "
    "after this delegation: do NOT carry them out. You are the subagent that was just invoked; do not delegate "
    "further. Use the facts already established above. Do ONLY the task below, then give its result as your final "
    "answer (only the parent sees it). Your task:"
)


def _close_in_flight_round(messages: list) -> None:
    """Answer the parent's in-flight tool calls inside the copy.

    The parent is mid tool round: its last assistant message carries the delegate_task call being executed,
    which can never be answered inside the child. Left unanswered, sequence repair prunes it, the parent's
    pending user request merges into the child's goal turn, and the child redoes the parent's whole job.
    Recording the hand-off keeps the request visibly handled by the parent.
    """
    from agent.message_sanitization import coalesce_tool_call_id, tool_call_id_variants, tool_result_id_variants

    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        end = idx + 1
        while end < len(messages) and messages[end].get("role") == "tool":
            end += 1
        answered = set().union(*(tool_result_id_variants(m.get("tool_call_id")) for m in messages[idx + 1:end]))
        messages[end:end] = [
            {
                "role": "tool",
                "tool_call_id": coalesce_tool_call_id(tc),
                "content": _HANDED_OFF if (tc.get("function") or {}).get("name") == "delegate_task" else _NOT_AVAILABLE,
            }
            for tc in msg["tool_calls"]
            if isinstance(tc, dict) and coalesce_tool_call_id(tc) and not tool_call_id_variants(tc) & answered
        ]
        return


def build_fork_history(parent_agent: Any) -> list:
    """Structural copy of the parent's live transcript with its in-flight tool round closed."""
    from agent.turn_finalizer import _clone_background_review_messages

    history = _clone_background_review_messages(list(getattr(parent_agent, "_session_messages", None) or []))
    _close_in_flight_round(history)
    return history


def frame_forked_goal(user_message: Any) -> Any:
    """Prefix a forked child's kickoff message (plain text or multimodal parts) with the fork notice."""
    if isinstance(user_message, list):
        return [{"type": "text", "text": _FORK_NOTICE}, *user_message]
    return f"{_FORK_NOTICE}\n\n{user_message}"
