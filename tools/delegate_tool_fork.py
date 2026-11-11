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


_PARITY_ATTRS = ("model", "provider", "base_url", "api_mode", "reasoning_config")


def adopt_parent_prefix(child: Any, parent_agent: Any, child_prompt: str) -> bool:
    """Same-model fork: send the parent's exact system prompt and ``tools[]`` so the fork's request
    shares the parent's cached prefix up to the hand-off (deepagents#7628). The subagent instructions
    move into the task message (``_delegate_fork_instructions``); dispatch stays limited to the child's
    own tool set, so advertising a blocked tool (delegate_task, memory, ...) never lets it run.
    A routed child (other model, provider, endpoint or reasoning setting) keeps its own prompt."""
    import copy

    from agent.background_review import _FROZEN_TOOL_SNAPSHOT_GENERATION
    from agent.prompt_cache_scope import resolve_prompt_cache_scope_safe

    parent_prompt, parent_tools = getattr(parent_agent, "_cached_system_prompt", None), getattr(parent_agent, "tools", None)
    if not (isinstance(parent_prompt, str) and parent_prompt and isinstance(parent_tools, list)):
        return False
    if any(getattr(child, a, None) != getattr(parent_agent, a, None) for a in _PARITY_ATTRS):
        return False
    own_names = set(getattr(child, "valid_tool_names", None) or ())
    child.tools = copy.deepcopy(parent_tools)
    child.valid_tool_names = own_names & {t["function"]["name"] for t in child.tools if isinstance(t, dict) and t.get("function")}
    # A between-turns MCP refresh or a compaction-boundary rebuild would republish the child's own tools[].
    child._tool_snapshot_generation, child._skip_mcp_refresh = _FROZEN_TOOL_SNAPSHOT_GENERATION, True
    child._cached_system_prompt = parent_prompt
    child.ephemeral_system_prompt = getattr(parent_agent, "ephemeral_system_prompt", None)
    child._delegate_fork_instructions = child_prompt
    # Scope-keyed caches (OpenAI prompt_cache_key, Portal affinity) route by the parent's scope, not the child id.
    if scope := resolve_prompt_cache_scope_safe(parent_agent):
        child._inherited_cache_scope = scope
    return True


def frame_forked_goal(user_message: Any, instructions: Any = None) -> Any:
    """Prefix a forked child's kickoff message (plain text or multimodal parts) with the fork notice and,
    for a prefix-sharing fork, the subagent instructions its system prompt no longer carries."""
    head = f"{instructions}\n\n{_FORK_NOTICE}" if isinstance(instructions, str) and instructions else _FORK_NOTICE
    if isinstance(user_message, list):
        return [{"type": "text", "text": head}, *user_message]
    return f"{head}\n\n{user_message}"
