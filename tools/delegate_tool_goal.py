"""Goal mode for delegation children (#124292): a bounded judge loop INSIDE a child.

The kanban goal loop (``hermes_cli.goals.run_kanban_goal_loop``) keeps a worker
on a card until a judge accepts the result; a plain delegation child had no such
loop — when it stopped between items, the parent had to notice and re-delegate
(manual continuation). ``delegate_task(goal_mode=True)`` opts a call's children
into the same shape: after each child turn the auxiliary goal judge evaluates
the child's latest response against the task goal; a continue verdict feeds the
continuation prompt back into the child's OWN conversation (append-only, same
session — prompt caching stays intact). Terminal states: the child finished;
the judge accepted; the judge ruled the goal unachievable; or the turn budget
expired — the partial summary returns with a structured remaining-work note so
the parent can re-dispatch or take over.

Per-call opt-in only (no config default); every continuation turn is
parent-visible (api_calls/cost fold into the child's entry). Judge failures
fail OPEN exactly like the kanban loop: the loop stops rather than burn budget
on an unreachable judge.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("tools.delegate_tool")

# Re-exported so callers (and tests) need only this module for the default.
from hermes_cli.goals import DEFAULT_MAX_TURNS as _GOALS_DEFAULT_MAX_TURNS  # noqa: E402  (lazy consumers below)


GOAL_CONTINUATION_TEMPLATE = (
    "[Continuing toward your task — the judge says it is not done yet]\n"
    "Task: {task}\n\n"
    "Judge's assessment: {reason}\n\n"
    "Continue working on this task from where you left off. Take the next "
    "concrete step. When the work is genuinely finished, reply with a final "
    "summary of what was accomplished. If you are blocked and need input from "
    "the parent, say so clearly and stop."
)

_BUDGET_NOTE = (
    "Goal-mode budget exhausted after {turns}/{max_turns} turns; the judge's "
    "last assessment was: {reason}"
)


def _judge_available() -> bool:
    """``judge_goal`` fails open (no auxiliary model -> ``"continue"``), which would
    burn the whole child budget against an unreachable judge; so goal_mode refuses
    to arm unless a judge is actually reachable (mirrors kanban's gate)."""
    try:
        from agent.auxiliary_client import get_text_auxiliary_client
        client, model = get_text_auxiliary_client("goal_judge")
    except Exception:
        return False
    return client is not None and bool(model)


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _apply_goal_mode(
    task_list: List[Dict[str, Any]], goal_mode: Any, goal_max_turns: Any
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Validate the call-level goal_mode flags onto every task dict. ``(task_list, None)``
    or ``(task_list, error)``; flags are call-level in the first cut — every child in the
    call gets the same budget."""
    if not goal_mode:
        return task_list, None
    if not _judge_available():
        return task_list, (
            "goal_mode requires the auxiliary goal judge (auxiliary.goal_judge) and none is "
            "configured; without it the loop cannot tell done from not-done."
        )
    if goal_max_turns is not None:
        try:
            turns = int(goal_max_turns)
        except (TypeError, ValueError):
            return task_list, f"goal_max_turns must be a positive integer, got {goal_max_turns!r}."
        if turns < 1:
            return task_list, f"goal_max_turns must be a positive integer, got {goal_max_turns!r}."
    for task in task_list:
        task["goal_mode"] = True
        if goal_max_turns is not None:
            task["goal_max_turns"] = int(goal_max_turns)
    return task_list, None


def _child_goal_cfg(task: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """Per-child goal-mode config from its task dict; None keeps the child on the plain path.
    Also gates model-supplied per-task flags (which bypass ``_apply_goal_mode``) on the judge
    actually being reachable — without it the loop would stop after turn one."""
    if not task.get("goal_mode"):
        return None
    if not _judge_available():
        logger.warning("goal_mode task ignored: no auxiliary goal judge configured")
        return None
    try:
        max_turns = int(task.get("goal_max_turns") or 0)
    except (TypeError, ValueError):
        max_turns = 0
    return {"max_turns": max_turns if max_turns >= 1 else _GOALS_DEFAULT_MAX_TURNS}


def _judge_child_turn(goal_text: str, response: str, session_id: str) -> Tuple[str, str]:
    """One between-turns judge call for a child. The judge runs OUTSIDE any agent turn:
    bind the child's relay-affinity scope (same shape as the kanban loop / handoff gates)
    so the relay does not reject the call (#113669). Fail-open -> (\"continue\", reason)."""
    from agent.portal_tags import get_affinity_scope, reset_affinity_scope, set_affinity_scope

    affinity_token = None if get_affinity_scope() else set_affinity_scope(f"delegation-child:{session_id}")
    try:
        from hermes_cli.goals import judge_goal
        verdict, reason, _parse_failed, _wait, _transport_failed = judge_goal(goal_text, response)
    except Exception as exc:
        logger.warning("goal-mode judge call failed (%s); continuing once", exc)
        return "continue", f"judge error: {type(exc).__name__}"
    finally:
        if affinity_token is not None:
            reset_affinity_scope(affinity_token)
    return verdict, reason


def _merge_turn_results(result: Dict[str, Any], continuation: Dict[str, Any]) -> Dict[str, Any]:
    """Fold one continuation turn into the running result (newest text wins, counters sum,
    histories concatenate — same shape the schema-retry fold uses)."""
    merged = dict(result)
    text = continuation.get("final_response") or ""
    if text.strip():
        merged["final_response"] = text
    try:
        merged["api_calls"] = int(result.get("api_calls", 0) or 0) + int(continuation.get("api_calls", 0) or 0)
    except (TypeError, ValueError):
        pass
    cont_messages = continuation.get("messages")
    if isinstance(cont_messages, list) and isinstance(result.get("messages"), list):
        merged["messages"] = result["messages"] + cont_messages
    # Terminal-state flags come from the newest turn: a continuation that got
    # interrupted/failed describes the run's actual end.
    for key in ("completed", "failed", "interrupted"):
        if key in continuation:
            merged[key] = continuation[key]
    if continuation.get("error"):
        merged["error"] = continuation["error"]
    return merged


def run_goal_continuations(
    *,
    child: Any,
    goal_text: str,
    first_result: Dict[str, Any],
    child_task_id: str,
    relay_text: Any,
    max_turns: int,
    session_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Drive the bounded judge loop over the child's own conversation.

    ``first_result`` is the child's first-turn ``run_conversation`` result. Each iteration:
    a failed/interrupted turn is terminal (never loop on a broken run); a completed run
    ends the loop; otherwise judge the latest response — continue feeds the continuation
    prompt back into ``child.run_conversation``, done sets ``completed`` on the merged
    result (the judge accepted the work, so the entry must not read ``max_iterations``),
    blocked stops with the reason. Budget expiry leaves the child's own terminal state
    and reports ``remaining_work`` for the parent.

    Returns ``(result, loop_info)``; ``loop_info`` rides ``result["_goal_loop"]`` into the
    parent-visible entry.
    """
    max_turns = int(max_turns or 0)
    if max_turns < 1:
        max_turns = _GOALS_DEFAULT_MAX_TURNS

    result = first_result
    turns_used = 1  # the first turn already consumed one unit of budget
    loop_info: Dict[str, Any] = {"enabled": True, "turns_used": turns_used, "max_turns": max_turns, "outcome": "completed"}

    while True:
        if result.get("failed") or result.get("error") or result.get("interrupted", False):
            loop_info["outcome"] = "stopped"
            return result, loop_info
        if result.get("completed", False):
            return result, loop_info
        if turns_used >= max_turns:
            loop_info["outcome"] = "budget_exhausted"
            return result, loop_info

        verdict, reason = _judge_child_turn(goal_text, result.get("final_response") or "", session_id)
        loop_info["last_reason"] = _truncate(reason, 500)
        if verdict == "wait":
            # A child cannot gain anything by parking between turns; waiting is continuing.
            verdict = "continue"
        logger.info(
            "Subagent %s goal-mode turn %d/%d verdict=%s reason=%s",
            child_task_id, turns_used, max_turns, verdict, _truncate(reason, 120),
        )

        if verdict == "blocked":
            loop_info["outcome"] = "blocked"
            loop_info["remaining_work"] = _truncate(reason, 500)
            return result, loop_info
        if verdict == "skipped":
            # Judge unreachable/empty goal: fail open by STOPPING (not looping) so an
            # unreachable judge cannot silently burn the remaining budget.
            loop_info["outcome"] = "judge_unavailable"
            return result, loop_info
        if verdict == "done":
            # The judge accepted the work: mark the run complete so the entry's
            # exit_reason contract stays truthful (max_iterations is only real exhaustion).
            result = dict(result)
            result["completed"] = True
            loop_info["outcome"] = "judge_done"
            return result, loop_info

        prompt = GOAL_CONTINUATION_TEMPLATE.format(task=_truncate(goal_text, 2000), reason=_truncate(reason, 400))
        continuation = None
        try:
            # Same identity as the main child turn: this runs on the parent worker's thread,
            # and an unmarked turn is misread as the dispatcher-owned worker by HERMES_KANBAN_* gates.
            from agent.delegation_context import delegated_child_context
            with delegated_child_context(str(getattr(child, "session_id", "") or "")):
                continuation = child.run_conversation(
                    user_message=prompt, task_id=child_task_id, stream_callback=relay_text,
                )
        except Exception as exc:
            logger.warning("Subagent %s goal-mode continuation turn failed: %s", child_task_id, exc)
        if not isinstance(continuation, dict) or not continuation:
            loop_info["outcome"] = "stopped"
            return result, loop_info
        result = _merge_turn_results(result, continuation)
        turns_used += 1
        loop_info["turns_used"] = turns_used


def _remaining_work_note(loop_info: Dict[str, Any]) -> Optional[str]:
    """Structured remaining-work note for a budget-exhausted child: the parent reads this
    to re-dispatch or take over."""
    if loop_info.get("outcome") != "budget_exhausted":
        return None
    reason = loop_info.get("last_reason")
    if reason:
        return _BUDGET_NOTE.format(turns=loop_info.get("turns_used"), max_turns=loop_info.get("max_turns"), reason=reason)
    return (
        f"Goal-mode budget exhausted after {loop_info.get('turns_used')}/{loop_info.get('max_turns')} turns "
        "without the judge accepting the result; inspect the child's summary for remaining work."
    )
