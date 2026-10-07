"""goal — the agent's own handle on its session goal (Muse-style).

The model declares, inspects, and closes the per-session goal itself. Backed by the same ``GoalManager`` / ``state_meta goal:<sid>``
row as ``/goal``, so the post-turn judge, gates, heartbeat and ``/goal status`` all see it.
Service-gated on ``goals.agent_tool`` (off by default): a profile that wants goal-owning agents
opts in; everyone else keeps the schema untouched.

Rules the handler enforces (never the prompt alone):
- ``create`` never overwrites a user-set or active goal — the user's ``/goal`` always wins.
- ``complete`` runs the completion audit (``judge_goal`` against the agent's stated evidence) and
  refuses when the verdict is not DONE, so an agent cannot wave its own goal through.
- The goal is tagged ``source="agent"``: ``self-set`` in the status line, parked (not nagged) on BLOCKED.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

_MIN_OBJECTIVE_CHARS = 12
_MAX_OBJECTIVE_CHARS = 600


def check_goal_tool_requirements() -> bool:
    """Opt-in per profile via ``goals.agent_tool: true``."""
    try:
        from hermes_cli.goals import agent_goal_tool_enabled
        return bool(agent_goal_tool_enabled())
    except Exception:
        return False


GOAL_SCHEMA = {
    "name": "goal",
    "description": (
        "Your session goal: one durable objective this conversation is working toward, judged "
        "after each of your replies and driven forward on the heartbeat until it is done. "
        "Call `get` first. Call `create` when you commit to a multi-step outcome the user wants "
        "and no goal is set — state the outcome, not the next step. Call `complete` only when "
        "you can cite evidence the outcome exists; the completion audit refuses otherwise. "
        "Never create a goal for a plain question, a one-shot answer, or work the user cancelled; "
        "a user-set /goal always wins and cannot be replaced here."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["get", "create", "complete"]},
            "objective": {
                "type": "string",
                "description": "For create: the outcome in one sentence, verifiable, 12–600 chars.",
            },
            "evidence": {
                "type": "string",
                "description": "For complete: what proves the outcome exists (commands run, URLs, ids).",
            },
            "max_turns": {
                "type": "integer",
                "description": "For create: optional continuation budget; default from goals.max_turns.",
            },
        },
        "required": ["action"],
    },
}


def _ok(**payload: Any) -> str:
    return json.dumps(payload)


def _manager(session_id: Optional[str]):
    from hermes_cli.goals import GoalManager
    sid = (session_id or "").strip()
    if not sid:
        return None
    return GoalManager(session_id=sid)


def _state_view(mgr) -> Dict[str, Any]:
    s = mgr.state
    if s is None or not mgr.has_goal():
        return {"goal": None}
    view: Dict[str, Any] = {
        "goal": s.goal, "status": s.status, "source": s.source,
        "turns_used": s.turns_used, "max_turns": s.max_turns,
        "last_verdict": s.last_verdict, "last_reason": s.last_reason,
    }
    if s.has_contract():
        c = s.contract
        view["contract"] = {k: getattr(c, k) for k in ("outcome", "verification", "stop_when") if getattr(c, k, None)}
    return view


def goal_tool(action: str, *, objective: str = "", evidence: str = "", max_turns: Optional[int] = None,
              session_id: Optional[str] = None) -> str:
    mgr = _manager(session_id)
    if mgr is None:
        return tool_error("goal tool needs a session; none is bound to this call")

    if action == "get":
        return _ok(**_state_view(mgr))

    if action == "create":
        objective = (objective or "").strip()
        if not (_MIN_OBJECTIVE_CHARS <= len(objective) <= _MAX_OBJECTIVE_CHARS):
            return tool_error(f"objective must be {_MIN_OBJECTIVE_CHARS}–{_MAX_OBJECTIVE_CHARS} chars")
        if not mgr.accepts_agent_goal():
            return tool_error("a goal is already set for this session; finish it with complete, "
                              "or the user clears it with /goal clear", **_state_view(mgr))
        from hermes_cli.goals import draft_contract
        try:
            contract = draft_contract(objective)
        except Exception as exc:  # contract is a quality aid, never a blocker
            logger.info("goal tool: draft_contract failed (%s)", exc)
            contract = None
        state = mgr.set_agent_goal(objective, contract=contract, max_turns=max_turns)
        if state is None:
            return tool_error("goal not set", **_state_view(mgr))
        return _ok(created=True, **_state_view(mgr))

    if action == "complete":
        if not mgr.has_goal():
            return tool_error("no goal to complete")
        evidence = (evidence or "").strip()
        if len(evidence) < _MIN_OBJECTIVE_CHARS:
            return tool_error("complete needs evidence that the outcome exists")
        from hermes_cli.goals import judge_goal
        s = mgr.state
        verdict, reason, parse_failed, _wd, transport_failed = judge_goal(
            s.goal, f"Completion claim with evidence:\n{evidence}",
            contract=s.contract if s.has_contract() else None,
        )
        if parse_failed or transport_failed:
            return tool_error(f"completion audit unavailable ({reason}); keep working or retry")
        if verdict != "done":
            return tool_error(f"completion audit refused: {verdict} — {reason}", verdict=verdict, reason=reason)
        mgr.mark_done(reason)
        return _ok(completed=True, reason=reason)

    return tool_error(f"unknown action {action!r}")


registry.register(
    name="goal", toolset="goal", schema=GOAL_SCHEMA, check_fn=check_goal_tool_requirements,
    handler=lambda args, **kw: goal_tool(
        str(args.get("action") or ""), objective=args.get("objective") or "", evidence=args.get("evidence") or "",
        max_turns=args.get("max_turns"), session_id=kw.get("session_id")),
    emoji="⊙",
)
