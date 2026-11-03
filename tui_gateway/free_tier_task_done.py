"""Detects a task finishing on its own on the free tier, which (re)arms the sign-in offer
(``hermes_cli.free_tier_offer``).

"On its own" = the turn completed with a reply, on the free tier, and nobody steered, redirected,
interrupted or typed into it while it ran. Answering a clarify card is not intervention: it is the
reply to a server request, which touches none of those paths. A crash resume is not a new task.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def is_clean_task(result: Any, agent: Any, *, user_input: bool, display_kind: str | None) -> bool:
    from hermes_cli.anon_auth import is_anonymous_agent
    if display_kind == "auto_continue" or user_input:
        return False
    if not isinstance(result, dict) or not result.get("completed"):
        return False
    if result.get("pending_steer") or result.get("user_intervened"):
        return False
    if not str(result.get("final_response") or "").strip():
        return False
    return is_anonymous_agent(agent)


def note_task_done(session: dict, result: Any, agent: Any, display_kind: str | None) -> None:
    """Record a clean free-tier task for the sign-in offer. Never fails the turn."""
    try:
        if is_clean_task(result, agent, user_input=bool(session.get("_turn_user_input")), display_kind=display_kind):
            from hermes_cli.free_tier_offer import record_task_done
            record_task_done()
    except Exception:
        logger.debug("sign-in offer task record failed", exc_info=True)
