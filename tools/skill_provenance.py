"""Skill write-origin provenance — ContextVar for distinguishing agent-sediment skill writes from foreground user-directed writes.

The curator only consolidates/prunes skills it autonomously created via the
background self-improvement review fork. Skills a user asks a foreground
agent to write belong to the user and must never be auto-curated.

This module exposes a ContextVar that run_agent.py sets before each tool
loop so tool handlers (e.g. skill_manage create) can check whether they
are executing inside the background-review fork.

The signal piggybacks on AIAgent._memory_write_origin, which is already
set to "background_review" for review-fork instances (see
_spawn_background_review in run_agent.py) and defaults to "assistant_tool"
for normal (foreground) agents.

Usage:
    from tools.skill_provenance import (
        set_current_write_origin,
        reset_current_write_origin,
        get_current_write_origin,
    )

    token = set_current_write_origin("background_review")
    try:
        ...  # tool runs here
    finally:
        reset_current_write_origin(token)

    # inside a tool:
    if get_current_write_origin() == "background_review":
        mark_agent_created(skill_name)
"""

import contextvars
import logging

logger = logging.getLogger(__name__)


_write_origin: contextvars.ContextVar[str] = contextvars.ContextVar(
    "skill_write_origin",
    default="foreground",
)

# The sentinel value the background review fork uses; mirrors
# run_agent.py's AIAgent._memory_write_origin override in
# _spawn_background_review().
BACKGROUND_REVIEW = "background_review"


def set_current_write_origin(origin: str) -> contextvars.Token[str]:
    """Bind the active write origin to the current context.

    Returns a Token the caller must pass to reset_current_write_origin
    in a finally block.

    Binding ``background_review`` also installs the shared read-before-write
    mark store on *this* context so later tool-worker snapshots can see
    ``skill_view`` marks. Creating that store inside a worker copy does
    not propagate back to the parent (or to the next tool call).
    """
    bound = origin or "foreground"
    token = _write_origin.set(bound)
    if bound == BACKGROUND_REVIEW:
        try:
            from tools.skill_manager_tool import ensure_background_review_read_marks

            ensure_background_review_read_marks()
        except Exception:
            # Do NOT swallow this silently: if installing the shared mark
            # store fails, the review fork regresses to exactly the bug this
            # PR fixes (skill_view marks stuck in worker copies, every later
            # skill_manage refused) with zero signal. The origin binding
            # itself still succeeds, so the caller has no other way to notice.
            # Log loudly enough that development runs surface it; the guard
            # remains fail-closed (writes without a readable store are
            # refused), which is the correct degradation.
            logger.warning(
                "Failed to install the background-review read-before-write "
                "mark store; skill_view marks may not survive tool-worker "
                "snapshots in this review fork.",
                exc_info=True,
            )
    return token


def reset_current_write_origin(token: contextvars.Token[str]) -> None:
    """Restore the prior write origin context."""
    _write_origin.reset(token)


def get_current_write_origin() -> str:
    """Return the active write origin.

    Default: "foreground" — any tool call made by a regular (non-review)
    agent, from the CLI, the gateway, cron, or a subagent.

    "background_review" — the self-improvement review fork; only skills
    created under this origin should be marked agent-created for curator
    management.
    """
    return _write_origin.get()


def is_background_review() -> bool:
    """Convenience: True iff the current write origin is the background
    review fork."""
    return get_current_write_origin() == BACKGROUND_REVIEW
