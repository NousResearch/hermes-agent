"""Skill write-origin provenance: a ContextVar separating background-review skill writes from foreground
user-directed writes (the curator only curates skills the self-improvement review fork created; skills a user
asked for belong to the user). run_agent.py binds the origin before each tool loop, mirroring
AIAgent._memory_write_origin: ``token = set_current_write_origin(...)`` / ``reset_current_write_origin(token)``."""

import contextvars

_write_origin: contextvars.ContextVar[str] = contextvars.ContextVar("skill_write_origin", default="foreground")
BACKGROUND_REVIEW = "background_review"  # sentinel used by run_agent._spawn_background_review


def set_current_write_origin(origin: str) -> contextvars.Token[str]:
    return _write_origin.set(origin or "foreground")


def reset_current_write_origin(token: contextvars.Token[str]) -> None:
    _write_origin.reset(token)


def get_current_write_origin() -> str:
    """"foreground" for any regular agent (CLI, gateway, cron, subagent); "background_review" for the review fork."""
    return _write_origin.get()


def is_background_review() -> bool:
    return get_current_write_origin() == BACKGROUND_REVIEW


# Attendedness is orthogonal to origin: an explicit ``/refine`` fork IS a background review (every
# curator / skill-ledger / approval guard keyed on ``is_background_review()`` must still apply), but a
# user asked for it, so the unattended-only memory delete gate (#105921) does not.
_review_attended: contextvars.ContextVar[bool] = contextvars.ContextVar("review_attended", default=False)


def set_review_attended(attended: bool) -> contextvars.Token[bool]:
    return _review_attended.set(bool(attended))


def reset_review_attended(token: contextvars.Token[bool]) -> None:
    _review_attended.reset(token)


def is_unattended_review() -> bool:
    return is_background_review() and not _review_attended.get()


# Entry trigger for a background-review fork, stamped from spawn_background_review_thread's entry
# classification (not from attendedness: a focus-only fork is unattended yet not automatic):
# ``"automatic"``, ``"focus_only"``, or ``"refine"``; ``None`` outside a review fork.
_review_trigger: contextvars.ContextVar[str | None] = contextvars.ContextVar("review_trigger", default=None)


def set_review_trigger(trigger: str | None) -> contextvars.Token[str | None]:
    return _review_trigger.set(trigger or None)


def reset_review_trigger(token: contextvars.Token[str | None]) -> None:
    _review_trigger.reset(token)


def get_review_trigger() -> str | None:
    return _review_trigger.get()
