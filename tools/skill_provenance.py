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
from typing import Any, Optional

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
    """
    return _write_origin.set(origin or "foreground")


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


# ── Background-review write eligibility ───────────────────────────────────
#
# One predicate with two consumers: skills_list() advertises its verdict as
# ``writable`` / ``blocked_because`` on every entry the review fork sees, and
# skill_manager_tool's write guard enforces the same verdict on the way in.
# Before they shared this function, discovery showed only name/description/
# category — so the review fork picked targets it was never allowed to write,
# hit a real refusal, and spent its budget retrying them.
#
# Each constant is a stable machine-readable reason code. They are the values
# that appear in ``blocked_because``; the human-facing refusal wording lives
# with the guard in tools/skill_manager_tool.py.

BLOCK_PINNED = "pinned"
BLOCK_EXTERNAL = "external"
BLOCK_PROTECTED_BUILTIN = "protected_builtin"
BLOCK_HUB_INSTALLED = "hub_installed"
BLOCK_BUNDLED = "bundled"
BLOCK_NOT_AGENT_CREATED = "not_agent_created"

BACKGROUND_REVIEW_BLOCK_REASONS = (
    BLOCK_PINNED,
    BLOCK_EXTERNAL,
    BLOCK_PROTECTED_BUILTIN,
    BLOCK_HUB_INSTALLED,
    BLOCK_BUNDLED,
    BLOCK_NOT_AGENT_CREATED,
)


def background_review_block_reason(
    name: str, skill_dir: Optional[Any] = None
) -> Optional[str]:
    """Return why autonomous review may not write to *name*, or None.

    This answers ownership only — "is this skill the review fork's to
    change" — and deliberately ignores the caller's origin. Callers that
    enforce (the write guard) check ``is_background_review()`` first;
    callers that advertise (``skills_list``) do the same before annotating.

    Ownership rules, checked in this order:

    1. ``pinned`` — pin blocks autonomous edits, not just deletion. The
       curator skips pinned skills on every auto-transition and there is no
       user in the loop here to consent to a write (issue #25839).
    2. ``external`` — the skill lives under ``skills.external_dirs``, which
       have an upstream owner.
    3. ``protected_builtin`` — a load-bearing built-in.
    4. ``hub_installed`` — installed via ``hermes skills install``.
    5. ``bundled`` — seeded from the skills shipped with Hermes.
    6. ``not_agent_created`` — a usage record exists and says the skill was
       authored by someone other than the agent.

    Args:
        name: The skill name.
        skill_dir: The skill's directory. Pass it whenever it is known — the
            ``external`` rule cannot be evaluated without a path, and is
            skipped when this is None.

    Every lookup is best-effort: a broken sidecar or an unreadable config
    must not invent a block, so each failing probe is logged and skipped.
    """
    try:
        from tools import skill_usage
        if skill_usage.get_record(name).get("pinned"):
            return BLOCK_PINNED
    except Exception:
        logger.debug("pinned skill lookup failed for %s", name, exc_info=True)

    if skill_dir is not None:
        try:
            from agent.skill_utils import is_external_skill_path
            if is_external_skill_path(skill_dir):
                return BLOCK_EXTERNAL
        except Exception:
            logger.debug("external skill lookup failed for %s", name, exc_info=True)

    try:
        from tools import skill_usage
        if skill_usage.is_protected_builtin(name):
            return BLOCK_PROTECTED_BUILTIN
        if skill_usage.is_hub_installed(name):
            return BLOCK_HUB_INSTALLED
        if skill_usage.is_bundled(name):
            return BLOCK_BUNDLED
        # Manually authored skills (created_by != "agent") are off-limits to
        # autonomous curation. This prevents the LLM consolidation pass from
        # archiving skills the user placed manually (e.g. via URL install or
        # direct SKILL.md authoring), which lack the `created_by: "agent"`
        # marker. A skill with no record at all is not blocked here — an
        # absent record carries no claim of user authorship.
        record = skill_usage.load_usage().get(name)
        if isinstance(record, dict) and not skill_usage._is_curator_managed_record(record):
            return BLOCK_NOT_AGENT_CREATED
    except Exception:
        logger.debug("owned skill lookup failed for %s", name, exc_info=True)

    return None
