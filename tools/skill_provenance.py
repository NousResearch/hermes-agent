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


# =========================================================================
# Draft/Pending/Confirmed Skill State Management
# =========================================================================
# 
# Skills can be in three states based on their frontmatter:
#   - draft: Created by agent in background_review, awaiting user confirmation
#   - pending: Explicitly deferred for later review
#   - confirmed: User-approved and active
#   - (no status field): Legacy/external skills or user-created skills
#
# The status field lives in SKILL.md frontmatter along with:
#   author: agent | user
#   confirmed_at: ISO timestamp (set when status moves to confirmed)

from datetime import datetime
from typing import Literal, Optional, Dict, Any

# State constants
DRAFT = "draft"
PENDING = "pending"
CONFIRMED = "confirmed"

Status = Literal["draft", "pending", "confirmed"]
Author = Literal["agent", "user"]

# ContextVar for tracking if we should suppress draft skills
_suppress_drafts: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "suppress_drafts", default=True
)


def set_suppress_drafts(suppress: bool) -> contextvars.Token[bool]:
    """Control whether draft skills are suppressed in discovery.
    
    Default: True (suppress drafts). Set to False when explicitly
    requesting draft skills.
    """
    return _suppress_drafts.set(suppress)


def reset_suppress_drafts(token: contextvars.Token[bool]) -> None:
    """Restore prior suppress_drafts setting."""
    _suppress_drafts.reset(token)


def should_suppress_drafts() -> bool:
    """Return True if draft skills should be excluded from discovery."""
    return _suppress_drafts.get()


def parse_skill_frontmatter(skill_md_path) -> Optional[Dict[str, Any]]:
    """Parse SKILL.md frontmatter, return dict or None on error."""
    try:
        from pathlib import Path
        import yaml as yaml_mod
        
        skill_md = Path(skill_md_path)
        with open(skill_md, 'r', encoding='utf-8') as f:
            content = f.read()
        if not content.startswith('---'):
            return None
        end_match = content.find('\n---\n', 3)
        if end_match < 0:
            return None
        yaml_str = content[3:end_match]
        return yaml_mod.safe_load(yaml_str) or {}
    except Exception:
        return None


def get_skill_status(skill_md_path) -> Optional[Status]:
    """Extract status field from skill frontmatter.
    
    Returns: "draft", "pending", "confirmed", or None (legacy/no status).
    """
    fm = parse_skill_frontmatter(skill_md_path)
    if fm and isinstance(fm, dict):
        status = fm.get('status')
        if status in (DRAFT, PENDING, CONFIRMED):
            return status
    return None


def get_skill_author(skill_md_path) -> Optional[Author]:
    """Extract author field from skill frontmatter.
    
    Returns: "agent", "user", or None (legacy/no author).
    """
    fm = parse_skill_frontmatter(skill_md_path)
    if fm and isinstance(fm, dict):
        author = fm.get('author')
        if author in ('agent', 'user'):
            return author
    return None


def get_confirmed_at(skill_md_path) -> Optional[str]:
    """Extract confirmed_at ISO timestamp from frontmatter, or None."""
    fm = parse_skill_frontmatter(skill_md_path)
    if fm and isinstance(fm, dict):
        return fm.get('confirmed_at')
    return None


def is_draft_skill(skill_md_path) -> bool:
    """Return True if skill has draft status."""
    return get_skill_status(skill_md_path) == DRAFT


def is_pending_skill(skill_md_path) -> bool:
    """Return True if skill has pending status."""
    return get_skill_status(skill_md_path) == PENDING


def is_confirmed_or_legacy(skill_md_path) -> bool:
    """Return True if skill is confirmed or has no status (legacy)."""
    status = get_skill_status(skill_md_path)
    return status is None or status == CONFIRMED
