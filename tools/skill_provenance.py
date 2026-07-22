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
    6. ``not_agent_created`` — no usage record positively claims agent
       authorship, or one of this skill's records says someone else wrote it.
       Ownership must be claimed; silence is not consent.

    Args:
        name: The skill name, in any spelling that reaches this skill —
            ``foo``, ``operations/foo`` and ``./foo`` are the same skill.
        skill_dir: The skill's directory. Pass it whenever it is known — the
            ``external`` rule cannot be evaluated without a path, and is
            skipped when this is None.

    Pin, protected-builtin, hub and bundled are all keyed by bare skill name,
    so each is probed against every alias. Probing only the caller's spelling
    let a categorized skill slip every one of them: a pinned
    ``operations/foo`` refused as ``foo`` and passed as ``operations/foo``.
    A block found under any alias blocks — guards fail closed.

    Every lookup is best-effort: a broken sidecar or an unreadable config
    must not invent a block, so each failing probe is logged and skipped.
    """
    aliases = skill_alias_names(name, skill_dir)

    try:
        from tools import skill_usage
        if any(skill_usage.get_record(alias).get("pinned") for alias in aliases):
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
        if any(skill_usage.is_protected_builtin(alias) for alias in aliases):
            return BLOCK_PROTECTED_BUILTIN
        if any(skill_usage.is_hub_installed(alias) for alias in aliases):
            return BLOCK_HUB_INSTALLED
        if any(skill_usage.is_bundled(alias) for alias in aliases):
            return BLOCK_BUNDLED
        # Ownership must be claimed, not assumed. Only a usage record that
        # positively says `created_by: "agent"` opts a skill into autonomous
        # curation; everything else — a user-authored skill, a URL install, a
        # skill nothing has recorded yet — stays off-limits.
        #
        # Requiring the positive marker also closes a ratchet. Telemetry
        # absence used to read as "not user-owned, so writable", but the
        # read-before-write guard forces a skill_view before any write, and
        # skill_view bumps telemetry — seeding a record with created_by=None.
        # A skill advertised as writable therefore became blocked the moment
        # the fork did the reading the write path demanded of it.
        usage = skill_usage.load_usage()
        records = [usage.get(alias) for alias in aliases]
        present = [rec for rec in records if isinstance(rec, dict)]
        # At least one record must claim agent authorship, and none may
        # contradict it. The "none may contradict" half answers the
        # basename-collision case: a user skill at ``operations/foo`` with its
        # own ``created_by: null`` record must not inherit ownership from an
        # unrelated top-level ``foo`` that happens to share a basename.
        if not present or not all(
            skill_usage._is_curator_managed_record(rec) for rec in present
        ):
            return BLOCK_NOT_AGENT_CREATED
    except Exception:
        logger.debug("owned skill lookup failed for %s", name, exc_info=True)

    return None


def canonical_skill_name(name: str, skill_dir: Optional[Any] = None) -> str:
    """The one key every store should use for this skill.

    ``.usage.json`` is a flat name→record map, and so is every other
    provenance store: the bundled manifest, the hub lock, and
    ``PROTECTED_BUILTIN_SKILLS`` are all keyed by bare skill name. The
    directory basename is therefore the canonical identity — it is what
    ``skill_manage(create, name=X, category=C)`` already records, and what
    the curator, ``archive_skill`` and ``_find_skill_dir`` already assume.

    Deriving it from the resolved directory is what makes caller spelling
    irrelevant: ``aio``, ``operations/aio`` and ``./aio`` all canonicalize to
    ``aio``. Writing telemetry under the caller's spelling instead used to
    mint a second record on the first categorized write, splitting the
    skill's ownership state in two.

    Falls back to the caller's basename when the directory is unknown.
    """
    from pathlib import PurePath

    if skill_dir is not None:
        try:
            basename = PurePath(str(skill_dir)).name
            if basename:
                return basename
        except Exception:
            logger.debug("canonical name from dir failed for %s", name, exc_info=True)
    try:
        return PurePath(name).name or name
    except Exception:
        return name


def _skill_dir_relative_to_root(skill_dir: Any) -> Optional[str]:
    """``skill_dir`` as a path relative to whichever skills root contains it.

    This is the alias legacy records may be filed under — the same string
    ``create`` returns as ``path`` (``"operations/aio"``). Returns None when
    the directory sits under no known root.
    """
    from pathlib import Path

    try:
        from agent.skill_utils import get_all_skills_dirs
        candidate = Path(str(skill_dir)).resolve()
    except Exception:
        logger.debug("relative-root derivation failed for %s", skill_dir, exc_info=True)
        return None

    for root in get_all_skills_dirs():
        try:
            relative = candidate.relative_to(Path(root).resolve())
        except (OSError, ValueError):
            continue
        if relative.parts:
            return relative.as_posix()
    return None


def skill_alias_names(name: str, skill_dir: Optional[Any] = None) -> list:
    """Every name this one skill answers to, for guard and record lookups.

    The canonical name comes first, then the spellings a record may already
    be filed under. Crucially the alias set is derived from the resolved
    **directory**, not only from the caller's text — text-derived aliases are
    one-directional. Given ``aio`` there is nothing in the string to suggest
    ``operations/aio``, so a pin stored under the categorized key slipped
    every bare-name write. Resolving the directory and asking where it sits
    relative to its root closes both directions.

    Aliases exist for reading legacy state only. Every write goes to
    ``canonical_skill_name`` so no new split can be created, and callers that
    read fail closed when two aliases disagree.

    Only exact key matches count, so this never widens a lookup past a record
    that actually names this skill.
    """
    from pathlib import PurePath

    aliases: list = []

    def _add(candidate) -> None:
        text = str(candidate).strip() if candidate else ""
        if text and text not in aliases:
            aliases.append(text)

    _add(canonical_skill_name(name, skill_dir))
    _add(name)
    if name:
        try:
            # Collapses "./foo" to "foo" and "a/./b" to "a/b" — the same
            # normalization the resolver applies before touching disk.
            _add(str(PurePath(name)))
        except Exception:
            logger.debug("alias normalization failed for %s", name, exc_info=True)
    if skill_dir is not None:
        _add(_skill_dir_relative_to_root(skill_dir))
    return aliases
