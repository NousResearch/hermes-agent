"""
Skill Quarantine — Ledger-based quarantine status tracking.

External skills are quarantined by default. The grant engine reads quarantine
status from the central ledger (not the filesystem — single query, no stat).

Quarantine lifecycle:
  1. skill.quarantined  — written by install pipeline when bundle enters quarantine
  2. skill.quarantine.reviewed — written by Denji/skill-research on manual review
  3. skill.quarantine.rejected — written if review fails (skill stays blocked)

A skill is considered quarantined if the most recent event for that skill name
is ``skill.quarantined`` and has not been superseded by ``skill.quarantine.reviewed``.

Usage:
    from tools.skill_quarantine import is_quarantined, quarantine_skill, promote_skill, reject_skill

    # Grant engine calls this before granting:
    if is_quarantined("some-external-skill"):
        return deny("quarantined — requires manual review by Denji/skill-research")
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from hermes_cli.profile_activity_ledger import append_event, query_events

logger = logging.getLogger(__name__)

EV_QUARANTINED = "skill.quarantined"
EV_REVIEWED = "skill.quarantine.reviewed"
EV_REJECTED = "skill.quarantine.rejected"
SOURCE = "skill.quarantine"


def _now() -> int:
    return int(time.time())


def _new_event_id(prefix: str = "quarantine") -> str:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{prefix}-{today}-{uuid.uuid4().hex[:8]}"


def quarantine_skill(
    skill_name: str,
    source: str = "external",
    identifier: str = "",
    review_required_by: str = "denji",
    metadata: Optional[dict] = None,
) -> str:
    """Record that a skill has been placed in quarantine.

    Call this from the install pipeline (skills_hub.py → do_install or
    quarantine_bundle) when an external skill is downloaded.

    Args:
        skill_name: Canonical skill name.
        source: Where it came from ("github", "skills-sh", "url", etc.).
        identifier: Source-specific ID.
        review_required_by: Profile that must review before promotion.
        metadata: Extra context (repo, trust_level, hash, etc.).

    Returns:
        The event_id for the quarantine record.
    """
    eid = _new_event_id()
    payload = {
        "source": source,
        "identifier": identifier,
        "review_required_by": review_required_by,
        "quarantined_at": _now(),
        **(metadata or {}),
    }
    append_event(
        source=SOURCE,
        event_type=EV_QUARANTINED,
        event_id=eid,
        actor_profile="skill-broker",
        target_profile=review_required_by,
        object_type="skill",
        object_id=skill_name,
        summary=f"Skill '{skill_name}' quarantined — requires review by {review_required_by}",
        payload=payload,
    )
    return eid


def promote_skill(
    skill_name: str,
    reviewer: str,
    notes: str = "",
) -> str:
    """Mark a quarantined skill as reviewed and safe.

    After this, ``is_quarantined(skill_name)`` returns False. Called by
    Denji/skill-research after manual review and native rewrite.

    Returns:
        The event_id for the promotion record.
    """
    return append_event(
        source=SOURCE,
        event_type=EV_REVIEWED,
        event_id=_new_event_id("review"),
        actor_profile=reviewer,
        target_profile="skill-broker",
        object_type="skill",
        object_id=skill_name,
        summary=f"Skill '{skill_name}' reviewed and promoted by {reviewer}",
        payload={
            "reviewer": reviewer,
            "reviewed_at": _now(),
            "notes": notes,
        },
    )


def reject_skill(
    skill_name: str,
    reviewer: str,
    reason: str,
) -> str:
    """Mark a quarantined skill as rejected.

    The skill stays blocked. The rejection reason is logged to the ledger
    so Denji can audit why skills were rejected.

    Returns:
        The event_id for the rejection record.
    """
    return append_event(
        source=SOURCE,
        event_type=EV_REJECTED,
        event_id=_new_event_id("reject"),
        actor_profile=reviewer,
        target_profile="skill-broker",
        object_type="skill",
        object_id=skill_name,
        summary=f"Skill '{skill_name}' rejected by {reviewer}: {reason[:120]}",
        payload={
            "reviewer": reviewer,
            "rejected_at": _now(),
            "reason": reason,
        },
    )


def is_quarantined(skill_name: str) -> bool:
    """True if skill_name is currently quarantined (not yet reviewed/promoted).

    Logic: query the most recent quarantine event for this skill. If it's
    ``skill.quarantined`` and has not been superseded by a more recent
    ``skill.quarantine.reviewed``, the skill is still quarantined.

    A ``skill.quarantine.rejected`` event keeps the skill blocked — it does
    NOT clear the quarantine. Only ``reviewed`` clears it.

    Fails CLOSED: any ledger error returns True so a broken ledger never
    silently unblocks a quarantined skill.
    """
    if not skill_name:
        return False
    try:
        events = query_events(
            event_types=[EV_QUARANTINED, EV_REVIEWED, EV_REJECTED],
            object_id=skill_name,
            limit=10,
        )
        if not events:
            return False

        # Walk newest-first — if the most recent event is quarantined or rejected,
        # the skill is blocked. If it's reviewed, it's cleared.
        for e in events:
            if e.get("event_type") == EV_REVIEWED:
                return False
            if e.get("event_type") in (EV_QUARANTINED, EV_REJECTED):
                return True

        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("is_quarantined failed for %s: %s", skill_name, exc)
        return True  # fail closed


def get_quarantine_info(skill_name: str) -> Optional[dict]:
    """Return metadata for the active quarantine, or None if not quarantined."""
    try:
        events = query_events(
            event_types=[EV_QUARANTINED, EV_REVIEWED, EV_REJECTED],
            object_id=skill_name,
            limit=1,
        )
        if not events:
            return None
        e = events[0]
        if e.get("event_type") == EV_REVIEWED:
            return None
        return {
            "skill_name": skill_name,
            "status": e.get("event_type"),
            "source": (e.get("payload") or {}).get("source", "unknown"),
            "identifier": (e.get("payload") or {}).get("identifier", ""),
            "review_required_by": (e.get("payload") or {}).get("review_required_by", "denji"),
            "quarantined_at": e.get("occurred_at"),
        }
    except Exception as exc:
        logger.debug("get_quarantine_info failed: %s", exc)
        return None


def list_quarantined() -> list[dict]:
    """Return all currently-quarantined skills for the review queue."""
    try:
        events = query_events(
            event_types=[EV_QUARANTINED, EV_REVIEWED, EV_REJECTED],
        )
        # Group by skill name, keep only unresolved
        seen: dict[str, Optional[str]] = {}  # skill_name -> most recent event_type
        for e in events:
            name = e.get("object_id")
            if not name:
                continue
            if name not in seen:
                seen[name] = e.get("event_type")

        quarantined = []
        for name, status in seen.items():
            if status in (EV_QUARANTINED, EV_REJECTED):
                info = get_quarantine_info(name)
                if info:
                    quarantined.append(info)
        return quarantined
    except Exception as exc:
        logger.debug("list_quarantined failed: %s", exc)
        return []
