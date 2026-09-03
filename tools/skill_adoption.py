"""Skill adoption -- recipient decision layer (Wisdom v1, M3).

Lets a recipient adopt an org-shared skill into their personal namespace
as a local, editable copy with provenance recording origin (author,
source commit, adopted-at). Declines are persisted so the same share
is not re-offered.

Design notes:
  - Adopted skills are ordinary local skills from the moment they land.
    The recipient's agent can patch them; local modification does not
    write back. Write-back is the proposal path the author already owns.
  - Declines are permanent within v1: a declined share never re-surfaces
    to the same person.
  - The adoption state file (`.adoption_state`) tracks adopted and
    declined skills per org, so the recipient can see what's new since
    their last review.

State file: ``~/.hermes/skills/.adoption_state`` (JSON).
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


def _state_file() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "skills" / ".adoption_state"


def _default_state() -> Dict[str, Any]:
    return {
        "adopted": {},
        "declined": {},
    }


def load_state() -> Dict[str, Any]:
    path = _state_file()
    if not path.exists():
        return _default_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            base = _default_state()
            base.update({k: v for k, v in data.items() if k in base})
            return base
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("skill_adoption: failed to read state: %s", e)
    return _default_state()


def save_state(data: Dict[str, Any]) -> None:
    path = _state_file()
    try:
        from utils import atomic_json_write

        atomic_json_write(path, data, indent=2, sort_keys=True)
    except Exception as e:
        logger.debug("skill_adoption: failed to save state: %s", e)


# ---------------------------------------------------------------------------
# Org skill listing
# ---------------------------------------------------------------------------


def _org_dir() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "skills" / "_org"


def _skills_dir() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "skills"


def list_org_skills(org_id: str) -> List[Dict[str, Any]]:
    """List skills in the org mirror with their adoption status.

    Returns [{name, rel_path, adopted, declined, has_local_copy}].
    """
    org_root = _org_dir() / org_id
    if not org_root.exists():
        return []

    state = load_state()
    adopted = state.get("adopted", {}).get(org_id, {})
    declined = set(state.get("declined", {}).get(org_id, []))

    skills = []
    for category_dir in sorted(org_root.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith("."):
            continue
        for skill_dir in sorted(category_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            if not (skill_dir / "SKILL.md").exists():
                continue
            rel = f"{category_dir.name}/{skill_dir.name}"
            local_path = _skills_dir() / category_dir.name / skill_dir.name
            skills.append(
                {
                    "name": skill_dir.name,
                    "rel_path": rel,
                    "adopted": rel in adopted,
                    "declined": rel in declined,
                    "has_local_copy": local_path.exists(),
                }
            )
    return skills


def pending_shares(org_id: str) -> List[Dict[str, Any]]:
    """List org skills the recipient hasn't decided on yet."""
    return [
        s
        for s in list_org_skills(org_id)
        if not s["adopted"] and not s["declined"]
    ]


# ---------------------------------------------------------------------------
# Adopt / decline
# ---------------------------------------------------------------------------


def adopt_skill(
    org_id: str,
    rel_path: str,
    *,
    source_commit: Optional[str] = None,
    author: Optional[str] = None,
) -> Dict[str, Any]:
    """Copy an org skill into the personal namespace with provenance.

    Args:
        org_id: The org whose mirror contains the skill.
        rel_path: Category/name path (e.g. "software-development/code-review").
        source_commit: The org HEAD commit at adopt time (for provenance).
        author: The skill's original author (for provenance).

    Returns:
        {ok, skill_name, dest} or {ok: False, error}.
    """
    org_skill_dir = _org_dir() / org_id / rel_path
    if not org_skill_dir.exists():
        return {"ok": False, "error": f"org skill not found: {rel_path}"}
    if not (org_skill_dir / "SKILL.md").exists():
        return {"ok": False, "error": f"org skill has no SKILL.md: {rel_path}"}

    skill_name = org_skill_dir.name
    category = org_skill_dir.parent.name
    dest = _skills_dir() / category / skill_name

    if dest.exists():
        return {
            "ok": False,
            "error": f"'{skill_name}' already exists in your personal skills",
        }

    # Copy the skill directory.
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(org_skill_dir, dest)

    # Write adoption provenance.
    now = datetime.now(timezone.utc)
    provenance = {
        "origin": "adopted",
        "org_id": org_id,
        "source_rel_path": rel_path,
        "source_commit": source_commit,
        "author": author,
        "adopted_at": now.isoformat(),
    }
    provenance_path = dest / ".adoption-provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    # Update state.
    state = load_state()
    if org_id not in state["adopted"]:
        state["adopted"][org_id] = {}
    state["adopted"][org_id][rel_path] = {
        "adopted_at": now.isoformat(),
        "source_commit": source_commit,
        "author": author,
    }
    # Remove from declined if it was previously declined.
    if org_id in state["declined"] and rel_path in state["declined"][org_id]:
        state["declined"][org_id].remove(rel_path)
    save_state(state)

    return {"ok": True, "skill_name": skill_name, "dest": str(dest)}


def decline_share(org_id: str, rel_path: str) -> Dict[str, Any]:
    """Persist a decline so the share is not re-offered.

    Returns {ok} or {ok: False, error}.
    """
    org_skill_dir = _org_dir() / org_id / rel_path
    if not org_skill_dir.exists():
        return {"ok": False, "error": f"org skill not found: {rel_path}"}

    state = load_state()
    if org_id not in state["declined"]:
        state["declined"][org_id] = []
    if rel_path not in state["declined"][org_id]:
        state["declined"][org_id].append(rel_path)
        save_state(state)

    return {"ok": True, "declined": rel_path}
