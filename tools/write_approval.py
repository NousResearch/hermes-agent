#!/usr/bin/env python3
"""Write-approval gate + pending store for memory, skill, and control-file writes.

A per-subsystem boolean ``write_approval`` gates the agent's cross-session writes —
**memory** (MEMORY.md / USER.md), **skills** (SKILL.md + files) and **config**
(config.yaml values) — from either origin (**foreground** turn or
**background_review** fork). One opt-in switch,
``agent.require_persistent_change_approval`` (#110429), turns the gate on for
EVERY subsystem at once. ``false`` (default) writes freely; effectively ``true``
never commits directly: it prompts inline (memory, interactive CLI only) or
**stages** the write under ``<HERMES_HOME>/pending/{memory,skills,config}/<id>.json``
for out-of-band review (``hermes pending``, ``/memory pending``, ``/skills pending``).
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

# Subsystem identifiers
MEMORY = "memory"
SKILLS = "skills"
CONFIG = "config"
_SUBSYSTEMS = (MEMORY, SKILLS, CONFIG)

# Per-subsystem config key. Intentionally a single boolean with no "block all writes"
# state — to disable a subsystem use its own enable flag (e.g. ``memory.memory_enabled``).
CONFIG_KEY = "write_approval"
_TRUTHY_STRINGS = frozenset({"on", "true", "yes", "1", "approve", "enabled"})

# One switch that arms every subsystem at once (#110429) instead of opting in one at a
# time. Off by default: when it is unset/false the per-subsystem booleans stay the only
# authority, so a default install behaves byte-for-byte as it did before.
GLOBAL_SECTION = "agent"
GLOBAL_KEY = "require_persistent_change_approval"
GLOBAL_SWITCH = f"{GLOBAL_SECTION}.{GLOBAL_KEY}"

# Where a staged write is reviewed, per subsystem (message text only).
_REVIEW_SURFACE = {
    MEMORY: "/memory pending (or `hermes pending`)",
    SKILLS: "/skills pending (or `hermes pending`)",
    CONFIG: "`hermes pending`",
}


# --- Config resolution ---

def persistent_change_approval_required() -> bool:
    """Read the global opt-in switch ``agent.require_persistent_change_approval``.

    Resolved per call (not frozen at import) so flipping it takes effect without a restart;
    any unset/invalid/unreadable value means OFF — the safe direction, since ON changes
    behavior for every subsystem."""
    try:
        from hermes_cli.config import load_config, cfg_get
        return _normalize_enabled(cfg_get(load_config(), GLOBAL_SECTION, GLOBAL_KEY, default=False))
    except Exception:
        return False


def write_approval_enabled(subsystem: str) -> bool:
    """True when ``subsystem``'s writes need approval: the global switch
    (``agent.require_persistent_change_approval``) OR ``<subsystem>.write_approval``.
    Any unset/invalid value means gate off."""
    if subsystem not in _SUBSYSTEMS:
        return False
    if persistent_change_approval_required():
        return True
    try:
        from hermes_cli.config import load_config, cfg_get
        return _normalize_enabled(cfg_get(load_config(), subsystem, CONFIG_KEY, default=False))
    except Exception:
        return False


def _normalize_enabled(value: Any) -> bool:
    """Coerce a config value to bool; unknown → False (gate off). The string branch
    covers hand-edited configs (YAML already parses bare on/off/yes/no)."""
    if isinstance(value, bool):
        return value
    return isinstance(value, str) and value.strip().lower() in _TRUTHY_STRINGS


# --- Pending store (file-backed) ---

def _pending_path(subsystem: str, pending_id: str) -> Path:
    return get_hermes_home() / "pending" / subsystem / f"{pending_id}.json"


def _pending_files(subsystem: str) -> list:
    d = _pending_path(subsystem, "").parent
    return list(d.glob("*.json")) if d.exists() else []


def stage_write(subsystem: str, payload: Dict[str, Any], *, summary: str, origin: str) -> Dict[str, Any]:
    """Persist a pending write and return its record (``id`` + metadata). ``payload`` is the exact
    kwargs to replay the write on approval; ``origin`` is ``foreground`` or ``background_review``.
    Best-effort: on disk failure it logs and still returns a record — the write is lost, which is
    the safe failure for an approval gate (nothing silently committed)."""
    pid = uuid.uuid4().hex[:8]
    record = {
        "id": pid, "subsystem": subsystem, "action": payload.get("action", ""),
        "summary": (summary or "").strip(), "origin": origin or "foreground",
        "created_at": time.time(), "payload": payload,
    }
    try:
        # 0600: a pending record can carry memory text, a skill body, or a config value that is
        # a credential, so it must not be umask-readable before it is reviewed.
        atomic_json_write(_pending_path(subsystem, pid), record, mode=0o600)
    except Exception as e:  # pragma: no cover - disk failure path
        logger.error("Failed to stage pending %s write: %s", subsystem, e, exc_info=True)
    return record


def list_pending(subsystem: str) -> List[Dict[str, Any]]:
    """Return all pending records for ``subsystem``, oldest first."""
    records: List[Dict[str, Any]] = []
    for p in _pending_files(subsystem):
        try:
            records.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            logger.warning("Skipping unreadable pending record: %s", p)
    records.sort(key=lambda r: r.get("created_at", 0))
    return records


def get_pending(subsystem: str, pending_id: str) -> Optional[Dict[str, Any]]:
    """Return a single pending record by id, or None."""
    path = _pending_path(subsystem, pending_id)
    if not path.exists():
        return None
    with suppress(Exception):
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def discard_pending(subsystem: str, pending_id: str) -> bool:
    """Delete a pending record. Returns True if it existed."""
    try:
        path = _pending_path(subsystem, pending_id)
        if path.exists():
            path.unlink()
            return True
    except Exception as e:  # pragma: no cover
        logger.error("Failed to discard pending %s/%s: %s", subsystem, pending_id, e)
    return False


def pending_count(subsystem: str) -> int:
    """Cheap count of pending records (for notification badges)."""
    d = _pending_path(subsystem, "").parent
    if not d.exists():
        return 0
    with suppress(Exception):
        return sum(1 for _ in d.glob("*.json"))
    return 0


def all_pending() -> List[Dict[str, Any]]:
    """Every pending record across every subsystem, oldest first — the whole review queue.
    Each record carries its own ``subsystem`` key (stamped by ``stage_write``)."""
    records: List[Dict[str, Any]] = []
    for sub in _SUBSYSTEMS:
        records.extend(list_pending(sub))
    records.sort(key=lambda r: r.get("created_at", 0))
    return records


def find_pending(pending_id: str) -> Optional[Dict[str, Any]]:
    """Look a pending record up by id across all subsystems (ids are unique hex)."""
    for sub in _SUBSYSTEMS:
        rec = get_pending(sub, pending_id)
        if rec is not None:
            return rec
    return None


# --- Approval replay (the ONLY path that applies a gated write) ---

def apply_control_pending(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Replay a staged ``config`` write (``hermes pending approve <id>``), bypassing the gate.

    ``set_config_value`` / ``unset_config_value`` still own validation and coercion, so an
    approved change goes through exactly the same code path a direct ``hermes config set``
    would have."""
    key = str(payload.get("key") or "").strip()
    if not key:
        return {"success": False, "error": "Staged config write has no key."}
    from hermes_cli.config import set_config_value, unset_config_value

    if payload.get("unset"):
        unset_config_value(key, _approved=True)
        return {"success": True, "message": f"Unset {key}."}
    set_config_value(key, str(payload.get("value") or ""), force=bool(payload.get("force")), _approved=True)
    return {"success": True, "message": f"Set {key}."}


def apply_pending(subsystem: str, record: Dict[str, Any], *, memory_store=None) -> Dict[str, Any]:
    """Apply ONE already-approved pending record, bypassing the gate; ``{"success", "error"?}``.

    The shared replay dispatcher behind every review surface (``/memory approve``,
    ``/skills approve``, ``hermes pending approve``), so adding a subsystem means adding a
    branch here rather than a second apply path. ``_exit_invalid`` inside ``hermes_cli.config``
    raises ``SystemExit``; that is caught and reported as a failure so one bad record cannot
    kill the batch."""
    payload = record.get("payload") or {}
    try:
        if subsystem == MEMORY:
            if memory_store is None:
                return {"success": False, "error": "memory store unavailable"}
            from tools.memory_tool import apply_memory_pending
            return apply_memory_pending(payload, memory_store)
        if subsystem == SKILLS:
            from tools.skill_manager_tool import apply_skill_pending
            return json.loads(apply_skill_pending(payload))
        if subsystem == CONFIG:
            return apply_control_pending(payload)
        return {"success": False, "error": f"Unknown subsystem '{subsystem}'."}
    except SystemExit as exc:  # hermes_cli.config._exit_invalid on a rejected write
        return {"success": False, "error": f"config write rejected (exit {exc.code})"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def approve_pending(subsystem: str, pending_id: str, *, memory_store=None) -> Dict[str, Any]:
    """Approve one record: apply it and drop it from the queue only on success."""
    record = get_pending(subsystem, pending_id)
    if record is None:
        return {"success": False, "error": f"No pending {subsystem} write with id '{pending_id}'."}
    result = apply_pending(subsystem, record, memory_store=memory_store)
    if result.get("success"):
        discard_pending(subsystem, pending_id)
    return result


# --- Write origin ---

def current_origin() -> str:
    """``foreground`` or ``background_review`` — reuses the skill-provenance ContextVar
    the background review fork sets; foreground turns leave it at the default."""
    with suppress(Exception):
        from tools.skill_provenance import get_current_write_origin
        return get_current_write_origin()
    return "foreground"


# --- Gate decision ---

@dataclass(slots=True, kw_only=True)
class GateDecision:
    """Result of evaluating the write gate; exactly one flag is True. ``allow``: do the real write;
    ``blocked``: user denied the inline prompt (``message`` says why); ``stage``: caller must
    ``stage_write`` the payload (``message`` is the user-facing "staged for approval" note)."""

    allow: bool = False
    blocked: bool = False
    stage: bool = False
    message: str = ""


def _staged(subsystem: str) -> GateDecision:
    source = GLOBAL_SWITCH if persistent_change_approval_required() else f"{subsystem}.{CONFIG_KEY}"
    where = _REVIEW_SURFACE.get(subsystem, "`hermes pending`")
    return GateDecision(stage=True, message=(
        f"Staged for approval ({source} is on). Not yet saved — review with {where}: "
        f"approve <id> to apply, reject <id> to drop."))


def evaluate_gate(subsystem: str, *, inline_summary: str = "", inline_detail: str = "") -> GateDecision:
    """Decide what to do with a pending write: gate off → allow; gate on + memory + foreground
    → inline prompt when an interactive channel exists, else stage; any other subsystem (skills,
    config) or a background origin → stage. Skill/config writes are staged rather than prompted
    inline: skills are too big to review inline, and a config write is frequently issued from a
    non-interactive process (``hermes config set``) with no approval channel to answer a prompt.
    The gate only ever delays a write, never silently refuses it; ``blocked`` is produced only
    when the user actively denies the inline prompt."""
    if not write_approval_enabled(subsystem):
        return GateDecision(allow=True)
    # A background write runs in a daemon thread with no user to prompt.
    if subsystem != MEMORY or current_origin() == "background_review":
        return _staged(subsystem)
    granted = _prompt_inline_memory_approval(inline_summary, inline_detail)
    if granted is None:
        return _staged(MEMORY)
    if granted:
        return GateDecision(allow=True)
    return GateDecision(blocked=True, message="Memory write denied by user. The change was not saved.")


def gate_or_stage(subsystem: str, payload: Dict[str, Any], *, summary: str,
                  inline_detail: str = "") -> Optional[str]:
    """Gate + stage in one call: ``None`` when the write may proceed, else the user/model-facing
    message for the blocked or staged outcome (and the payload is in the pending store).

    The single call-site shape for non-memory subsystems, so adding a control-file surface
    costs one line at the write point instead of a copy of the staging dance."""
    decision = evaluate_gate(subsystem, inline_summary=summary, inline_detail=inline_detail)
    if decision.allow:
        return None
    if decision.blocked:
        return decision.message
    detail = f"{summary}: {inline_detail[:120]}" if inline_detail else summary
    stage_write(subsystem, payload, summary=detail, origin=current_origin())
    return decision.message


def _prompt_inline_memory_approval(summary: str, detail: str) -> Optional[bool]:
    """Prompt inline for a memory write: True approved, False denied, None → stage. Uses the per-thread
    CLI approval callback (``tools.terminal_tool.set_approval_callback``) directly, not
    ``prompt_dangerous_approval``: that wrapper falls back to ``input()`` (deadlock-prone under
    prompt_toolkit; silent deny in gateway sessions) and turns callback errors into a deny, whereas
    here a missing channel or failed prompt must stage instead.

    See #15216.
    """
    try:
        from tools.terminal_tool import _get_approval_callback
    except Exception:
        return None
    callback = _get_approval_callback()
    if callback is None:
        return None
    header = summary.strip() or "Save to memory?"
    try:
        from tools.approval_prompt import callback_accepts
        extra = {"title": "Save to memory?"} if callback_accepts(callback, "title") else {}
        choice = callback(detail.strip() or header, f"Save to memory: {header}", allow_permanent=False, **extra)
    except Exception as e:
        logger.error("Inline memory approval prompt failed: %s", e)
        return None
    # unknown outcome → stage rather than drop
    return {"once": True, "session": True, "deny": False}.get(choice)


# --- Skill-specific helpers (gist + diff for the review affordances) ---

_GIST_TEMPLATES = {"write_file": "write {file_path} in '{name}'", "remove_file": "remove {file_path} from '{name}'",
                   "delete": "delete skill '{name}'"}


def skill_gist(action: str, name: str, *, content: str = "", file_path: str = "",
               old_string: str = "", new_string: str = "") -> str:
    """One-line heuristic gist (no model call) for a pending skill write: create/edit use
    the frontmatter ``description:``; patch/write_file describe the size of the change."""
    if action in {"create", "edit"} and content:
        desc = _frontmatter_description(content)
        size = f"{len(content) // 1024 + 1} KB" if len(content) >= 1024 else f"{len(content)} chars"
        return f"{'create' if action == 'create' else 'rewrite'} '{name}'{f' — {desc}' if desc else ''} ({size})"
    if action == "patch":
        removed = old_string.count("\n") + 1 if old_string else 0
        added = new_string.count("\n") + 1 if new_string else 0
        return f"patch '{name}' {file_path or 'SKILL.md'} (+{added}/-{removed} lines)"
    return _GIST_TEMPLATES.get(action, "{action} '{name}'").format(action=action, name=name, file_path=file_path)


def _frontmatter_description(content: str) -> str:
    """Extract the ``description:`` value from SKILL.md YAML frontmatter (≤140 chars)."""
    m = re.search(r"^description:\s*(.+)$", content, re.MULTILINE)
    return m.group(1).strip().strip("'\"")[:140] if m else ""


def _find_skill_path(name: str) -> Optional[Path]:
    """Directory of an installed skill, or None if unknown / lookup unavailable."""
    try:
        from tools.skill_manager_tool import _find_skill
    except Exception:
        return None
    # Only the import is guarded (as on main); a lookup failure propagates.
    found = _find_skill(name)
    return found["path"] if found else None


def skill_pending_diff(record: Dict[str, Any]) -> str:
    """Full content (create) or unified diff vs. the on-disk skill (edit/patch/write_file),
    rendered by /skills diff <id> on surfaces that can show it."""
    payload = record.get("payload", {})
    action = payload.get("action", "")
    name = payload.get("name", "")
    if action == "create":
        return payload.get("content") or ""
    if action not in {"edit", "patch", "write_file"}:
        return {"remove_file": f"remove file: {payload.get('file_path')} from skill '{name}'",
                "delete": f"delete skill '{name}'"}.get(action, f"({action} on '{name}')")

    # patch/write_file target a file inside the skill; edit always targets SKILL.md.
    target_label, current = "SKILL.md", ""
    skill_dir = _find_skill_path(name)
    if skill_dir:
        if action != "edit":
            target_label = payload.get("file_path") or "SKILL.md"
        with suppress(Exception):
            p = skill_dir / target_label
            current = p.read_text(encoding="utf-8") if p.exists() else ""

    if action == "patch":
        old_s, new_s = payload.get("old_string") or "", payload.get("new_string") or ""
        new = current.replace(old_s, new_s) if current else f"(patch {old_s!r} → {new_s!r})"
    else:
        new = payload.get("content" if action == "edit" else "file_content") or ""
    diff = difflib.unified_diff(current.splitlines(keepends=True), new.splitlines(keepends=True),
                                fromfile=f"a/{target_label}", tofile=f"b/{target_label}")
    return "".join(diff) or "(no textual change)"


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def is_background() -> bool:
    return current_origin() == "background_review"
# ---- END PLUGIN-COMPAT ----
