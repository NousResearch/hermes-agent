#!/usr/bin/env python3
"""Write-approval gate + pending store for memory and skill writes.

A per-subsystem boolean ``write_approval`` gates the agent's cross-session writes —
**memory** (MEMORY.md / USER.md) and **skills** (SKILL.md + files) — from either
origin (**foreground** turn or **background_review** fork). ``false`` (default)
writes freely; ``true`` never commits directly: it prompts inline (memory,
interactive CLI only) or **stages** the write under
``<HERMES_HOME>/pending/{memory,skills}/<id>.json`` for out-of-band review.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
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
_SUBSYSTEMS = (MEMORY, SKILLS)

# Per-subsystem config key. Intentionally a single boolean with no "block all writes"
# state — to disable a subsystem use its own enable flag (e.g. ``memory.memory_enabled``).
CONFIG_KEY = "write_approval"
_TRUTHY_STRINGS = frozenset({"on", "true", "yes", "1", "approve", "enabled"})


# --- Config resolution ---

def write_approval_enabled(subsystem: str) -> bool:
    """Read ``<subsystem>.write_approval``; any unset/invalid value means gate off."""
    if subsystem not in _SUBSYSTEMS:
        return False
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


def stage_write(subsystem: str, payload: dict[str, Any], *, summary: str, origin: str) -> dict[str, Any]:
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
        atomic_json_write(_pending_path(subsystem, pid), record)
    except Exception as e:  # pragma: no cover - disk failure path
        logger.error("Failed to stage pending %s write: %s", subsystem, e, exc_info=True)
    return record


def list_pending(subsystem: str) -> list[dict[str, Any]]:
    """Return all pending records for ``subsystem``, oldest first."""
    records: list[dict[str, Any]] = []
    for p in _pending_files(subsystem):
        try:
            record = json.loads(p.read_text(encoding="utf-8-sig"))
            if not isinstance(record, dict):
                raise ValueError(f"expected a JSON object, got {type(record).__name__}")
            records.append(record)
        except Exception:
            logger.warning("Skipping unreadable pending record: %s", p)
    records.sort(key=lambda r: r.get("created_at", 0))
    return records


def get_pending(subsystem: str, pending_id: str) -> Optional[dict[str, Any]]:
    """Return a single pending record by id, or None."""
    path = _pending_path(subsystem, pending_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else None
    except Exception:
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


def _slash_review_surface() -> bool:
    """Whether the ACTIVE turn has a human who can type ``/<subsystem> pending``: the interactive
    CLI, TUI/desktop (exec route), and chat-gateway sessions all answer it now. Headless worker
    contexts — cron (``HERMES_CRON_SESSION``), kanban (``HERMES_KANBAN_TASK``) and the unattended
    programmatic platforms (api_server, webhook delivery) — have nobody at a prompt, so the staged
    hint must not name a command nobody can deliver (#98330); it names the pending dir instead."""
    def _env(name: str) -> str:
        try:
            from gateway.session_context import get_session_env
            return get_session_env(name, "") or ""
        except Exception:  # standalone/tests: process env is the fallback
            return os.environ.get(name, "") or ""
    if _env("HERMES_KANBAN_TASK").strip():
        return False
    try:
        from utils import is_truthy_value
        if is_truthy_value(_env("HERMES_CRON_SESSION")):
            return False
    except Exception:
        pass
    platform = (_env("HERMES_SESSION_PLATFORM") or os.environ.get("HERMES_PLATFORM", "")).strip().lower()
    return platform not in {"webhook", "msgraph_webhook", "api_server"}


def _staged(subsystem: str) -> GateDecision:
    command = f"/{subsystem} pending"
    if _slash_review_surface():
        where = f"review with {command}"
    else:
        where = (f"pending records live in {_pending_path(subsystem, '').parent} "
                 f"(review from an interactive CLI session with {command})")
    return GateDecision(stage=True, message=(f"Staged for approval ({subsystem}.write_approval is on). "
                                             f"Not yet saved — {where}."))


def evaluate_gate(subsystem: str, *, inline_summary: str = "", inline_detail: str = "") -> GateDecision:
    """Decide what to do with a pending write: gate off → allow; gate on + skills (any origin) or
    background → stage; gate on + memory + foreground → inline prompt when an interactive channel
    exists, else stage. The gate only ever delays a write, never silently refuses it; ``blocked``
    is produced only when the user actively denies the inline prompt."""
    if not write_approval_enabled(subsystem):
        return GateDecision(allow=True)
    # Skills are too big to review inline; a background write runs in a daemon thread with no user.
    if subsystem == SKILLS or current_origin() == "background_review":
        return _staged(subsystem)
    granted = _prompt_inline_memory_approval(inline_summary, inline_detail)
    if granted is None:
        return _staged(MEMORY)
    if granted:
        return GateDecision(allow=True)
    return GateDecision(blocked=True, message="Memory write denied by user. The change was not saved.")


def _prompt_inline_memory_approval(summary: str, detail: str) -> Optional[bool]:
    """Prompt inline for a memory write: True approved, False denied, None → stage. Uses the per-thread
    CLI approval callback (``tools.terminal_tool.set_approval_callback``) directly, not
    ``prompt_dangerous_approval``: that wrapper falls back to ``input()`` (deadlock-prone under
    prompt_toolkit; silent deny in gateway sessions) and turns callback errors into a deny, whereas
    here a missing channel or failed prompt must stage instead.

    ``hermes chat -q`` (and so every kanban worker), cron and unattended platforms can register a callback too,
    but nobody answers it: stage at once instead of waiting the approval timeout. See #15216.
    """
    from tools.approval_context import _no_user_can_answer
    if _no_user_can_answer():
        return None
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


def _staged_base(name: str, target_label: str, staged: Optional[dict[str, dict[str, str]]]) -> str:
    """What an op diffs against: the content earlier ops of the same batch left in this file,
    else the on-disk copy (empty for a file that does not exist yet)."""
    files = (staged or {}).get(name)
    if files and target_label in files:
        return files[target_label]
    skill_dir = _find_skill_path(name)
    if not skill_dir:
        return ""
    with suppress(Exception):
        p = skill_dir / target_label
        if p.exists():
            return p.read_text(encoding="utf-8-sig")
    return ""


def skill_pending_diff(
    record: dict[str, Any], staged: Optional[dict[str, dict[str, str]]] = None
) -> str:
    """Full content (create) or unified diff vs. the base file (edit/patch/write_file),
    rendered by /skills diff <id> on surfaces that can show it. ``staged`` carries the file
    contents the earlier ops of a batch left behind, so op *i* diffs against ops 1..*i-1* and
    disk is read only for files no earlier op touched. A staged batch renders each op's diff
    under its gist header (single-op path below is unchanged)."""
    payload = record.get("payload", {})
    action = payload.get("action", "")
    name = payload.get("name", "")
    if action == "batch":
        return _batch_pending_diff(payload)
    if action == "create":
        return payload.get("content") or ""
    if action not in {"edit", "patch", "write_file"}:
        return {"remove_file": f"remove file: {payload.get('file_path')} from skill '{name}'",
                "delete": f"delete skill '{name}'"}.get(action, f"({action} on '{name}')")

    # patch/write_file target a file inside the skill; edit always targets SKILL.md.
    target_label = "SKILL.md" if action == "edit" else (payload.get("file_path") or "SKILL.md")
    current = _staged_base(name, target_label, staged)

    if action == "patch":
        old_s, new_s = payload.get("old_string") or "", payload.get("new_string") or ""
        if not current:
            new = f"(patch {old_s!r} → {new_s!r})"
        else:
            # Fold through the same matcher approve will run, so the preview can't
            # fabricate a result the approve path would reject (repeated anchor without
            # replace_all, whitespace-only anchor, escape drift, old_string == new_string).
            folded, patch_err = _fold_patch(current, old_s, new_s, payload.get("replace_all"))
            if patch_err:
                return f"(patch would fail: {patch_err})"
            new = folded
    else:
        new = payload.get("content" if action == "edit" else "file_content") or ""
    label = str(target_label).lstrip('/')  # keep the a//b// join single-slashed (#134718)
    diff = difflib.unified_diff(current.splitlines(keepends=True), new.splitlines(keepends=True),
                                fromfile=f"a/{label}", tofile=f"b/{label}")
    return "".join(diff) or "(no textual change)"


def _fold_patch(base: str, old_string: str, new_string: str, replace_all: Any = False
                ) -> tuple[str, Optional[str]]:
    """Fold one patch op through ``fuzzy_find_and_replace`` — the SAME matcher the approve
    path runs (``_patch_skill`` / ``apply_skill_pending``) — so a previewed diff is exactly
    what an approval would commit, and a patch approval would reject (repeated anchor with
    ``replace_all`` unset, whitespace-only anchor, escape drift, ``old_string == new_string``)
    renders as an explicit failed-patch note instead of a fabricated folded result.

    Returns ``(folded_content, None)`` or ``(base, error)``; base is returned unchanged on
    error so the caller can still render the unchanged content, matching the engine's own
    ``(content, 0, None, error)`` contract. Local import: keeps write_approval importable
    without the tool chain (fuzzy_match is core-adjacent but guarded for safety anyway).
    """
    from tools.fuzzy_match import fuzzy_find_and_replace
    folded, match_count, _strategy, error = fuzzy_find_and_replace(
        base, old_string, str(new_string), bool(replace_all))
    if error or match_count == 0:
        return base, error or "Could not find a match for old_string in the file"
    return folded, None


def _fold_staged(staged: dict[str, dict[str, str]], action: str, name: str,
                 op: dict[str, Any]) -> None:
    """Record what one op leaves in the staged view the next op diffs against."""
    if not name:
        return
    if action == "delete":
        staged.pop(name, None)
        return
    label = "SKILL.md" if action == "edit" else (op.get("file_path") or "SKILL.md")
    files = staged.setdefault(name, {})
    if action in {"create", "edit"}:
        files["SKILL.md"] = op.get("content") or ""
    elif action == "write_file":
        files[label] = op.get("file_content") or ""
    elif action == "patch":
        base = _staged_base(name, label, staged)
        folded, patch_err = _fold_patch(base, op.get("old_string") or "",
                                        op.get("new_string") or "", op.get("replace_all"))
        if patch_err is None:
            files[label] = folded
        # On error the staged label keeps its prior content: approve would abort the batch
        # at this op, so nothing later in the preview should diff against a folded result
        # that will never be committed.
    elif action == "remove_file":
        files.pop(label, None)


def _batch_pending_diff(payload: dict[str, Any]) -> str:
    """Per-op diffs for a staged ``batch`` payload (skill_manage operations[]). Ops apply in
    order, so each op reuses the single-op path above with the earlier ops' staged content as
    its base (disk only for files no earlier op touched) — a patch to a skill an earlier op
    created shows a real diff, which is the case this renderer exists for."""
    ops = [op for op in (payload.get("operations") or []) if isinstance(op, dict)]
    total = len(ops)
    staged: dict[str, dict[str, str]] = {}
    parts = []
    for i, op in enumerate(ops):
        op_action, op_name = op.get("action", ""), op.get("name") or ""
        gist = skill_gist(op_action, op_name, content=op.get("content") or "",
                          file_path=op.get("file_path") or "",
                          old_string=op.get("old_string") or "",
                          new_string=op.get("new_string") or "")
        diff = skill_pending_diff({"payload": {**op, "name": op_name}}, staged)
        parts.append(f"## op {i + 1}/{total}: {gist}\n\n{diff}")
        _fold_staged(staged, op_action, op_name, op)
    return "\n\n".join(parts) or "(empty batch)"
