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

import contextlib
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

logger = logging.getLogger(__name__)

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows uses msvcrt
    fcntl = None
try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX uses fcntl
    msvcrt = None

# Subsystem identifiers
MEMORY = "memory"
SKILLS = "skills"
_SUBSYSTEMS = (MEMORY, SKILLS)
PENDING = "pending"
NEEDS_REVIEW = "needs_review"
_MAX_FAILURE_ERROR_CHARS = 2000

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


def _write_pending_record(path: Path, record: Dict[str, Any]) -> None:
    """Atomically replace one pending record while its record lock is held."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


@contextlib.contextmanager
def _pending_record_lock(subsystem: str, pending_id: str):
    """Serialize approval, failure disposition, and rejection for one record."""
    path = _pending_path(subsystem, pending_id)
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if fcntl is None and msvcrt is None:
        yield
        return
    with open(lock_path, "a+", encoding="utf-8") as fd:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
            return

        fd.seek(0, os.SEEK_END)
        if fd.tell() == 0:
            fd.write("0")
            fd.flush()
        fd.seek(0)
        msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            fd.seek(0)
            msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)


def pending_status(record: Dict[str, Any]) -> str:
    """Return a pending record's lifecycle status, including legacy records."""
    return record.get("status", PENDING)


def stage_write(subsystem: str, payload: Dict[str, Any], *, summary: str, origin: str) -> Dict[str, Any]:
    """Persist a pending write and return its record (``id`` + metadata). ``payload`` is the exact
    kwargs to replay the write on approval; ``origin`` is ``foreground`` or ``background_review``.
    Best-effort: on disk failure it logs and still returns a record — the write is lost, which is
    the safe failure for an approval gate (nothing silently committed)."""
    pid = uuid.uuid4().hex[:8]
    record = {
        "id": pid, "subsystem": subsystem, "action": payload.get("action", ""),
        "summary": (summary or "").strip(), "origin": origin or "foreground",
        "created_at": time.time(), "payload": payload, "status": PENDING,
    }
    try:
        path = _pending_path(subsystem, pid)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_pending_record(path, record)
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
        if not path.exists():
            return False
        with _pending_record_lock(subsystem, pending_id):
            if path.exists():
                path.unlink()
                return True
    except Exception as e:  # pragma: no cover
        logger.error("Failed to discard pending %s/%s: %s", subsystem, pending_id, e)
    return False


def process_pending(
    subsystem: str,
    pending_id: str,
    apply_fn,
    *,
    allow_needs_review: bool = False,
    track_failure: bool = False,
) -> Dict[str, Any]:
    """Apply one record while serializing its state transition.

    Legacy records without ``status`` are active. Memory failures are retained as
    ``needs_review`` so bulk approval cannot replay the same deterministic failure;
    an explicit approval by ID may retry that record after the underlying state is
    repaired. The record lock covers apply plus discard/retention, preventing a
    concurrent rejection or successful approval from being undone by a late write.
    """
    path = _pending_path(subsystem, pending_id)
    if not path.exists():
        return {"state": "missing"}
    try:
        with _pending_record_lock(subsystem, pending_id):
            if not path.exists():
                return {"state": "missing"}
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Unable to read pending %s/%s: %s", subsystem, pending_id, exc)
                return {"state": "unreadable", "error": "pending record could not be read"}

            status = pending_status(record)
            if status != PENDING and not (allow_needs_review and status == NEEDS_REVIEW):
                return {"state": "skipped", "record": record}

            try:
                ok, message = apply_fn(record)
            except Exception as exc:  # pragma: no cover - apply helpers return errors
                logger.error("Pending %s/%s application failed: %s", subsystem, pending_id, exc, exc_info=True)
                ok, message = False, str(exc)
            if ok:
                try:
                    path.unlink()
                except Exception as exc:  # pragma: no cover - filesystem failure
                    logger.error("Applied pending %s/%s but could not discard it: %s", subsystem, pending_id, exc,
                                 exc_info=True)
                    return {
                        "state": "failed",
                        "record": record,
                        "error": "write applied, but its approval record could not be removed; it was not retried",
                    }
                return {"state": "applied", "record": record}

            message = str(message or "approval failed")
            if track_failure:
                try:
                    failure_count = int(record.get("failure_count", 0)) + 1
                except (TypeError, ValueError):
                    failure_count = 1
                record = {
                    **record,
                    "status": NEEDS_REVIEW,
                    "failure_count": failure_count,
                    "last_error": message[:_MAX_FAILURE_ERROR_CHARS],
                    "failed_at": time.time(),
                }
                try:
                    _write_pending_record(path, record)
                except Exception as exc:  # pragma: no cover - filesystem failure
                    logger.error("Pending %s/%s failed and could not record disposition: %s", subsystem, pending_id,
                                 exc, exc_info=True)
            return {"state": "failed", "record": record, "error": message}
    except Exception as exc:  # pragma: no cover - lock/filesystem failure
        logger.error("Failed to process pending %s/%s: %s", subsystem, pending_id, exc, exc_info=True)
        return {"state": "failed", "error": str(exc)}


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


def _staged(subsystem: str) -> GateDecision:
    where = "/skills pending" if subsystem == SKILLS else "/memory pending"
    return GateDecision(stage=True, message=(f"Staged for approval ({subsystem}.write_approval is on). "
                                             f"Not yet saved — review with {where}."))


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
        choice = callback(detail.strip() or header, f"Save to memory: {header}", allow_permanent=False)
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
