#!/usr/bin/env python3
"""Write-approval gate + pending store for memory and skill writes.

Background
----------
The agent writes to two persistent stores that survive across sessions:

  * **memory** — MEMORY.md / USER.md, small (~200 char) declarative entries
  * **skills** — SKILL.md + supporting files, potentially huge (10-100 KB)

Both stores are written from two origins:

  * **foreground** — a normal agent turn (user is present / chatting)
  * **background_review** — the self-improvement review fork that runs after a
    turn and autonomously decides what to save (the source of the
    "wrong assumptions" users complained about)

This module lets the user gate those writes per-subsystem with a boolean
``write_approval``:

  * ``false`` (default) — write freely (the pre-gate behaviour)
  * ``true``            — require approval: do not commit the write; either
    prompt inline (memory, interactive CLI only) or **stage** it to a pending
    store and surface it for the user to approve or reject out-of-band

The size asymmetry between memory and skills is real and unavoidable: a memory
entry can be reviewed inline in a chat bubble; a 100 KB SKILL.md cannot. So
the gate stages BOTH to disk, but review affordances differ by subsystem
(see ``hermes_cli`` slash handlers): memory shows full content, skills show
metadata + a one-line gist + a ``diff`` escape hatch (CLI/dashboard/file).

Staging is mandatory for background-origin writes (a daemon thread cannot
block on an interactive prompt) and for gateway sessions (no inline prompt
channel — review happens via ``/memory pending``). Foreground CLI memory
writes prompt inline via the dangerous-command approval callback; skill
writes always stage (too big to eyeball mid-loop).

Pending records live under ``<HERMES_HOME>/pending/{memory,skills}/<id>.json``
so they survive process restarts and can be reviewed from CLI, gateway, or the
web dashboard.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Subsystem identifiers
MEMORY = "memory"
SKILLS = "skills"
_SUBSYSTEMS = (MEMORY, SKILLS)

# Config key (per subsystem). A single boolean: the approval gate is OFF by
# default (writes flow freely, the pre-gate behaviour), and ON means stage /
# prompt every write for the user's approval. There is intentionally no third
# "block all writes" state — to disable a subsystem entirely use its own
# enable flag (e.g. ``memory.memory_enabled: false``).
CONFIG_KEY = "write_approval"
SKILL_MAX_PENDING_KEY = "write_approval_max_pending"
SKILL_TTL_DAYS_KEY = "write_approval_ttl_days"
DEFAULT_SKILL_MAX_PENDING = 100
DEFAULT_SKILL_TTL_DAYS = 30

_skill_state_guard = threading.Lock()
_skill_states: Dict[Path, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def write_approval_enabled(subsystem: str) -> bool:
    """Return whether the approval gate is enabled for ``subsystem``.

    Reads ``<subsystem>.write_approval`` from config.yaml. Defaults to
    ``False`` (gate off — writes flow freely) for any unset / invalid value so
    existing installs keep their current behaviour until the user opts in.
    """
    if subsystem not in _SUBSYSTEMS:
        return False
    try:
        from hermes_cli.config import load_config, cfg_get
        cfg = load_config()
        raw = cfg_get(cfg, subsystem, CONFIG_KEY, default=False)
    except Exception:
        return False
    return _normalize_enabled(raw)


def _normalize_enabled(value: Any) -> bool:
    """Coerce a config value to a bool. Default (unknown) is False (gate off).

    Accepts real bools and the usual truthy/falsey strings. YAML 1.1 parses
    bare ``on``/``off``/``yes``/``no`` as bools already, so the string branch
    is mostly for hand-edited configs.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"on", "true", "yes", "1", "approve", "enabled"}
    return False


# ---------------------------------------------------------------------------
# Pending store (file-backed)
# ---------------------------------------------------------------------------

def _pending_dir(subsystem: str) -> Path:
    return get_hermes_home() / "pending" / subsystem


def _skill_pending_state() -> Dict[str, Any]:
    directory = _pending_dir(SKILLS).resolve()
    with _skill_state_guard:
        state = _skill_states.get(directory)
        if state is None:
            state = {
                "lock": threading.RLock(),
                "expired_count": 0,
                "overflow_count": 0,
            }
            _skill_states[directory] = state
        return state


def _resolve_skill_pending_policy() -> tuple[int, float]:
    """Return the active bounded-retention policy for staged skill writes."""
    try:
        from hermes_cli.config import load_config, cfg_get

        cfg = load_config()
        max_pending_raw = cfg_get(cfg, SKILLS, SKILL_MAX_PENDING_KEY, default=DEFAULT_SKILL_MAX_PENDING)
        ttl_days_raw = cfg_get(cfg, SKILLS, SKILL_TTL_DAYS_KEY, default=DEFAULT_SKILL_TTL_DAYS)
    except Exception:
        return DEFAULT_SKILL_MAX_PENDING, float(DEFAULT_SKILL_TTL_DAYS)

    return (
        _normalize_positive_int(max_pending_raw, DEFAULT_SKILL_MAX_PENDING),
        _normalize_positive_number(ttl_days_raw, float(DEFAULT_SKILL_TTL_DAYS)),
    )


def _normalize_positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)) and _is_finite_number(value) and value > 0:
        return max(1, int(value))
    return default


def _normalize_positive_number(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)) and _is_finite_number(value) and value > 0:
        return float(value)
    return default


def _is_finite_number(value: int | float) -> bool:
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def _coerce_created_at(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and _is_finite_number(value):
        return float(value)
    return None


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def _effective_pending_id(path: Path, record: Dict[str, Any]) -> str:
    raw = record.get("id")
    return path.stem if not isinstance(raw, str) or raw != path.stem else raw


def _load_pending_entries(subsystem: str) -> List[Dict[str, Any]]:
    d = _pending_dir(subsystem)
    if not d.exists():
        return []

    entries: List[Dict[str, Any]] = []
    for path in d.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Skipping unreadable pending record: %s", path)
            continue
        if not isinstance(record, dict):
            logger.warning("Skipping unreadable pending record: %s", path)
            continue

        pending_id = _effective_pending_id(path, record)
        if subsystem == SKILLS and record.get("id") != pending_id:
            record = {**record, "id": pending_id}

        entries.append(
            {
                "record": record,
                "path": path,
                "pending_id": pending_id,
                "created_at": _coerce_created_at(record.get("created_at")),
                "mtime": _safe_mtime(path),
            }
        )
    return entries


def _pending_entry_order(entry: Dict[str, Any]) -> tuple[float, str]:
    sort_time = entry["created_at"] if entry["created_at"] is not None else entry["mtime"]
    return sort_time, entry["path"].name


def _unlink_snapshot_entry(entry: Dict[str, Any], reason: str) -> bool:
    path = entry["path"]
    pending_id = entry["pending_id"]
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return True
    except Exception as e:  # pragma: no cover - filesystem failure path
        logger.error("Failed to discard %s pending %s/%s: %s", reason, SKILLS, pending_id, e)
        return False


def _basic_pending_snapshot(subsystem: str) -> Dict[str, Any]:
    entries = _load_pending_entries(subsystem)
    if subsystem == MEMORY:
        records = [entry["record"] for entry in entries]
        records.sort(key=lambda record: record.get("created_at", 0))
    else:
        entries.sort(key=_pending_entry_order)
        records = [entry["record"] for entry in entries]
    return {
        "records": records,
        "expired_ids": [],
        "overflow_ids": [],
        "entries": entries,
    }


def _skill_pending_snapshot() -> Dict[str, Any]:
    entries = _load_pending_entries(SKILLS)
    if not entries:
        return {"records": [], "expired_ids": [], "overflow_ids": [], "entries": []}

    entries.sort(key=_pending_entry_order)
    max_pending, ttl_days = _resolve_skill_pending_policy()
    ttl_seconds = ttl_days * 86400.0
    now = time.time()

    active_entries: List[Dict[str, Any]] = []
    expired_ids: List[str] = []

    for entry in entries:
        created_at = entry["created_at"]
        if created_at is None or (now - created_at) <= ttl_seconds:
            active_entries.append(entry)
            continue
        if _unlink_snapshot_entry(entry, "expired"):
            expired_ids.append(entry["pending_id"])
        else:
            active_entries.append(entry)

    overflow_ids: List[str] = []
    overflow = max(0, len(active_entries) - max_pending)
    if overflow:
        removed_paths = set()
        for entry in active_entries[:overflow]:
            if _unlink_snapshot_entry(entry, "overflow"):
                overflow_ids.append(entry["pending_id"])
                removed_paths.add(entry["path"])
        if removed_paths:
            active_entries = [entry for entry in active_entries if entry["path"] not in removed_paths]

    if expired_ids or overflow_ids:
        affected_ids = expired_ids + overflow_ids
        logger.info(
            "Cleaned pending %s writes, expired=%d overflow=%d retained=%d affected_ids=%s",
            SKILLS,
            len(expired_ids),
            len(overflow_ids),
            len(active_entries),
            affected_ids,
        )

    return {
        "records": [entry["record"] for entry in active_entries],
        "expired_ids": expired_ids,
        "overflow_ids": overflow_ids,
        "entries": active_entries,
    }


def pending_snapshot(subsystem: str) -> Dict[str, Any]:
    """Return the active pending-record snapshot and any cleanup report."""
    if subsystem == SKILLS:
        state = _skill_pending_state()
        with state["lock"]:
            return _skill_pending_snapshot()
    return _basic_pending_snapshot(subsystem)


def pending_review_snapshot(subsystem: str) -> Dict[str, Any]:
    """Return a user-facing snapshot, including deferred stage cleanup counts."""
    if subsystem != SKILLS:
        return pending_snapshot(subsystem)

    state = _skill_pending_state()
    with state["lock"]:
        expired_count = state["expired_count"]
        overflow_count = state["overflow_count"]
        state["expired_count"] = 0
        state["overflow_count"] = 0
        try:
            snapshot = _skill_pending_snapshot()
        except Exception:
            state["expired_count"] += expired_count
            state["overflow_count"] += overflow_count
            raise
        snapshot["expired_count"] = expired_count + len(snapshot["expired_ids"])
        snapshot["overflow_count"] = overflow_count + len(snapshot["overflow_ids"])
        return snapshot


@contextmanager
def pending_operation_lock(subsystem: str):
    """Serialize skill cleanup with approval and rejection operations."""
    if subsystem == SKILLS:
        state = _skill_pending_state()
        with state["lock"]:
            yield
        return
    yield


def stage_write(subsystem: str, payload: Dict[str, Any],
                *, summary: str, origin: str) -> Dict[str, Any]:
    """Persist a pending write and return a short record describing it.

    Args:
        subsystem: ``memory`` or ``skills``.
        payload: the exact kwargs needed to replay the write when approved
            (e.g. ``{"action": "add", "target": "user", "content": "..."}``
            for memory, or the full ``skill_manage`` kwargs for skills).
        summary: a one-line human-readable description shown in pending lists.
            For skills this is the LLM/heuristic gist; for memory it can be the
            entry text itself.
        origin: ``foreground`` or ``background_review`` — recorded for audit.

    Returns a dict with ``id`` and metadata. Best-effort: on disk failure it
    logs and still returns a record (the write is simply lost, which is the
    safe failure for an approval gate — nothing is silently committed).
    """
    pid = uuid.uuid4().hex[:8]
    record = {
        "id": pid,
        "subsystem": subsystem,
        "action": payload.get("action", ""),
        "summary": (summary or "").strip(),
        "origin": origin or "foreground",
        "created_at": time.time(),
        "payload": payload,
    }
    try:
        d = _pending_dir(subsystem)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{pid}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as e:  # pragma: no cover - disk failure path
        logger.error("Failed to stage pending %s write: %s", subsystem, e, exc_info=True)
    else:
        if subsystem == SKILLS:
            state = _skill_pending_state()
            with state["lock"]:
                try:
                    snapshot = _skill_pending_snapshot()
                    state["expired_count"] += len(snapshot["expired_ids"])
                    state["overflow_count"] += len(snapshot["overflow_ids"])
                except Exception as e:  # pragma: no cover - maintenance failure path
                    logger.error("Failed to maintain pending %s writes: %s", subsystem, e, exc_info=True)
    return record


def list_pending(subsystem: str) -> List[Dict[str, Any]]:
    """Return all pending records for ``subsystem``, oldest first."""
    if subsystem == SKILLS:
        return pending_snapshot(SKILLS)["records"]
    return _basic_pending_snapshot(subsystem)["records"]


def get_pending(subsystem: str, pending_id: str) -> Optional[Dict[str, Any]]:
    """Return a single pending record by id, or None."""
    if subsystem == SKILLS:
        snapshot = pending_snapshot(SKILLS)
        for entry in snapshot["entries"]:
            if entry["pending_id"] == pending_id:
                return entry["record"]
        return None
    path = _pending_dir(subsystem) / f"{pending_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def discard_pending(subsystem: str, pending_id: str) -> bool:
    """Delete a pending record. Returns True if it existed."""
    with pending_operation_lock(subsystem):
        path = _pending_dir(subsystem) / f"{pending_id}.json"
        try:
            if path.exists():
                path.unlink()
                return True
        except Exception as e:  # pragma: no cover
            logger.error("Failed to discard pending %s/%s: %s", subsystem, pending_id, e)
        return False


def pending_count(subsystem: str) -> int:
    """Cheap count of pending records (for notification badges)."""
    if subsystem == SKILLS:
        return len(pending_snapshot(SKILLS)["records"])
    d = _pending_dir(subsystem)
    if not d.exists():
        return 0
    try:
        return sum(1 for _ in d.glob("*.json"))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Write origin
# ---------------------------------------------------------------------------

def current_origin() -> str:
    """Return the active write origin: ``foreground`` or ``background_review``.

    Reuses the skill-provenance ContextVar, which the background review fork
    already sets (see ``agent.background_review`` /
    ``AIAgent._spawn_background_review``). Foreground agent turns leave it at
    the default ``foreground``.
    """
    try:
        from tools.skill_provenance import get_current_write_origin
        return get_current_write_origin()
    except Exception:
        return "foreground"


def is_background() -> bool:
    return current_origin() == "background_review"


# ---------------------------------------------------------------------------
# Gate decision
# ---------------------------------------------------------------------------

class GateDecision:
    """Result of evaluating the write gate for a single write attempt.

    Exactly one of the boolean flags is True:
      * ``allow``  — proceed with the real write (gate off, or an inline
        approval was granted).
      * ``blocked`` — refuse the write (the user denied an inline approval
        prompt). ``message`` explains why; surface it to the agent.
      * ``stage``  — do not write; the caller should stage the payload via
        ``stage_write`` (gate on, and no inline prompt is available — gateway,
        background review, script, or any skill write). ``message`` is the
        user-facing "staged for approval" note.
    """

    __slots__ = ("allow", "blocked", "stage", "message")

    def __init__(self, *, allow=False, blocked=False, stage=False, message=""):
        self.allow = allow
        self.blocked = blocked
        self.stage = stage
        self.message = message


def evaluate_gate(subsystem: str, *, inline_summary: str = "",
                  inline_detail: str = "") -> GateDecision:
    """Decide what to do with a pending write for ``subsystem``.

    Args:
        subsystem: ``memory`` or ``skills``.
        inline_summary: short description used as the inline approval prompt
            header (memory foreground path only).
        inline_detail: full content shown in the inline prompt (memory entries
            are small; skills never take the inline path).

    Decision matrix:
        gate off (default)                    → allow (writes flow freely)
        gate on, memory + interactive CLI     → inline approve/deny prompt
        gate on, memory + gateway/script/bg   → stage
        gate on, skills (any origin)          → stage (too big to review inline)

    Note: there is no config-driven "blocked" outcome — the gate only ever
    delays a write for approval, never silently refuses it. ``blocked`` is
    still produced when the user *actively denies* an inline prompt.
    """
    if not write_approval_enabled(subsystem):
        return GateDecision(allow=True)

    background = is_background()

    # Skills always stage — a SKILL.md is too large to review inline, and a
    # background skill write happens in a daemon thread with no user present.
    if subsystem == SKILLS or background:
        where = "/skills pending" if subsystem == SKILLS else "/memory pending"
        return GateDecision(
            stage=True,
            message=(
                f"Staged for approval ({subsystem}.write_approval is on). "
                f"Not yet saved — review with {where}."
            ),
        )

    # Memory + foreground: if an interactive approval channel exists (a CLI
    # approval callback registered on this thread), prompt inline — entries
    # are small enough to show in full. Otherwise (gateway, script, batch,
    # no listener) stage instead of forcing a blind deny.
    if _interactive_approval_available():
        granted = _prompt_inline_memory_approval(inline_summary, inline_detail)
        if granted is True:
            return GateDecision(allow=True)
        if granted is False:
            return GateDecision(
                blocked=True,
                message="Memory write denied by user. The change was not saved.",
            )
        # granted is None → prompt failed; fall through to staging.

    return GateDecision(
        stage=True,
        message=(
            "Staged for approval (memory.write_approval is on). "
            "Not yet saved — review with /memory pending."
        ),
    )


def _interactive_approval_available() -> bool:
    """True when a foreground memory write can be approved inline.

    Inline prompting requires a per-thread approval callback registered by the
    interactive CLI (``tools.terminal_tool.set_approval_callback``). Every
    other surface stages instead:

    * **Gateway/API sessions** — the dangerous-command ``/approve`` round-trip
      lives in the pending-approval queue (``submit_pending`` +
      ``_await_gateway_decision``), which ``prompt_dangerous_approval`` never
      reaches; trying to prompt from a gateway session would hit the
      ``input()`` fallback and silently deny. Staging gives the user a real
      review affordance (``/memory pending``) instead.
    * Scripts, cron, and background threads — no user present.
    """
    try:
        from tools.terminal_tool import _get_approval_callback
        return _get_approval_callback() is not None
    except Exception:
        return False


def _prompt_inline_memory_approval(summary: str, detail: str) -> Optional[bool]:
    """Prompt the user inline to approve a memory write.

    Returns True (approved), False (denied), or None (no interactive prompt
    available / prompt failed → caller should stage instead).

    Reuses the per-thread CLI approval callback registered for dangerous
    commands (``tools.terminal_tool.set_approval_callback``). The callback is
    invoked directly — NOT via ``prompt_dangerous_approval`` — because that
    wrapper falls back to ``input()`` (deadlock-prone under prompt_toolkit,
    see #15216) and converts callback errors into a silent deny; here a
    failed prompt must stage the write instead.
    """
    try:
        from tools.terminal_tool import _get_approval_callback
    except Exception:
        return None

    callback = _get_approval_callback()
    if callback is None:
        # No interactive channel on this thread — stage rather than risk the
        # input() fallback (deadlock under prompt_toolkit, EOF-deny in tests).
        return None

    header = summary.strip() or "Save to memory?"
    body = detail.strip()
    description = f"Save to memory: {header}"
    command = body if body else header
    # Invoke the callback directly instead of via prompt_dangerous_approval:
    # that wrapper swallows callback exceptions into "deny", which would
    # silently refuse the write. Direct invocation lets a crashed prompt fall
    # back to staging (the gate only ever delays a write, never drops it).
    try:
        choice = callback(command, description, allow_permanent=False)
    except Exception as e:
        logger.error("Inline memory approval prompt failed: %s", e)
        return None

    if choice in {"once", "session"}:
        return True
    if choice == "deny":
        return False
    # Any other outcome (e.g. timeout that returns "deny" already handled) →
    # treat unknown as no-decision so we stage rather than silently drop.
    return None


# ---------------------------------------------------------------------------
# Skill-specific helpers (gist + diff for the review affordances)
# ---------------------------------------------------------------------------

def skill_gist(action: str, name: str, *, content: str = "",
               file_path: str = "", old_string: str = "",
               new_string: str = "") -> str:
    """Build a one-line human gist for a pending skill write.

    Heuristic, no model call — the gist surfaces enough to decide approve/reject
    in a chat bubble, while the full diff stays behind /skills diff (CLI/
    dashboard/file). For create/edit it pulls the frontmatter ``description:``;
    for patch/write_file it describes the size of the change.
    """
    if action in {"create", "edit"} and content:
        desc = _frontmatter_description(content)
        size = f"{len(content) // 1024 + 1} KB" if len(content) >= 1024 else f"{len(content)} chars"
        verb = "create" if action == "create" else "rewrite"
        if desc:
            return f"{verb} '{name}' — {desc} ({size})"
        return f"{verb} '{name}' ({size})"
    if action == "patch":
        target = file_path or "SKILL.md"
        removed = old_string.count("\n") + 1 if old_string else 0
        added = new_string.count("\n") + 1 if new_string else 0
        return f"patch '{name}' {target} (+{added}/-{removed} lines)"
    if action == "write_file":
        return f"write {file_path} in '{name}'"
    if action == "remove_file":
        return f"remove {file_path} from '{name}'"
    if action == "delete":
        return f"delete skill '{name}'"
    return f"{action} '{name}'"


def _frontmatter_description(content: str) -> str:
    """Extract the ``description:`` value from SKILL.md YAML frontmatter."""
    import re
    m = re.search(r"^description:\s*(.+)$", content, re.MULTILINE)
    if not m:
        return ""
    desc = m.group(1).strip().strip("'\"")
    return desc[:140]


def skill_pending_diff(record: Dict[str, Any]) -> str:
    """Build a full unified diff (or full content) for a staged skill write.

    Used by /skills diff <id> on a surface that can render it (CLI pager, web
    dashboard, or by opening the pending JSON file). For create this is the new
    file content; for edit/patch it is a unified diff against the current
    on-disk skill.
    """
    import difflib
    payload = record.get("payload", {})
    action = payload.get("action", "")
    name = payload.get("name", "")

    if action == "create":
        return (payload.get("content") or "")

    # Resolve current on-disk content for diffable actions.
    try:
        from tools.skill_manager_tool import _find_skill
    except Exception:
        _find_skill = None  # type: ignore

    current = ""
    target_label = "SKILL.md"
    if _find_skill is not None:
        found = _find_skill(name)
        if found:
            base = found["path"]
            if action == "edit":
                p = base / "SKILL.md"
            elif action in {"patch", "write_file"}:
                rel = payload.get("file_path") or "SKILL.md"
                p = base / rel
                target_label = rel
            else:
                p = base / "SKILL.md"
            try:
                if p.exists():
                    current = p.read_text(encoding="utf-8")
            except Exception:
                current = ""

    if action == "edit":
        new = payload.get("content") or ""
    elif action == "patch":
        old_s = payload.get("old_string") or ""
        new_s = payload.get("new_string") or ""
        new = current.replace(old_s, new_s) if current else f"(patch {old_s!r} → {new_s!r})"
    elif action == "write_file":
        new = payload.get("file_content") or ""
    elif action == "remove_file":
        return f"remove file: {payload.get('file_path')} from skill '{name}'"
    elif action == "delete":
        return f"delete skill '{name}'"
    else:
        return f"({action} on '{name}')"

    diff = difflib.unified_diff(
        current.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{target_label}",
        tofile=f"b/{target_label}",
    )
    text = "".join(diff)
    return text or "(no textual change)"
