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

This module lets the user gate **foreground** writes per-subsystem with a
boolean ``write_approval``. Background-review-origin writes always stage for
approval regardless of that foreground setting:

  * ``false`` (default) — foreground writes freely (the pre-gate behaviour)
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
import os
import re
import stat
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Subsystem identifiers
MEMORY = "memory"
SKILLS = "skills"
_SUBSYSTEMS = (MEMORY, SKILLS)
_PENDING_ID_RE = re.compile(r"^[0-9a-f]{8}$")

# Config key (per subsystem). The boolean gates foreground writes: OFF by
# default preserves foreground pre-gate behaviour, and ON means stage / prompt
# foreground writes for the user's approval. Background-review-origin writes
# always stage. There is intentionally no third "block all writes" state — to
# disable a subsystem entirely use its own enable flag (e.g.
# ``memory.memory_enabled: false``).
CONFIG_KEY = "write_approval"


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def write_approval_enabled(subsystem: str) -> bool:
    """Return whether the approval gate is enabled for ``subsystem``.

    Reads ``<subsystem>.write_approval`` from config.yaml. Defaults to
    ``False`` (foreground gate off — foreground writes flow freely) for any
    unset / invalid value so existing installs keep their foreground behaviour
    until the user opts in. Background-review-origin writes are staged before
    this config is consulted.
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


def _valid_subsystem(subsystem: str) -> bool:
    return subsystem in _SUBSYSTEMS


def _safe_pending_dir(subsystem: str) -> Optional[Path]:
    """Return the private pending directory, rejecting symlinked queue roots."""
    if not _valid_subsystem(subsystem):
        return None
    directory = _pending_dir(subsystem)
    # The queue and its immediate parent are application-owned directories.
    # Following either as a symlink would send proposals or claims outside the
    # configured Hermes home.
    for candidate in (directory, directory.parent):
        if _is_path_redirect(candidate):
            logger.error("Refusing symlinked pending directory: %s", candidate)
            return None
    return directory


def _is_path_redirect(path: Path) -> bool:
    """Return whether an existing path can redirect queue writes elsewhere.

    ``Path.is_junction`` is only available on newer Python versions.  The
    Windows reparse-point attribute covers junctions on the older supported
    runtimes too.  A missing directory is safe to create; an unreadable
    existing path fails closed because its type cannot be established.
    """
    try:
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    except FileNotFoundError:
        return False
    except OSError:
        logger.error("Could not inspect pending directory redirect state: %s", path)
        return True


def valid_pending_id(pending_id: str) -> bool:
    """True only for IDs issued by :func:`stage_write`.

    Pending IDs are part of slash-command input, so accepting a path-like ID
    would let callers escape the per-subsystem pending directory on Windows or
    POSIX.  Keep the current eight-lowercase-hex creation contract explicit.
    """
    return isinstance(pending_id, str) and bool(_PENDING_ID_RE.fullmatch(pending_id))


def _pending_path(subsystem: str, pending_id: str) -> Optional[Path]:
    if not _valid_subsystem(subsystem) or not valid_pending_id(pending_id):
        return None
    directory = _safe_pending_dir(subsystem)
    if directory is None:
        return None
    return directory / f"{pending_id}.json"


def _valid_record(record: Any, subsystem: str, pending_id: str) -> bool:
    """Return whether a disk record still belongs to this queue entry."""
    return (
        isinstance(record, dict)
        and record.get("id") == pending_id
        and record.get("subsystem") == subsystem
        and isinstance(record.get("payload"), dict)
    )


def stage_write(subsystem: str, payload: Dict[str, Any],
                *, summary: str, origin: str) -> Optional[Dict[str, Any]]:
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

    Returns a dict with ``id`` and metadata, or ``None`` when the proposal
    cannot be persisted.  A lost proposal must never be reported to a caller
    as successfully staged.
    """
    if not _valid_subsystem(subsystem):
        logger.error("Refusing to stage unknown pending subsystem: %r", subsystem)
        return None
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
        from utils import atomic_json_write

        directory = _safe_pending_dir(subsystem)
        if directory is None:
            return None
        # Pending records are new immutable queue entries.  Publish through a
        # private sibling plus a no-clobber hard link so an injected final-path
        # symlink is never followed by atomic_json_write's normal symlink
        # preservation behavior.
        destination = directory / f"{pid}.json"
        staging = directory / f".{pid}.{uuid.uuid4().hex}.stage"
        atomic_json_write(staging, record)
        try:
            if staging.is_symlink() or not staging.is_file():
                return None
            os.link(staging, destination)
        finally:
            try:
                staging.unlink()
            except OSError:
                pass
    except Exception as e:  # pragma: no cover - disk failure path
        logger.error("Failed to stage pending %s write: %s", subsystem, e, exc_info=True)
        return None
    return record


def list_pending(subsystem: str) -> List[Dict[str, Any]]:
    """Return all pending records for ``subsystem``, oldest first."""
    d = _safe_pending_dir(subsystem)
    if d is None or not d.exists():
        return []
    records: List[Dict[str, Any]] = []
    for p in d.glob("*.json"):
        if p.is_symlink() or not p.is_file() or not valid_pending_id(p.stem):
            continue
        try:
            record = json.loads(p.read_text(encoding="utf-8"))
            if _valid_record(record, subsystem, p.stem):
                records.append(record)
        except Exception:
            logger.warning("Skipping unreadable pending record: %s", p)
    records.sort(key=lambda r: r.get("created_at", 0))
    return records


def get_pending(subsystem: str, pending_id: str) -> Optional[Dict[str, Any]]:
    """Return a single pending record by id, or None."""
    path = _pending_path(subsystem, pending_id)
    if path is None or not path.exists() or path.is_symlink():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        return record if _valid_record(record, subsystem, pending_id) else None
    except Exception:
        return None


def discard_pending(subsystem: str, pending_id: str) -> bool:
    """Delete a pending record. Returns True if it existed."""
    path = _pending_path(subsystem, pending_id)
    if path is None or path.is_symlink():
        return False
    try:
        if path.exists():
            path.unlink()
            return True
    except Exception as e:  # pragma: no cover
        logger.error("Failed to discard pending %s/%s: %s", subsystem, pending_id, e)
    return False


PendingClaim = Tuple[Dict[str, Any], Path]


def claim_pending(subsystem: str, pending_id: str) -> Optional[PendingClaim]:
    """Atomically remove one actionable proposal from the approval queue.

    The source is renamed to a unique sibling claim file with raw
    :func:`os.replace`.  Claim publication intentionally has no copy fallback:
    a claimant that did not win the one-step rename must not apply the payload.
    A stranded claim is non-actionable, which is safer than automatic replay
    after an interrupted mutation.
    """
    source = _pending_path(subsystem, pending_id)
    if source is None or source.is_symlink() or not source.is_file():
        return None
    claim = source.with_name(f".{pending_id}.{uuid.uuid4().hex}.claim")
    try:
        os.replace(source, claim)
    except (FileNotFoundError, OSError) as e:
        logger.info("Pending %s/%s could not be claimed: %s", subsystem, pending_id, e)
        return None
    if claim.is_symlink() or not claim.is_file():
        logger.error("Claimed pending %s/%s is not a regular file", subsystem, pending_id)
        return None
    try:
        record = json.loads(claim.read_text(encoding="utf-8"))
    except Exception as e:
        # Keep unreadable claimed input non-actionable for manual recovery.
        logger.error("Claimed pending %s/%s is unreadable: %s", subsystem, pending_id, e)
        return None
    if not _valid_record(record, subsystem, pending_id):
        logger.error("Claimed pending %s/%s has invalid content", subsystem, pending_id)
        return None
    return record, claim


def release_claim(subsystem: str, pending_id: str, claim_path: Path) -> Optional[bool]:
    """Return a known-failed claim without overwriting a new queue record.

    Returns ``True`` when the claim was requeued and removed, ``False`` when
    no actionable record was published, and ``None`` when the requeue was
    published but only stale-claim cleanup failed.  The latter remains safe to
    retry because the raw pending JSON is already actionable again.
    """
    destination = _pending_path(subsystem, pending_id)
    if destination is None or destination.exists() or claim_path.is_symlink():
        return False
    expected_prefix = f".{pending_id}."
    if claim_path.parent != destination.parent or not (
        claim_path.name.startswith(expected_prefix) and claim_path.name.endswith(".claim")
    ):
        return False
    try:
        # ``os.replace`` would overwrite a concurrently created pending record
        # after the exists() check.  A hard link is an atomic no-clobber publish
        # on the same queue filesystem; only then can the private claim vanish.
        os.link(claim_path, destination)
    except OSError as e:
        logger.error("Failed to release pending claim %s/%s: %s", subsystem, pending_id, e)
        return False
    try:
        claim_path.unlink()
        return True
    except OSError as e:
        logger.error("Released pending claim %s/%s but could not clean it up: %s", subsystem, pending_id, e)
        return None


def complete_claim(claim_path: Path) -> bool:
    """Finalize a known terminal claim without making it actionable again."""
    if claim_path.is_symlink():
        return False
    try:
        claim_path.unlink()
        return True
    except OSError as e:
        logger.error("Failed to clean up terminal pending claim %s: %s", claim_path, e)
        return False


def pending_count(subsystem: str) -> int:
    """Cheap count of pending records (for notification badges)."""
    d = _safe_pending_dir(subsystem)
    if d is None or not d.exists():
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
        background-review origin (any config) → stage
        gate off (foreground default)         → allow (writes flow freely)
        gate on, memory + interactive CLI     → inline approve/deny prompt
        gate on, memory + gateway/script      → stage
        gate on, skills (any origin)          → stage (too big to review inline)

    Note: there is no config-driven "blocked" outcome — the gate only ever
    delays a write for approval, never silently refuses it. ``blocked`` is
    still produced when the user *actively denies* an inline prompt.
    """
    background = is_background()

    # Autonomous review forks (including the curator) must never persist a
    # model-proposed memory or skill change merely because the optional
    # foreground approval gate is off.  The origin comes from the host-owned
    # ContextVar, not tool arguments, so it is a trustworthy boundary shared
    # by both background actors.
    if background:
        where = "/skills pending" if subsystem == SKILLS else "/memory pending"
        return GateDecision(
            stage=True,
            message=(
                "Staged for approval (background review writes require approval). "
                f"Not yet saved — review with {where}."
            ),
        )

    if not write_approval_enabled(subsystem):
        return GateDecision(allow=True)

    # Skills always stage — a SKILL.md is too large to review inline.
    if subsystem == SKILLS:
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
