#!/usr/bin/env python3
"""Write-approval gate + pending store for memory and skill writes.

A per-subsystem boolean ``write_approval`` gates the agent's cross-session writes —
**memory** (MEMORY.md / USER.md) and **skills** (SKILL.md + files) — from either
origin (**foreground** turn or **background_review** fork). The gate is enforced
by the subsystem tools that own these writes (``memory``, ``skill_manage``);
the generic file tools (``write_file``/``patch``) do not consult it (#60440
tracks closing that gap). ``false`` (default) writes freely; ``true`` never
commits directly: it prompts inline (memory, interactive CLI only) or **stages**
the write under
``<HERMES_HOME>/pending/{memory,skills}/<id>.json`` for out-of-band review.

Every surface decides a staged write through :func:`resolve_pending` (GUIs first take
:func:`preview_pending`). A decision claims the record, applies only while each target still
equals the revision it is checked against (the base recorded at staging, or for a previewed skill
write the revisions the reviewer saw), and leaves a durable outcome under
``<HERMES_HOME>/pending/resolved/<subsystem>/<id>.json``. An approval that stopped mid-write is
``needs_review``: never applied again, listed by :func:`list_needs_review`, and closed by a human
who checked the files with an explicit ``reject`` of that id.
"""

from __future__ import annotations

import difflib
import errno
import hashlib
import json
import logging
import math
import os
import re
import stat
import time
import unicodedata
import uuid
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeGuard

from hermes_constants import get_hermes_home, hermes_home_key, mkdir_under_hermes_home
from utils import atomic_json_write
from utils.files import fsync_directory

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

_PENDING_ID_RE = re.compile(r"[0-9a-f]{8}")  # how stage_write mints ids: uuid4().hex[:8]
_MAX_RECORD_BYTES = 16 * 1024 * 1024


def _pending_dir(subsystem: str) -> Path:
    """``<HERMES_HOME>/pending/<subsystem>``; raises when a symlink inside the store leads elsewhere
    (a planted link would make reads, deletes and approvals act on files outside it)."""
    if subsystem not in _SUBSYSTEMS:
        raise ValueError(f"unknown write-approval subsystem {subsystem!r}")
    root = get_hermes_home().resolve() / "pending"
    d = root / subsystem
    if d.resolve() != d or any(os.path.lexists(p) and not p.is_dir() for p in (root, d)):
        raise ValueError("pending store is not a plain profile directory")
    return d


def _pending_path(subsystem: str, pending_id: str) -> Path:
    """Path of one pending record. Ids are typed by users (``/skills reject <id>``), so only the
    minted format is accepted: anything else (``../x``, ``/abs``, ``a/b``) raises instead of
    naming a file outside the store, and so does a record file that is a symlink."""
    if not isinstance(pending_id, str) or not _PENDING_ID_RE.fullmatch(pending_id):
        raise ValueError(f"invalid pending id {pending_id!r}")
    path = _pending_dir(subsystem) / f"{pending_id}.json"
    if path.is_symlink():
        raise ValueError(f"pending record {pending_id!r} is a symlink")
    return path


def _record_bytes(path: Path) -> bytes:
    """Read only a bounded, regular queue record, without following a link."""
    info = path.lstat()
    if info.st_size > _MAX_RECORD_BYTES:
        raise ValueError("pending record is too large")
    with _open_regular(path, info) as file:
        raw = file.read(_MAX_RECORD_BYTES + 1)
    if len(raw) > _MAX_RECORD_BYTES:
        raise ValueError("pending record is too large")
    return raw


def _read_record(path: Path) -> tuple[bytes, dict[str, Any]]:
    """A queue record's exact bytes and its JSON object; ValueError if it is not one."""
    raw = _record_bytes(path)
    record = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(record, dict):
        raise ValueError(f"expected a JSON object, got {type(record).__name__}")
    return raw, record


@contextmanager
def _try_claim_lock(pending: Path):
    """Try once to own a decision. The sidecar must not be replaced or unlinked while in use."""
    path = pending.with_name(f".{pending.stem}.lock")
    if os.path.lexists(path) and not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("invalid decision lock file")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("invalid decision lock file")
        try:
            if os.name == "nt":
                import msvcrt
                # Locks a byte at offset zero, even past EOF.
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if not (isinstance(exc, BlockingIOError) or
                    (os.name == "nt" and exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK})):
                raise
            yield False
        else:
            try:
                yield True
            finally:
                if os.name == "nt":
                    with suppress(OSError):
                        os.lseek(fd, 0, os.SEEK_SET)
                        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    finally:
        os.close(fd)


def _recover_abandoned_claim(subsystem: str, claimed: Path) -> None:
    """Recover only an unlocked claim, without waiting on a live decision."""
    with suppress(OSError, ValueError):
        if not os.path.lexists(claimed):
            return
        with _try_claim_lock(_pending_path(subsystem, claimed.stem)) as owned:
            if owned:
                _recover_locked(subsystem, claimed)


def _recover_locked(subsystem: str, claimed: Path) -> None:
    """No mark means no write began. A non-final or unreadable mark forbids replay."""
    with suppress(OSError, ValueError):
        if not stat.S_ISREG(claimed.lstat().st_mode):
            return
        pid, back = claimed.stem, claimed.with_suffix(".json")
        outcome = _read_outcome(subsystem, pid, strict=True)
        status = outcome.get("status")
        if status in _FINAL:
            claimed.unlink()
        elif status == _NEEDS_REVIEW:
            return  # the claimed record stays for the human who checks the result
        elif os.path.lexists(_outcome_path(subsystem, pid)):
            _write_outcome(subsystem, pid, _NEEDS_REVIEW, outcome, reason=_INTERRUPTED)
            logger.warning("write_approval: the approval of %s/%s stopped mid-write; marked needs_review",
                           subsystem, pid)
        elif not os.path.lexists(back):
            os.replace(claimed, back)


def _release_abandoned_claims(subsystem: str, d: Path) -> None:
    for p in d.glob("*.claimed"):
        if _PENDING_ID_RE.fullmatch(p.stem):
            _recover_abandoned_claim(subsystem, p)


def _pending_files(subsystem: str) -> list:
    try:
        d = _pending_dir(subsystem)
    except ValueError:
        logger.warning("write_approval: ignoring the %s pending store", subsystem, exc_info=True)
        return []
    _release_abandoned_claims(subsystem, d)
    return [p for p in d.glob("*.json")
            if _PENDING_ID_RE.fullmatch(p.stem) and not p.is_symlink()] if d.exists() else []


def stage_write(subsystem: str, payload: dict[str, Any], *, summary: str, origin: str) -> dict[str, Any]:
    """Persist a pending write and return its record (``id`` + metadata). ``payload`` is the exact
    kwargs to replay the write on approval; ``origin`` is ``foreground`` or ``background_review``.
    Best-effort: on disk failure it logs and still returns a record — the write is lost, which is
    the safe failure for an approval gate (nothing silently committed). A skill record also
    carries ``base``: the revision of every file it reads or changes (see :func:`_skill_revisions`)."""
    pid = uuid.uuid4().hex[:8]
    record = {
        "id": pid, "subsystem": subsystem, "action": payload.get("action", ""),
        "summary": (summary or "").strip(), "origin": origin or "foreground",
        "created_at": time.time(), "payload": payload,
    }
    try:
        if subsystem == SKILLS:
            record["base"] = _skill_revisions(_skill_targets(payload))
        mkdir_under_hermes_home(_pending_dir(subsystem))
        for _ in range(16):
            # Share the per-id lock with resolve/recovery so no state transition can
            # hide a reservation between checking its names and publishing the record.
            pending = _pending_path(subsystem, pid)
            with _try_claim_lock(pending) as owned:
                if owned and not any(os.path.lexists(path) for path in (
                    pending, pending.with_suffix(".claimed"), _outcome_path(subsystem, pid),
                )):
                    record["id"] = pid
                    atomic_json_write(pending, record)
                    return record
            pid = uuid.uuid4().hex[:8]
        raise OSError("could not reserve a pending write id")
    except Exception as e:  # disk or skill-lookup failure, or a store refused as a link
        logger.error("Failed to stage pending %s write: %s", subsystem, e, exc_info=True)
        record["error"] = f"the pending {subsystem} store could not be written"  # callers report it, not "staged"
    return record


def list_pending(subsystem: str) -> list[dict[str, Any]]:
    """Return all pending records for ``subsystem``, oldest first. A record's ``id`` is its file
    name, never the id written inside it (approve and discard act on that id)."""
    records: list[dict[str, Any]] = []
    for p in _pending_files(subsystem):
        try:
            _, record = _read_record(p)
            records.append({**record, "id": p.stem})
        except Exception:
            logger.warning("Skipping unreadable pending record: %s", p)
    records.sort(key=lambda r: r.get("created_at") if isinstance(r.get("created_at"), (int, float)) else 0)
    return records


def get_pending(subsystem: str, pending_id: str) -> Optional[dict[str, Any]]:
    """Return a single pending record by id, or None (also for an id that was never minted)."""
    try:
        path = _pending_path(subsystem, pending_id)
    except ValueError:
        return None
    if not path.exists():
        return None
    try:
        _, data = _read_record(path)
        return {**data, "id": pending_id}
    except Exception:
        return None


def _claim_pending(subsystem: str, pending_id: str) -> Optional[_Claim]:
    """With the decision lock held, claim and bind the record to its exact bytes."""
    try:
        src = _pending_path(subsystem, pending_id)
        claimed = src.with_suffix(".claimed")
        if os.path.lexists(claimed) or not stat.S_ISREG(src.lstat().st_mode):
            return None
        os.replace(src, claimed)
    except (OSError, ValueError):
        return None
    try:
        fsync_directory(claimed.parent)
        raw, record = _read_record(claimed)
    except (OSError, ValueError):
        _finish_claim(subsystem, pending_id, applied=False)
        return None
    return _Claim(subsystem, pending_id, {**record, "id": pending_id}, hashlib.sha256(raw).hexdigest())


def _finish_claim(subsystem: str, pending_id: str, *, applied: bool) -> None:
    """Applied: drop the claimed record. Not applied: put it back in the queue."""
    try:
        src = _pending_path(subsystem, pending_id)
        claimed = src.with_suffix(".claimed")
        if applied:
            claimed.unlink(missing_ok=True)
        else:
            os.replace(claimed, src)
    except (OSError, ValueError):
        logger.warning("write_approval: could not finish the claim on %s/%s", subsystem, pending_id, exc_info=True)


def discard_pending(subsystem: str, pending_id: str) -> bool:
    """Delete a pending record. Returns True if it existed."""
    try:
        path = _pending_path(subsystem, pending_id)
        if path.exists():
            path.unlink()
            return True
    except ValueError:
        return False  # not a minted id: no such record
    except Exception as e:  # pragma: no cover
        logger.error("Failed to discard pending %s/%s: %s", subsystem, pending_id, e)
    return False


def pending_count(subsystem: str) -> int:
    """Cheap count of pending records (for notification badges)."""
    with suppress(Exception):
        return len(_pending_files(subsystem))
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
        where = (f"pending records live in {_pending_dir(subsystem)} "
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


def _skill_file_refused(skill_dir: Path, target_label: str) -> bool:
    """A staged file_path is the model's: apply the checks approval applies before reading it."""
    from tools.skill_manager_tool import _resolve_supporting_file
    return target_label != "SKILL.md" and _resolve_supporting_file(Path(skill_dir), target_label)[1] is not None


def _staged_base(name: str, target_label: str, staged: Optional[dict[str, dict[str, str]]]) -> str:
    """What an op diffs against: the content earlier ops of the same batch left in this file,
    else the on-disk copy (empty for a file that does not exist yet or may not be read)."""
    files = (staged or {}).get(name)
    if files and target_label in files:
        return files[target_label]
    skill_dir = _find_skill_path(name)
    if not skill_dir or _skill_file_refused(skill_dir, target_label):
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
    skill_dir = _find_skill_path(name)
    refused = skill_dir and _skill_file_refused(skill_dir, target_label)
    if refused:
        return f"(refused: {action} path {target_label!r} is outside the skill's allowed files)"
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
    diff = difflib.unified_diff(current.splitlines(keepends=True), new.splitlines(keepends=True),
                                fromfile=f"a/{target_label}", tofile=f"b/{target_label}")
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
        files[label] = ""  # later ops see the removal, not the original file


def _batch_pending_diff(payload: dict[str, Any]) -> str:
    """Per-op diffs for a staged ``batch`` payload (skill_manage operations[]), each under its
    gist header (see :func:`_skill_op_previews`)."""
    previews = _skill_op_previews(payload)
    total = len(previews)
    return "\n\n".join(f"## op {i}/{total}: {p['gist']}\n\n{p['diff']}"
                       for i, p in enumerate(previews, 1)) or "(empty batch)"


def _skill_op_previews(payload: dict[str, Any], snapshot: Optional[_SkillSnapshot] = None) -> list[dict[str, Any]]:
    """Gist + diff per op of a staged skill write. Ops apply in order, so each op diffs against what
    the earlier ops left (disk only for files no earlier op touched). That way a patch to a skill an
    earlier op created shows a real diff."""
    batch = payload.get("action") == "batch"
    raw_ops = payload.get("operations") if batch else [payload]
    ops = [op for op in raw_ops if isinstance(op, dict)] if isinstance(raw_ops, list) else []
    staged: dict[str, dict[str, str]] = {}
    planned_files: dict[tuple[str, str], Optional[str]] = {}
    existing = set(snapshot.existing) if snapshot else set()
    previews = []
    for op in ops:
        op_action, op_name = op.get("action", ""), op.get("name") or payload.get("name") or ""
        gist = skill_gist(op_action, op_name, content=op.get("content") or "",
                          file_path=op.get("file_path") or "",
                          old_string=op.get("old_string") or "",
                          new_string=op.get("new_string") or "")
        # A content-form patch rewrites SKILL.md, the target _skill_targets binds.
        render_action = "edit" if op_action == "patch" and op.get("content") else op_action
        planned_op = {**op, "action": render_action, "name": op_name}
        blocker = None
        if snapshot is not None:
            diff, blocker = _plan_skill_op(planned_op, snapshot, existing, planned_files)
        else:
            diff = skill_pending_diff({"payload": planned_op}, staged)
            _fold_staged(staged, render_action, op_name, op)
        previews.append({"action": op_action, "name": op_name, "file_path": op.get("file_path"),
                         "gist": gist, "diff": diff, "blocker": blocker})
    return previews


def _plan_skill_op(op: dict[str, Any], snapshot: _SkillSnapshot, existing: set[str],
                   files: dict[tuple[str, str], Optional[str]]) -> tuple[str, Optional[str]]:
    """Follow one op against a snapshot and ordered in-memory state, with a structured blocker.

    Dynamic write guards and security scans still run in the applier, never during preview.
    """
    from tools import skill_manager_tool as smt

    def blocked(reason: str) -> tuple[str, str]:
        return f"(refused: {reason})", reason

    action, name = op["action"], op["name"]
    targets = _skill_targets(op)
    if not targets or not isinstance(name, str) or not name:
        return blocked("This operation has no skill target.")
    key = targets[0]
    _, label = key
    if snapshot.revisions.get(name, {}).get(label) in _UNPREVIEWABLE:
        return blocked(f"Cannot safely preview '{name}/{label}'.")
    if action == "create" and name in existing:
        return blocked(f"Skill '{name}' already exists, so this create cannot apply.")
    if action != "create" and name not in existing:
        return blocked(f"Skill '{name}' does not exist.")
    if label not in ("SKILL.md", _WHOLE_SKILL) or action in ("write_file", "remove_file"):
        if error := smt._validate_file_path(op.get("file_path") or ""):
            return blocked(error)
    before = files.get(key, snapshot.text.get(key))
    present = before is not None if key in files else snapshot.revisions.get(name, {}).get(label) != _ABSENT
    if action in {"delete", "remove_file"}:
        if not present:
            return blocked(f"Target '{name}/{label}' does not exist.")
        if action == "delete":
            existing.discard(name)
            paths = snapshot.removals.get(name, [])
            return "\n".join([f"delete skill '{name}'", *[f"- {path}" for path in paths]]), None
        files[key] = None
        return f"remove file: {label} from skill '{name}'", None
    if action == "patch":
        if not present:
            return blocked(f"File not found: {label}")
        if not isinstance(op.get("old_string"), str) or not isinstance(op.get("new_string"), str):
            return blocked("Both patch strings are required.")
        after, error = _fold_patch(before or "", op["old_string"], op["new_string"], op.get("replace_all"))
        if error:
            return f"(patch would fail: {error})", error
    elif action in {"create", "edit", "write_file"}:
        after = op.get("file_content" if action == "write_file" else "content")
        if not isinstance(after, str):
            return blocked("The operation requires text content.")
    else:
        return blocked("Unknown skill operation.")
    if error := smt._validate_content_size(after, label=label):
        return blocked(error)
    if (action in ("create", "edit") or (action == "patch" and not op.get("file_path"))) and (
        error := smt._validate_frontmatter(after)
    ):
        return blocked(error)
    files[key] = after
    if action == "create":
        existing.add(name)
        return after, None
    diff = difflib.unified_diff((before or "").splitlines(keepends=True), after.splitlines(keepends=True),
                                fromfile=f"a/{label}", tofile=f"b/{label}")
    return "".join(diff) or "(no textual change)", None


# --- Staged base revisions (skills) ---
#
# A staged skill write records the revision of every file it reads or changes, so approval can
# refuse a write whose target changed after staging (an anchor that still matches would otherwise
# patch text the reviewer never saw). Memory needs no such map: a staged patch/remove pins its
# full ``matched_entry``, which the store checks under its own lock, and an add has no target.

_ABSENT = "absent"
_WHOLE_SKILL = "*"  # a delete's target: every file of the skill
_UNPREVIEWABLE = frozenset({"unreadable", "not-previewable", "refused"})


def _skill_targets(payload: dict[str, Any], *, text_only: bool = False,
                   destructive_only: bool = False) -> list[tuple[str, str]]:
    """``(skill name, file)`` each op reads or changes, in op order."""
    ops = payload.get("operations") if payload.get("action") == "batch" else [payload]
    targets = []
    for op in ops if isinstance(ops, list) else []:
        if not isinstance(op, dict):
            continue
        action = op.get("action")
        if text_only and action not in {"patch", "edit", "write_file"}:
            continue
        if destructive_only and action not in {"delete", "remove_file"}:
            continue
        if action == "delete":
            label = _WHOLE_SKILL
        elif action in {"create", "edit"} or (action == "patch" and op.get("content")):
            label = "SKILL.md"
        else:
            label = op.get("file_path") or "SKILL.md"
        targets.append((str(op.get("name") or payload.get("name") or ""), Path(str(label)).as_posix()))
    return targets


def _ambiguous_skill_targets(targets: list[tuple[str, str]]) -> bool:
    """Do not guess whether differently cased or normalized labels alias on this volume."""
    seen: dict[tuple[str, str], tuple[str, str]] = {}

    def normalized(value: str) -> str:
        return unicodedata.normalize("NFC", value).casefold()

    for name, label in targets:
        key = (normalized(name), "/".join(normalized(part.rstrip(" .")) for part in label.split("/")))
        if seen.setdefault(key, (name, label)) != (name, label):
            return True
    return _aliased_skill_names({name for name, _ in targets if name}, normalized)


def _aliased_skill_names(names: set[str], normalized) -> bool:
    """Two spellings of one skill (``foo`` and ``cat/foo``, or a name found through another skill's
    frontmatter) would be checked and previewed apart while replay edits one folder. Same leaf name
    covers a skill the batch creates; same resolved folder covers the rest."""
    from tools import skill_manager_tool
    if len(names) < 2:
        return False
    if len({normalized(Path(name).name) for name in names}) < len(names):
        return True
    found = [skill_manager_tool._find_skill(name) for name in names]
    dirs = [Path(hit["path"]).resolve() for hit in found if hit]
    return len(set(dirs)) < len(dirs)


def _skill_revisions(targets: list[tuple[str, str]]) -> dict[str, dict[str, str]]:
    return _capture_skill_targets(targets).revisions


@dataclass(slots=True)
class _SkillSnapshot:
    revisions: dict[str, dict[str, str]]
    text: dict[tuple[str, str], str]
    existing: set[str]
    removals: dict[str, list[str]]


def _capture_skill_targets(targets: list[tuple[str, str]],
                           text_targets: Optional[set[tuple[str, str]]] = None) -> _SkillSnapshot:
    """Take each target once, before rendering. Hash the raw bytes, not re-encoded display text."""
    dirs: dict[str, Optional[Path]] = {}
    revisions: dict[str, dict[str, str]] = {}
    text = {}
    removals: dict[str, list[str]] = {}
    for name, label in targets:
        files = revisions.setdefault(name, {})
        if label in files:
            continue
        if name not in dirs:
            dirs[name] = _find_skill_path(name) if name else None
        skill_dir = dirs[name]
        if label == _WHOLE_SKILL and skill_dir is not None:
            files[label], removals[name] = _tree_snapshot(Path(skill_dir))
            continue
        revision, raw = _target_bytes(skill_dir, label)
        if text_targets and (name, label) in text_targets:
            if raw is not None:
                try:
                    # Match read_text(encoding='utf-8-sig') and its universal newlines.
                    text[name, label] = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
                except UnicodeDecodeError:
                    revision = "not-previewable"  # never matches a revision at approval
            elif revision == "unreadable":
                revision = "not-previewable"
        files[label] = revision
    return _SkillSnapshot(revisions, text, {name for name, path in dirs.items() if path is not None}, removals)


def _target_bytes(skill_dir: Optional[Path], label: str) -> tuple[str, Optional[bytes]]:
    if skill_dir is None:
        return _ABSENT, None
    if _skill_file_refused(skill_dir, label):
        return "refused", None
    try:
        # A profile or external skills root may be reached through a symlink. Refuse
        # redirects only within the resolved skill, not aliases in its ancestry.
        path = Path(skill_dir).resolve(strict=True) / label
        if path.resolve(strict=False) != path:
            return "refused", None
    except (OSError, RuntimeError):
        return "unreadable", None
    return _file_bytes(path)


def _file_bytes(path: Path) -> tuple[str, Optional[bytes]]:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return ("unreadable" if os.path.lexists(path) else _ABSENT), None
    except OSError:
        return "unreadable", None
    try:
        with _open_regular(path, info) as file:
            raw = file.read()
        return "sha256:" + hashlib.sha256(raw).hexdigest(), raw
    except OSError:
        return "unreadable", None


@contextmanager
def _open_regular(path: Path, info: os.stat_result):
    if not stat.S_ISREG(info.st_mode) or getattr(info, "st_reparse_tag", 0):
        raise OSError("not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as file:
        def fingerprint(s):
            # Windows path-based stat can infer execute bits from the extension;
            # fstat on the same handle does not have to report those bits.
            return (s.st_dev, s.st_ino, stat.S_IFMT(s.st_mode), s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        expected = fingerprint(info)
        if fingerprint(os.fstat(file.fileno())) != expected:
            raise OSError("file changed while opening")
        yield file
        if fingerprint(os.fstat(file.fileno())) != expected:
            raise OSError("file changed while reading")


def _tree_snapshot(skill_dir: Path) -> tuple[str, list[str]]:
    """Hash and display one no-follow manifest, including empty directories and link entries."""
    from tools.skill_manager_guards import _is_path_redirect

    digest = hashlib.sha256()
    removed: list[str] = []
    try:
        if _is_path_redirect(skill_dir):
            return "not-previewable", []
        skill_dir = skill_dir.resolve(strict=True)
        stack = [skill_dir]
        while stack:
            directory = stack.pop()
            info = directory.lstat()
            if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_reparse_tag", 0):
                raise OSError("not a plain skill directory")
            for path in sorted(directory.iterdir()):
                rel = path.relative_to(skill_dir).as_posix()
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    row = (rel, "link", os.readlink(path))
                    removed.append(rel + " (symlink)")
                elif stat.S_ISDIR(info.st_mode) and not getattr(info, "st_reparse_tag", 0):
                    row = (rel, "dir")
                    removed.append(rel + "/")
                    stack.append(path)
                else:
                    with _open_regular(path, info) as file:
                        row = (rel, "file", info.st_size, hashlib.file_digest(file, "sha256").hexdigest())
                    removed.append(rel)
                digest.update(json.dumps(row).encode("ascii") + b"\n")
    except (OSError, RuntimeError):
        return "not-previewable", []
    return "tree:v2:" + digest.hexdigest(), removed


def _is_revision_map(value: Any) -> TypeGuard[dict[str, dict[str, str]]]:
    return isinstance(value, dict) and all(
        isinstance(name, str) and isinstance(files, dict)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in files.items())
        for name, files in value.items())


def _changed_targets(expected: dict[str, dict[str, str]], actual: dict[str, dict[str, str]]) -> list[str]:
    """Every target whose revision is not ``expected`` now: ``skill/file``, or ``skill`` for a delete."""
    def flat(revisions):
        return {(name, label): rev for name, files in revisions.items() for label, rev in files.items()}
    want, have = flat(expected), flat(actual)
    return [name if label == _WHOLE_SKILL else f"{name}/{label}"
            for name, label in sorted(k for k in want.keys() | have.keys() if want.get(k) != have.get(k))]


def _skill_drift(expected: dict[str, dict[str, str]], actual: dict[str, dict[str, str]], since: str) -> Optional[str]:
    changed = _changed_targets(expected, actual)
    if not changed:
        return None
    return f"changed since it was {since}: {', '.join(changed)}. Nothing was applied; review it again."


def _target_subset(revisions: dict[str, dict[str, str]],
                   targets: list[tuple[str, str]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, label in targets:
        if label in revisions.get(name, {}):
            result.setdefault(name, {})[label] = revisions[name][label]
    return result


def _destructive_drift(base: Any, actual: dict[str, dict[str, str]],
                       targets: list[tuple[str, str]]) -> Optional[str]:
    if not targets:
        return None
    staged = _target_subset(base, targets) if _is_revision_map(base) else {}
    if any(label not in staged.get(name, {}) for name, label in targets):
        return "This removal lacks a staged base. Reject it and stage a new proposal."
    return _skill_drift(staged, _target_subset(actual, targets), "staged")


def _has_unpreviewable_targets(revisions: dict[str, dict[str, str]]) -> bool:
    return any(rev in _UNPREVIEWABLE for files in revisions.values() for rev in files.values())


# --- Resolver: one decision per pending write, from any surface ---

_DECISIONS = {"approve": "applied", "reject": "rejected"}
_FINAL = frozenset(_DECISIONS.values())
_APPLYING, _NEEDS_REVIEW = "applying", "needs_review"
_CLAIM_WAIT_SECONDS = 5.0  # a decision that finds the record claimed waits this long for the outcome
_CLAIM_POLL_SECONDS = 0.05
_INTERRUPTED = ("An earlier approval stopped after it began writing, so this write may be partly "
                "applied. It will not be applied again: check the result by hand.")
_CLOSED_INTERRUPTED = ("Its earlier approval stopped partway, so the files may already be partly changed, "
                       "and nothing was rolled back. Check them by hand.")
_NON_ATOMIC = "This batch cannot be applied atomically. Reject it and stage separate proposals."
_UNBASED = ("This write was staged before Hermes recorded what it changes. Preview it and approve "
            "with the preview's expect, or reject it.")


def _outcome_path(subsystem: str, pending_id: str) -> Path:
    """``<HERMES_HOME>/pending/resolved/<subsystem>/<id>.json``: a decision's durable outcome
    (status, timestamps, record hash; never the proposal). Same id and link rules as the queue."""
    _pending_dir(subsystem)  # checks the subsystem name and that its queue isn't a link
    if not isinstance(pending_id, str) or not _PENDING_ID_RE.fullmatch(pending_id):
        raise ValueError(f"invalid pending id {pending_id!r}")
    d = get_hermes_home().resolve() / "pending" / "resolved" / subsystem
    if d.resolve() != d or (os.path.lexists(d) and not d.is_dir()):
        raise ValueError(f"outcome store {d} is not a plain profile directory")
    path = d / f"{pending_id}.json"
    if path.is_symlink():
        raise ValueError(f"outcome record {pending_id!r} is a symlink")
    return path


def _read_outcome(subsystem: str, pending_id: str, *, strict: bool = False) -> dict[str, Any]:
    """The recorded outcome, or {} when there is none (or it cannot be read). ``strict``: only a
    missing or malformed outcome is {}; any other read error is raised, so a caller about to act on
    the outcome (recovery) leaves it alone instead of overwriting a final one."""
    try:
        data = json.loads(_record_bytes(_outcome_path(subsystem, pending_id)).decode("utf-8"))
    except FileNotFoundError:
        return {}
    except OSError:
        if strict:
            raise
        return {}
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _write_outcome(subsystem: str, pending_id: str, status: str, fields: dict[str, Any], **extra) -> None:
    """Atomic and fsynced (file and directory): an ``applying`` mark must survive a crash."""
    atomic_json_write(_outcome_path(subsystem, pending_id),
                      {**fields, "id": pending_id, "subsystem": subsystem, "status": status,
                       "decided_at": time.time(), **extra}, fsync_dir=True)


def _reply(subsystem: str, pending_id: Any, status: str, **fields) -> dict[str, Any]:
    return {"id": pending_id, "subsystem": subsystem, "status": status, **fields}


def _decided(subsystem: str, pending_id: str) -> Optional[dict[str, Any]]:
    """The reply for a write that was already decided, or None."""
    outcome = _read_outcome(subsystem, pending_id)
    status = outcome.get("status")
    if status in _FINAL:
        return _reply(subsystem, pending_id, "already_resolved", prior=status, decided_at=outcome.get("decided_at"))
    if status == _NEEDS_REVIEW:
        return _reply(subsystem, pending_id, _NEEDS_REVIEW, reason=outcome.get("reason") or _INTERRUPTED)
    return None


def _profile_identity() -> str:
    return hashlib.sha256(hermes_home_key().encode("utf-8", "surrogateescape")).hexdigest()[:16]


def _expect_refusal(expect: Any) -> Optional[str]:
    if not (isinstance(expect, dict) and isinstance(expect.get("record_sha256"), str)
            and _is_revision_map(expect.get("targets"))):
        return "malformed expect: pass the expect preview_pending returned, unchanged."
    if expect.get("profile") != _profile_identity():
        return "this preview was taken in another profile."
    return None


@dataclass(frozen=True, slots=True)
class _Claim:
    subsystem: str
    pending_id: str
    record: dict[str, Any]
    record_sha256: str

    def record_outcome(self, status: str, **extra) -> None:
        _write_outcome(self.subsystem, self.pending_id, status,
                       {"staged_at": self.record.get("created_at"), "record_sha256": self.record_sha256}, **extra)

    def refuse(self, reason: str) -> dict[str, Any]:
        """Back to the queue untouched, as a conflict."""
        _finish_claim(self.subsystem, self.pending_id, applied=False)
        return _reply(self.subsystem, self.pending_id, "conflict", reason=reason)


def resolve_pending(subsystem: str, pending_id: str, decision: str, *, expect: Optional[dict[str, Any]] = None,
                    memory_store=None, legacy_unchecked: bool = False,
                    close_interrupted: bool = False) -> dict[str, Any]:
    """Decide one pending write. Every surface (CLI, gateway, TUI, GUI) goes through here.

    ``decision`` is ``approve`` or ``reject``. The reply is ``{"id", "subsystem", "status", ...}``,
    where ``status`` is one of:

    - ``applied``, with ``result`` (what the applier returned);
    - ``rejected``;
    - ``already_resolved``, with ``prior`` (``applied`` or ``rejected``);
    - ``conflict``, with ``reason``: nothing was written, and a write that was pending still is;
    - ``needs_review``, with ``reason``: an earlier approval stopped partway. It is never applied
      again. A ``reject`` with ``close_interrupted=True`` and no ``expect`` closes it as ``rejected``
      and returns a ``warning`` that the files may be partly changed; nothing is rolled back. Only
      pass that for an id a person named, never for a bulk reject.

    A decision holds the record's OS lock from the claim until its outcome is saved, so recovery
    never touches a claim someone is still working on. Recovery covers a crashed or killed process,
    not a power cut. Stop processes from older versions, which don't take this lock, before
    upgrading a profile.

    An approval only writes while every target still matches the revision it is checked against,
    and it checks under the same locks the appliers take (the skill mutation locks, or the memory
    store's lock, which re-checks each pinned ``matched_entry``). Without ``expect`` that revision is
    the one recorded at staging. With ``expect`` from :func:`preview_pending`, the approval is also
    tied to the profile and the exact record bytes, and a non-destructive skill write is checked
    against the files the reviewer previewed instead. That is how a stacked proposal (files changed
    by an earlier approval) can still be approved once its diff against the current files has been
    shown. Anything that changes after the preview is a conflict. Delete and remove_file must always
    match their staged revisions. A reject checks the record but ignores changes to its targets. The
    origin and payload always come from the record, never from the caller.

    A non-destructive skill record staged before base revisions existed has nothing to check, so it
    is refused without ``expect`` unless the caller passes ``legacy_unchecked``. The slash commands
    pass it, so records queued by an older version stay approvable. A removal staged without a base
    is always refused: reject it and stage it again.
    """
    if decision not in _DECISIONS:
        raise ValueError(f"decision must be 'approve' or 'reject', not {decision!r}")
    try:
        pending = _pending_path(subsystem, pending_id)
    except ValueError as exc:
        return _reply(subsystem, pending_id, "conflict", reason=str(exc))
    if expect is not None and (refusal := _expect_refusal(expect)):
        return _reply(subsystem, pending_id, "conflict", reason=refusal)
    claimed = pending.with_suffix(".claimed")
    # Only allocate a lock for an existing proposal; recheck pending in case recovery just requeued it.
    if not (os.path.lexists(pending) or os.path.lexists(claimed) or os.path.lexists(pending)):
        return _decided(subsystem, pending_id) or _reply(
            subsystem, pending_id, "conflict", reason=f"no pending {subsystem} write with id '{pending_id}'.")
    deadline = time.monotonic() + _CLAIM_WAIT_SECONDS
    while True:
        with ExitStack() as lock:
            try:
                owned = lock.enter_context(_try_claim_lock(pending))
            except (OSError, ValueError):
                return _reply(subsystem, pending_id, "conflict", reason="cannot acquire the decision lock.")
            if owned:
                _recover_locked(subsystem, claimed)
                if (prior := _decided(subsystem, pending_id)) is not None:
                    if (prior["status"] == _NEEDS_REVIEW and decision == "reject" and close_interrupted
                            and expect is None):
                        return _close_interrupted(subsystem, pending_id, claimed)
                    return prior
                if decision == "approve" and subsystem == MEMORY and memory_store is None:
                    return _reply(subsystem, pending_id, "conflict", reason="memory store unavailable")
                claim = _claim_pending(subsystem, pending_id)
                if claim is None:
                    return _reply(subsystem, pending_id, "conflict", reason="the pending record cannot be read.")
                return _resolve_owned(claim, decision, expect, memory_store, legacy_unchecked)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _reply(subsystem, pending_id, "conflict", reason="being decided by another client.")
        time.sleep(min(_CLAIM_POLL_SECONDS, remaining))


def _resolve_owned(claim: _Claim, decision: str, expect: Optional[dict[str, Any]], memory_store,
                   legacy_unchecked: bool) -> dict[str, Any]:
    if expect is not None and expect["record_sha256"] != claim.record_sha256:
        return claim.refuse("the pending write changed since it was previewed. Preview it again.")
    if decision == "reject":
        return _reject_claimed(claim)
    payload = claim.record.get("payload")
    if not isinstance(payload, dict):
        return claim.refuse("the pending record has no payload.")
    if claim.subsystem == MEMORY:
        from tools import memory_tool
        return _apply_claimed(claim, lambda: memory_tool.apply_memory_pending(payload, memory_store))
    return _approve_skill_claim(claim, payload, expect, legacy_unchecked)


def _reject_claimed(claim: _Claim) -> dict[str, Any]:
    try:
        claim.record_outcome("rejected")
    except (OSError, ValueError) as exc:
        logger.warning("write_approval: could not record a reject", exc_info=True)
        return claim.refuse(f"the decision could not be recorded: {exc}")
    _finish_claim(claim.subsystem, claim.pending_id, applied=True)
    return _reply(claim.subsystem, claim.pending_id, "rejected")


def _close_interrupted(subsystem: str, pending_id: str, claimed: Path) -> dict[str, Any]:
    """Reject a ``needs_review`` write: record ``rejected`` (keeping why it stopped) and drop the claim.
    Writes nothing to the targets and undoes nothing."""
    try:
        outcome = _read_outcome(subsystem, pending_id, strict=True)  # a read error must not lose the reason
        _write_outcome(subsystem, pending_id, "rejected",
                       {k: outcome[k] for k in ("staged_at", "record_sha256") if k in outcome},
                       interrupted=outcome.get("reason") or _INTERRUPTED)
    except (OSError, ValueError) as exc:
        return _reply(subsystem, pending_id, _NEEDS_REVIEW, reason=f"the decision could not be recorded: {exc}")
    # A crash before this unlink is settled by _recover_abandoned_claim: the outcome is final.
    with suppress(OSError):
        claimed.unlink(missing_ok=True)
    return _reply(subsystem, pending_id, "rejected", warning=_CLOSED_INTERRUPTED)


def _non_atomic_batch(payload: dict[str, Any]) -> bool:
    """A batch with a delete among other ops: its rollback can't restore the deleted skill."""
    ops = payload.get("operations") if payload.get("action") == "batch" else None
    return (isinstance(ops, list) and len(ops) > 1
            and any(isinstance(op, dict) and op.get("action") == "delete" for op in ops))


def _approve_skill_claim(claim: _Claim, payload: dict[str, Any], expect: Optional[dict[str, Any]],
                         legacy_unchecked: bool) -> dict[str, Any]:
    from tools import skill_manager_tool
    with ExitStack() as locks:
        try:
            refusal = _skill_claim_refusal(claim, payload, expect, legacy_unchecked, locks)
        except OSError as exc:  # a lookup or lock failure: nothing was written yet
            logger.warning("write_approval: could not check %s/%s", claim.subsystem, claim.pending_id, exc_info=True)
            refusal = f"the skill files could not be checked: {exc}"
        if refusal:
            return claim.refuse(refusal)
        origin = claim.record.get("origin") or "foreground"
        return _apply_claimed(
            claim, lambda: json.loads(skill_manager_tool.apply_skill_pending(payload, origin=origin)))


def _skill_claim_refusal(claim: _Claim, payload: dict[str, Any], expect: Optional[dict[str, Any]],
                         legacy_unchecked: bool, locks: ExitStack) -> Optional[str]:
    """Why this skill write can't be applied now, or None. Takes the skill locks on ``locks``, so
    the targets are checked under the same locks the applier then runs under."""
    if _non_atomic_batch(payload):
        return _NON_ATOMIC
    targets = _skill_targets(payload)
    if _ambiguous_skill_targets(targets):
        return "Use one consistent spelling for every skill name and file path in this batch."
    base = claim.record.get("base")
    if expect is not None:
        expected, since = expect["targets"], "previewed"
    elif base is not None:
        expected, since = base, "staged"
    elif legacy_unchecked:
        expected, since = None, ""
    else:
        return _UNBASED
    if expected is not None and not _is_revision_map(expected):
        return "the pending record's base revisions are malformed."
    locks.enter_context(_target_locks(targets))
    actual = _skill_revisions(targets)
    if _has_unpreviewable_targets(actual):
        return "A skill target cannot be safely previewed or checked."
    if (drift := _destructive_drift(base, actual, _skill_targets(payload, destructive_only=True))):
        return drift
    if expected is not None and (drift := _skill_drift(expected, actual, since)):
        return drift
    return None


def _target_locks(targets: list[tuple[str, str]]):
    from tools import skill_manager_tool
    # A name the applier rejects gets no lock file.
    names = {name for name, _ in targets
             if name and skill_manager_tool._validate_name(Path(name).name) is None}
    return skill_manager_tool._skill_mutation_locks(names)


def _apply_claimed(claim: _Claim, apply) -> dict[str, Any]:
    """Mark the claim ``applying`` durably, run the applier, record the outcome. A refusal from the
    applier (moved pin, char limit, security scan) is a conflict: the mark is cleared and the record
    goes back to the queue. An exception or a failed rollback leaves the write's state unknown:
    ``needs_review``."""
    subsystem, pid = claim.subsystem, claim.pending_id
    try:
        claim.record_outcome(_APPLYING)
    except (OSError, ValueError) as exc:
        logger.warning("write_approval: could not mark %s/%s applying", subsystem, pid, exc_info=True)
        return claim.refuse(f"the decision could not be recorded: {exc}")
    try:
        result = apply()
    except Exception:  # the write may have partly landed: never requeue it, never re-apply it
        logger.exception("write_approval: applying %s/%s raised", subsystem, pid)
        return _mark_needs_review(claim, _INTERRUPTED)
    if not result.get("success"):
        if subsystem == SKILLS and result.get("rollback_failed") is True:
            return _mark_needs_review(claim, "The skill batch rollback failed, so some changes may remain. "
                                      "It will not be retried: check the files and local logs, then reject it.")
        refusal = result.get("error") or "the write was refused."
        try:
            _outcome_path(subsystem, pid).unlink(missing_ok=True)  # before the requeue: never both
        except (OSError, ValueError):
            # Requeueing beside a live mark would break that rule; keep the claim for a human instead.
            logger.warning("write_approval: could not clear the applying mark of %s/%s", subsystem, pid,
                           exc_info=True)
            return _mark_needs_review(claim, f"The write was refused ({refusal}), but its in-progress mark "
                                      "could not be cleared, so it will not be retried. Nothing was written "
                                      "by this approval: check the files, then reject it.")
        return claim.refuse(refusal)
    try:
        claim.record_outcome("applied")
    except (OSError, ValueError):
        # The write landed; the claim and its mark stay, so later access reports needs_review.
        logger.exception("write_approval: %s/%s was applied but its outcome was not recorded", subsystem, pid)
        return _reply(subsystem, pid, "applied", result=result)
    _finish_claim(subsystem, pid, applied=True)
    return _reply(subsystem, pid, "applied", result=result)


def _mark_needs_review(claim: _Claim, reason: str) -> dict[str, Any]:
    subsystem, pid = claim.subsystem, claim.pending_id
    try:
        claim.record_outcome(_NEEDS_REVIEW, reason=reason)
    except (OSError, ValueError):
        # Leave the original applying mark and claim intact for stale-claim recovery.
        logger.exception("write_approval: could not record needs_review for %s/%s", subsystem, pid)
    return _reply(subsystem, pid, _NEEDS_REVIEW, reason=reason)


# --- needs_review: list it, and close it once a human checked the files ---------------------------

def list_needs_review(subsystem: str) -> list[dict[str, Any]]:
    """Writes whose approval stopped mid-apply, oldest decision first: ``id``, ``summary``, ``action``,
    ``origin``, ``staged_at``, ``decided_at`` and ``reason``, read from the outcome and the claimed record
    the approval left behind (never the payload). Abandoned claims are settled first, so an approval that
    died while ``applying`` shows up here. Such a write keeps its claimed record until it is rejected."""
    try:
        d = _pending_dir(subsystem)
    except ValueError:
        logger.warning("write_approval: ignoring the %s pending store", subsystem, exc_info=True)
        return []
    _release_abandoned_claims(subsystem, d)
    items = []
    for claimed in d.glob("*.claimed"):
        pid = claimed.stem
        if not _PENDING_ID_RE.fullmatch(pid) or claimed.is_symlink():
            continue
        outcome = _read_outcome(subsystem, pid)
        if outcome.get("status") != _NEEDS_REVIEW:
            continue
        try:
            _, record = _read_record(claimed)
        except (OSError, ValueError):
            record = {}
        # Stored gists may quote memory text or a proposed skill description.
        action = str(record.get("action") or "write")
        summary = f"{'Memory' if subsystem == MEMORY else 'Skill'} {action}"
        items.append({"id": pid, "summary": summary,
                      "action": str(record.get("action") or ""), "origin": record.get("origin") or "foreground",
                      "staged_at": outcome.get("staged_at", record.get("created_at")),
                      "decided_at": outcome.get("decided_at"), "reason": outcome.get("reason") or _INTERRUPTED})
    items.sort(key=lambda r: r.get("decided_at") or 0)
    return items


def _staged_time(record: dict[str, Any]) -> Optional[float]:
    """The record's ``created_at`` as a float, or None if it is missing, a bool, not a finite number
    or outside +/-1e11 seconds (a millisecond timestamp, say). The list and the preview both use it."""
    created = record.get("created_at")
    if isinstance(created, bool) or not isinstance(created, (int, float)) or not math.isfinite(created):
        return None
    return float(created) if -100_000_000_000 < created < 100_000_000_000 else None


def preview_pending(subsystem: str, pending_id: str) -> dict[str, Any]:
    """What approving ``pending_id`` would do right now, for a GUI to show before anyone decides.

    The reply has ``id``, ``subsystem``, ``summary``, ``origin``, ``created_at``, ``status``, ``reason``,
    ``ops``, ``changed_since_staged``, ``notice`` and ``expect``. ``created_at`` is the staging time
    (epoch seconds, a float), or None when the record has no usable time or no record was loaded.
    Each op has a ``gist`` and ``diff`` for skills, or ``action``, ``content``, ``old_text`` and
    ``matched_entry`` for memory.
    For a skill ``create`` with no blocker, ``content`` holds the full ``SKILL.md`` and ``diff`` is None.

    ``status`` is ``ready``, ``stale`` (something changed since staging and the write no longer
    applies, such as a moved memory entry or a skill patch whose anchor is gone) or ``conflict`` (the
    write can't apply at all, such as a refused path or a create for a skill that now exists). A
    skill write whose files changed since staging but whose ops all still apply is ``ready``;
    ``changed_since_staged`` lists those files (``skill/file``, or ``skill`` for a delete; always
    empty for memory), and ``notice`` says the diffs are against the current files.

    ``expect`` is opaque and JSON-serialisable: hand it back to :func:`resolve_pending` unchanged.
    It holds the profile, a hash of the exact record bytes and the current target revisions. A write
    that is no longer pending comes back as ``already_resolved`` (with ``prior``), ``needs_review``
    or ``conflict``, and has no ``expect``."""
    try:
        pending = _pending_path(subsystem, pending_id)
        _recover_abandoned_claim(subsystem, pending.with_suffix(".claimed"))
        raw, record = _read_record(pending)
    except (OSError, ValueError):
        raw, record = b"", {}
    if not isinstance(record.get("payload"), dict):
        reply = _decided(subsystem, pending_id) or _reply(
            subsystem, pending_id, "conflict", reason=f"no pending {subsystem} write with id '{pending_id}'.")
        return {"reason": None, **reply, "summary": "", "origin": "", "created_at": None, "ops": [],
                "changed_since_staged": [], "notice": None, "expect": None}
    payload = record["payload"]
    changed: list[str] = []
    if subsystem == MEMORY:
        ops, targets = _memory_op_previews(payload), {}
        status, reason = _memory_preview_status(payload)
    else:
        affected = _skill_targets(payload)
        with _target_locks(affected):
            snapshot = _capture_skill_targets(affected, set(_skill_targets(payload, text_only=True)))
        targets = snapshot.revisions
        ops = _skill_op_previews(payload, snapshot)
        for op in ops:
            if op["action"] == "create" and op["blocker"] is None:
                op["content"], op["diff"] = op["diff"], None
        base = record.get("base")
        changed = _changed_targets(base, targets) if _is_revision_map(base) else []
        if _non_atomic_batch(payload):
            status, reason = "conflict", _NON_ATOMIC
        elif _ambiguous_skill_targets(affected):
            status, reason = "conflict", "Use one consistent spelling for every skill name and file path in this batch."
        elif _has_unpreviewable_targets(targets):
            status, reason = "conflict", "A skill target cannot be safely previewed or checked."
        elif (drift := _destructive_drift(base, targets, _skill_targets(payload, destructive_only=True))):
            status, reason = ("stale" if changed else "conflict"), drift
        else:
            status, reason = _skill_preview_status(ops, changed)
    notice = (f"Changed since this write was staged: {', '.join(changed)}. Each diff is against the "
              "current file, and approving applies the write to it." if changed and status == "ready" else None)
    return {**_reply(subsystem, pending_id, status, reason=reason),
            "summary": record.get("summary", ""), "origin": record.get("origin", "foreground"),
            "created_at": _staged_time(record), "ops": ops,
            "changed_since_staged": changed, "notice": notice,
            "expect": {"profile": _profile_identity(), "record_sha256": hashlib.sha256(raw).hexdigest(),
                       "targets": targets}}


def _skill_preview_status(ops: list[dict[str, Any]], changed: list[str]) -> tuple[str, Optional[str]]:
    if not ops:
        return "conflict", "The proposal contains no skill operations."
    for op in ops:
        if failure := op.get("blocker"):
            if op["action"] == "patch" and changed:
                return "stale", f"changed since it was staged: {', '.join(changed)}; {failure}"
            return "conflict", failure
    return "ready", None


def _memory_op_previews(payload: dict[str, Any]) -> list[dict[str, Any]]:
    ops = payload.get("operations") if payload.get("action") == "batch" else [payload]
    return [{"action": op.get("action", ""), "target": payload.get("target") or "memory",
             "content": op.get("content") or op.get("new_text") or "", "old_text": op.get("old_text") or "",
             "matched_entry": op.get("matched_entry")}
            for op in (ops if isinstance(ops, list) else []) if isinstance(op, dict)]


def _memory_preview_status(payload: dict[str, Any]) -> tuple[str, Optional[str]]:
    from tools.memory_tool import FAILURE_CLASS, destructive_ops, load_on_disk_store

    operations = payload.get("operations") if payload.get("action") == "batch" else [payload]
    target = payload.get("target")
    if not isinstance(target, str) or target not in {"memory", "user"}:
        return "conflict", "The staged memory target is invalid."
    if not isinstance(operations, list) or not operations or any(not isinstance(op, dict) for op in operations):
        return "conflict", "The staged memory operations are invalid."
    if any(not op.get("matched_entry") for op in destructive_ops(payload)):
        return "conflict", "a destructive op predates entry pinning: reject it and recreate the change."
    # Each pin is checked against the state left by preceding operations, not the
    # initial state. Use the same lock and working-copy loop as the actual applier.
    token = FAILURE_CLASS.set("other")
    try:
        result = load_on_disk_store().resolve_batch_entries(target, operations)
        if result.get("success"):
            return "ready", None
        if FAILURE_CLASS.get() == "stale_entry":
            return "stale", "an entry this write targets changed since it was staged."
        return "conflict", result.get("error") or "The staged memory operations cannot be applied."
    finally:
        FAILURE_CLASS.reset(token)
