"""The ``/undo`` pointer at ``/rollback``: which checkpoint restores the files the undone turns changed.

``/undo`` only rewinds the conversation; it never touches files. When checkpoints are on, the
snapshot taken before an undone turn's first write is still in the store, so ``/undo`` names the
files changed since then and the ``/rollback <N>`` that restores them. ``N`` is the 1-based position
in ``list_checkpoints(cwd)`` (newest first), the same list every surface's ``/rollback N`` indexes.

Read-only over the checkpoint store: one ``git log`` and one name-only diff of the tree against the
matched snapshot (staged through the project's own index and reset afterwards, as ``/rollback diff``
does). Nothing is recorded: the match is by time alone, so an unsure match yields no number.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

_MAX_NAMED_FILES = 8


@dataclass(frozen=True)
class UndoRollbackHint:
    """``number`` is what ``/rollback`` takes; None when the span has a checkpoint but no safe number."""

    number: Optional[int]
    files: list[str] = field(default_factory=list)
    more: int = 0

    def as_dict(self) -> dict:
        """Wire shape for clients that only print the line: the fields plus the rendered ``text``."""
        return {"number": self.number, "files": list(self.files), "more": self.more,
                "text": format_undo_rollback_hint(self)}


def _epoch(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) and value > 0 else None
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.strip()).timestamp()
        except ValueError:
            return None
    return None


def undo_span_start(target_timestamp, kept_history=()) -> Optional[float]:
    """When the undone span began: the undone user message's stamp, or the newest kept message's when that is
    later. A gateway stamps a message with its platform send time, so a prompt queued while the previous turn
    ran is stamped before that turn's checkpoints; the kept turn's last row bounds them out."""
    from agent.context_compressor import is_compaction_summary_message

    start = _epoch(target_timestamp)
    for message in kept_history or ():
        if not isinstance(message, dict) or is_compaction_summary_message(message):
            continue  # a summary row is stamped when the compaction ran, not when its turns did
        if (when := _epoch(message.get("timestamp"))) is not None:
            start = when if start is None else max(start, when)
    return start


def _checkpoint_rows(mgr, cwd: str) -> list[tuple[str, float]]:
    """``(hash, epoch seconds)`` newest first: exactly the rows ``list_checkpoints`` numbers, minus its per-row shortstat."""
    from tools import checkpoint_manager as cm

    abs_dir = str(cm._normalize_path(cwd))
    store = cm._store_path()
    if not (store / "HEAD").exists():
        return []
    ok, out, _ = cm._run_git(["log", cm._ref_name(cm._project_hash(abs_dir)), "--format=%H|%aI",
                              "-n", str(mgr.max_snapshots)], store, abs_dir, allowed_returncodes={128, 129})
    rows = []
    for line in (out.splitlines() if ok else []):
        sha, _, stamp = line.partition("|")
        if (when := _epoch(stamp)) is not None:
            rows.append((sha, when))
    return rows


def _span_intact(cwd: str, rows: list[tuple[str, float]], oldest: int, floor: int) -> bool:
    """True when no checkpoint taken at or after ``floor`` can have been pruned. Pruning only ever drops a
    project's OLDEST snapshots, so a retained checkpoint older than the span proves the span whole;
    otherwise the project's prune ledger must show every drop predates the span. A project an older
    build registered has no ledger: unprovable, so no number."""
    import json

    from tools import checkpoint_manager as cm

    if any(when < floor for _sha, when in rows[oldest + 1:]):
        return True
    abs_dir = str(cm._normalize_path(cwd))
    try:
        meta = json.loads(cm._project_meta_path(cm._store_path(), cm._project_hash(abs_dir)).read_text(
            encoding="utf-8-sig"))
    except (OSError, ValueError):
        return False
    return isinstance(meta, dict) and bool(meta.get("prune_ledger")) and int(meta.get("pruned_through") or 0) < floor


def _changed_files(cwd: str, commit: str) -> Optional[list[str]]:
    """Paths changed in ``cwd`` since ``commit`` (``_safe_restore_plan``'s name-only diff), None on failure."""
    from tools import checkpoint_manager as cm
    from tools.checkpoint_pruning import store_lock

    abs_dir = str(cm._normalize_path(cwd))
    store = cm._store_path()
    dir_hash = cm._project_hash(abs_dir)
    index_file = cm._index_path(store, dir_hash)
    with store_lock(cm._resolve_checkpoint_base()):
        cm._run_git(["add", "-A"], store, abs_dir, timeout=cm._GIT_TIMEOUT * 2, index_file=index_file)
        ok, names, _ = cm._run_git(["diff", "--name-only", "-z", commit, "--cached"], store, abs_dir,
                                   index_file=index_file)
        cm._run_git(["read-tree", cm._ref_name(dir_hash)], store, abs_dir, index_file=index_file,
                    allowed_returncodes={128})
    return sorted(filter(None, names.split("\x00"))) if ok else None


def undo_rollback_hint(mgr, cwd: str, since, *, task_id: Optional[str] = "default") -> Optional[UndoRollbackHint]:
    """The hint for undoing every turn from the user message stamped ``since`` onward.

    Picks the OLDEST checkpoint taken during that span (the snapshot before its first write), so the
    restore covers every undone turn. Checkpoint times have whole-second precision: one in the same
    second as ``since`` may belong to the turn before; and pruning (count or size cap) may have dropped an
    older checkpoint of the span (``_span_intact``). Both drop the number: a wrong one is worse than none.
    None: checkpoints off or refused here, no timestamp, no checkpoint in the span, or nothing changed.
    ``task_id=None``: the caller already applied ``unsupported_backend_reason`` under its own scope.
    """
    if mgr is None or not getattr(mgr, "enabled", False) or (since_ts := _epoch(since)) is None:
        return None
    try:
        if task_id is not None and mgr.unsupported_backend_reason(task_id):
            return None
        rows = _checkpoint_rows(mgr, cwd)
        floor = math.floor(since_ts)
        in_span = [i for i, (_sha, when) in enumerate(rows) if when >= floor]
        if not in_span:
            return None
        oldest = in_span[-1]
        unsure = (rows[oldest][1] == floor or (oldest == len(rows) - 1 and len(rows) >= mgr.max_snapshots)
                  or not _span_intact(cwd, rows, oldest, floor))
        files = _changed_files(cwd, rows[oldest][0])
    except Exception as exc:  # health: allow BLE001 -- an advisory line must never fail the /undo it follows
        logger.debug("undo rollback hint skipped: %s", exc)
        return None
    if not files:
        return None
    return UndoRollbackHint(number=None if unsure else oldest + 1, files=files[:_MAX_NAMED_FILES],
                            more=max(0, len(files) - _MAX_NAMED_FILES))


def checkpoint_epoch(cwd: str, commit: str) -> Optional[float]:
    """When checkpoint ``commit`` of ``cwd``'s project was taken (epoch seconds), None when unreadable."""
    from tools import checkpoint_manager as cm

    store = cm._store_path()
    if not commit or not (store / "HEAD").exists():
        return None
    try:
        ok, out, _ = cm._run_git(["log", "-1", "--format=%at", commit], store, str(cm._normalize_path(cwd)),
                                 allowed_returncodes={128})
        return float(out) if ok and out.strip() else None
    except (OSError, ValueError):
        return None


def rollback_rewinds_newest_turn(history: list, user_indices: list[int], taken: Optional[float]) -> bool:
    """Whether a whole-tree ``/rollback`` to a checkpoint taken at ``taken`` (``checkpoint_epoch``) also undoes
    the newest kept user turn (main's "so the agent's context matches the restored files"): only when that
    turn made the checkpoint or came after it, i.e. one of its rows is no older than the checkpoint's second.
    After ``/undo`` already removed the turn that made it, the newest kept turn ended before the checkpoint
    and stays. Unknown times keep main's behaviour."""
    from agent.context_compressor import is_compaction_summary_message

    if not user_indices or taken is None:
        return bool(user_indices)
    stamps = [_epoch(m.get("timestamp")) for m in history[user_indices[-1]:]
              if isinstance(m, dict) and not is_compaction_summary_message(m)]
    if not stamps or None in stamps:
        return True
    return max(stamps) >= math.floor(taken)


def undo_hint_payload(mgr, cwd: str, target_timestamp, kept_history=(), *,
                      task_id: Optional[str] = "default") -> Optional[dict]:
    """``undo_rollback_hint`` over a rewind's target stamp and kept prefix, as the wire dict (None: no line)."""
    hint = undo_rollback_hint(mgr, cwd, undo_span_start(target_timestamp, kept_history), task_id=task_id)
    return hint.as_dict() if hint is not None else None


def format_undo_rollback_hint(hint: Optional[UndoRollbackHint]) -> str:
    """The one localized line every surface prints after its own ``/undo`` output; ``""`` for no hint."""
    if hint is None:
        return ""
    from agent.i18n import t

    files = ", ".join(hint.files)
    if hint.more:
        files += " " + t("shared.undo_rollback.more", count=hint.more)
    number = hint.number
    if number:
        return t("shared.undo_rollback.restore", files=files, number=number)
    return t("shared.undo_rollback.list", files=files)
