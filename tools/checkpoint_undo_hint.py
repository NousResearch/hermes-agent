"""The ``/undo`` pointer at ``/rollback``: which checkpoint restores the files the undone turns changed.

``/undo`` only rewinds the conversation; it never touches files. When checkpoints are on, the
snapshot the project was at when the FIRST undone turn started writing is still in the store, so
``/undo`` names the files changed since then and the ``/rollback <hash>`` that restores them. The
short hash is what every surface's ``/rollback`` accepts (``tools/checkpoint_refs.py``) and, unlike a
list position, it does not move when the next checkpoint is taken.

The target comes from the take record (``tools/checkpoint_takes.py``), not from commit times: a turn
whose take found nothing changed reused an older snapshot, which git history alone cannot show, and
git times are whole seconds. The first recorded take at or after the span start names the target by
tree; the hint then finds that tree among the retained snapshots, so a snapshot the pruner dropped
is simply not found. Whenever the record cannot prove the target (an older build's project, a gap,
a failed take, a pruned snapshot), the line says to run ``/rollback`` and pick one: a wrong command
is worse than none.

Read-only over the store: ``git log`` plus a name-only diff of the snapshot's tree against the
working tree, through a throwaway index; it writes no objects, refs or project index.
"""
from __future__ import annotations

import logging
import math
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MAX_NAMED_FILES = 8


@dataclass(frozen=True)
class UndoRollbackHint:
    """``ref`` is what ``/rollback`` takes (a short checkpoint hash); None when no safe one is known."""

    ref: Optional[str]
    files: list[str] = field(default_factory=list)
    more: int = 0

    def as_dict(self) -> dict:
        """Wire shape for clients that only print the line: the fields plus the rendered ``text``."""
        return {"ref": self.ref, "files": list(self.files), "more": self.more,
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


def _snapshots(mgr, abs_dir: str) -> list[tuple[str, str, int]]:
    """``(commit, tree, author epoch)`` of the retained snapshots ``list_checkpoints`` shows, newest first."""
    from tools import checkpoint_manager as cm

    store = cm._store_path()
    if not (store / "HEAD").exists():
        return []
    ok, out, _ = cm._run_git(["log", cm._ref_name(cm._project_hash(abs_dir)), "--format=%H|%T|%at",
                              "-n", str(mgr.max_snapshots)], store, abs_dir, allowed_returncodes={128, 129})
    rows = []
    for line in (out.splitlines() if ok else []):
        sha, tree, epoch = (line.split("|") + ["", ""])[:3]
        if sha and tree and epoch.isdigit():
            rows.append((sha, tree, int(epoch)))
    return rows


def _span_target(abs_dir: str, since: float, rows: list) -> Optional[tuple[str, int]]:
    """``(tree, author epoch)`` the project was at when the span's first take ran; None when unprovable."""
    from tools.checkpoint_takes import project_takes

    record = project_takes(abs_dir)
    if record is None or record[1] > since:
        return None  # the record does not reach back to the span start
    seen = {(tree, epoch) for _when, tree, epoch in record[0]}
    if any((tree, epoch) not in seen for _sha, tree, epoch in rows if epoch >= math.floor(since)):
        return None  # a snapshot of the span the record never saw: a take went unrecorded (another process)
    first = next(((tree, epoch) for when, tree, epoch in record[0] if when >= since), None)
    return first if first and first[0] else None  # empty tree: that take failed or never completed


def _changed_files(abs_dir: str, tree: str) -> Optional[list[str]]:
    """Paths whose working-tree state differs from ``tree``: tracked changes and deletions plus new files,
    through a throwaway index (no object, ref or project index is written), None on failure."""
    from tools import checkpoint_manager as cm

    store = cm._store_path()
    excludes = cm._exclude_store_args(store.parent, abs_dir)
    with tempfile.TemporaryDirectory(prefix="undo-hint-", dir=store) as scratch:
        index = Path(scratch) / "index"
        ok, _, _ = cm._run_git(["read-tree", tree], store, abs_dir, index_file=index)
        if not ok:
            return None
        ok, changed, _ = cm._run_git(["diff", "--name-only", "-z", *excludes], store, abs_dir, index_file=index)
        ok_new, new, _ = cm._run_git(["ls-files", "--others", "--exclude-standard", "-z", *excludes],
                                     store, abs_dir, index_file=index)
    if not (ok and ok_new):
        return None
    return sorted({p for p in (changed + "\x00" + new).split("\x00") if p})


def undo_rollback_hint(mgr, cwd: str, since, *, task_id: Optional[str] = "default") -> Optional[UndoRollbackHint]:
    """The hint for undoing every turn from the user message stamped ``since`` onward.

    The target is the snapshot the project was at when the span's FIRST take ran (``_span_target``), so
    the restore brings back the files as the first undone turn found them. None: checkpoints off or
    refused here, no timestamp, no take in the span, or nothing changed since. ``ref`` None: the span
    took a checkpoint but the record cannot prove which one.
    ``task_id=None``: the caller already applied ``unsupported_backend_reason`` under its own scope.
    """
    from tools import checkpoint_manager as cm

    if mgr is None or not getattr(mgr, "enabled", False) or (since_ts := _epoch(since)) is None:
        return None
    try:
        if task_id is not None and mgr.unsupported_backend_reason(task_id):
            return None
        abs_dir = str(cm._normalize_path(cwd))
        rows = _snapshots(mgr, abs_dir)
        target = _span_target(abs_dir, since_ts, rows)
        if not target and not any(epoch >= math.floor(since_ts) for _sha, _tree, epoch in rows):
            return None  # no take and no snapshot in the span: nothing for /rollback to restore
        match = next((sha for sha, tree, epoch in rows if target and (tree, epoch) == target), None)
        # No provable target: name the files against the oldest snapshot still in the span, else the newest.
        in_span = [tree for _sha, tree, epoch in rows if epoch >= math.floor(since_ts)] or [r[1] for r in rows[:1]]
        files = _changed_files(abs_dir, next((t for s, t, _e in rows if s == match), None) or in_span[-1]) \
            if in_span else None
    except Exception as exc:  # health: allow BLE001 -- an advisory line must never fail the /undo it follows
        logger.debug("undo rollback hint skipped: %s", exc)
        return None
    if not files:
        return None
    # An all-digit short hash would read as a list number (tools/checkpoint_refs.py): print it whole.
    ref = (match[:8] if not match[:8].isdigit() else match) if match else None
    return UndoRollbackHint(ref=ref, files=files[:_MAX_NAMED_FILES], more=max(0, len(files) - _MAX_NAMED_FILES))


def checkpoint_taken_at(cwd: str, commit: str) -> Optional[tuple[float, bool]]:
    """``(time, exact)``: the LAST time the project was seen in checkpoint ``commit``'s state, to the
    sub-second from the take record (``exact``), else the commit's git second. A later take that reused the
    snapshot moves it forward: the turn that took it then left the files as they are in it. None: unreadable."""
    from tools import checkpoint_manager as cm
    from tools.checkpoint_takes import project_takes

    store = cm._store_path()
    if not commit or not (store / "HEAD").exists():
        return None
    abs_dir = str(cm._normalize_path(cwd))
    ok, out, _ = cm._run_git(["log", "-1", "--format=%T|%at", commit], store, abs_dir, allowed_returncodes={128})
    tree, _, epoch = out.partition("|") if ok else ("", "", "")
    if not tree or not epoch.isdigit():
        return None
    record = project_takes(abs_dir)
    seen = max((w for w, t, e in record[0] if t == tree and e == int(epoch)), default=None) if record else None
    if seen is not None and record[1] <= seen:  # no unrecorded take since: nothing later can have reused it
        return seen, True
    return float(epoch), False


def rollback_rewinds_newest_turn(history: list, user_indices: list[int], taken) -> bool:
    """Whether a whole-tree ``/rollback`` to a checkpoint taken at ``taken`` (``checkpoint_taken_at``) also
    undoes the newest kept user turn (main's "so the agent's context matches the restored files"): only when
    that turn made the checkpoint or came after it. After ``/undo`` already removed the turn that made it,
    the newest kept turn ended before the checkpoint and stays.

    With only the checkpoint's whole second known, a turn row inside that same second may be either side of
    it: that is ambiguous, and an ambiguous turn is KEPT (losing a turn the user kept is the worse mistake).
    Unknown times keep main's behaviour."""
    from agent.context_compressor import is_compaction_summary_message

    if not user_indices or taken is None:
        return bool(user_indices)
    when, exact = taken
    stamps = [_epoch(m.get("timestamp")) for m in history[user_indices[-1]:]
              if isinstance(m, dict) and not is_compaction_summary_message(m)]
    known = [s for s in stamps if s is not None]
    if not stamps or len(known) != len(stamps):
        return True
    newest = max(known)
    return newest >= when if exact else newest >= math.floor(when) + 1


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
    if hint.ref:
        return t("shared.undo_rollback.restore", files=files, ref=hint.ref)
    return t("shared.undo_rollback.list", files=files)
