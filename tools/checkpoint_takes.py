"""Per-project record of checkpoint *takes*: when each snapshot attempt ran and which snapshot it left.

``/undo``'s ``/rollback`` pointer needs the files as they were when the first undone turn started
writing. A take that finds nothing changed creates no commit, so git history alone cannot tell that
the turn reused an older snapshot, and git commit times are whole seconds. Each entry here is
``[time.time() at the attempt, tree sha, author epoch]`` of the snapshot the project was at once the
attempt finished: the new one, or the unchanged tip it reused. Tree plus author second survive the
pruner's history rewrite (it keeps both, while every commit hash changes), so an entry still finds
its snapshot after a prune and finds nothing once that snapshot was dropped.

An attempt is written BEFORE it runs, with an empty tree, and completed after: a take that failed or
whose completion could not be written stays visibly unknown, and an attempt that never got the store
lock (busy: another process's take or prune) is noted like an unwritable one. ``takes_since`` is the time from which
the record is complete. It moves forward for a project an older build snapshotted, for entries the
cap dropped, and past an attempt this process could not record at all. The record lives in the
project's metadata JSON, so it is renamed and deleted with the project.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Optional

_TAKES, _SINCE = "takes", "takes_since"
_MAX_TAKES = 256
# Attempts this process could not write at all, by metadata path: the next write moves ``takes_since`` past them.
_UNRECORDED: dict[str, float] = {}


def turn_key(timestamp) -> str:
    """The identity of one user turn every surface can rebuild from its transcript row: ``""`` when unknown."""
    from tools.checkpoint_undo_hint import _epoch

    when = _epoch(timestamp)
    return repr(when) if when is not None else ""


def unique_turn_key(timestamp, rows) -> str:
    """``turn_key(timestamp)``, or ``""`` when more than one user row in ``rows`` (every row the surface knows,
    the undone ones included) carries that key. Stamps are not unique per turn (a platform's event time can be
    whole seconds), and a take under a shared key proves neither turn made it."""
    key = turn_key(timestamp)
    same = sum(1 for m in rows or () if isinstance(m, dict) and m.get("role") == "user"
               and turn_key(m.get("timestamp")) == key)
    return key if key and same <= 1 else ""


def stored_turn_key(db, session_id: str, timestamp) -> str:
    """``unique_turn_key`` over every stored row of the session's lineage, rewound ones included; ``""``
    when the transcript cannot be read (the hint is advisory: no proof, no hash)."""
    try:
        rows = db.get_messages_as_conversation(session_id, include_ancestors=True, include_inactive=True)
    except Exception:  # health: allow BLE001 -- an advisory line must never fail the /undo it follows
        return ""
    return unique_turn_key(timestamp, rows)


def bind_checkpoint_turn(agent, user_msg: dict) -> None:
    """Name the turn whose takes the agent's checkpoint manager records next by the stamp its user row is
    persisted with (the persist override wins, as in ``agent/session_persistence.py``). A key this manager
    already bound for an earlier turn binds ``""``: those takes then prove no turn, rather than the wrong one."""
    mgr = getattr(agent, "_checkpoint_mgr", None)
    if mgr is None:
        return
    stamp = getattr(agent, "_persist_user_message_timestamp", None)
    key = turn_key(stamp if stamp is not None else (user_msg or {}).get("timestamp"))
    bound = getattr(mgr, "_bound_turn_keys", None)
    if bound is None:
        bound = mgr._bound_turn_keys = {}
    mgr.turn_key = "" if key in bound else key
    bound[key] = None
    if len(bound) > _MAX_TAKES:  # older turns' takes have left the record anyway
        del bound[next(iter(bound))]


def _entries(raw) -> list[tuple[float, str, int, str]]:
    out = []
    for item in raw if isinstance(raw, list) else ():
        if (isinstance(item, list) and len(item) == 4 and isinstance(item[0], (int, float))
                and not isinstance(item[0], bool) and math.isfinite(item[0])
                and isinstance(item[1], str) and isinstance(item[2], int) and isinstance(item[3], str)):
            out.append((float(item[0]), item[1], item[2], item[3]))
    return out


def _since(meta: dict) -> Optional[float]:
    since = meta.get(_SINCE)
    if isinstance(since, bool) or not isinstance(since, (int, float)) or not math.isfinite(since):
        return None
    return float(since)


def note_unrecorded(meta_path: Path, when: float) -> None:
    """Attempt ``when`` ran but nothing on disk says so (no lock, unwritable metadata): the record is
    incomplete up to it, here at once and on disk from the next write."""
    key = str(meta_path)
    _UNRECORDED[key] = max(_UNRECORDED.get(key, 0.0), when)


def record_take(meta_path: Path, when: float, tree: str = "", author_epoch: int = 0, *,
                had_history=True, turn: str = "") -> None:
    """Write attempt ``when`` (``tree=""``: started) or complete it with the snapshot it left. The caller
    holds ``store_lock``. Never raises: a failed write is remembered and closes the gap on the next one.
    ``had_history`` (a bool or a callable) is read only for the project's first recorded take."""
    key = str(meta_path)
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        if not isinstance(meta, dict):
            raise ValueError("project metadata is not an object")
        takes, since = _entries(meta.get(_TAKES)), _since(meta)
        if since is None:  # the project's first recorded take: any earlier (an older build's) are unknown
            takes, since = [], (when if (had_history() if callable(had_history) else had_history) else 0.0)
        if key in _UNRECORDED:
            since = max(since, _UNRECORDED[key])
        takes = [t for t in takes if not (t[0] == when and not t[1])]  # completing replaces the started row
        takes.append((when, tree, int(author_epoch), turn))
        takes.sort()
        if len(takes) > _MAX_TAKES:
            since = max(since, takes[-_MAX_TAKES - 1][0])
            takes = takes[-_MAX_TAKES:]
        meta[_TAKES], meta[_SINCE] = [list(t) for t in takes], since
        tmp = meta_path.with_name(f".{meta_path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(meta), encoding="utf-8")
        os.replace(tmp, meta_path)
    except (OSError, ValueError):
        if not tree:  # the started row is missing: nothing on disk says this attempt ran
            note_unrecorded(meta_path, when)
        return
    _UNRECORDED.pop(key, None)


def load_takes(meta_path: Path) -> Optional[tuple[list[tuple[float, str, int, str]], float]]:
    """``(entries oldest first, takes_since)``, or None when the project has no readable record."""
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict) or (since := _since(meta)) is None:
        return None
    if str(meta_path) in _UNRECORDED:
        since = max(since, _UNRECORDED[str(meta_path)])
    return sorted(_entries(meta.get(_TAKES))), since


def _tip(store: Path, abs_dir: str, ref: str) -> Optional[tuple[str, int]]:
    """``(tree, author epoch)`` of the project's newest snapshot, None when it has none."""
    from tools import checkpoint_manager as cm

    ok, out, _ = cm._run_git(["log", "-1", "--format=%T|%at", ref], store, abs_dir, allowed_returncodes={128})
    tree, _, epoch = out.partition("|") if ok else ("", "", "")
    return (tree, int(epoch)) if tree and epoch.isdigit() else None


def recorded_take(mgr, abs_dir: str, reason: str, **kwargs) -> bool:
    """``mgr._take(abs_dir, reason, **kwargs)`` (the caller holds ``store_lock``) with the attempt written
    down first and completed with the snapshot the project is at afterwards, taken or reused."""
    from tools import checkpoint_manager as cm

    store = cm._store_path()
    dir_hash = cm._project_hash(abs_dir)
    meta_path = cm._project_meta_path(store, dir_hash)
    ref = cm._ref_name(dir_hash)
    # Only a project's FIRST recorded take reads whether it had history (an older build's snapshots), and only
    # before the take: once a record exists that git log is skipped. Read late it says True: the safe answer.
    def had_history() -> bool:
        return (store / "HEAD").exists() and _tip(store, abs_dir, ref) is not None

    when, turn = time.time(), getattr(mgr, "turn_key", "") or ""
    if meta_path.exists():
        record_take(meta_path, when, had_history=had_history, turn=turn)
    else:
        had_history = had_history()
    mgr._take_reused = False
    ok = mgr._take(abs_dir, reason, **kwargs)
    # A take that failed (too many files, a git error) left the tip, not the files: its row stays unknown.
    tip = _tip(store, abs_dir, ref) if ok or mgr._take_reused else None
    record_take(meta_path, when, *(tip or ("", 0)), had_history=had_history, turn=turn)
    return ok


def note_unrecorded_attempt(abs_dir: str, when: float) -> None:
    """``note_unrecorded`` for ``abs_dir``'s project in the active profile's store."""
    from tools import checkpoint_manager as cm

    note_unrecorded(cm._project_meta_path(cm._store_path(), cm._project_hash(abs_dir)), when)


def project_takes(abs_dir: str):
    """``load_takes`` for ``abs_dir``'s project in the active profile's store."""
    from tools import checkpoint_manager as cm

    return load_takes(cm._project_meta_path(cm._store_path(), cm._project_hash(abs_dir)))
