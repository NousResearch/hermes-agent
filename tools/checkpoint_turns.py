"""Per-turn file revert for ``/undo``: put back the files Hermes wrote in the undone turns.

``/undo`` rewinds the conversation to before a user turn; with checkpoints on it also restores the
files Hermes' file tools wrote since that turn began. Two records in the shared checkpoint store make
that exact without guessing from timestamps:

* turn marks (``turns/<project>.json``): the tree a project was in when a turn first touched it,
  keyed by the turn's user-message timestamp (the same value its transcript row stores) and session;
* the agent-write ledger: each file's last Hermes-written hash, now also tagged with that turn.

Reverting turn T restores every ledger file written by T or a later turn of the same conversation
to the mark of the FIRST undone turn that wrote that file (so work another writer added between two
undone turns is not erased). A file whose content changed since Hermes wrote it (a hand edit) is
kept, the same rule as the safe ``/rollback``. A file absent from that tree is only deleted when the
tree could have held it (not excluded, not oversize then or now, not inside a nested repo).

Only Hermes' file tools (``write_file``/``patch``) feed the ledger; files a terminal command changed
are left to ``/rollback``, which restores whole checkpoints.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

_TURNS_DIRNAME = "turns"
_MAX_MARKS = 500
# Git's empty tree: the pre-state of a project that had no files when the turn began.
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def _marks_path(store: Path, dir_hash: str) -> Path:
    return store / _TURNS_DIRNAME / f"{dir_hash}.json"


def _load_marks(store: Path, dir_hash: str) -> list[dict]:
    try:
        data = json.loads(_marks_path(store, dir_hash).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return []
    return [m for m in data if isinstance(m, dict)] if isinstance(data, list) else []


def record_turn_mark(store: Path, working_dir: str, tree: str, turn: dict,
                     excluded: Iterable[str] = ()) -> None:
    """Remember ``tree`` as ``working_dir``'s state when ``turn`` (``turn_ts``, ``session``) first
    touched it; ``excluded``: paths that existed then but the size cap kept out of ``tree`` (undo
    must never read their absence as "Hermes created it"). Caller holds the store lock. Best effort:
    a lost mark only makes that turn's files unrevertible by ``/undo`` (``/rollback`` still has the
    checkpoint)."""
    from tools.checkpoint_manager import _project_hash
    dir_hash = _project_hash(working_dir)
    marks = _load_marks(store, dir_hash)
    mark = {"turn_ts": turn["turn_ts"], "session": turn.get("session") or "", "tree": tree, "ts": time.time()}
    if excluded := sorted(set(excluded)):
        mark["excluded"] = excluded
    marks.append(mark)
    path = _marks_path(store, dir_hash)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(marks[-_MAX_MARKS:]), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        logger.debug("Could not record turn mark for %s", working_dir, exc_info=True)


def _in_scope(entry: dict, since: float, sessions: set[str], until: Optional[float] = None) -> bool:
    turn_ts = entry.get("turn_ts")
    return (isinstance(turn_ts, (int, float)) and turn_ts >= since and (until is None or turn_ts <= until)
            and entry.get("session") in sessions)


def revert_turns(mgr, since: float, sessions: Iterable[str], until: Optional[float] = None) -> dict:
    """Restore the files Hermes wrote in turns starting at or after ``since`` (a user-message
    timestamp) of the conversation ``sessions`` (its session ids, compression lineage included).
    ``until``: the newest undone turn's timestamp, so a turn another client started after the
    rewind (its writes are newer) is never reverted with it.

    Returns ``{"reverted": [abs paths], "skipped_user_edits": [...], "not_captured": [...],
    "errors": [...]}``; every list is empty when the turns wrote nothing (or checkpoints are off).
    """
    from tools.checkpoint_manager import _resolve_checkpoint_base
    from tools.checkpoint_pruning import PruneError, store_lock
    out: dict[str, list[str]] = _empty()
    sessions = {s for s in sessions if s}
    if mgr is None or not getattr(mgr, "enabled", False) or not sessions:
        return out  # no conversation named: never revert another conversation's writes
    try:
        with store_lock(_resolve_checkpoint_base()):
            _revert_locked(mgr, (float(since), until), sessions, out)
    except (PruneError, OSError) as exc:
        out["errors"].append(str(exc))
    return out


def _revert_locked(mgr, window: tuple[float, Optional[float]], sessions: set[str], out: dict) -> None:
    since, until = window
    from tools.checkpoint_manager import _list_projects, _load_ledger, _store_path
    store = _store_path()
    if not (store / "HEAD").exists():
        return
    workdirs = {meta["_hash"]: meta.get("workdir") for meta in _list_projects(store)}
    ledgers = sorted((store / "ledgers").glob("*.json")) if (store / "ledgers").is_dir() else []
    for ledger_path in ledgers:
        dir_hash = ledger_path.stem
        ledger = _load_ledger(store, dir_hash)
        files = sorted(p for p, e in ledger.items() if isinstance(e, dict) and _in_scope(e, since, sessions, until))
        if not files:
            continue
        marks = [m for m in _load_marks(store, dir_hash) if _in_scope(m, since, sessions, until)]
        workdir = workdirs.get(dir_hash)
        if not marks or not workdir or not Path(workdir).is_dir():
            out["not_captured"].extend(files)  # no snapshot of this project before the turn
            continue
        baselines = {f: _baseline_mark(marks, _first_write_ts(ledger[f], window, sessions)) for f in files}
        _revert_project(mgr, store, workdir, dir_hash, baselines, ledger, out, since)


def _first_write_ts(entry: dict, window: tuple[float, Optional[float]], sessions: set[str]) -> float:
    """The earliest undone turn that wrote this file: its latest write or an older one in history."""
    writes = [entry, *(h for h in entry.get("history", []) if isinstance(h, dict))]
    return min(w["turn_ts"] for w in writes if _in_scope(w, window[0], sessions, window[1]))


def _baseline_mark(marks: list[dict], first_ts: float) -> dict:
    """That turn's own mark (the tree before it touched the project), else the latest earlier one."""
    earlier = [m for m in marks if m["turn_ts"] <= first_ts] or marks
    return max(earlier, key=lambda m: (m["turn_ts"], -m.get("ts", 0)))


def _revert_project(mgr, store: Path, workdir: str, dir_hash: str, baselines: dict[str, dict],
                    ledger: dict, out: dict, since: float) -> None:
    from tools.checkpoint_manager import _hash_file, _ref_name, _run_git, _save_ledger
    restore: dict[str, list[str]] = {}  # baseline tree -> paths to check out of it
    delete = []
    for abs_path, mark in baselines.items():
        path = Path(abs_path)
        try:
            rel = path.relative_to(workdir).as_posix()
        except ValueError:
            out["not_captured"].append(abs_path)
            continue
        current = _hash_file(path)
        if current is not None and current != ledger[abs_path].get("sha256"):
            out["skipped_user_edits"].append(abs_path)  # changed since Hermes wrote it: keep
            continue
        tree = mark["tree"]
        in_tree, _, _ = _run_git(["cat-file", "-e", f"{tree}:{rel}"], store, workdir,
                                 allowed_returncodes={1, 128})
        if in_tree:
            restore.setdefault(tree, []).append(rel)
        elif rel not in mark.get("excluded", ()) and _could_be_captured(mgr, store, workdir, tree, rel, path):
            delete.append(rel)  # Hermes created it in the undone turns
        else:
            out["not_captured"].append(abs_path)
    if not restore and not delete:
        return
    # "Undo the undo": /rollback lists this snapshot like any other.
    mgr._take(workdir, "pre-undo snapshot", prune=False)
    restored = []
    for tree, rels in restore.items():
        # A scratch index: the project's own index tracks its ref tip, not this older tree.
        scratch = store / "indexes" / f"{dir_hash}.undo"
        try:
            ok, _, err = _run_git(["checkout", tree, "--", *rels], store, workdir, index_file=scratch)
        finally:
            scratch.unlink(missing_ok=True)
        if ok:
            restored.extend(rels)
        else:
            out["errors"].append(err or "checkout failed")
    deleted = []
    for rel in delete:
        try:
            (Path(workdir) / rel).unlink(missing_ok=True)
            deleted.append(rel)
        except OSError as exc:
            out["errors"].append(f"{rel}: {exc}")
    done = [str(Path(workdir) / rel) for rel in (*restored, *deleted)]
    out["reverted"].extend(done)
    for abs_path in done:
        _rewind_ledger_entry(ledger, abs_path, since)
    _save_ledger(store, dir_hash, ledger)
    # The pre-undo snapshot skipped pruning so the restore could not lose its tree; budgets now.
    mgr._prune(store, workdir, _ref_name(dir_hash))


def _rewind_ledger_entry(ledger: dict, abs_path: str, since: float) -> None:
    """After a revert the file holds either an earlier Hermes write (that write's entry comes back,
    so undoing its turn later still reverts it) or content Hermes never wrote (no entry: a later
    /undo or safe /rollback leaves it alone)."""
    from tools.checkpoint_manager import _hash_file
    entry = ledger.pop(abs_path, None) or {}
    current = _hash_file(Path(abs_path))
    for older in reversed(entry.get("history", [])):
        if (isinstance(older, dict) and isinstance(older.get("turn_ts"), (int, float))
                and older["turn_ts"] < since and current is not None and older.get("sha256") == current):
            index = entry["history"].index(older)
            ledger[abs_path] = {**older, "ts": entry.get("ts"), "history": entry["history"][:index]}
            return


def _could_be_captured(mgr, store: Path, workdir: str, tree: str, rel: str, path: Path) -> bool:
    """Whether a path missing from ``tree`` was really absent then, rather than kept out of every
    checkpoint (store excludes, the size cap, a nested repository): only then is deleting it a revert."""
    from tools.checkpoint_manager import _gitlink_paths, _run_git
    if mgr._exceeds_size_cap(path):
        return False
    ignored, _, _ = _run_git(["check-ignore", "-q", "--no-index", rel], store, workdir,
                             allowed_returncodes={1, 128})
    if ignored:
        return False
    ok, tree_out, _ = _run_git(["ls-tree", "-r", "-z", tree], store, workdir)
    return ok and not any(rel == repo or rel.startswith(repo + "/") for repo in _gitlink_paths(tree_out))


def undo_session_ids(db, *session_ids: Optional[str]) -> set[str]:
    """The session ids a conversation's turns ran under: each given id plus its compression lineage."""
    ids = {s for s in session_ids if s}
    for sid in list(ids):
        try:
            ids.update(db.get_compression_lineage(sid) if db is not None else ())
        except Exception:  # health: allow BLE001 -- a lineage read failure narrows the revert, never widens it
            logger.debug("undo: lineage lookup failed for %s", sid, exc_info=True)
    return ids


def _empty() -> dict:
    return {"reverted": [], "skipped_user_edits": [], "not_captured": [], "errors": []}


def undo_turn_files(mgr, since, db, *session_ids: Optional[str], until: Optional[float] = None) -> dict:
    """``/undo``'s file half for an in-process surface: revert what the conversation's turns from
    the rewound user row's stored timestamp ``since`` onward wrote (``begin_turn`` keyed them by
    it). Empty when checkpoints are off, ``since`` is unknown, or the turns wrote nothing."""
    if mgr is None or not getattr(mgr, "enabled", False) or not isinstance(since, (int, float)):
        return _empty()
    return revert_turns(mgr, float(since), undo_session_ids(db, *session_ids), until)


def undo_row_files(mgr, db, row_id: Optional[int], sessions: Iterable[str]) -> dict:
    """Same, addressed by the rewound user row's durable id (a remote client names it). The row
    must belong to ``sessions`` (the caller's conversation), so a client cannot name another
    conversation's turn."""
    if mgr is None or not getattr(mgr, "enabled", False) or db is None or not isinstance(row_id, int):
        return _empty()
    row = db._read_one("SELECT timestamp, session_id, message_uid FROM messages WHERE id = ? AND role = 'user'",
                       (row_id,))
    lineage = undo_session_ids(db, *sessions)
    if row is None or row["session_id"] not in lineage:
        return _empty()
    since = _turn_started_at(db, row, lineage)
    return undo_turn_files(mgr, since, db, row["session_id"], until=_newest_undone_turn(db, since, lineage))


def _turn_started_at(db, row, lineage: set[str]) -> float:
    """The timestamp the turn's writes are keyed by. A compaction restatement replaces the in-flight
    request with a row stamped at compaction time but keeps its ``message_uid``, so the earliest row
    of the conversation carrying that uid is the one the turn began with."""
    uid = row["message_uid"]
    if not uid:
        return row["timestamp"]
    marks = ",".join("?" * len(lineage))
    first = db._read_one(f"SELECT MIN(timestamp) AS ts FROM messages WHERE message_uid = ? AND role = 'user' "
                         f"AND session_id IN ({marks})", (uid, *sorted(lineage)))
    return min(row["timestamp"], first["ts"]) if first is not None and first["ts"] is not None else row["timestamp"]


def _newest_undone_turn(db, since: float, lineage: set[str]) -> Optional[float]:
    """The newest user turn the rewind soft-deleted (rows go ``active = 0``): the upper bound of the
    revert. None when there is none to read (then nothing bounds it, as for an in-process undo)."""
    marks = ",".join("?" * len(lineage))
    row = db._read_one(f"SELECT MAX(timestamp) AS ts FROM messages WHERE role = 'user' AND active = 0 "
                       f"AND timestamp >= ? AND session_id IN ({marks})", (since, *sorted(lineage)))
    return row["ts"] if row is not None and row["ts"] is not None else None


def summarize(result: dict, cwd: Optional[str] = None, limit: int = 5) -> list[str]:
    """User-facing lines for a revert result (empty when nothing was reverted or kept)."""
    def show(paths: list[str]) -> str:
        names = []
        for p in paths[:limit]:
            try:
                names.append(str(Path(p).relative_to(cwd)) if cwd else p)
            except ValueError:
                names.append(p)
        more = f" (+{len(paths) - limit} more)" if len(paths) > limit else ""
        return ", ".join(names) + more
    from agent.i18n import t
    lines = []
    if result.get("reverted"):
        lines.append(t("checkpoints.undo.reverted", count=len(result["reverted"]), files=show(result["reverted"])))
    if result.get("skipped_user_edits"):
        lines.append(t("checkpoints.undo.kept_edits", files=show(result["skipped_user_edits"])))
    if result.get("not_captured"):
        lines.append(t("checkpoints.undo.not_captured", files=show(result["not_captured"])))
    if result.get("errors"):
        lines.append(t("checkpoints.undo.failed", error="; ".join(result["errors"][:3])))
    return lines
