"""Per-turn file revert for ``/undo``: put back the files Hermes wrote in the undone turns.

``/undo`` rewinds the conversation to before a user turn; with checkpoints on it also restores the
files Hermes' file tools wrote since that turn began. Two records in the shared checkpoint store make
that exact without guessing from timestamps:

* turn marks (``turns/<project>.json``): the tree a project was in when a turn first touched it,
  keyed by the turn's user-message timestamp (the same value its transcript row stores) and session;
* the agent-write ledger: each file's last Hermes-written hash, tagged with that turn, plus ``pre``:
  the file as it was right before the turn's first write to it (existed or not, its hash and blob id).

Reverting turn T touches only what Hermes can prove it wrote, file by file, and reports the rest:

* a file whose content changed since Hermes wrote it (a hand edit) is kept, as the safe ``/rollback`` does;
* a file another conversation or a newer turn wrote after the first undone write, or that changed
  between two undone writes, is skipped (``skipped_other_writers``): its pre-state is not "before these turns";
* a file the undone turns created (``pre.existed`` false) is deleted; absence from a snapshot never
  implies that (ignored, oversize and nested-repo files are absent from every snapshot);
* an edited file is restored only from the writing turn's own snapshot, and only when that snapshot
  holds exactly the recorded pre-write blob; otherwise it is reported ``not_captured``;
* nothing is changed unless the pre-undo snapshot holds the file's current content (undo the undo).

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


def record_turn_mark(store: Path, working_dir: str, tree: str, turn: dict) -> None:
    """Remember ``tree`` as ``working_dir``'s state when ``turn`` (``turn_ts``, ``session``) first
    touched it. Caller holds the store lock. Best effort:
    a lost mark only makes that turn's files unrevertible by ``/undo`` (``/rollback`` still has the
    checkpoint)."""
    from tools.checkpoint_manager import _project_hash
    dir_hash = _project_hash(working_dir)
    marks = _load_marks(store, dir_hash)
    mark = {"turn_ts": turn["turn_ts"], "session": turn.get("session") or "", "tree": tree, "ts": time.time()}
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
        workdir = workdirs.get(dir_hash)
        if not workdir or not Path(workdir).is_dir():
            out["not_captured"].extend(files)  # no snapshot of this project before the turn
            continue
        marks = _load_marks(store, dir_hash)
        plans = {}
        for abs_path in files:
            plan = _plan_file(ledger[abs_path], window, sessions, marks)
            if isinstance(plan, str):
                out[plan].append(abs_path)
            else:
                plans[abs_path] = plan
        _revert_project(mgr, store, workdir, dir_hash, plans, ledger, out, since)


def _plan_file(entry: dict, window: tuple[float, Optional[float]], sessions: set[str], marks: list[dict]):
    """What undoing ``window`` may do to one file, from its ledger writes alone: ``(pre, marks)``
    (``pre``: the file before the earliest undone write; ``marks``: that turn's own pre-state trees),
    or the result key naming why the file is left alone. Hermes only proves what it wrote: any write
    outside the undone turns after the first undone one, or a change between two undone writes,
    means the pre-state is not the file's state \"before these turns\" alone."""
    from tools.checkpoint_manager import _LEDGER_TURN_HISTORY
    history = [h for h in entry.get("history", []) if isinstance(h, dict)]
    writes = [*history, entry]
    first = next(i for i, w in enumerate(writes) if _in_scope(w, window[0], sessions, window[1]))
    if first == 0 and len(history) >= _LEDGER_TURN_HISTORY:
        return "not_captured"  # older undone writes may have aged out of the history
    later = writes[first:]
    if any(not _in_scope(w, window[0], sessions, window[1]) for w in later):
        return "skipped_other_writers"  # another conversation (or a newer turn) wrote it since
    pres = [w.get("pre") for w in later]
    if not all(isinstance(p, dict) and isinstance(p.get("existed"), bool) for p in pres):
        return "not_captured"  # its pre-write state was never recorded
    if any(pres[i]["sha256"] != later[i - 1].get("sha256") for i in range(1, len(later))):
        return "skipped_other_writers"  # changed outside Hermes between two undone turns
    turn = (later[0]["turn_ts"], later[0].get("session") or "")
    own = sorted((m for m in marks if (m.get("turn_ts"), m.get("session") or "") == turn), key=lambda m: m.get("ts", 0))
    return pres[0], own


def tree_oid(store: Path, workdir: str, tree: Optional[str], rel: str) -> Optional[str]:
    """Blob id ``tree`` holds at ``rel``, or None when it holds none."""
    from tools.checkpoint_manager import _git_out
    return (_git_out(["rev-parse", "--verify", "--quiet", f"{tree}:{rel}"], store, workdir, {1, 128}) or None) if tree else None


def worktree_oid(store: Path, workdir: str, rel: str) -> Optional[str]:
    """Blob id a snapshot would store for the file at ``rel`` now (the same clean conversion
    ``add`` applies), or None when it is missing or unreadable."""
    from tools.checkpoint_manager import _git_out
    if not (Path(workdir) / rel).is_file():
        return None
    return _git_out(["hash-object", "--", rel], store, workdir) or None


def _revert_project(mgr, store: Path, workdir: str, dir_hash: str, plans: dict[str, tuple],
                    ledger: dict, out: dict, since: float) -> None:
    from tools.checkpoint_manager import _LITERAL_PATHSPECS, _hash_file, _ref_name, _ref_tip, _run_git, _save_ledger
    restore: dict[str, list[str]] = {}  # baseline tree -> paths to check out of it
    delete = []
    for abs_path, (pre, own_marks) in plans.items():
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
        if not pre["existed"]:
            delete.append(rel)  # recorded at write time: the undone turns created it
            continue
        # Only the writing turn's own snapshot, and only when it holds the exact pre-write bytes:
        # an older turn's snapshot (or one taken before an intervening change) is not \"before".
        tree = next((m["tree"] for m in own_marks
                     if pre.get("oid") and tree_oid(store, workdir, m["tree"], rel) == pre["oid"]), None)
        if tree is None:
            out["not_captured"].append(abs_path)
        else:
            restore.setdefault(tree, []).append(rel)
    if not restore and not delete:
        return
    # "Undo the undo": /rollback lists this snapshot like any other. A file is changed only when
    # the snapshot now at the tip provably holds its current bytes.
    mgr._take(workdir, "pre-undo snapshot", prune=False)
    tip = _ref_tip(store, workdir, _ref_name(dir_hash))
    restore = {tree: kept for tree, rels in restore.items()
               if (kept := _preserved(store, workdir, tip, rels, out))}
    delete = _preserved(store, workdir, tip, delete, out)
    restored = []
    for tree, rels in restore.items():
        # A scratch index: the project's own index tracks its ref tip, not this older tree.
        scratch = store / "indexes" / f"{dir_hash}.undo"
        try:
            ok, _, err = _run_git(["checkout", tree, "--", *rels], store, workdir, index_file=scratch,
                                  extra_env=_LITERAL_PATHSPECS)
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


def _preserved(store: Path, workdir: str, tip: Optional[str], rels: list[str], out: dict) -> list[str]:
    """The ``rels`` whose current bytes the pre-undo snapshot holds (a missing file needs none);
    the rest are left untouched and reported."""
    kept = []
    for rel in rels:
        current = worktree_oid(store, workdir, rel)
        if current is None or tree_oid(store, workdir, tip, rel) == current:
            kept.append(rel)
        else:
            out["errors"].append(f"{rel}: no pre-undo snapshot holds its current content; left unchanged")
    return kept


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
    return {"reverted": [], "skipped_user_edits": [], "skipped_other_writers": [], "not_captured": [], "errors": []}


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
    if result.get("skipped_other_writers"):
        lines.append(t("checkpoints.undo.other_writers", files=show(result["skipped_other_writers"])))
    if result.get("not_captured"):
        lines.append(t("checkpoints.undo.not_captured", files=show(result["not_captured"])))
    if result.get("errors"):
        lines.append(t("checkpoints.undo.failed", error="; ".join(result["errors"][:3])))
    return lines
