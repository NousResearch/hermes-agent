"""Per-turn file revert for ``/undo``: put back the files Hermes wrote in the undone turns.

``/undo`` rewinds the conversation to before a user turn; with checkpoints on it also restores the
files Hermes' file tools wrote in the undone turns. The proof is the agent-write ledger's per-write
records (``tools/checkpoint_write_ledger.py``), each appended AT WRITE TIME with a store-wide sequence
number, the writing turn (its user message's ``message_uid``), and the exact pre- and post-images.

The undone turns are a set of turn identities (the rewound row's uid and every later turn's), never a
timestamp range: two turns may share a platform timestamp. For each file, revert only when

* every write since the first undone one belongs to the undone turns, none interleaved with another
  writer, and each undone write starts from the bytes the previous one left (else
  ``skipped_other_writers``);
* the file holds exactly what the last undone write left (else ``skipped_user_edits``, a hand edit);
* the first undone write's pre-image is available: "did not exist" (delete), or a blob a checkpoint
  stored (restore); else ``not_captured``;
* the two halves of a V4A move both qualify (else both ``skipped_linked``).

Nothing changes unless the pre-undo snapshot holds the file's current bytes (undo the undo), and each
file is re-checked immediately before it is written, under the file tools' path lock and the store
lock, so an edit landing during the snapshot is kept. The undone writes are then dropped from the
ledger by sequence number. Only Hermes' file tools feed the ledger; files a terminal command changed
are left to ``/rollback``, which restores whole checkpoints.
"""
from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

# A write's membership in the undone turns: True, False, or None when its turn cannot be told
# apart from a retained one (a write recorded without a turn uid, sharing a retained turn's stamp).
Scope = Callable[[dict], Optional[bool]]


@dataclass
class _Plan:
    rel: str
    seqs: set
    pre: dict
    post: dict
    links: set = field(default_factory=set)


def _state(path: Path) -> dict:
    from tools.checkpoint_manager import _hash_file
    if not (path.exists() or path.is_symlink()):
        return {"existed": False, "sha256": None}
    return {"existed": True, "sha256": _hash_file(path)}


def _same(a: Any, b: Any) -> bool:
    return isinstance(a, dict) and isinstance(b, dict) and a.get("existed") == b.get("existed") and (
        not a.get("existed") or (a.get("sha256") is not None and a.get("sha256") == b.get("sha256")))


def revert_turns(mgr, since: float, sessions: Iterable[str], until: Optional[float] = None) -> dict:
    """Restore the files Hermes wrote in turns of the conversation ``sessions`` (its session ids,
    compression lineage included) that started at or after ``since`` and, with ``until``, no later.
    The in-process fallback when no stored row names the turn; :func:`undo_row_files` is exact.

    Returns ``{"reverted": [abs paths], "skipped_user_edits": [...], "skipped_other_writers": [...],
    "skipped_linked": [...], "not_captured": [...], "errors": [...]}``.
    """
    sessions = {s for s in sessions if s}
    if mgr is None or not getattr(mgr, "enabled", False) or not sessions:
        return _empty()  # no conversation named: never revert another conversation's writes
    low = float(since)

    def scope(w: dict) -> bool:
        ts = w.get("turn_ts")
        return (w.get("session") in sessions and isinstance(ts, (int, float)) and ts >= low
                and (until is None or ts <= until))
    return _revert(mgr, scope)


def _revert(mgr, scope: Scope) -> dict:
    from tools.checkpoint_pruning import PruneError
    out = _empty()
    try:
        _revert_all(mgr, scope, out)
    except (PruneError, OSError) as exc:
        out["errors"].append(str(exc))
    return out


def _revert_all(mgr, scope: Scope, out: dict) -> None:
    from tools.checkpoint_manager import _list_projects, _load_ledger, _store_path
    from tools.checkpoint_write_ledger import store_lock_waiting, writes_of
    store = _store_path()
    if not (store / "ledgers").is_dir():
        return
    has_store = (store / "HEAD").exists()  # without one no checkpoint holds any pre-write bytes
    workdirs = {meta["_hash"]: meta.get("workdir") for meta in _list_projects(store)} if has_store else {}
    for ledger_path in sorted((store / "ledgers").glob("*.json")):
        dir_hash = ledger_path.stem
        # Unlocked read only to learn which paths to lock; the plan is re-made under the locks.
        paths = [p for p, e in _load_ledger(store, dir_hash).items() if any(scope(w) is not False for w in writes_of(e))]
        if not paths:
            continue
        # Path locks before the store lock: the file tools take them in that order.
        with _path_locks(paths), store_lock_waiting():
            _revert_project(mgr, store, dir_hash, workdirs.get(dir_hash), scope, out)


@contextlib.contextmanager
def _path_locks(paths: list[str]):
    from tools import file_state
    with contextlib.ExitStack() as stack:
        for path in sorted(set(paths)):
            stack.enter_context(file_state.lock_path(path))
        yield


def _plan_file(path: str, entry: dict, scope: Scope):
    """What undoing ``scope`` may do to one file, from its write records: a :class:`_Plan`, the
    result key naming why it is left alone, or None when the undone turns never wrote it."""
    from tools.checkpoint_write_ledger import writes_of
    writes = writes_of(entry)
    first = next((i for i, w in enumerate(writes) if scope(w) is not False), None)
    if first is None:
        return None
    if any(scope(w) is None for w in writes[first:]):
        return "skipped_other_writers"  # a retained turn may have written it: never guess
    if first == 0 and entry.get("evicted"):
        return "not_captured"  # older writes aged out: an undone one may be among them
    tail = writes[first:]
    if not all(scope(w) is True and not w.get("interleaved") for w in tail):
        return "skipped_other_writers"  # another conversation or a later turn wrote it since
    pres = [w.get("pre") for w in tail]
    if not all(isinstance(p, dict) and isinstance(p.get("existed"), bool) for p in pres):
        return "not_captured"  # its pre-write state was never recorded
    if not all(_same(pres[i], tail[i - 1].get("post")) for i in range(1, len(tail))):
        return "skipped_other_writers"  # changed outside Hermes between two undone writes
    if not _same(_state(Path(path)), tail[-1].get("post")):
        return "skipped_user_edits"  # changed since Hermes wrote it: keep
    if pres[0]["existed"] and not pres[0].get("oid"):
        return "not_captured"  # a symlink, an oversize or unreadable file: no checkpoint holds it
    return _Plan(rel="", seqs={w.get("seq") for w in tail}, pre=pres[0], post=tail[-1]["post"],
                 links={w["link"] for w in tail if w.get("link")})


def _plan_project(store: Path, workdir: str, ledger: dict, scope: Scope, out: dict) -> dict[str, _Plan]:
    plans: dict[str, _Plan] = {}
    for path, entry in ledger.items():
        plan = _plan_file(path, entry, scope)
        if plan is None:
            continue
        if isinstance(plan, str):
            out[plan].append(path)
            continue
        try:
            plan.rel = Path(path).relative_to(workdir).as_posix()
        except ValueError:
            out["not_captured"].append(path)
            continue
        oid = plan.pre.get("oid")
        if oid and not _blob_exists(store, workdir, oid):
            out["not_captured"].append(path)  # no checkpoint ever stored those bytes (or it was pruned)
            continue
        plans[path] = plan
    return _close_links(plans, ledger, out)


def _blob_exists(store: Path, workdir: str, oid: str) -> bool:
    from tools.checkpoint_manager import _run_git
    return _run_git(["cat-file", "-e", f"{oid}^{{blob}}"], store, workdir, allowed_returncodes={1, 128})[0]


def _close_links(plans: dict[str, _Plan], ledger: dict, out: dict) -> dict[str, _Plan]:
    """Keep a moved file only together with its other half: a link whose every path is planned."""
    from tools.checkpoint_write_ledger import writes_of
    members: dict[str, set] = {}
    for path, entry in ledger.items():
        for w in writes_of(entry):
            if w.get("link"):
                members.setdefault(w["link"], set()).add(path)
    dropped: set[str] = set()
    changed = True
    while changed:
        changed = False
        for path, plan in plans.items():
            if path in dropped:
                continue
            if any(not (members.get(link, set()) <= set(plans) - dropped) for link in plan.links):
                dropped.add(path)
                changed = True
    out["skipped_linked"].extend(sorted(dropped))
    return {p: plan for p, plan in plans.items() if p not in dropped}


def _groups(plans: dict[str, _Plan]) -> list[list[str]]:
    """Planned paths, the halves of each move together."""
    groups: list[set] = []
    for path, plan in plans.items():
        joined = [g for g in groups if any(plans[p].links & plan.links for p in g)] if plan.links else []
        merged = {path}.union(*joined)
        groups = [g for g in groups if g not in joined] + [merged]
    return [sorted(g) for g in groups]


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


def _revert_project(mgr, store: Path, dir_hash: str, workdir: Optional[str], scope: Scope, out: dict) -> None:
    from tools.checkpoint_manager import _load_ledger, _ref_name, _ref_tip, _save_ledger
    ledger = _load_ledger(store, dir_hash)
    if not workdir or not Path(workdir).is_dir():
        out["not_captured"].extend(p for p, e in ledger.items() if _plan_file(p, e, scope) is not None)
        return
    plans = _plan_project(store, workdir, ledger, scope, out)
    if not plans:
        return
    # "Undo the undo": /rollback lists this snapshot like any other. A file is changed only when
    # the snapshot now at the tip provably holds its current bytes (a failed snapshot holds none).
    mgr._take(workdir, "pre-undo snapshot", prune=False)
    tip = _ref_tip(store, workdir, _ref_name(dir_hash))
    unsaved = [plan.rel for plan in plans.values() if not _preserved(store, workdir, tip, plan.rel)]
    if unsaved:  # the snapshot failed (or missed a file): change nothing in this project
        out["errors"].append(f"{', '.join(unsaved[:5])}: no pre-undo snapshot holds the current content; "
                             "no file was changed")
        return
    for group in _groups(plans):
        # Re-checked right before writing: an edit that landed during the snapshot is kept.
        if not all(_same(_state(Path(p)), plans[p].post) for p in group):
            out["skipped_user_edits"].extend(group)
            continue
        for path in group:
            if _apply(store, workdir, path, plans[path], out):
                out["reverted"].append(path)
                _drop_writes(ledger, path, plans[path])
    _save_ledger(store, dir_hash, ledger)
    # The pre-undo snapshot skipped pruning so the restore could not lose its blobs; budgets now.
    mgr._prune(store, workdir, _ref_name(dir_hash))


def _preserved(store: Path, workdir: str, tip: Optional[str], rel: str) -> bool:
    """Whether the pre-undo snapshot holds the file's current bytes (a missing file needs none)."""
    path = Path(workdir) / rel
    if not (path.exists() or path.is_symlink()):
        return True
    current = worktree_oid(store, workdir, rel)
    return current is not None and tree_oid(store, workdir, tip, rel) == current


def _apply(store: Path, workdir: str, path: str, plan: _Plan, out: dict) -> bool:
    """Write the first undone write's pre-image: delete, or the stored blob."""
    from tools.checkpoint_manager import _LITERAL_PATHSPECS, _hash_file, _run_git
    target = Path(path)
    if not plan.pre["existed"]:
        try:
            target.unlink(missing_ok=True)
            return True
        except OSError as exc:
            out["errors"].append(f"{plan.rel}: {exc}")
            return False
    scratch = store / "indexes" / f"undo-{plan.pre['oid'][:16]}"
    mode = "100755" if plan.pre.get("exec") else "100644"
    try:
        ok, _, err = _run_git(["update-index", "--add", "--cacheinfo", f"{mode},{plan.pre['oid']},{plan.rel}"],
                              store, workdir, index_file=scratch, extra_env=_LITERAL_PATHSPECS)
        if ok:
            ok, _, err = _run_git(["checkout-index", "-f", "--", plan.rel], store, workdir, index_file=scratch,
                                  extra_env=_LITERAL_PATHSPECS)
    finally:
        scratch.unlink(missing_ok=True)
    if not ok:
        out["errors"].append(err or f"{plan.rel}: checkout failed")
        return False
    if _hash_file(target) != plan.pre.get("sha256"):
        out["errors"].append(f"{plan.rel}: restored content differs from the recorded pre-write bytes")
        return False
    return True


def _drop_writes(ledger: dict, path: str, plan: _Plan) -> None:
    """Remove exactly the undone writes. The file now holds the first one's pre-image: an earlier
    Hermes write when it left those bytes (so undoing its turn later still reverts it), else content
    Hermes never wrote (no hash: a later safe /rollback leaves it alone)."""
    from tools.checkpoint_write_ledger import writes_of
    entry = ledger.get(path)
    keep = [w for w in writes_of(entry) if w.get("seq") not in plan.seqs]
    if not keep:
        ledger.pop(path, None)
        return
    entry["writes"] = keep
    if _same(plan.pre, keep[-1].get("post")):
        entry["sha256"] = keep[-1]["post"].get("sha256")
    else:
        entry.pop("sha256", None)


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
    return {"reverted": [], "skipped_user_edits": [], "skipped_other_writers": [], "skipped_linked": [],
            "not_captured": [], "errors": []}


def undo_turn_files(mgr, since, db, *session_ids: Optional[str], until: Optional[float] = None) -> dict:
    """``/undo``'s file half for an in-process surface with no stored row to name the turn: revert
    what the conversation's turns from the rewound user message's timestamp ``since`` onward wrote.
    Empty when checkpoints are off, ``since`` is unknown, or the turns wrote nothing."""
    if mgr is None or not getattr(mgr, "enabled", False) or not isinstance(since, (int, float)):
        return _empty()
    return revert_turns(mgr, float(since), undo_session_ids(db, *session_ids), until)


def undo_row_files(mgr, db, row_id: Optional[int], sessions: Iterable[str]) -> dict:
    """Same, addressed by the rewound user row's durable id: revert exactly the writes of that turn
    and every later one of the conversation. The row must belong to ``sessions`` (the caller's
    conversation), so a client cannot name another conversation's turn."""
    if mgr is None or not getattr(mgr, "enabled", False) or db is None or not isinstance(row_id, int):
        return _empty()
    scope = _row_scope(db, row_id, undo_session_ids(db, *sessions))
    return _empty() if scope is None else _revert(mgr, scope)


def _row_scope(db, row_id: int, lineage: set[str]) -> Optional[Scope]:
    """The undone turns' identities: user rows of the conversation from the rewound turn's first row
    on (a compaction restatement keeps its ``message_uid``, so its original row counts), minus any
    identity a retained earlier row also carries. A write is undone when its turn's uid is, or, for a
    turn begun without one, when its exact (timestamp, session) is and no retained turn shares it."""
    row = db._read_one("SELECT id, session_id, message_uid FROM messages WHERE id = ? AND role = 'user'", (row_id,))
    if row is None or row["session_id"] not in lineage:
        return None
    marks = ",".join("?" * len(lineage))
    rows = db._read_all(f"SELECT id, timestamp, session_id, message_uid, absorbed_message_uids, active FROM messages "
                        f"WHERE role = 'user' AND session_id IN ({marks})", tuple(sorted(lineage)))
    uid = row["message_uid"]
    start = min((r["id"] for r in rows if uid and r["message_uid"] == uid), default=row["id"])
    kept = [r for r in rows if r["id"] < start]
    # Later rows the rewind soft-deleted; an active one is a turn another client started after it.
    undone = [r for r in rows if r["id"] >= start and (not r["active"] or r["id"] == row["id"] or r["message_uid"] == uid)]
    uids = _row_uids(undone) - _row_uids(kept)
    pairs = {(r["timestamp"], r["session_id"]) for r in undone}
    kept_pairs = {(r["timestamp"], r["session_id"]) for r in kept}

    def scope(w: dict) -> Optional[bool]:
        if w.get("uid"):
            return w["uid"] in uids
        pair = (w.get("turn_ts"), w.get("session"))
        return None if pair in pairs and pair in kept_pairs else pair in pairs
    return scope


def _row_uids(rows) -> set[str]:
    """Each row's uid plus those of the user messages a merge folded into it (their turns' writes)."""
    from agent.message_metadata import uid_list
    out: set[str] = set()
    for r in rows:
        out.update(u for u in (r["message_uid"],) if u)
        try:
            out.update(uid_list(json.loads(r["absorbed_message_uids"] or "[]")))
        except ValueError:
            pass
    return out


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
    keys = (("reverted", "checkpoints.undo.reverted"), ("skipped_user_edits", "checkpoints.undo.kept_edits"),
            ("skipped_other_writers", "checkpoints.undo.other_writers"), ("skipped_linked", "checkpoints.undo.linked"),
            ("not_captured", "checkpoints.undo.not_captured"))
    lines = [t(key, count=len(result[name]), files=show(result[name])) for name, key in keys if result.get(name)]
    if result.get("errors"):
        lines.append(t("checkpoints.undo.failed", error="; ".join(result["errors"][:3])))
    return lines
