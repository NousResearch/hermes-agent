"""The agent-write ledger's per-write records: who wrote a file, and the exact bytes before and after.

``/undo`` reverts a file only on proof recorded AT WRITE TIME, never reconstructed later from
timestamps or a re-read of the disk. Each successful ``write_file`` / ``patch`` appends one record
per path to ``ledger[path]["writes"]``, in write order:

* ``seq``: a per-store counter taken under the store lock, so records sort by when they landed;
* ``uid`` (the user turn's ``message_uid``) or ``turn_ts`` + ``session``: the turn that wrote it;
* ``pre``: the file right before the write, captured inside the file tool's per-path lock
  (``existed``, ``sha256``, the snapshot blob id ``oid`` when a checkpoint can hold it);
* ``post``: what the write left (``existed``, ``sha256`` of the bytes the TOOL wrote);
* ``link``: shared by the two halves of a V4A move, which ``/undo`` reverts together or not at all.

The file tools record through :func:`recording`, which the tool executor binds around a
``write_file`` / ``patch`` dispatch; outside it (another surface calling the tools, a container
backend) nothing is recorded and ``/undo`` treats the file as not Hermes'. The top-level
``sha256`` stays the last Hermes-written hash safe ``/rollback`` reads.
"""
from __future__ import annotations

import contextlib
import json
import logging
import stat
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# Records kept per file; once older ones drop, an undo reaching past them is reported, not guessed.
MAX_WRITES_PER_FILE = 50
_SEQ_NAME = ".seq"
_LOCK_WAIT_S = 10.0

_ACTIVE: ContextVar = ContextVar("hermes_checkpoint_write_recorder", default=None)


@contextlib.contextmanager
def recording(mgr):
    """Bind ``mgr`` (an enabled CheckpointManager with a turn) for the file tool running inside."""
    token = _ACTIVE.set(mgr if mgr is not None and getattr(mgr, "enabled", False) else None)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def active():
    return _ACTIVE.get()


def last_seq(ledger: dict, path: str) -> int:
    writes = writes_of(ledger.get(path))
    return writes[-1].get("seq", 0) if writes else 0


def capture(paths: Iterable[str]) -> dict[str, dict]:
    """Each path's pre-image right now (call inside the tool's path lock, before the write)."""
    mgr = _ACTIVE.get()
    if mgr is None:
        return {}
    return {str(p): pre for p in dict.fromkeys(str(p) for p in paths if p) if (pre := mgr.capture_pre(p)) is not None}


def record(pres: dict[str, dict], posts: dict[str, dict], links: Optional[dict[str, str]] = None) -> None:
    """Append one record per path the tool call changed (call inside the same lock, after the write)."""
    mgr = _ACTIVE.get()
    if mgr is None or not pres:
        return
    mgr.record_exact_writes([(path, pre, posts.get(path), (links or {}).get(path)) for path, pre in pres.items()])


def post_of(path: str, sha256: Optional[str]) -> dict:
    """The post-image of a write: ``sha256`` is the tool's own hash of the bytes it wrote; only when
    the tool had none is the file hashed here, still inside the write's critical section."""
    if sha256 is None:
        from tools.checkpoint_manager import _hash_file
        sha256 = _hash_file(Path(path))
    return {"existed": True, "sha256": sha256}


def v4a_posts(patch_text: str, writes, pres: dict[str, dict]) -> tuple[dict[str, dict], dict[str, str]]:
    """Each path's state after a V4A patch, replayed op by op from the pre-images and the tool's own
    written hashes (``PatchResult._writes``), plus the shared link of every move's two halves."""
    from tools.patch_parser import OperationType, parse_v4a_patch
    ops, err = parse_v4a_patch(patch_text)
    if err:
        return {}, {}
    state: dict[str, dict] = {}
    pending = list(writes or ())
    links: dict[str, str] = {}
    for op in ops:
        if op.operation in (OperationType.UPDATE, OperationType.ADD):
            hit = next((w for w in pending if w[0] == op.file_path), None)
            if hit is not None:
                pending.remove(hit)
            state[op.file_path] = post_of(op.file_path, hit[2] if hit else None)
        elif op.operation is OperationType.DELETE:
            state[op.file_path] = {"existed": False, "sha256": None}
        elif op.operation is OperationType.MOVE and op.new_path:
            moved = state.get(op.file_path) or pres.get(op.file_path)
            state[op.new_path] = ({"existed": True, "sha256": moved.get("sha256")} if moved and moved.get("existed")
                                  else post_of(op.new_path, None))
            state[op.file_path] = {"existed": False, "sha256": None}
            link = links.get(op.file_path) or links.get(op.new_path) or uuid.uuid4().hex
            links[op.file_path] = links[op.new_path] = link
    return state, links


def move_links(patch_text) -> dict[str, str]:
    """``{path: link}`` for both halves of every ``*** Move File:`` header in a V4A patch."""
    if not isinstance(patch_text, str) or "Move" not in patch_text:
        return {}
    from tools.patch_parser import OperationType, parse_v4a_patch
    ops, err = parse_v4a_patch(patch_text)
    links: dict[str, str] = {}
    for op in [] if err else ops:
        if op.operation is OperationType.MOVE and op.new_path:
            links[op.file_path] = links[op.new_path] = uuid.uuid4().hex
    return links


def pre_image(mgr, path: Path, workdir: str) -> dict:
    """``existed`` / ``sha256`` of ``path`` as it is now, plus ``oid``: its blob id, kept only when the
    project's newest checkpoint (the one taken before this tool ran) holds exactly these bytes. Without
    it (a symlink, a file over the size cap, a snapshot that failed or predates a change) undo reports
    the file instead of guessing where its earlier content lives."""
    from tools.checkpoint_manager import _hash_file
    existed = path.exists() or path.is_symlink()
    pre: dict = {"existed": existed, "sha256": None, "oid": None}
    if not existed:
        return pre
    pre["sha256"] = _hash_file(path)
    if pre["sha256"] is None or path.is_symlink() or mgr._exceeds_size_cap(path):
        return pre
    from tools.checkpoint_manager import _project_hash, _ref_name, _ref_tip, _store_path
    from tools.checkpoint_turns import tree_oid, worktree_oid
    store, rel = _store_path(), path.relative_to(workdir).as_posix()
    oid = worktree_oid(store, workdir, rel)
    tip = _ref_tip(store, workdir, _ref_name(_project_hash(workdir))) if oid else None
    if oid and tree_oid(store, workdir, tip, rel) == oid:
        pre["oid"] = oid
        pre["exec"] = bool(path.stat().st_mode & stat.S_IXUSR)
    return pre


@contextlib.contextmanager
def store_lock_waiting(timeout: float = _LOCK_WAIT_S):
    """The checkpoint store lock, waiting up to ``timeout`` for a snapshot or undo to finish (a write
    record dropped because the lock was busy would leave that write unrevertible)."""
    from tools.checkpoint_manager import _resolve_checkpoint_base
    from tools.checkpoint_pruning import PruneError, store_lock
    deadline = time.monotonic() + timeout
    while True:
        try:
            lock = store_lock(_resolve_checkpoint_base())
            lock.__enter__()
            break
        except PruneError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)
    try:
        yield
    finally:
        lock.__exit__(None, None, None)


def next_seq(store: Path) -> int:
    """The next per-store write sequence number. Caller holds the store lock."""
    path = store / "ledgers" / _SEQ_NAME
    try:
        seq = int(path.read_text(encoding="utf-8")) + 1
    except (OSError, ValueError):
        seq = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(str(seq), encoding="utf-8")
    tmp.replace(path)
    return seq


def append(ledger: dict, path: str, write: dict) -> None:
    """Add ``write`` to ``path``'s record (keeping the newest :data:`MAX_WRITES_PER_FILE`). A record
    another Hermes writer landed after this write's pre-image was taken makes that pre-image no proof
    of "before this write" alone: the write is flagged ``interleaved`` and ``/undo`` skips it."""
    entry = ledger.get(path) if isinstance(ledger.get(path), dict) else {}
    writes = [w for w in entry.get("writes", []) if isinstance(w, dict)]
    pre = write.get("pre")
    seen = pre.pop("after_seq", None) if isinstance(pre, dict) else None
    if writes and (seen is None or writes[-1].get("seq", 0) > seen):
        write["interleaved"] = True
    writes.append(write)
    evicted = bool(entry.get("evicted")) or len(writes) > MAX_WRITES_PER_FILE
    ledger[path] = {"sha256": write["post"].get("sha256"), "ts": time.time(),
                    "writes": writes[-MAX_WRITES_PER_FILE:], **({"evicted": True} if evicted else {})}


def writes_of(entry) -> list[dict]:
    return [w for w in entry.get("writes", []) if isinstance(w, dict)] if isinstance(entry, dict) else []


def dump(ledger: dict) -> str:
    return json.dumps(ledger)
