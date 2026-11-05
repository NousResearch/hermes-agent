"""``/undo`` reverts a file only on ownership recorded at write time (review of PR 149206, round 3).

Real CheckpointManager + real git store + real ``SessionDB`` under a temp checkpoint base; writes go
through the real file tools (``write_file_tool`` / ``patch_tool``) with the checkpoint manager bound
the way the tool executor binds it (``checkpoint_write_ledger.recording``), after the same
``_ensure_file_checkpoint`` the executor runs, and the post-tool ``record_agent_write`` hook.
"""
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.tool_executor import _ensure_file_checkpoint
from tools import checkpoint_manager as cm
from tools import checkpoint_write_ledger as write_ledger
from tools.checkpoint_turns import undo_row_files


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    monkeypatch.setattr(cm, "CHECKPOINT_BASE", tmp_path / "home" / "checkpoints")
    monkeypatch.setenv("HERMES_DISABLE_FILE_STATE_GUARD", "1")  # several "turns" share one task id here
    work = tmp_path / "proj"
    work.mkdir()
    (work / "keep.txt").write_text("original\n")
    db = SessionDB(db_path=tmp_path / "state.db")
    for sid in ("a", "b", "old"):
        db.create_session(sid, source="cli")
    monkeypatch.chdir(work)
    yield SimpleNamespace(work=work, db=db)
    db.close()


def _turn(env, session: str, ts: float = 100.0):
    """A user turn as the agent loop starts it: the stored row's timestamp, session and uid."""
    row = env.db.append_message(session, "user", f"turn in {session}", timestamp=ts)
    stored = env.db._read_one("SELECT timestamp, message_uid FROM messages WHERE id = ?", (row,))
    mgr = cm.CheckpointManager(enabled=True)
    mgr.new_turn()
    mgr.begin_turn(stored["timestamp"], session, stored["message_uid"])
    return mgr, row


def _tool(mgr, name: str, args: dict):
    from tools.file_tools import patch_tool, write_file_tool
    _ensure_file_checkpoint(SimpleNamespace(_checkpoint_mgr=mgr), name, args, "default")
    with write_ledger.recording(mgr):
        result = (write_file_tool if name == "write_file" else patch_tool)(**args)
    assert not json.loads(result).get("error"), result
    from agent.tool_dispatch_helpers import _extract_landed_file_mutation_paths
    for p in _extract_landed_file_mutation_paths(name, args, result):
        mgr.record_agent_write(p)
    return result


def _write(mgr, path: Path, text: str):
    return _tool(mgr, "write_file", {"path": str(path), "content": text})


def _rewind(env, session: str, row: int) -> None:
    with env.db._lock:
        env.db._conn.execute("UPDATE messages SET active = 0 WHERE session_id = ? AND id >= ?", (session, row))
        env.db._conn.commit()


def test_a_turn_sharing_its_timestamp_with_a_retained_turn_keeps_that_turns_files(env):
    """Finding 1: two platform messages with one timestamp; undoing the second keeps the first's file."""
    m1, _r1 = _turn(env, "a", 100.0)
    first = env.work / "first.txt"
    _write(m1, first, "first\n")
    m2, r2 = _turn(env, "a", 100.0)
    second = env.work / "second.txt"
    _write(m2, second, "second\n")
    _rewind(env, "a", r2)

    result = undo_row_files(m2, env.db, r2, {"a"})

    assert first.read_text() == "first\n"  # the retained turn's file survives
    assert not second.exists() and result["reverted"] == [str(second)]


def test_undoing_an_older_started_turn_never_hands_a_later_writer_s_file_to_an_older_owner(env):
    """Finding 2: A started before B but wrote after it; undo A, then undo an older turn: B's bytes stay."""
    shared = env.work / "shared.txt"
    old, r_old = _turn(env, "old", 50.0)
    _write(old, shared, "shared\n")
    a, r_a = _turn(env, "a", 100.0)
    b, r_b = _turn(env, "b", 200.0)
    _write(b, shared, "shared\n")  # same bytes as old's write
    _write(a, env.work / "a.txt", "A\n")
    _write(a, shared, "A\n")
    _rewind(env, "a", r_a)

    first = undo_row_files(a, env.db, r_a, {"a"})
    assert shared.read_text() == "shared\n" and str(shared) in first["reverted"]
    _rewind(env, "old", r_old)
    second = undo_row_files(old, env.db, r_old, {"old"})

    assert shared.read_text() == "shared\n"  # B wrote it after old's write: never put back to original
    assert str(shared) in second["skipped_other_writers"] and str(shared) not in second["reverted"]
    _rewind(env, "b", r_b)
    assert str(shared) in undo_row_files(b, env.db, r_b, {"b"})["reverted"]  # B still owns its write
    assert shared.read_text() == "shared\n"


def test_a_late_post_write_hook_never_claims_another_writers_bytes(env):
    """Finding 3: A's tool wrote, B wrote and recorded, then A's post-tool hook ran: undo A keeps B."""
    from tools.file_tools import write_file_tool
    path = env.work / "keep.txt"
    a, r_a = _turn(env, "a", 100.0)
    b, _ = _turn(env, "b", 200.0)
    _ensure_file_checkpoint(SimpleNamespace(_checkpoint_mgr=a), "write_file", {"path": str(path)}, "default")
    with write_ledger.recording(a):
        write_file_tool(str(path), "A\n")
    _write(b, path, "B\n")
    a.record_agent_write(str(path))  # A's delayed hook sees B's bytes on disk
    _rewind(env, "a", r_a)

    result = undo_row_files(a, env.db, r_a, {"a"})

    assert path.read_text() == "B\n"
    assert result["reverted"] == [] and str(path) in result["skipped_other_writers"]


def test_a_hand_edit_during_the_pre_undo_snapshot_is_kept(env):
    """Finding 4: the user edits the file while the pre-undo snapshot is being taken."""
    path = env.work / "keep.txt"
    a, r_a = _turn(env, "a", 100.0)
    _write(a, path, "A\n")
    _rewind(env, "a", r_a)
    real_take = a._take

    def take_after_edit(*args, **kwargs):
        path.write_text("USER EDIT DURING UNDO\n")
        return real_take(*args, **kwargs)
    a._take = take_after_edit

    result = undo_row_files(a, env.db, r_a, {"a"})

    assert path.read_text() == "USER EDIT DURING UNDO\n"
    assert result["reverted"] == [] and str(path) in result["skipped_user_edits"]


def test_a_moved_file_is_restored_whole_and_deleted_files_come_back(env):
    """Finding 6: a V4A Move restores the source and removes the destination together; a V4A
    Delete puts the deleted file back."""
    notes, doomed = env.work / "user_notes.txt", env.work / "doomed.txt"
    notes.write_text("PRECIOUS USER DATA\n")
    doomed.write_text("keep me\n")
    a, r_a = _turn(env, "a", 100.0)
    renamed = env.work / "renamed.txt"
    _tool(a, "patch", {"mode": "patch", "patch": (
        f"*** Begin Patch\n*** Move File: {notes} -> {renamed}\n*** Delete File: {doomed}\n*** End Patch\n")})
    assert renamed.exists() and not notes.exists() and not doomed.exists()
    _rewind(env, "a", r_a)

    result = undo_row_files(a, env.db, r_a, {"a"})

    assert notes.read_text() == "PRECIOUS USER DATA\n" and not renamed.exists()
    assert doomed.read_text() == "keep me\n"
    assert sorted(result["reverted"]) == sorted(map(str, (doomed, notes, renamed)))


def test_a_moved_file_whose_destination_was_edited_reverts_neither_half(env):
    """Finding 6 (skip side): one half cannot be reverted, so neither is; the content stays on disk."""
    notes = env.work / "user_notes.txt"
    notes.write_text("PRECIOUS USER DATA\n")
    a, r_a = _turn(env, "a", 100.0)
    renamed = env.work / "renamed.txt"
    _tool(a, "patch", {"mode": "patch", "patch": f"*** Begin Patch\n*** Move File: {notes} -> {renamed}\n*** End Patch\n"})
    renamed.write_text("PRECIOUS USER DATA\nplus a hand edit\n")
    _rewind(env, "a", r_a)

    result = undo_row_files(a, env.db, r_a, {"a"})

    assert renamed.read_text() == "PRECIOUS USER DATA\nplus a hand edit\n" and not notes.exists()
    assert result["reverted"] == [] and str(notes) in result["skipped_linked"]


def test_a_normal_turn_still_reverts_its_edits_and_deletes_what_it_created(env):
    """Control: the ordinary single-turn undo through the real tools."""
    keep = env.work / "keep.txt"
    a, r_a = _turn(env, "a", 100.0)
    _write(a, keep, "edited\n")
    _tool(a, "patch", {"mode": "replace", "path": str(keep), "old_string": "edited", "new_string": "patched"})
    made = env.work / "made.txt"
    _write(a, made, "created\n")
    _rewind(env, "a", r_a)

    result = undo_row_files(a, env.db, r_a, {"a"})

    assert keep.read_text() == "original\n" and not made.exists()
    assert sorted(result["reverted"]) == sorted([str(keep), str(made)])
    assert cm._load_ledger(cm._store_path(), cm._project_hash(str(env.work))) == {}  # exactly its writes dropped


def test_concurrent_writers_record_in_write_order(env):
    """Sequence numbers come from one store-wide counter: every record is distinct and ordered."""
    managers = [_turn(env, s, 100.0 + i)[0] for i, s in enumerate(("a", "b"))]
    errors = []

    def worker(mgr, name):
        try:
            for i in range(5):
                _write(mgr, env.work / f"{name}{i}.txt", f"{name}{i}\n")
        except Exception as exc:  # surfaced below
            errors.append(exc)
    threads = [threading.Thread(target=worker, args=(m, n)) for m, n in zip(managers, "ab")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ledger = cm._load_ledger(cm._store_path(), cm._project_hash(str(env.work)))
    seqs = [w["seq"] for e in ledger.values() for w in write_ledger.writes_of(e)]
    assert not errors and len(seqs) == 10 and len(set(seqs)) == 10
