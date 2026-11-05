"""``/undo`` file revert edge cases from review: per-file baselines, uncaptured oversize files,
repeated writes in one turn, snapshot budgets, the undo window's upper bound, compaction restatement.

Real CheckpointManager + real git store under a temp HERMES_HOME, driven the way the agent loop
and the file tools drive it (``begin_turn`` → ``ensure_checkpoint`` → write → ``record_agent_write``).
"""
from pathlib import Path

import pytest

from tools import checkpoint_manager as cm
from tools.checkpoint_turns import revert_turns


@pytest.fixture()
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "CHECKPOINT_BASE", tmp_path / "home" / "checkpoints")
    work = tmp_path / "proj"
    work.mkdir()
    (work / "keep.txt").write_text("original\n")
    return work


def _turn(mgr, ts: float, session: str = "s1") -> None:
    mgr.new_turn()
    mgr.begin_turn(ts, session)


def _agent_write(mgr, work: Path, name: str, text: str) -> Path:
    mgr.ensure_checkpoint(str(work), "before write_file")
    path = work / name
    path.write_text(text)
    mgr.record_agent_write(str(path))
    return path


def test_undo_n_keeps_a_file_someone_else_created_between_the_undone_turns(project):
    mgr = cm.CheckpointManager(enabled=True)
    _turn(mgr, 100.0)
    _agent_write(mgr, project, "keep.txt", "turn1\n")
    (project / "b.txt").write_text("the user made this\n")  # between the turns, not Hermes
    _turn(mgr, 200.0)
    _agent_write(mgr, project, "b.txt", "turn2 edit\n")

    result = revert_turns(mgr, 100.0, {"s1"})  # /undo 2

    assert (project / "keep.txt").read_text() == "original\n"
    assert (project / "b.txt").read_text() == "the user made this\n"  # back to its own pre-edit state
    assert sorted(Path(p).name for p in result["reverted"]) == ["b.txt", "keep.txt"]


def test_undo_never_deletes_a_file_the_size_cap_kept_out_of_the_checkpoint(project):
    mgr = cm.CheckpointManager(enabled=True, max_file_size_mb=1)
    big = project / "data.bin"
    big.write_bytes(b"x" * (2 * 1024 * 1024))  # exists before the turn, too big to capture
    _turn(mgr, 100.0)
    _agent_write(mgr, project, "data.bin", "small replacement\n")

    result = revert_turns(mgr, 100.0, {"s1"})

    assert big.exists()  # not mistaken for a file the turn created
    assert result["reverted"] == [] and result["not_captured"] == [str(big)]


def test_many_writes_in_one_turn_never_evict_an_earlier_turns_undo(project):
    mgr = cm.CheckpointManager(enabled=True)
    _turn(mgr, 100.0)
    _agent_write(mgr, project, "keep.txt", "turn1\n")
    _turn(mgr, 200.0)
    for i in range(12):
        _agent_write(mgr, project, "keep.txt", f"turn2 patch {i}\n")

    assert [Path(p).name for p in revert_turns(mgr, 200.0, {"s1"})["reverted"]] == ["keep.txt"]
    assert (project / "keep.txt").read_text() == "turn1\n"
    assert [Path(p).name for p in revert_turns(mgr, 100.0, {"s1"})["reverted"]] == ["keep.txt"]
    assert (project / "keep.txt").read_text() == "original\n"


def test_undo_keeps_the_snapshot_count_budget(project):
    mgr = cm.CheckpointManager(enabled=True, max_snapshots=1)
    _turn(mgr, 100.0)
    _agent_write(mgr, project, "keep.txt", "turn1\n")
    revert_turns(mgr, 100.0, {"s1"})

    # The ref itself, not list_checkpoints(), which already stops at max_snapshots.
    ref = cm._ref_name(cm._project_hash(str(project)))
    assert cm._run_git(["rev-list", "--count", ref], cm._store_path(), str(project))[1] == "1"


def test_a_turn_newer_than_the_undone_ones_is_not_reverted_with_them(project):
    mgr = cm.CheckpointManager(enabled=True)
    _turn(mgr, 100.0)
    _agent_write(mgr, project, "keep.txt", "turn1\n")
    _turn(mgr, 300.0)  # started by another client after the rewind
    newer = _agent_write(mgr, project, "newer.txt", "not undone\n")

    revert_turns(mgr, 100.0, {"s1"}, until=100.0)

    assert (project / "keep.txt").read_text() == "original\n"
    assert newer.read_text() == "not undone\n"


def test_undo_by_row_finds_a_turn_restated_by_compaction(project, tmp_path):
    """Compaction restates an in-flight request as a new row stamped at compaction time but keeps
    its ``message_uid``; undoing that row reverts the writes the turn made under the original stamp."""
    from hermes_state import SessionDB
    from tools.checkpoint_turns import undo_row_files
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="cli")
    original = db.append_message("s1", "user", "edit keep.txt")
    stored = db._read_one("SELECT timestamp, message_uid FROM messages WHERE id = ?", (original,))
    ts, uid = stored["timestamp"], stored["message_uid"]
    mgr = cm.CheckpointManager(enabled=True)
    _turn(mgr, ts, "s1")
    _agent_write(mgr, project, "keep.txt", "agent\n")
    restated = db.append_message("s1", "user", "[restated] edit keep.txt")
    with db._lock:
        db._conn.execute("UPDATE messages SET message_uid = ?, timestamp = ? WHERE id = ?", (uid, ts + 50, restated))
        db._conn.execute("UPDATE messages SET active = 0 WHERE id = ?", (original,))
        db._conn.commit()

    assert [Path(p).name for p in undo_row_files(mgr, db, restated, {"s1"})["reverted"]] == ["keep.txt"]
    assert (project / "keep.txt").read_text() == "original\n"


def test_diff_never_stages_an_oversize_file(project):
    mgr = cm.CheckpointManager(enabled=True, max_file_size_mb=1)
    mgr.ensure_checkpoint(str(project), "first")
    (project / "huge.bin").write_bytes(b"y" * (2 * 1024 * 1024))
    target = mgr.list_checkpoints(str(project))[0]["hash"]

    assert "huge.bin" not in mgr.diff(str(project), target).get("stat", "")


def test_newly_excluded_browser_profile_leaves_later_snapshots(project, monkeypatch):
    profile = project / "chrome_cdp_profile"
    profile.mkdir()
    (profile / "Cookies").write_text("cookie db\n")
    old = [e for e in cm.DEFAULT_EXCLUDES if e != "chrome_cdp_profile/"]
    monkeypatch.setattr(cm, "DEFAULT_EXCLUDES", old)  # a store created before the exclude existed
    mgr = cm.CheckpointManager(enabled=True)
    mgr.ensure_checkpoint(str(project), "old snapshot")
    monkeypatch.undo()
    monkeypatch.setattr(cm, "CHECKPOINT_BASE", project.parent / "home" / "checkpoints")
    (project / "keep.txt").write_text("changed\n")
    mgr.new_turn()
    mgr.ensure_checkpoint(str(project), "new snapshot")

    newest = mgr.list_checkpoints(str(project))[0]["hash"]
    store = cm._store_path()
    _, files, _ = cm._run_git(["ls-tree", "-r", "--name-only", newest], store, str(project))
    assert "chrome_cdp_profile/Cookies" not in files.split("\n")
