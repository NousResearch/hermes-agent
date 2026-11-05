"""``/undo`` never deletes or overwrites a file it cannot prove Hermes wrote in the undone turns.

Real CheckpointManager + real git store under a temp checkpoint base, driven through the same
entry point the agent loop uses before a file tool runs (``agent.tool_executor._ensure_file_checkpoint``),
then the write and ``record_agent_write`` the file tools' result hook makes.
"""
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.tool_executor import _ensure_file_checkpoint
from tools import checkpoint_manager as cm
from tools.checkpoint_turns import revert_turns


@pytest.fixture()
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "CHECKPOINT_BASE", tmp_path / "home" / "checkpoints")
    work = tmp_path / "proj"
    work.mkdir()
    (work / "keep.txt").write_text("original\n")
    return work


def _turn(mgr, ts: float, session: str = "a") -> None:
    mgr.new_turn()
    mgr.begin_turn(ts, session)


def _tool_write(mgr, work: Path, name: str, text: str) -> Path:
    path = work / name
    _ensure_file_checkpoint(SimpleNamespace(_checkpoint_mgr=mgr), "write_file", {"path": str(path)}, "default")
    path.write_text(text)
    mgr.record_agent_write(str(path))
    return path


def test_undo_treats_a_glob_looking_file_name_as_one_file(project):
    (project / "a*.txt").write_text("original wildcard\n")
    (project / "another.txt").write_text("original other\n")
    mgr = cm.CheckpointManager(enabled=True)
    _turn(mgr, 100.0)
    _tool_write(mgr, project, "a*.txt", "agent wildcard\n")
    (project / "another.txt").write_text("USER EDIT MUST SURVIVE\n")

    revert_turns(mgr, 100.0, {"a"})

    assert (project / "a*.txt").read_text() == "original wildcard\n"
    assert (project / "another.txt").read_text() == "USER EDIT MUST SURVIVE\n"


def test_undo_never_deletes_an_ignored_file_that_existed_before_the_turn(project):
    (project / ".gitignore").write_text("private.txt\n")
    (project / "private.txt").write_text("preexisting user data\n")
    mgr = cm.CheckpointManager(enabled=True)
    _turn(mgr, 100.0)
    _tool_write(mgr, project, ".gitignore", "")
    private = _tool_write(mgr, project, "private.txt", "agent edit\n")

    result = revert_turns(mgr, 100.0, {"a"})

    assert private.exists()  # absent from every snapshot, but NOT created by the turn
    assert str(private) in result["not_captured"]
    assert (project / ".gitignore").read_text() == "private.txt\n"


def test_undo_never_falls_back_to_an_older_turns_snapshot(project):
    mgr = cm.CheckpointManager(enabled=True)
    _turn(mgr, 100.0)
    _tool_write(mgr, project, "keep.txt", "A first\n")
    (project / "added.txt").write_text("user-created between turns\n")
    nested = project / "nested"
    nested.mkdir()
    subprocess.run(["git", "init", "-q", str(nested)], check=True, stdin=subprocess.DEVNULL)  # snapshots now fail
    _turn(mgr, 200.0)
    added = _tool_write(mgr, project, "added.txt", "agent edit\n")

    result = revert_turns(mgr, 100.0, {"a"})

    assert added.read_text() == "agent edit\n"  # turn 2 has no snapshot of its own: skipped, reported
    assert str(added) in result["not_captured"]
    # The pre-undo snapshot cannot be taken either, so nothing else is changed (reported instead).
    assert (project / "keep.txt").read_text() == "A first\n"
    assert result["reverted"] == [] and result["errors"]


def test_undo_skips_a_file_another_conversation_wrote_in_between(project):
    mgr = cm.CheckpointManager(enabled=True)
    _turn(mgr, 100.0, "a")
    _tool_write(mgr, project, "keep.txt", "A first\n")
    _turn(mgr, 200.0, "b")
    _tool_write(mgr, project, "keep.txt", "B unrelated conversation\n")
    _turn(mgr, 300.0, "a")
    keep = _tool_write(mgr, project, "keep.txt", "A last\n")

    result = revert_turns(mgr, 100.0, {"a"})

    assert keep.read_text() == "A last\n"  # restoring pre-A content would erase B's work
    assert result["reverted"] == [] and result["skipped_other_writers"] == [str(keep)]


def test_undo_still_reverts_and_deletes_what_the_turn_provably_wrote(project):
    """Control: the guards above leave the ordinary case working."""
    mgr = cm.CheckpointManager(enabled=True)
    _turn(mgr, 100.0)
    keep = _tool_write(mgr, project, "keep.txt", "edited\n")
    made = _tool_write(mgr, project, "made.txt", "created\n")

    result = revert_turns(mgr, 100.0, {"a"})

    assert keep.read_text() == "original\n" and not made.exists()
    assert sorted(Path(p).name for p in result["reverted"]) == ["keep.txt", "made.txt"]
