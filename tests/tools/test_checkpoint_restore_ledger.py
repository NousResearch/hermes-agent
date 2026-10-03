"""A rollback is a Hermes write: later safe rollbacks must recognize it."""

from pathlib import Path

import pytest

from tools.checkpoint_manager import CheckpointManager


@pytest.fixture
def history(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    work = tmp_path / "project"
    work.mkdir()
    (work / "nested").mkdir()
    target = work / "nested" / "data.txt"
    sibling = work / "notes.txt"
    manager = CheckpointManager(enabled=True, max_snapshots=50)
    checkpoints = []
    for version in ("original", "intermediate"):
        for path in (target, sibling):
            path.write_text(version, encoding="utf-8")
            manager.record_agent_write(str(path))
        manager.new_turn()
        assert manager.ensure_checkpoint(str(work), version)
        checkpoints.append(manager.list_checkpoints(str(work))[0]["hash"])
    for path in (target, sibling):
        path.write_text("latest", encoding="utf-8")
        manager.record_agent_write(str(path))
    return manager, work, target, sibling, checkpoints


@pytest.mark.parametrize("mode", ["safe", "full", "file", "directory"])
@pytest.mark.parametrize("destination", ["older", "undo"])
def test_safe_restore_recognizes_previous_rollback(history, mode, destination):
    manager, work, target, _, checkpoints = history
    untracked = work / "personal.txt"
    untracked.write_text("user-owned", encoding="utf-8")
    restore_path = {"file": str(target.relative_to(work)), "directory": "nested"}.get(mode)
    result = manager.restore(
        str(work), checkpoints[1], safe=mode == "safe",
        file_path=restore_path,
    )
    assert result["success"]
    assert target.read_text(encoding="utf-8") == "intermediate"
    undo = manager.list_checkpoints(str(work))[0]["hash"]
    # Reopening the manager also verifies the ownership update is durable.
    manager = CheckpointManager(enabled=True, max_snapshots=50)
    result = manager.restore(
        str(work), checkpoints[0] if destination == "older" else undo, safe=True,
    )
    assert result["success"]
    assert target.relative_to(work).as_posix() in result["restored_files"]
    assert untracked.name not in result["restored_files"]
    if destination == "older":
        assert untracked.name in result["skipped_user_edits"]
    assert untracked.read_text(encoding="utf-8") == "user-owned"
    assert target.read_text(encoding="utf-8") == (
        "original" if destination == "older" else "latest"
    )


def test_rollback_does_not_adopt_skipped_or_subsequent_user_edits(history):
    manager, work, target, sibling, checkpoints = history
    sibling.write_text("user edit before rollback", encoding="utf-8")
    result = manager.restore(str(work), checkpoints[1], safe=True)
    assert result["success"]
    assert result["skipped_user_edits"] == [sibling.name]
    assert target.read_text(encoding="utf-8") == "intermediate"
    target.write_text("user edit after rollback", encoding="utf-8")
    result = manager.restore(str(work), checkpoints[0], safe=True)
    assert result["success"]
    assert set(result["skipped_user_edits"]) == {target.relative_to(work).as_posix(), sibling.name}
    assert not result["restored_files"]
    assert target.read_text(encoding="utf-8") == "user edit after rollback"
    assert sibling.read_text(encoding="utf-8") == "user edit before rollback"
