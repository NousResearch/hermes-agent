"""Safe rollback applies its exact file plan without expanding Git pathspecs."""

from tools import checkpoint_manager as checkpoints


def test_safe_restore_does_not_expand_planned_names(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints, "CHECKPOINT_BASE", tmp_path / "checkpoints")
    work = tmp_path / "project"
    work.mkdir()
    (work / "AGENTS.md").write_text("Project instructions\n", encoding="utf-8")
    agent_file = work / "a[12].txt"
    user_file = work / "a1.txt"
    for path in (agent_file, user_file):
        path.write_text("original\n", encoding="utf-8")

    manager = checkpoints.CheckpointManager(enabled=True)
    assert manager.ensure_checkpoint(str(work))
    target = manager.list_checkpoints(str(work))[0]["hash"]
    for path in (agent_file, user_file):
        path.write_text("agent edit\n", encoding="utf-8")
        manager.record_agent_write(str(path))
    user_file.write_text("user edit\n", encoding="utf-8")

    result = manager.restore(str(work), target, safe=True)

    assert result["success"]
    assert user_file.read_text(encoding="utf-8") == "user edit\n"
    assert agent_file.read_text(encoding="utf-8") == "original\n"
    assert result["restored_files"] == [agent_file.name]
    assert result["skipped_user_edits"] == [user_file.name]


def test_file_scoped_restore_keeps_explicit_glob_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints, "CHECKPOINT_BASE", tmp_path / "checkpoints")
    work = tmp_path / "project"
    work.mkdir()
    selected = [work / "a1.txt", work / "a2.txt"]
    untouched = work / "other.txt"
    for path in (*selected, untouched):
        path.write_text("original\n", encoding="utf-8")

    manager = checkpoints.CheckpointManager(enabled=True)
    assert manager.ensure_checkpoint(str(work))
    target = manager.list_checkpoints(str(work))[0]["hash"]
    for path in (*selected, untouched):
        path.write_text("changed\n", encoding="utf-8")

    result = manager.restore(str(work), target, file_path="a*.txt")

    assert result["success"]
    assert all(path.read_text(encoding="utf-8") == "original\n" for path in selected)
    assert untouched.read_text(encoding="utf-8") == "changed\n"
