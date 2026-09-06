"""Checkpoint file names must survive Git's machine-readable path output."""

from tools import checkpoint_manager as checkpoints


def test_safe_restore_preserves_exact_paths_and_user_edits(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints, "CHECKPOINT_BASE", tmp_path / "checkpoints")
    work = tmp_path / "project"
    work.mkdir()
    (work / "AGENTS.md").write_text("Project instructions\n", encoding="utf-8")
    names = [" leading.txt", "报告.txt", "ordinary.txt"]
    for name in names:
        (work / name).write_text("original\n", encoding="utf-8")
    preserved = work / " 用户修改.txt"
    preserved.write_text("original\n", encoding="utf-8")
    manager = checkpoints.CheckpointManager(enabled=True)
    assert manager.ensure_checkpoint(str(work))
    target = manager.list_checkpoints(str(work))[0]["hash"]

    for name in names:
        (work / name).write_text("agent edit\n", encoding="utf-8")
        manager.record_agent_write(str(work / name))
    preserved.write_text("agent edit\n", encoding="utf-8")
    manager.record_agent_write(str(preserved))
    preserved.write_text("user edit\n", encoding="utf-8")

    result = manager.restore(str(work), target, safe=True)

    assert result["success"]
    assert set(result["restored_files"]) == set(names)
    assert result["skipped_user_edits"] == [preserved.name]
    assert all((work / name).read_text(encoding="utf-8") == "original\n" for name in names)
    assert preserved.read_text(encoding="utf-8") == "user edit\n"


def test_size_cap_applies_to_leading_space_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints, "CHECKPOINT_BASE", tmp_path / "checkpoints")
    work = tmp_path / "project"
    work.mkdir()
    (work / "small.txt").write_text("keep\n", encoding="utf-8")
    name = " oversized.bin"
    (work / name).write_bytes(b"x" * (1024 * 1024 + 1))
    manager = checkpoints.CheckpointManager(enabled=True, max_file_size_mb=1)
    assert manager.ensure_checkpoint(str(work))
    target = manager.list_checkpoints(str(work))[0]["hash"]
    store = checkpoints._store_path()

    included, _, _ = checkpoints._run_git(
        ["cat-file", "-e", f"{target}:small.txt"], store, str(work)
    )
    oversized_included, _, _ = checkpoints._run_git(
        ["cat-file", "-e", f"{target}:{name}"], store, str(work), allowed_returncodes={128}
    )
    assert included
    assert not oversized_included
