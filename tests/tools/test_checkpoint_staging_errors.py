"""Failed staging must not expose a stale index as a successful diff or plan."""

import pytest


@pytest.mark.parametrize("operation", ["diff", "safe_restore_plan", "restore"])
def test_index_lock_failure_does_not_report_clean_worktree(
    tmp_path, monkeypatch, operation
):
    from tools import checkpoint_manager as cm

    monkeypatch.setattr(cm, "CHECKPOINT_BASE", tmp_path / "checkpoints")
    project = tmp_path / "project"
    project.mkdir()
    source = project / "report.txt"
    source.write_text("before", encoding="utf-8")
    manager = cm.CheckpointManager(enabled=True)
    assert manager.ensure_checkpoint(str(project))
    commit = manager.list_checkpoints(str(project))[0]["hash"]
    source.write_text("after", encoding="utf-8")
    manager.record_agent_write(str(source))
    refs = cm._project_refs(str(project))
    index_lock = refs.index_file.with_name(refs.index_file.name + ".lock")
    index_lock.write_text("held by another writer", encoding="utf-8")
    kwargs = {"safe": True} if operation == "restore" else {}
    result = getattr(manager, operation)(str(project), commit, **kwargs)
    assert not result["success"], result
    assert source.read_text(encoding="utf-8") == "after"
    # A failed operation must not remove a lock that may belong to another writer.
    assert index_lock.read_text(encoding="utf-8") == "held by another writer"
