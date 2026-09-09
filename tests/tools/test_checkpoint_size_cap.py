"""Real history retention and failure propagation at the public pruning callers."""
from __future__ import annotations

import os

import pytest

from tools import checkpoint_manager as checkpoints


@pytest.fixture
def history(tmp_path, monkeypatch):
    base = tmp_path / "checkpoints"
    monkeypatch.setattr(checkpoints, "CHECKPOINT_BASE", base)
    project = min((tmp_path / name for name in ("project-a", "project-b")),
                  key=lambda path: checkpoints._project_hash(str(path)))
    project.mkdir()
    manager = checkpoints.CheckpointManager(enabled=True, max_total_size_mb=0)
    blob = project / "old.bin"
    blob.write_bytes(os.urandom(1_300_000))
    assert manager.ensure_checkpoint(str(project), "large-oldest")
    blob.unlink()
    for index in range(5):
        (project / "small.txt").write_text(str(index), encoding="utf-8")
        manager.new_turn()
        assert manager.ensure_checkpoint(str(project), f"small-{index}")
    return base, project, manager


@pytest.mark.parametrize("entry", ["maintenance", "snapshot"])
def test_size_cap_retains_every_snapshot_after_the_only_large_one(history, entry):
    base, project, manager = history
    store = checkpoints._store_path(base)
    assert checkpoints._dir_size_bytes(store) > 1024 * 1024
    expected = [f"small-{index}" for index in reversed(range(5))]
    if entry == "maintenance":
        result = checkpoints.prune_checkpoints(retention_days=0, checkpoint_base=base, max_total_size_mb=1)
        assert result["errors"] == 0
    else:
        manager.max_total_size_mb = 1
        (project / "new.txt").write_text("new snapshot", encoding="utf-8")
        manager.new_turn()
        assert manager.ensure_checkpoint(str(project), "newest")
        expected.insert(0, "newest")
    assert [row["reason"] for row in manager.list_checkpoints(str(project))] == expected
    assert checkpoints._dir_size_bytes(store) <= 1024 * 1024
    assert (project / "small.txt").read_text(encoding="utf-8") == "4"


def test_reclaim_stops_before_touching_another_projects_history(history):
    base, first, manager = history
    # Real ref sorting chooses the first victim. Pick the other project so
    # this is a discriminating reclaim-before-next-project assertion.
    other = next(first.parent / name for name in ("project-a", "project-b") if first.name != name)
    other.mkdir()
    for index in range(3):
        (other / "small.txt").write_text(str(index), encoding="utf-8")
        manager.new_turn()
        assert manager.ensure_checkpoint(str(other), f"other-{index}")
    before = manager.list_checkpoints(str(other))
    result = checkpoints.prune_checkpoints(retention_days=0, checkpoint_base=base, max_total_size_mb=1)
    assert result["errors"] == 0
    assert manager.list_checkpoints(str(other)) == before
    assert len(manager.list_checkpoints(str(first))) == 5


@pytest.mark.parametrize("failure", ["gc", "reflog", "for-each-ref", "rev-list", "log", "update-ref", "delete-ref"])
def test_maintenance_stops_on_failed_git_without_losing_a_second_snapshot(history, monkeypatch, failure):
    import json

    base, project, manager = history
    original = checkpoints._run_git
    rewrites = []
    retention = 0
    if failure == "delete-ref":
        meta = checkpoints._project_meta_path(checkpoints._store_path(base), checkpoints._project_hash(str(project)))
        state = json.loads(meta.read_text(encoding="utf-8"))
        state["last_touch"] = 1
        meta.write_text(json.dumps(state), encoding="utf-8")
        retention = 1

    def fail_one(args, *rest, **kwargs):
        if args[0] == failure or (failure == "delete-ref" and args[:2] == ["update-ref", "-d"]):
            return False, "", "injected Git failure"
        result = original(args, *rest, **kwargs)
        if args[0] == "update-ref" and result[0]:
            rewrites.append(args)
        return result

    with monkeypatch.context() as patcher:
        patcher.setattr(checkpoints, "_run_git", fail_one)
        result = checkpoints.prune_checkpoints(retention_days=retention, checkpoint_base=base, max_total_size_mb=1)
    assert result["errors"] > 0
    assert len(rewrites) <= 1
    assert len(manager.list_checkpoints(str(project))) >= 5
    if failure == "delete-ref":
        assert meta.is_file()
        assert result["deleted_stale"] == 0


def test_snapshot_pruning_does_not_enter_size_pass_after_failed_count_reclaim(history, monkeypatch, caplog):
    _, project, manager = history
    manager.max_snapshots = 5
    manager.max_total_size_mb = 1
    (project / "later.txt").write_text("newest", encoding="utf-8")
    manager.new_turn()
    original = checkpoints._run_git
    changed_refs = []

    def refuse_gc(args, *rest, **kwargs):
        if args[0] == "gc":
            return False, "", "injected reclamation failure"
        result = original(args, *rest, **kwargs)
        if args[0] == "update-ref" and result[0]:
            changed_refs.append(args)
        return result

    monkeypatch.setattr(checkpoints, "_run_git", refuse_gc)
    assert manager.ensure_checkpoint(str(project), "latest")
    assert len(changed_refs) == 2  # The new checkpoint plus one count trim.
    assert "injected reclamation failure" in caplog.text
    assert len(manager.list_checkpoints(str(project))) == 5


def test_restore_keeps_its_target_alive_while_taking_the_safety_snapshot(history):
    _, project, manager = history
    target = manager.list_checkpoints(str(project))[-1]["hash"]
    manager.max_snapshots = 1
    (project / "small.txt").write_text("changed before restore", encoding="utf-8")
    result = manager.restore(str(project), target)
    assert result["success"], result
    assert (project / "old.bin").stat().st_size == 1_300_000
