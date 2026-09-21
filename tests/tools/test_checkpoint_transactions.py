"""Shared-store publication/maintenance invariants (regression for #99780)."""

import multiprocessing
from pathlib import Path

import pytest

from tools import checkpoint_manager as cpm


def _paused_snapshot(base, work, entered, release, results):
    cpm.CHECKPOINT_BASE = Path(base)
    run_git = cpm._run_git

    def pause_after_commit(args, *pos, **kw):
        result = run_git(args, *pos, **kw)
        if args[0] == "commit-tree" and result[0]:
            # Model deferred GC from a preceding ref rewrite. The new commit is
            # real but still unreferenced: precisely the vulnerable prune window.
            cpm._mark_gc_pending(cpm._store_path(Path(base)))
            entered.set()
            if not release.wait(30):
                raise RuntimeError("snapshot release timed out")
        return result

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(cpm, "_run_git", pause_after_commit)
        results.put(cpm.CheckpointManager(enabled=True).ensure_checkpoint(work, "paused writer"))


def _maintenance(base, results):
    cpm._MAINTENANCE_LOCK_TIMEOUT = 0.1
    cpm._INTERACTIVE_LOCK_TIMEOUT = 0.1
    base = Path(base)
    results.put({
        "prune": cpm.prune_checkpoints(retention_days=0, delete_orphans=False, checkpoint_base=base),
        "auto": cpm.maybe_auto_prune_checkpoints(checkpoint_base=base),
        "legacy": cpm.clear_legacy(base),
        "clear": cpm.clear_all(base),
    })


def _finish(process):
    process.join(10)
    if process.is_alive():
        process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join(10)


@pytest.mark.parametrize("lane", [
    pytest.param("linux", marks=pytest.mark.linux_only),
    pytest.param("macos", marks=pytest.mark.macos_only),
    pytest.param("windows", marks=pytest.mark.windows_only),
])
def test_unpublished_snapshot_excludes_maintenance(tmp_path, lane):
    """Real spawned processes must serialize, including clear's lock-inode lifetime."""
    ctx = multiprocessing.get_context("spawn")
    base, work = tmp_path / "checkpoints", tmp_path / "project"
    work.mkdir()
    (work / "main.py").write_text("content\n", encoding="utf-8")
    entered, release = ctx.Event(), ctx.Event()
    snapshots, maintenance = ctx.Queue(), ctx.Queue()
    writer = ctx.Process(target=_paused_snapshot, args=(str(base), str(work), entered, release, snapshots))
    pruner = ctx.Process(target=_maintenance, args=(str(base / ".." / base.name), maintenance))
    writer.start()
    try:
        assert entered.wait(20), "writer did not reach the unpublished-commit window"
        # A lexical alias must contend on the same store, not a second lock.
        pruner.start()
        pruner.join(10)
        assert pruner.exitcode == 0
        blocked = maintenance.get(timeout=5)
        for operation in ("prune", "legacy", "clear"):
            assert blocked[operation].get("lock_error"), blocked
        assert blocked["auto"]["skipped"] and blocked["auto"].get("error")
        assert not (base / ".last_prune").exists(), "contention must not claim the prune interval"
        assert writer.is_alive()
        release.set()
        writer.join(10)
        assert writer.exitcode == 0
        assert snapshots.get(timeout=5) is True
        result = cpm.prune_checkpoints(retention_days=0, delete_orphans=False, checkpoint_base=base)
        assert result["errors"] == 0 and "lock_error" not in result
        ok, _, err = cpm._run_git(["fsck", "--no-dangling"], cpm._store_path(base), str(work))
        assert ok, err
        lock_path = base.parent / f".{base.name}.transaction.lock"
        before = lock_path.stat()
        assert cpm.clear_all(base)["deleted"]
        after = lock_path.stat()
        assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
    finally:
        release.set()
        _finish(writer)
        if pruner.pid is not None:
            _finish(pruner)
        for queue in (snapshots, maintenance):
            queue.close()
            queue.join_thread()


@pytest.mark.parametrize("operation", ["snapshot", "rewrite"])
def test_unreadable_root_tree_is_not_published(tmp_path, monkeypatch, operation):
    base, work = tmp_path / "checkpoints", tmp_path / "project"
    work.mkdir()
    target = work / "main.py"
    target.write_text("original\n", encoding="utf-8")
    monkeypatch.setattr(cpm, "CHECKPOINT_BASE", base)
    manager = cpm.CheckpointManager(enabled=True)
    assert manager.ensure_checkpoint(str(work), "original")
    store, ref = cpm._store_path(base), cpm._ref_name(cpm._project_hash(str(work)))
    original = cpm._ref_tip(store, str(work), ref)
    assert original is not None
    target.write_text("changed\n", encoding="utf-8")
    run_git = cpm._run_git

    def remove_tree_after_commit(args, *pos, **kw):
        result = run_git(args, *pos, **kw)
        if args[0] == "commit-tree" and result[0]:
            tree = args[1]
            (store / "objects" / tree[:2] / tree[2:]).unlink()
        return result

    monkeypatch.setattr(cpm, "_run_git", remove_tree_after_commit)
    if operation == "snapshot":
        manager.new_turn()
        assert manager.ensure_checkpoint(str(work), "corrupt tree") is False
    else:
        assert cpm._rewrite_ref_to(store, str(work), ref, [original]) is False
    assert cpm._ref_tip(store, str(work), ref) == original
