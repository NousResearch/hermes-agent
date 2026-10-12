"""Real history retention and failure propagation at the public pruning callers."""
from __future__ import annotations

import os

import pytest

import tools.checkpoint_manager as checkpoints
from tools import checkpoint_maintenance


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


def test_maintenance_size_cap_retains_every_snapshot_after_the_only_large_one(history):
    base, project, manager = history
    store = checkpoints._store_path(base)
    assert checkpoints._dir_size_bytes(store) > 1024 * 1024
    result = checkpoint_maintenance.prune_checkpoints(retention_days=0, checkpoint_base=base, max_total_size_mb=1)
    assert result["errors"] == 0
    assert [row["reason"] for row in manager.list_checkpoints(str(project))] == [f"small-{i}" for i in reversed(range(5))]
    assert checkpoints._dir_size_bytes(store) <= 1024 * 1024
    assert (project / "small.txt").read_text(encoding="utf-8") == "4"


def test_snapshot_over_cap_drops_one_round_and_leaves_the_gc_to_the_prune(history):
    """A checkpoint never waits on ``git gc``: over the cap it rewrites the ref once and marks
    the store gc-pending; the periodic prune reclaims the objects and clears the marker."""
    from tools.checkpoint_pruning import GC_PENDING_NAME

    base, project, manager = history
    store = checkpoints._store_path(base)
    manager.max_total_size_mb = 1
    gc_calls = []
    original = checkpoints._run_git

    def count_gc(args, *rest, **kwargs):
        if args[0] == "gc":
            gc_calls.append(args)
        return original(args, *rest, **kwargs)

    (project / "new.txt").write_text("new snapshot", encoding="utf-8")
    manager.new_turn()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(checkpoints, "_run_git", count_gc)
        assert manager.ensure_checkpoint(str(project), "newest")
    assert gc_calls == []
    assert (store / GC_PENDING_NAME).exists()
    assert [row["reason"] for row in manager.list_checkpoints(str(project))] == \
        ["newest"] + [f"small-{i}" for i in reversed(range(5))]
    # Objects are still in the pack until the prune runs.
    assert checkpoints._dir_size_bytes(store) > 1024 * 1024

    result = checkpoint_maintenance.prune_checkpoints(retention_days=0, checkpoint_base=base, max_total_size_mb=1)
    assert result["errors"] == 0
    assert not (store / GC_PENDING_NAME).exists()
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
    result = checkpoint_maintenance.prune_checkpoints(retention_days=0, checkpoint_base=base, max_total_size_mb=1)
    assert result["errors"] == 0
    assert manager.list_checkpoints(str(other)) == before
    assert len(manager.list_checkpoints(str(first))) == 5


@pytest.mark.parametrize("failure", ["gc", "reflog", "for-each-ref", "rev-list", "log", "update-ref", "delete-ref"])
def test_maintenance_stops_on_failed_git_without_losing_a_second_snapshot(history, monkeypatch, failure):
    import json

    base, project, manager = history
    original = checkpoint_maintenance._run_git
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
        patcher.setattr(checkpoint_maintenance, "_run_git", fail_one)
        result = checkpoint_maintenance.prune_checkpoints(retention_days=retention, checkpoint_base=base, max_total_size_mb=1)
    assert result["errors"] > 0
    assert len(rewrites) <= 1
    assert len(manager.list_checkpoints(str(project))) >= 5
    if failure == "delete-ref":
        assert meta.is_file()
        assert result["deleted_stale"] == 0


def test_snapshot_pruning_stops_at_a_failed_count_rewrite(history, monkeypatch, caplog):
    """A count trim that cannot rewrite its ref aborts the take-path pruning before the size
    round runs — never a partial history, never a second rewrite on top of a failed one."""
    _, project, manager = history
    manager.max_snapshots = 5
    manager.max_total_size_mb = 1
    (project / "later.txt").write_text("newest", encoding="utf-8")
    manager.new_turn()
    original = checkpoints._run_git
    changed_refs = []

    def refuse_rewrite(args, *rest, **kwargs):
        # The rewrite re-creates commits under their original dates; the snapshot's own
        # commit-tree carries no extra_env.
        if args[0] == "commit-tree" and kwargs.get("extra_env"):
            return False, "", "injected rewrite failure"
        result = original(args, *rest, **kwargs)
        if args[0] == "update-ref" and result[0]:
            changed_refs.append(args)
        return result

    monkeypatch.setattr(checkpoints, "_run_git", refuse_rewrite)
    assert manager.ensure_checkpoint(str(project), "latest")
    assert len(changed_refs) == 1  # The new checkpoint only; the trim failed before update-ref.
    assert "injected rewrite failure" in caplog.text
    store = checkpoints._store_path(checkpoints.CHECKPOINT_BASE)
    ref = checkpoints._ref_name(checkpoints._project_hash(str(project)))
    ok, out, _ = original(["rev-list", "--count", ref], store, str(project))
    assert ok and int(out) == 7  # Nothing trimmed: the failed rewrite left history whole.


def test_restore_keeps_its_target_alive_while_taking_the_safety_snapshot(history):
    _, project, manager = history
    target = manager.list_checkpoints(str(project))[-1]["hash"]
    manager.max_snapshots = 1
    (project / "small.txt").write_text("changed before restore", encoding="utf-8")
    result = manager.restore(str(project), target)
    assert result["success"], result
    assert (project / "old.bin").stat().st_size == 1_300_000


def _corrupt_parent_object(store, ref):
    """Break one ref's parent link the way the reported stores are broken: the ref's
    history stays readable one commit deep but its full walk hits a missing object."""
    import subprocess as sp

    chain = sp.run(["git", "--git-dir", str(store), "rev-list", ref],
                   capture_output=True, text=True, check=True).stdout.split()
    victim, parent = chain[-2], chain[-1]
    obj = store / "objects" / parent[:2] / parent[2:]
    if not obj.is_file():  # packed: repack first so the loose object exists to delete
        sp.run(["git", "--git-dir", str(store), "repack", "-a", "-d"], check=True,
               capture_output=True)
        sp.run(["git", "--git-dir", str(store), "prune-packed"], check=True,
               capture_output=True)
        raise AssertionError(f"{parent} stayed packed; fixture needs a loose parent")
    obj.unlink()
    return victim, parent


def test_one_unwalkable_ref_no_longer_stops_the_size_round_for_every_project(history):
    """A missing parent object makes one ref's full walk fail (rc=128). That ref must be
    skipped, not fatal: the other projects' snapshots keep converging under the cap and
    the periodic prune still reports success."""
    from tools.checkpoint_pruning import PruneError

    base, project, manager = history
    store = checkpoints._store_path(base)
    other = None
    for name in ("project-a", "project-b"):
        candidate = base.parent / name
        if str(candidate) != str(project):
            candidate.mkdir(parents=True, exist_ok=True)
            for index in range(3):
                (candidate / "small.txt").write_text(str(index), encoding="utf-8")
                manager.new_turn()
                assert manager.ensure_checkpoint(str(candidate), f"other-{index}")
            other = candidate
    assert other is not None

    ref = checkpoints._ref_name(checkpoints._project_hash(str(project)))
    _corrupt_parent_object(store, ref)
    import subprocess as sp
    with pytest.raises(sp.CalledProcessError):
        sp.run(["git", "--git-dir", str(store), "rev-list", "--reverse", ref],
               check=True, capture_output=True)

    result = checkpoint_maintenance.prune_checkpoints(
        retention_days=0, checkpoint_base=base, max_total_size_mb=1)
    assert result["errors"] == 0
    # the healthy ref's history still converged: started at 3, the size round must have
    # dropped at least the oldest (exact depth depends on remeasured object sizes)
    remaining = [row["reason"] for row in manager.list_checkpoints(str(other))]
    assert len(remaining) < 3, remaining
    # and the periodic prune still reclaimed the store under the cap
    assert checkpoints._dir_size_bytes(store) <= 1024 * 1024


def test_take_path_size_round_repairs_the_broken_ref_and_trims_the_healthy_one(history):
    """The take-path size round must still do its job when a sibling ref is unwalkable:
    the broken ref gets rebuilt onto a readable root and the healthy ref drops its oldest
    snapshot — over-cap stores keep converging instead of aborting on the first rc=128."""
    base, project, manager = history
    store = checkpoints._store_path(base)
    other = None
    for name in ("project-a", "project-b"):
        candidate = base.parent / name
        if str(candidate) != str(project):
            candidate.mkdir(parents=True, exist_ok=True)
            for index in range(4):
                (candidate / "small.txt").write_text(str(index), encoding="utf-8")
                manager.new_turn()
                assert manager.ensure_checkpoint(str(candidate), f"other-{index}")
            other = candidate
    assert other is not None

    other_ref = checkpoints._ref_name(checkpoints._project_hash(str(other)))
    _corrupt_parent_object(store, other_ref)

    manager.max_total_size_mb = 1
    (project / "late.txt").write_text("late", encoding="utf-8")
    manager.new_turn()
    assert manager.ensure_checkpoint(str(project), "late")

    import subprocess as sp
    # healthy ref: the size round dropped its oldest snapshot (4 -> newest only under 1MB cap)
    other_count = sp.run(["git", "--git-dir", str(store), "rev-list", "--count", other_ref],
                         check=False, capture_output=True, text=True)
    assert other_count.returncode == 0, other_count.stderr
    assert int(other_count.stdout) < 5, other_count.stdout
    # broken ref: repaired onto a fresh readable root
    walk = sp.run(["git", "--git-dir", str(store), "rev-list", "--reverse", other_ref],
                  check=False, capture_output=True, text=True)
    assert walk.returncode == 0, walk.stderr


def test_take_path_repairs_its_own_broken_ref_before_the_size_round(history):
    """The take path must heal THIS project's own unwalkable ref: a single-project store
    with auto-prune off (the reported configuration) has no sibling take to fix it — if
    the count trim aborted the round, the ref would stay broken and the store over cap
    forever (every take just logs the same warning)."""
    from tools.checkpoint_pruning import GC_PENDING_NAME

    base, project, manager = history
    store = checkpoints._store_path(base)
    ref = checkpoints._ref_name(checkpoints._project_hash(str(project)))
    _corrupt_parent_object(store, ref)

    import subprocess as sp
    broken = sp.run(["git", "--git-dir", str(store), "rev-list", "--reverse", ref],
                    check=False, capture_output=True, text=True)
    assert broken.returncode != 0  # fixture sanity: the own ref really is unwalkable

    manager.max_total_size_mb = 1
    (project / "late.txt").write_text("late", encoding="utf-8")
    manager.new_turn()
    assert manager.ensure_checkpoint(str(project), "late")

    walk = sp.run(["git", "--git-dir", str(store), "rev-list", "--reverse", ref],
                  check=False, capture_output=True, text=True)
    assert walk.returncode == 0, walk.stderr  # repaired during the take
    count = sp.run(["git", "--git-dir", str(store), "rev-list", "--count", ref],
                   check=False, capture_output=True, text=True)
    assert count.returncode == 0, count.stderr
    # repair alone rebuilds the readable prefix (6 incl. the new take); a smaller count
    # proves the size round landed real drops on top of the repair
    assert int(count.stdout) < 6, count.stdout
    assert (store / GC_PENDING_NAME).exists()  # the drop was real and the gc is deferred
    reasons = [row["reason"] for row in manager.list_checkpoints(str(project))]
    assert reasons[0] == "late" and all(r.startswith("small-") for r in reasons[1:]), reasons
