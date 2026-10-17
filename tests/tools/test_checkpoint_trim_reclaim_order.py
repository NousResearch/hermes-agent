"""Regression test: a take whose trim rewrites history must not pay a full gc
when the store already fits (the deferred-GC design)."""

from tools.checkpoint_pruning import Pruner


def test_take_that_trims_does_not_reclaim_when_store_fits(tmp_path):
    """CheckpointManager._prune runs trim() before drop_one_round(). A project at
    the snapshot cap trims on every take; the marker just created by this take
    must not trigger a full git gc on the take path."""
    store = tmp_path / "store.git"
    store.mkdir()
    packed = {"bytes": 40}
    git_calls = []

    def run_git(args, *_args, **_kwargs):
        git_calls.append(list(args))
        if args[0] == "rev-list":
            return True, "\n".join(f"c{i}" for i in range(25)), ""
        if args[0] == "log":
            return True, "tree\x002026-01-01T00:00:00+00:00\x002026-01-01T00:00:00+00:00\x00msg", ""
        if args[0] == "commit-tree":
            return True, "newsha", ""
        if args[0] == "gc":
            packed["bytes"] = 5
        return True, "", ""

    pruner = Pruner(
        run_git=run_git,
        store=store,
        cwd=str(tmp_path),
        timeout=1,
        dir_size=lambda _path: packed["bytes"],
        refs_prefix="refs/checkpoints/",
    )

    pruner.trim("refs/checkpoints/proj", 20)   # rewrites, marks .gc-pending
    assert pruner.drop_one_round(50) is False  # store fits: no reclaim, no drop

    assert "gc" not in [call[0] for call in git_calls]
