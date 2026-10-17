"""Checkpoint size-pruning behavior around deferred object reclamation."""

from tools.checkpoint_pruning import GC_PENDING_NAME, Pruner


def _pruner(tmp_path, *, size, git_calls):
    store = tmp_path / "store.git"
    store.mkdir()
    packed_size = {"bytes": size}

    def run_git(args, *_args, **_kwargs):
        git_calls.append(list(args))
        if args[0] == "gc":
            packed_size["bytes"] = 5
        return True, "", ""

    pruner = Pruner(
        run_git=run_git,
        store=store,
        cwd=str(tmp_path),
        timeout=1,
        dir_size=lambda _path: packed_size["bytes"],
        refs_prefix="refs/checkpoints/",
    )
    return store, pruner


def test_pending_gc_is_reclaimed_before_rewriting_history_when_packed_store_fits(
    tmp_path,
):
    git_calls = []
    store, pruner = _pruner(tmp_path, size=100, git_calls=git_calls)
    (store / GC_PENDING_NAME).touch()

    assert pruner.drop_one_round(cap_bytes=50) is False

    assert [call[0] for call in git_calls] == ["reflog", "gc"]
    assert not (store / GC_PENDING_NAME).exists()


def test_under_cap_without_pending_gc_does_not_call_git(tmp_path):
    git_calls = []
    _, pruner = _pruner(tmp_path, size=40, git_calls=git_calls)

    assert pruner.drop_one_round(cap_bytes=50) is False
    assert git_calls == []


def test_non_positive_cap_does_not_reclaim_or_rewrite(tmp_path):
    git_calls = []
    store, pruner = _pruner(tmp_path, size=100, git_calls=git_calls)
    (store / GC_PENDING_NAME).touch()

    assert pruner.drop_one_round(cap_bytes=0) is False
    assert git_calls == []
    assert (store / GC_PENDING_NAME).exists()
