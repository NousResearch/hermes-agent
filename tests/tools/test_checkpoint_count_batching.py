"""Regression for #135009: bounded count trimming amortizes real Git rewrites."""
from __future__ import annotations

import pytest

from tools import checkpoint_manager as checkpoints
from tools.checkpoint_pruning import GC_PENDING_NAME


@pytest.mark.parametrize("limit", [5, 10])
def test_count_trim_batches_rewrites_and_preserves_visible_restore_targets(tmp_path, monkeypatch, limit):
    base = tmp_path / "checkpoints"
    project = tmp_path / "project"
    project.mkdir()
    document = project / "document.txt"
    monkeypatch.setattr(checkpoints, "CHECKPOINT_BASE", base)
    manager = checkpoints.CheckpointManager(enabled=True, max_snapshots=limit, max_total_size_mb=0)
    original = checkpoints._run_git
    rewritten = []
    gc_calls = []

    def record_git(args, *rest, **kwargs):
        if args[0] == "commit-tree" and kwargs.get("extra_env"):
            rewritten.append(args)
        if args[0] == "gc":
            gc_calls.append(args)
        return original(args, *rest, **kwargs)

    monkeypatch.setattr(checkpoints, "_run_git", record_git)

    def take(index):
        document.write_text(f"revision {index}\n", encoding="utf-8")
        manager.new_turn()
        assert manager.ensure_checkpoint(str(project), f"step-{index}")

    for index in range(limit):
        take(index)
    assert not rewritten
    store = checkpoints._store_path(base)
    ref = checkpoints._ref_name(checkpoints._project_hash(str(project)))
    slack = limit // 5

    # Two full batches prove that the steady state does not rebuild every turn.
    for batch in range(2):
        for offset in range(slack + 1):
            index = limit + batch * (slack + 1) + offset
            before = len(rewritten)
            take(index)
            if offset < slack:
                assert len(rewritten) == before, "taking a checkpoint rewrote retained history before the batch filled"
            else:
                assert len(rewritten) - before == limit
            ok, count, error = original(["rev-list", "--count", ref], store, str(project))
            assert ok, error
            assert int(count) == (limit + offset + 1 if offset < slack else limit)
            visible = manager.list_checkpoints(str(project))
            assert [row["reason"] for row in visible] == [
                f"step-{i}" for i in range(index, index - limit, -1)
            ]
            assert not gc_calls, "checkpoint takes must leave GC to maintenance"

    visible = manager.list_checkpoints(str(project))
    latest = limit + 2 * (slack + 1) - 1
    assert [row["reason"] for row in visible] == [f"step-{i}" for i in range(latest, latest - limit, -1)]
    assert (store / GC_PENDING_NAME).exists()
    oldest = visible[-1]
    restored = manager.restore(str(project), oldest["hash"])
    assert restored["success"], restored
    assert document.read_text(encoding="utf-8") == f"revision {latest - limit + 1}\n"
