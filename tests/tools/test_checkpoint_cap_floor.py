"""A cap that cannot be reached without destroying history is a policy floor, not an error.

``Pruner.enforce_size`` answered only "does the store fit now", so the one shape where it
legitimately cannot — every project already down to its last retained checkpoint — was counted
into ``errors`` beside real Git/OS failures. ``hermes checkpoints prune`` then printed
``Errors: 1`` for a healthy store and explained nothing (the reason only reached a
``logger.warning`` with no console handler on that path).

The periodic prune reports that floor in ``cap_not_met`` instead, and ``errors`` stays reserved
for Git/OS failures, so the counter keeps meaning what its name says.
"""
from __future__ import annotations

import os

import pytest

import tools.checkpoint_manager as checkpoints
from tools import checkpoint_maintenance, checkpoint_pruning

# The outcome constants are what the fix adds, so they are read defensively: a run of this file
# against the unfixed module must fail on the behaviour, not on an import at collection time.
CAP_MET = getattr(checkpoint_pruning, "CAP_MET", None)
CAP_FLOOR = getattr(checkpoint_pruning, "CAP_FLOOR", None)
CAP_BUDGET = getattr(checkpoint_pruning, "CAP_BUDGET", None)


def _pruner(base):
    return checkpoint_pruning.Pruner(
        checkpoints._run_git, checkpoints._store_path(base), str(base),
        checkpoints._GIT_TIMEOUT, checkpoints._dir_size_bytes, checkpoints._REFS_PREFIX)


@pytest.fixture
def floor_store(tmp_path, monkeypatch):
    """One project whose ONLY snapshot is larger than the cap: nothing is droppable."""
    base = tmp_path / "checkpoints"
    monkeypatch.setattr(checkpoints, "CHECKPOINT_BASE", base)
    project = tmp_path / "project-a"
    project.mkdir()
    manager = checkpoints.CheckpointManager(enabled=True, max_total_size_mb=0)
    (project / "big.bin").write_bytes(os.urandom(1_300_000))
    assert manager.ensure_checkpoint(str(project), "only")
    return base, project, manager


@pytest.fixture
def droppable_store(tmp_path, monkeypatch):
    """A large oldest snapshot followed by small ones: dropping it does reach the cap."""
    base = tmp_path / "checkpoints"
    monkeypatch.setattr(checkpoints, "CHECKPOINT_BASE", base)
    project = tmp_path / "project-a"
    project.mkdir()
    manager = checkpoints.CheckpointManager(enabled=True, max_total_size_mb=0)
    blob = project / "big.bin"
    blob.write_bytes(os.urandom(1_300_000))
    assert manager.ensure_checkpoint(str(project), "big-oldest")
    blob.unlink()
    for index in range(5):
        (project / "small.txt").write_text(str(index), encoding="utf-8")
        manager.new_turn()
        assert manager.ensure_checkpoint(str(project), f"small-{index}")
    return base, project, manager


def test_a_floor_store_is_not_reported_as_an_error(floor_store):
    """The reported symptom: a healthy store, ``Errors: 1``."""
    base, project, manager = floor_store
    store = checkpoints._store_path(base)
    assert checkpoints._dir_size_bytes(store) > 1024 * 1024

    result = checkpoint_maintenance.prune_checkpoints(
        retention_days=0, checkpoint_base=base, max_total_size_mb=1)

    # On the unfixed module this reads 1: the unreachable cap was counted as a failure.
    assert result["errors"] == 0
    assert result.get("cap_not_met", 0) == 1
    # The floor means "nothing left but history", so the history is still there.
    assert [row["reason"] for row in manager.list_checkpoints(str(project))] == ["only"]


def test_a_reachable_cap_reports_neither_a_floor_nor_an_error(droppable_store):
    """Control: the counter must not swallow a cap that really was enforced."""
    base, _project, _manager = droppable_store
    store = checkpoints._store_path(base)
    assert checkpoints._dir_size_bytes(store) > 1024 * 1024

    result = checkpoint_maintenance.prune_checkpoints(
        retention_days=0, checkpoint_base=base, max_total_size_mb=1)

    assert result["errors"] == 0
    assert result.get("cap_not_met", 0) == 0
    assert checkpoints._dir_size_bytes(store) <= 1024 * 1024


def test_enforce_size_names_which_outcome_stopped_it(floor_store):
    if CAP_FLOOR is None or CAP_MET is None:
        pytest.skip("the fix's outcome constants are absent")
    base, _project, _manager = floor_store
    pruner = _pruner(base)

    assert pruner.enforce_size(0) == CAP_MET            # no cap configured: nothing to enforce
    assert pruner.enforce_size(1024 * 1024) == CAP_FLOOR


def test_the_rewrite_budget_is_its_own_outcome(tmp_path, monkeypatch):
    """Not every unreachable cap is a floor: a store still shrinking must say so."""
    if CAP_BUDGET is None:
        pytest.skip("the fix's outcome constants are absent")
    base = tmp_path / "checkpoints"
    monkeypatch.setattr(checkpoints, "CHECKPOINT_BASE", base)
    project = tmp_path / "project-a"
    project.mkdir()
    manager = checkpoints.CheckpointManager(enabled=True, max_total_size_mb=0)
    for index in range(3):
        (project / f"file-{index}.bin").write_bytes(os.urandom(700_000))
        manager.new_turn()
        assert manager.ensure_checkpoint(str(project), f"commit-{index}")

    monkeypatch.setattr(checkpoint_pruning, "_REWRITE_ROUNDS", 1)
    assert _pruner(base).enforce_size(1024 * 1024) == CAP_BUDGET


def test_the_empty_result_carries_the_counter(tmp_path, monkeypatch):
    """``maybe_auto_prune_checkpoints`` shapes its own dict: the key must exist there too."""
    missing = tmp_path / "missing"
    monkeypatch.setattr(checkpoints, "CHECKPOINT_BASE", missing)

    out = checkpoint_maintenance.maybe_auto_prune_checkpoints(checkpoint_base=missing)

    result = out["result"]
    assert isinstance(result, dict)
    assert result["cap_not_met"] == 0
