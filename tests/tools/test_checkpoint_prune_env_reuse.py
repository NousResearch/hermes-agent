"""One checkpoint operation selects its Git environment once, not once per project.

Regression for the Oct 20 2026 over-cap edit latency.  ``_prune`` runs ``Pruner`` after every
checkpoint take, and on the reporting store (377 project refs, one snapshot each) every one of
those 377 ``rev-list`` calls re-ran ``selected_git_env`` -> PM's ``ensure`` + config parsing:
134.7 s for a read-only sweep that drops nothing, against 3.3 s when the same sweep reuses one
selected environment.  The sweep is inherently one Git call per project; selecting the Git to
use is not.

The scope is a contextvar set and reset around a single take/restore.  Unrelated operations --
this profile's next edit, or another profile's -- therefore still select for themselves and see
whatever the selector and profile context say then.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

import tools.checkpoint_manager as checkpoints

REFS_PREFIX = "refs/hermes"
# The reporting store's project count: enough refs that a per-project selection is obvious.
SWEEP_PROJECTS = 377


def _refs(store) -> dict[str, str]:
    """``{refname: objectname}`` for every checkpoint ref in *store*."""
    ok, out, err = checkpoints._run_git(
        ["for-each-ref", "--format=%(refname) %(objectname)", REFS_PREFIX],
        store, str(store.parent),
    )
    assert ok, err
    return dict(line.split(" ", 1) for line in out.splitlines() if line.strip())


def _count_selections(monkeypatch) -> list[int]:
    """Count how often a checkpoint operation asks PM/the machine which Git to use."""
    real = checkpoints.selected_git_env
    calls: list[int] = []

    def counting(base_env=None):
        calls.append(1)
        return real(base_env)

    monkeypatch.setattr(checkpoints, "selected_git_env", counting)
    return calls


@pytest.fixture
def manager_store(tmp_path, monkeypatch):
    """A real checkpoint store with one project, seeded through the public manager."""
    base = tmp_path / "checkpoints"
    monkeypatch.setattr(checkpoints, "CHECKPOINT_BASE", base)
    work = tmp_path / "project"
    work.mkdir()
    (work / "seed.txt").write_text("seed", encoding="utf-8")
    manager = checkpoints.CheckpointManager(enabled=True, max_total_size_mb=0)
    assert manager.ensure_checkpoint(str(work), "seed")
    return base, work, manager


@pytest.fixture
def wide_store(tmp_path, monkeypatch):
    """An over-cap shaped store whose 377 project refs each hold exactly one snapshot."""
    git = shutil.which("git")
    assert git, "this regression needs a real Git on PATH"
    base = tmp_path / "checkpoints"
    base.mkdir()
    monkeypatch.setattr(checkpoints, "CHECKPOINT_BASE", base)
    store = base / "store"
    work = tmp_path / "project"
    work.mkdir()
    (work / "seed.txt").write_text("seed", encoding="utf-8")

    subprocess.run([git, "init", "--bare", "-q", str(store)], check=True, capture_output=True)
    for key, value in (("user.email", "hermes@local"), ("user.name", "Hermes Checkpoint")):
        assert checkpoints._run_git(["config", key, value], store, str(work))[0]
    ok, tree, err = checkpoints._run_git(["mktree"], store, str(work))
    assert ok, err
    ok, commit, err = checkpoints._run_git(["commit-tree", tree, "-m", "seed"], store, str(work))
    assert ok, err
    lines = "".join(
        f"create {REFS_PREFIX}/{index:016x} {commit}\n" for index in range(SWEEP_PROJECTS)
    )
    subprocess.run(
        [git, f"--git-dir={store}", "update-ref", "--stdin"],
        input=lines, text=True, check=True, capture_output=True, cwd=str(base),
    )
    return store, work


def test_a_take_selects_the_git_environment_once(manager_store, monkeypatch):
    """One edit's take is one operation: one selection for all of its Git calls."""
    _, work, manager = manager_store
    calls = _count_selections(monkeypatch)

    (work / "next.txt").write_text("next", encoding="utf-8")
    manager.new_turn()
    assert manager.ensure_checkpoint(str(work), "next")
    assert len(calls) == 1, f"one take selected Git {len(calls)} times"

    # Not a process-wide or lifetime cache: the next edit selects again.
    (work / "third.txt").write_text("third", encoding="utf-8")
    manager.new_turn()
    assert manager.ensure_checkpoint(str(work), "third")
    assert len(calls) == 2, "a separate operation must select for itself"


def test_restore_selects_the_git_environment_once(manager_store, monkeypatch):
    """Restore (including its pre-rollback safety snapshot) is one operation too."""
    _, work, manager = manager_store
    (work / "second.txt").write_text("second", encoding="utf-8")
    manager.new_turn()
    assert manager.ensure_checkpoint(str(work), "second")
    target = manager.list_checkpoints(str(work))[-1]["hash"]

    calls = _count_selections(monkeypatch)
    result = manager.restore(str(work), target)
    assert result["success"], result
    assert len(calls) == 1, f"one restore selected Git {len(calls)} times"


def test_over_cap_all_single_snapshot_sweep_selects_once_and_keeps_what_it_took(
    wide_store, monkeypatch,
):
    """377 single-snapshot refs over the cap: one selection, nothing dropped, newest visible."""
    store, work = wide_store
    before = _refs(store)
    assert len(before) == SWEEP_PROJECTS
    # The size round is what walks every project; the synthetic store is small on disk.
    monkeypatch.setattr(checkpoints, "_dir_size_bytes", lambda path: 2 * 1024 * 1024)
    calls = _count_selections(monkeypatch)

    manager = checkpoints.CheckpointManager(enabled=True, max_total_size_mb=1)
    (work / "next.txt").write_text("next", encoding="utf-8")
    selections = len(calls)
    assert manager.ensure_checkpoint(str(work), "latest")
    assert len(calls) - selections == 1, (
        f"a {SWEEP_PROJECTS}-project sweep selected Git {len(calls) - selections} times"
    )

    # Every project had a single snapshot, so the round could drop nothing: the store's refs are
    # exactly as they were plus the one just written, and none was rewritten to hide it.
    after = _refs(store)
    new_ref = checkpoints._ref_name(checkpoints._project_hash(str(work)))
    assert {name: sha for name, sha in after.items() if name != new_ref} == before
    assert new_ref in after
    assert [row["reason"] for row in manager.list_checkpoints(str(work))] == ["latest"]

    # The next take is a new operation: it selects for itself, and retention still applies.
    (work / "later.txt").write_text("later", encoding="utf-8")
    manager.new_turn()
    selections = len(calls)
    assert manager.ensure_checkpoint(str(work), "newest")
    assert len(calls) - selections == 1, "a separate operation must select for itself"
    assert [row["reason"] for row in manager.list_checkpoints(str(work))] == ["newest"]
    assert {name: sha for name, sha in _refs(store).items()
            if name not in {new_ref}} == before


def test_per_call_index_store_and_extra_env_survive_reuse(tmp_path, monkeypatch):
    """The shared environment is a template: per-call index, store and overrides still apply."""
    work = tmp_path / "work"
    work.mkdir()
    store = tmp_path / "store"
    other_store = tmp_path / "other-store"
    index_file = tmp_path / "indexes" / "project"

    monkeypatch.setattr(checkpoints, "selected_git_env", lambda base_env=None: dict(base_env or {}))
    with checkpoints._operation_git_env():
        indexed = checkpoints._git_env(store, str(work), index_file=index_file)
        plain = checkpoints._git_env(store, str(work))
        moved = checkpoints._git_env(other_store, str(work))

    assert indexed["GIT_INDEX_FILE"] == str(index_file)
    assert "GIT_INDEX_FILE" not in plain, "a per-project index leaked into the shared template"
    assert plain["GIT_DIR"] == str(store)
    assert moved["GIT_DIR"] == str(other_store)
    for key in ("GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM"):
        assert plain[key] == os.devnull


def test_extra_env_applies_to_one_call_only(manager_store, monkeypatch):
    """``extra_env`` overrides one call; the reused environment does not keep them."""
    base, work, _ = manager_store
    store = checkpoints._store_path(base)
    monkeypatch.setattr(checkpoints, "selected_git_env", lambda base_env=None: dict(base_env or {}))

    with checkpoints._operation_git_env():
        ok, dated, err = checkpoints._run_git(
            ["var", "GIT_AUTHOR_IDENT"], store, str(work),
            extra_env={"GIT_AUTHOR_DATE": "@1234567890 +0000"},
        )
        ok_plain, plain, err_plain = checkpoints._run_git(
            ["var", "GIT_AUTHOR_IDENT"], store, str(work),
        )

    assert ok, err
    assert ok_plain, err_plain
    assert "1234567890" in dated
    assert "1234567890" not in plain


def test_cold_first_store_init_selects_once_and_the_operation_after_selects_again(
    tmp_path, monkeypatch,
):
    """The first-ever checkpoint creates its store inside the operation: one selection.

    ``_init_store`` runs from the take, i.e. inside the operation scope.  It therefore has to
    reuse a copy of that operation's selection instead of asking PM/the machine a second time,
    or the cold path keeps the extra lookup the once-per-operation contract forbids.
    """
    base = tmp_path / "checkpoints"
    monkeypatch.setattr(checkpoints, "CHECKPOINT_BASE", base)
    store = checkpoints._store_path(base)
    work = tmp_path / "project"
    work.mkdir()
    (work / "seed.txt").write_text("seed", encoding="utf-8")
    assert not (store / "HEAD").exists(), "this regression needs a store that does not exist yet"

    # The counter goes in before anything initialises the store, so the cold init is counted.
    calls = _count_selections(monkeypatch)
    manager = checkpoints.CheckpointManager(enabled=True, max_total_size_mb=0)
    assert manager.ensure_checkpoint(str(work), "cold")
    assert (store / "HEAD").exists(), "the cold store was never created"
    assert len(calls) == 1, f"cold first take selected Git {len(calls)} times"
    assert checkpoints._OPERATION_GIT_ENV.get() is None, "the operation scope outlived the take"

    # A separate operation selects for itself: the cold init left nothing process-wide behind.
    (work / "next.txt").write_text("next", encoding="utf-8")
    manager.new_turn()
    assert manager.ensure_checkpoint(str(work), "next")
    assert len(calls) == 2, "a separate operation must select for itself"
