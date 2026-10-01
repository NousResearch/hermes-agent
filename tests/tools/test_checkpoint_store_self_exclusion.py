"""The checkpoint store must never stage itself.

When the checkpointed working directory is HERMES_HOME (or a profile home under it), the
shadow store at ``<home>/checkpoints/store`` sits inside the tree being snapshotted. Without
an exclusion every snapshot stages the store's own packfiles and per-project index, so each
checkpoint carries the previous one and the store grows by about its own size per snapshot
until ``git add -A`` times out and leaves an index lock behind.
"""

import os
import subprocess
from pathlib import Path

import pytest

from tools.checkpoint_manager import (
    CheckpointManager,
    _project_hash,
    _ref_name,
    _store_path,
)
import tools.checkpoint_manager as _cm

# Tolerant lookup so RED on unmodified main fails inside each test, not at collection.
_stage_all_args = getattr(_cm, "_stage_all_args", None)


def _snapshot_paths(store: Path, workdir: Path) -> list[str]:
    ref = _ref_name(_project_hash(str(workdir)))
    out = subprocess.run(
        ["git", f"--git-dir={store}", "ls-tree", "-r", "--name-only", ref],
        capture_output=True, text=True, check=True,
    ).stdout
    return out.split()


@pytest.fixture()
def home_as_workdir(tmp_path, monkeypatch):
    """A HERMES_HOME that is also the checkpointed working directory, with the store inside it."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text("checkpoints:\n  enabled: true\n")
    (home / "notes.md").write_text("one\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", home / "checkpoints")
    # Use the system git so PM's lazy acquisition does not bootstrap a runtime under HOME.
    monkeypatch.setattr(
        "tools.checkpoint_manager.selected_git_env",
        lambda base=None: dict(base if base is not None else os.environ),
    )
    return home


class TestStageAllArgs:
    def test_store_outside_workdir_is_plain_add(self, tmp_path):
        assert _stage_all_args(tmp_path / "checkpoints", tmp_path / "project") == ["add", "-A"]

    def test_store_inside_workdir_is_excluded_by_pathspec(self, tmp_path):
        args = _stage_all_args(tmp_path / "home" / "checkpoints", tmp_path / "home")
        assert args[:3] == ["add", "-A", "--"]
        assert ":(exclude)checkpoints" in args
        assert ":(exclude).checkpoints.lock" in args

    def test_nested_base_uses_relative_posix_path(self, tmp_path):
        args = _stage_all_args(tmp_path / "home" / "state" / "ckpt", tmp_path / "home")
        assert ":(exclude)state/ckpt" in args
        assert ":(exclude)state/.ckpt.lock" in args

    def test_base_equal_to_workdir_is_plain_add(self, tmp_path):
        # Degenerate: excluding "." would stage nothing; leave the add alone.
        assert _stage_all_args(tmp_path / "home", tmp_path / "home") == ["add", "-A"]


class TestStoreNeverStagesItself:
    def test_snapshots_exclude_the_store_and_its_lock(self, home_as_workdir):
        home = home_as_workdir
        mgr = CheckpointManager(enabled=True, max_snapshots=50)
        store = _store_path()
        assert store.is_relative_to(home)

        assert mgr.ensure_checkpoint(str(home), "first") is True
        first = _snapshot_paths(store, home)
        assert "notes.md" in first
        assert not [p for p in first if p.startswith("checkpoints/")], first
        assert ".checkpoints.lock" not in first

        # The second snapshot is the one that used to stage the first snapshot's objects.
        (home / "notes.md").write_text("two\n")
        mgr._checkpointed_dirs.clear()
        assert mgr.ensure_checkpoint(str(home), "second") is True
        second = _snapshot_paths(store, home)
        assert "notes.md" in second
        assert not [p for p in second if p.startswith("checkpoints/")], second
        assert ".checkpoints.lock" not in second

    def test_diff_and_restore_paths_do_not_stage_the_store(self, home_as_workdir):
        home = home_as_workdir
        mgr = CheckpointManager(enabled=True, max_snapshots=50)
        store = _store_path()
        assert mgr.ensure_checkpoint(str(home), "first") is True
        commit = mgr.list_checkpoints(str(home))[0]["hash"]

        (home / "notes.md").write_text("two\n")
        mgr.record_agent_write(str(home / "notes.md"))
        diff = mgr.diff(str(home), commit)
        assert diff["success"], diff
        assert "notes.md" in diff["diff"]
        assert "checkpoints/" not in diff["diff"]

        plan = mgr._safe_restore_plan(str(home), commit)
        assert plan["success"], plan
        changed = plan.get("restore", []) + plan.get("skipped", [])
        assert "notes.md" in changed
        assert not [p for p in changed if p.startswith("checkpoints/")], changed

    def test_project_outside_home_unaffected(self, home_as_workdir, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        (project / "a.py").write_text("x = 1\n")
        mgr = CheckpointManager(enabled=True, max_snapshots=50)
        assert mgr.ensure_checkpoint(str(project), "ctl") is True
        assert _snapshot_paths(_store_path(), project) == ["a.py"]
