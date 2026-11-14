"""Checkpoint recovery and data-cleanup locks against real Git and native holders."""

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli.data_cleanup_holders import quiescent_home
from tools.checkpoint_manager import CheckpointManager, _project_hash, _ref_name
from tools.checkpoint_pruning import store_lock


@pytest.fixture(scope="module", autouse=True)
def native_tool_store(tmp_path_factory):
    # Reuse the real development tool store independently of each data home.
    runtime = tmp_path_factory.getbasetemp().parent / "pm-runtime"
    previous = os.environ.get("HERMES_RUNTIME_DIR")
    os.environ["HERMES_RUNTIME_DIR"] = str(runtime)
    try:
        yield runtime
    finally:
        if previous is None:
            os.environ.pop("HERMES_RUNTIME_DIR")
        else:
            os.environ["HERMES_RUNTIME_DIR"] = previous


@contextmanager
def checkpoint_world(tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    shutil.copyfile(ROOT / "README.md", project / "README.md")
    token = set_hermes_home_override(home)
    try:
        manager = CheckpointManager(enabled=True)
        assert manager.ensure_checkpoint(str(project), "snapshot the actual repository README")
        yield home, project, manager
    finally:
        reset_hermes_home_override(token)


def git(store, *args):
    return subprocess.run(
        ["git", "--git-dir=" + str(store), *args],
        check=True, capture_output=True, text=True, timeout=30,
    ).stdout.strip()


def test_data_cleanup_and_plain_lock_do_not_repair_corrupt_history(tmp_path):
    with checkpoint_world(tmp_path) as (home, project, _manager):
        base = home / "checkpoints"
        store = base / "store"
        loose = store / _ref_name(_project_hash(str(project)))
        with loose.open("wb"):
            pass  # Truncate the actual Git-created ref as an interrupted write would.
        with store_lock(base):
            assert loose.stat().st_size == 0
        with quiescent_home(home):
            assert loose.stat().st_size == 0
        assert not list((base / "corrupt-refs").rglob(loose.name))
        assert (project / "README.md").read_bytes() == (ROOT / "README.md").read_bytes()


def test_real_packed_predecessor_is_recovered_without_losing_the_damaged_ref(tmp_path):
    with checkpoint_world(tmp_path) as (home, project, manager):
        store = home / "checkpoints/store"
        ref = _ref_name(_project_hash(str(project)))
        packed_tip = git(store, "rev-parse", ref)
        git(store, "pack-refs", "--all")
        shutil.copyfile(ROOT / "AGENTS.md", project / "AGENTS.md")
        manager.new_turn()
        assert manager.ensure_checkpoint(str(project), "snapshot the actual repository instructions")
        loose = store / ref
        assert loose.is_file()
        with loose.open("wb"):
            pass
        manager.new_turn()
        assert manager.ensure_checkpoint(str(project), "recover the interrupted ref write")
        recovered_tip = git(store, "rev-parse", ref)
        assert recovered_tip != packed_tip
        git(store, "merge-base", "--is-ancestor", packed_tip, recovered_tip)
        preserved = list((home / "checkpoints/corrupt-refs").rglob(loose.name))
        assert len(preserved) == 1 and preserved[0].stat().st_size == 0
        assert git(store, "show", ref + ":AGENTS.md") == (ROOT / "AGENTS.md").read_text(encoding="utf-8").strip()
