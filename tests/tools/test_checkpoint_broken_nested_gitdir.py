"""A nested repository recorded as a gitlink whose .git later breaks must not
stop every later checkpoint of the directory."""
import os
import subprocess

import pytest

from tools import checkpoint_manager as cm
from utils import rmtree_readonly


@pytest.fixture
def broken_nested(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "CHECKPOINT_BASE", tmp_path / "checkpoints")
    work = tmp_path / "workspace"
    nested = work / "tool"
    nested.mkdir(parents=True)
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    for args in (["init", "-q"], ["config", "user.name", "Test"],
                 ["config", "user.email", "test@example.invalid"]):
        subprocess.run(["git", *args], cwd=nested, env=env, check=True, capture_output=True)
    (nested / "main.py").write_text("committed\n")
    subprocess.run(["git", "add", "."], cwd=nested, env=env, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "seed"], cwd=nested, env=env, check=True, capture_output=True)
    (work / "notes.txt").write_text("before\n")
    mgr = cm.CheckpointManager(enabled=True)
    assert mgr.ensure_checkpoint(str(work), "baseline")
    # The nested .git stops being a repository (e.g. an interrupted rm -rf):
    # git now dies while checking the recorded gitlink for changes.
    rmtree_readonly(nested / ".git" / "objects")
    return mgr, work


def _show(work, commit, path):
    ok, out, _ = cm._run_git(["show", f"{commit}:{path}"], cm._store_path(), str(work))
    return out if ok else None


def test_snapshot_still_captures_the_rest_of_the_directory(broken_nested):
    mgr, work = broken_nested
    (work / "notes.txt").write_text("after\n")
    mgr.new_turn()
    assert mgr.ensure_checkpoint(str(work), "edit notes")
    entry = mgr.list_checkpoints(str(work))[0]
    assert entry["reason"] == "edit notes [nested git repos not captured: tool]"
    assert _show(work, entry["hash"], "notes.txt") == "after"


def test_later_snapshots_keep_working(broken_nested):
    mgr, work = broken_nested
    for text in ("second\n", "third\n"):
        (work / "notes.txt").write_text(text)
        mgr.new_turn()
        assert mgr.ensure_checkpoint(str(work), text.strip())
    assert [e["reason"].split(" [")[0] for e in mgr.list_checkpoints(str(work))[:2]] == ["third", "second"]
