"""Tests for tools/checkpoint_tool.py — CheckpointStore, dispatcher, and convenience wrapper."""

import json
import shutil
import pytest
from pathlib import Path

from tools.checkpoint_tool import (
    CheckpointStore,
    checkpoint_tool,
    check_checkpoint_requirements,
    take_checkpoint,
    _shadow_repo_path,
    CHECKPOINT_BASE,
    DEFAULT_EXCLUDES,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture()
def work_dir(tmp_path):
    """A temporary working directory that acts as the user's project root."""
    d = tmp_path / "project"
    d.mkdir()
    (d / "app.py").write_text("print('hello')\n")
    return d


@pytest.fixture()
def checkpoint_base(tmp_path):
    """Isolated checkpoint base directory — never writes to ~/.hermes/."""
    return tmp_path / "checkpoints"


@pytest.fixture()
def store(work_dir, checkpoint_base, monkeypatch):
    """A CheckpointStore with shadow repo redirected to tmp_path."""
    monkeypatch.setattr("tools.checkpoint_tool.CHECKPOINT_BASE", checkpoint_base)
    return CheckpointStore(str(work_dir))


# =========================================================================
# Shadow repo initialisation
# =========================================================================

class TestShadowRepoInit:
    def test_shadow_repo_created_on_first_take(self, store):
        result = store.take("initial commit")
        assert result["success"] is True
        assert (store.shadow_repo / "HEAD").exists()

    def test_shadow_repo_path_is_deterministic(self, work_dir, checkpoint_base, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_tool.CHECKPOINT_BASE", checkpoint_base)
        path1 = _shadow_repo_path(str(work_dir))
        path2 = _shadow_repo_path(str(work_dir))
        assert path1 == path2

    def test_different_dirs_get_different_shadow_repos(self, tmp_path, checkpoint_base, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_tool.CHECKPOINT_BASE", checkpoint_base)
        dir_a = tmp_path / "project_a"
        dir_b = tmp_path / "project_b"
        assert _shadow_repo_path(str(dir_a)) != _shadow_repo_path(str(dir_b))

    def test_shadow_repo_does_not_contain_git_dir_in_project(self, work_dir, store):
        store.take("initial commit")
        # The user's working directory must not gain a .git folder
        assert not (work_dir / ".git").exists()

    def test_shadow_repo_has_info_exclude(self, store):
        store.take("initial commit")
        exclude_file = store.shadow_repo / "info" / "exclude"
        assert exclude_file.exists()
        content = exclude_file.read_text()
        assert "node_modules/" in content
        assert ".env" in content
        assert "dist/" in content

    def test_second_take_does_not_reinit(self, store):
        store.take("first")
        # Modify shadow repo's HEAD to detect if re-init would overwrite it
        head_before = (store.shadow_repo / "HEAD").read_text()
        store.take("second")
        head_after = (store.shadow_repo / "HEAD").read_text()
        assert head_before == head_after  # HEAD format unchanged by re-init


# =========================================================================
# CheckpointStore.take()
# =========================================================================

class TestCheckpointStoreTake:
    def test_take_returns_success(self, store):
        result = store.take("initial commit")
        assert result["success"] is True

    def test_take_returns_commit_hash(self, store):
        result = store.take("initial commit")
        assert "commit_hash" in result
        # Short git hash: 7 hex chars by default, up to 16
        assert len(result["commit_hash"]) >= 7
        assert all(c in "0123456789abcdef" for c in result["commit_hash"])

    def test_take_includes_working_dir(self, store, work_dir):
        result = store.take("initial commit")
        assert result["working_dir"] == str(work_dir)

    def test_take_empty_reason_rejected(self, store):
        result = store.take("")
        assert result["success"] is False
        assert "reason" in result["error"].lower()

    def test_take_whitespace_reason_rejected(self, store):
        result = store.take("   ")
        assert result["success"] is False

    def test_take_idempotent_no_change(self, store):
        first = store.take("initial commit")
        assert first["success"] is True
        first_hash = first["commit_hash"]

        # No files changed — second take should return the existing hash
        second = store.take("nothing changed")
        assert second["success"] is True
        assert second["commit_hash"] == first_hash
        assert second["files_changed"] == 0

    def test_take_new_commit_after_file_change(self, store, work_dir):
        first = store.take("initial")
        (work_dir / "app.py").write_text("print('changed')\n")
        second = store.take("after change")
        assert second["success"] is True
        assert second["commit_hash"] != first["commit_hash"]

    def test_take_multiple_files_tracked(self, store, work_dir):
        (work_dir / "utils.py").write_text("# utils\n")
        (work_dir / "config.json").write_text('{"key": "value"}\n')
        result = store.take("added utils and config")
        assert result["success"] is True
        assert result["files_changed"] >= 2

    def test_take_excludes_node_modules(self, store, work_dir):
        nm = work_dir / "node_modules" / "lodash"
        nm.mkdir(parents=True)
        (nm / "index.js").write_text("module.exports = {};\n")
        result = store.take("with node_modules present")
        assert result["success"] is True
        # Restore and verify node_modules was not tracked:
        # delete the dir, restore, it should remain absent
        shutil.rmtree(str(work_dir / "node_modules"))
        store.restore(result["commit_hash"])
        assert not (work_dir / "node_modules").exists()

    def test_take_excludes_dot_env(self, store, work_dir):
        (work_dir / ".env").write_text("SECRET=hunter2\n")
        result = store.take("with .env present")
        assert result["success"] is True
        (work_dir / ".env").unlink()
        store.restore(result["commit_hash"])
        # .env was excluded — restore should not recreate it
        assert not (work_dir / ".env").exists()


# =========================================================================
# CheckpointStore.restore()
# =========================================================================

class TestCheckpointStoreRestore:
    def test_restore_reverts_file_content(self, store, work_dir):
        (work_dir / "app.py").write_text("version 1\n")
        snapshot = store.take("version 1")
        assert snapshot["success"] is True
        commit_hash = snapshot["commit_hash"]

        (work_dir / "app.py").write_text("version 2\n")
        assert (work_dir / "app.py").read_text() == "version 2\n"

        result = store.restore(commit_hash)
        assert result["success"] is True
        assert (work_dir / "app.py").read_text() == "version 1\n"

    def test_restore_returns_commit_hash(self, store, work_dir):
        snap = store.take("initial")
        result = store.restore(snap["commit_hash"])
        assert result["commit_hash"] == snap["commit_hash"]

    def test_restore_includes_reason(self, store, work_dir):
        snap = store.take("my restore reason")
        result = store.restore(snap["commit_hash"])
        assert result["success"] is True
        assert "my restore reason" in result["reason"]

    def test_restore_nonexistent_hash_fails(self, store, work_dir):
        store.take("initial")
        result = store.restore("deadbeef")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_restore_empty_hash_fails(self, store):
        result = store.restore("")
        assert result["success"] is False
        assert "commit_hash" in result["error"].lower()

    def test_restore_whitespace_hash_fails(self, store):
        result = store.restore("   ")
        assert result["success"] is False

    def test_restore_does_not_touch_excluded_paths(self, store, work_dir):
        # .env exists before snapshot but is excluded
        env_file = work_dir / ".env"
        env_file.write_text("BEFORE=1\n")
        snap = store.take("with env")
        # Change .env after snapshot
        env_file.write_text("AFTER=2\n")
        store.restore(snap["commit_hash"])
        # .env was never tracked so restore should not revert it
        assert env_file.read_text() == "AFTER=2\n"

    def test_restore_no_checkpoints_fails(self, store):
        result = store.restore("abc1234")
        assert result["success"] is False

    def test_restore_reverts_deleted_file(self, store, work_dir):
        snap = store.take("before delete")
        (work_dir / "app.py").unlink()
        store.restore(snap["commit_hash"])
        assert (work_dir / "app.py").exists()

    def test_restore_to_older_of_two_checkpoints(self, store, work_dir):
        (work_dir / "app.py").write_text("state A\n")
        snap_a = store.take("state A")

        (work_dir / "app.py").write_text("state B\n")
        store.take("state B")

        store.restore(snap_a["commit_hash"])
        assert (work_dir / "app.py").read_text() == "state A\n"


# =========================================================================
# CheckpointStore.list()
# =========================================================================

class TestCheckpointStoreList:
    def test_list_empty_when_no_checkpoints(self, store):
        result = store.list()
        assert result["success"] is True
        assert result["checkpoints"] == []
        assert result["count"] == 0

    def test_list_includes_working_dir(self, store, work_dir):
        result = store.list()
        assert result["working_dir"] == str(work_dir)

    def test_list_returns_one_checkpoint(self, store):
        store.take("only checkpoint")
        result = store.list()
        assert result["success"] is True
        assert result["count"] == 1

    def test_list_returns_newest_first(self, store, work_dir):
        store.take("first checkpoint")
        (work_dir / "app.py").write_text("changed\n")
        store.take("second checkpoint")

        result = store.list()
        assert result["success"] is True
        assert result["count"] == 2
        # Newest first
        assert result["checkpoints"][0]["reason"] == "second checkpoint"
        assert result["checkpoints"][1]["reason"] == "first checkpoint"

    def test_list_each_entry_has_required_fields(self, store):
        store.take("test entry")
        result = store.list()
        entry = result["checkpoints"][0]
        assert "commit_hash" in entry
        assert "timestamp" in entry
        assert "reason" in entry

    def test_list_reason_matches_take_reason(self, store):
        store.take("specific reason string")
        result = store.list()
        assert result["checkpoints"][0]["reason"] == "specific reason string"

    def test_list_timestamp_is_iso_format(self, store):
        store.take("timestamped")
        result = store.list()
        ts = result["checkpoints"][0]["timestamp"]
        # ISO 8601 date starts with YYYY-
        assert ts[:4].isdigit()
        assert "-" in ts

    def test_list_respects_limit(self, store, work_dir):
        for i in range(5):
            (work_dir / "app.py").write_text(f"version {i}\n")
            store.take(f"checkpoint {i}")

        result = store.list(limit=2)
        assert result["success"] is True
        assert len(result["checkpoints"]) == 2

    def test_list_commit_hash_matches_take_hash(self, store):
        snap = store.take("known hash")
        result = store.list()
        listed_hash = result["checkpoints"][0]["commit_hash"]
        # The listed short hash should be a prefix of or equal to the take hash
        assert snap["commit_hash"].startswith(listed_hash) or listed_hash.startswith(snap["commit_hash"])


# =========================================================================
# checkpoint_tool() dispatcher
# =========================================================================

class TestCheckpointToolDispatcher:
    @pytest.fixture(autouse=True)
    def _patch_base(self, checkpoint_base, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_tool.CHECKPOINT_BASE", checkpoint_base)

    def test_unknown_action_returns_error(self, work_dir):
        result = json.loads(checkpoint_tool("frobnicate", working_dir=str(work_dir)))
        assert result["success"] is False
        assert "Unknown action" in result["error"]

    def test_take_requires_reason(self, work_dir):
        result = json.loads(checkpoint_tool("take", working_dir=str(work_dir)))
        assert result["success"] is False
        assert "reason" in result["error"].lower()

    def test_restore_requires_commit_hash(self, work_dir):
        result = json.loads(checkpoint_tool("restore", working_dir=str(work_dir)))
        assert result["success"] is False
        assert "commit_hash" in result["error"].lower()

    def test_nonexistent_working_dir_returns_error(self, tmp_path):
        missing = str(tmp_path / "does_not_exist")
        result = json.loads(checkpoint_tool("take", working_dir=missing, reason="test"))
        assert result["success"] is False
        assert "not a directory" in result["error"].lower() or "not exist" in result["error"].lower()

    def test_working_dir_defaults_to_cwd(self, monkeypatch, work_dir):
        monkeypatch.chdir(work_dir)
        result = json.loads(checkpoint_tool("list"))
        assert result["success"] is True
        assert result["working_dir"] == str(work_dir)

    def test_valid_take_returns_json(self, work_dir):
        result = json.loads(checkpoint_tool("take", working_dir=str(work_dir), reason="via dispatcher"))
        assert result["success"] is True
        assert "commit_hash" in result

    def test_valid_list_returns_json(self, work_dir):
        checkpoint_tool("take", working_dir=str(work_dir), reason="first")
        result = json.loads(checkpoint_tool("list", working_dir=str(work_dir)))
        assert result["success"] is True
        assert isinstance(result["checkpoints"], list)

    def test_valid_restore_returns_json(self, work_dir):
        snap = json.loads(checkpoint_tool("take", working_dir=str(work_dir), reason="to restore"))
        result = json.loads(
            checkpoint_tool("restore", working_dir=str(work_dir), commit_hash=snap["commit_hash"])
        )
        assert result["success"] is True

    def test_list_limit_clamped_to_100(self, work_dir):
        result = json.loads(checkpoint_tool("list", working_dir=str(work_dir), limit=9999))
        # Should not error — limit is silently clamped
        assert result["success"] is True


# =========================================================================
# check_checkpoint_requirements()
# =========================================================================

class TestCheckpointRequirements:
    def test_available_when_git_installed(self):
        # git is required to run the rest of this suite anyway
        if not shutil.which("git"):
            pytest.skip("git not installed")
        assert check_checkpoint_requirements() is True

    def test_unavailable_when_git_missing(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _cmd: None)
        assert check_checkpoint_requirements() is False


# =========================================================================
# take_checkpoint() convenience wrapper
# =========================================================================

class TestTakeCheckpointWrapper:
    def test_does_not_raise_on_invalid_working_dir(self):
        # Should silently swallow errors — never propagate to caller
        take_checkpoint("/nonexistent/path/xyz", "reason")  # must not raise

    def test_does_not_raise_on_empty_reason(self, work_dir, checkpoint_base, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_tool.CHECKPOINT_BASE", checkpoint_base)
        take_checkpoint(str(work_dir), "")  # must not raise

    def test_does_not_raise_when_git_missing(self, work_dir, checkpoint_base, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_tool.CHECKPOINT_BASE", checkpoint_base)
        monkeypatch.setattr("shutil.which", lambda _cmd: None)
        take_checkpoint(str(work_dir), "no git installed")  # must not raise

    def test_successful_wrapper_creates_shadow_repo(self, work_dir, checkpoint_base, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_tool.CHECKPOINT_BASE", checkpoint_base)
        take_checkpoint(str(work_dir), "via wrapper")
        shadow_repo = _shadow_repo_path(str(work_dir))
        assert (shadow_repo / "HEAD").exists()
