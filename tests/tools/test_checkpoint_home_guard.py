import pytest

from tools import checkpoint_manager as checkpoints


def test_ensure_checkpoint_refuses_home_through_symlink_alias(tmp_path, monkeypatch):
    """HOME itself is never a snapshot root — even when it resolves through a
    symlink alias (macOS /tmp -> /private/tmp) that makes the raw home string
    differ from its canonical form, which let the old string guard pass and
    snapshot the whole home directory."""
    real_home = tmp_path / "realhome"
    real_home.mkdir()
    alias = tmp_path / "home-alias"
    alias.symlink_to(real_home)
    monkeypatch.setattr(checkpoints.Path, "home", staticmethod(lambda: alias))

    def _fake_take(self, working_dir, reason):
        return True  # would snapshot HOME — the bug this guard prevents

    monkeypatch.setattr(checkpoints.CheckpointManager, "_take", _fake_take)
    manager = checkpoints.CheckpointManager(enabled=True)
    manager._git_available = True
    assert manager.ensure_checkpoint(str(alias), reason="test") is False


def test_ensure_checkpoint_refuses_filesystem_root():
    manager = checkpoints.CheckpointManager(enabled=True)
    manager._git_available = True
    assert manager.ensure_checkpoint("/", reason="test") is False


def test_ensure_checkpoint_allows_normal_project_dir(tmp_path, monkeypatch):
    work = tmp_path / "project"
    work.mkdir()

    def _fake_take(self, working_dir, reason):
        return True

    monkeypatch.setattr(checkpoints.CheckpointManager, "_take", _fake_take)
    manager = checkpoints.CheckpointManager(enabled=True)
    manager._git_available = True
    assert manager.ensure_checkpoint(str(work), reason="test") is True
