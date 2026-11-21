"""Remote-clock sync-back regressions for #140024."""

import io
import logging
import tarfile

import pytest

from tools.environments.file_sync import FileSyncManager


def _manager(tmp_path, remote_time, file_time):
    manifest = tmp_path / "skills" / ".bundled_manifest"
    manifest.parent.mkdir()
    manifest.write_bytes(b"{}")

    def download(dest):
        data = b"remote skill"
        with tarfile.open(dest, "w") as archive:
            info = tarfile.TarInfo("root/.hermes/skills/cat/new/SKILL.md")
            info.size = len(data)
            info.mtime = file_time
            archive.addfile(info, io.BytesIO(data))

    manager = FileSyncManager(
        get_files_fn=lambda: [(str(manifest), "/root/.hermes/skills/.bundled_manifest")],
        upload_fn=lambda host, remote: None,
        delete_fn=lambda paths: None,
        bulk_download_fn=download,
        get_remote_time_fn=remote_time,
    )
    return manager


@pytest.mark.parametrize("remote_epoch", [1_000_000_000, 2_000_000_000])
@pytest.mark.parametrize("offset", [0, 30])
def test_new_file_survives_host_remote_clock_skew(tmp_path, remote_epoch, offset):
    manager = _manager(tmp_path, lambda: remote_epoch, remote_epoch + offset)
    manager.sync(force=True)
    manager.sync_back(hermes_home=tmp_path / "home")
    assert (tmp_path / "skills/cat/new/SKILL.md").read_bytes() == b"remote skill"


def test_preserved_archive_mtime_is_skipped_with_recovery_warning(tmp_path, caplog):
    manager = _manager(tmp_path, lambda: 1_000_000_000, 1_000_000_000 - 86400 * 30)
    manager.sync(force=True)
    with caplog.at_level(logging.WARNING, logger="tools.environments.file_sync"):
        manager.sync_back(hermes_home=tmp_path / "home")
    assert not (tmp_path / "skills/cat/new/SKILL.md").exists()
    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("/root/.hermes/skills/cat/new/SKILL.md" in w and "2592000" in w for w in warnings)


def test_first_successful_remote_boundary_survives_later_sync(tmp_path):
    epochs = iter([1_000_000_000, 1_000_000_100])
    manager = _manager(tmp_path, lambda: next(epochs), 1_000_000_030)
    manager.sync(force=True)
    manager.sync(force=True)
    manager.sync_back(hermes_home=tmp_path / "home")
    assert (tmp_path / "skills/cat/new/SKILL.md").read_bytes() == b"remote skill"


def test_failed_push_does_not_commit_remote_boundary(tmp_path, caplog):
    epochs = iter([1_000_000_000, 1_000_000_100])
    manager = _manager(tmp_path, lambda: next(epochs), 1_000_000_030)

    def fail_upload(host, remote):
        raise OSError("upload failed")

    manager._upload_fn = fail_upload
    manager.sync(force=True)
    manager._upload_fn = lambda host, remote: None
    manager.sync(force=True)
    with caplog.at_level(logging.WARNING, logger="tools.environments.file_sync"):
        manager.sync_back(hermes_home=tmp_path / "home")
    assert not (tmp_path / "skills/cat/new/SKILL.md").exists()
    assert any("70s before first push" in r.message for r in caplog.records)


def test_ssh_constructor_uses_remote_clock_for_sync_back(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools.environments import ssh

    manager = _manager(tmp_path, lambda: 1_000_000_000, 1_000_000_030)
    mapping = manager._get_files_fn()
    monkeypatch.setattr(ssh, "_ensure_ssh_available", lambda: None)
    monkeypatch.setattr(ssh.SSHEnvironment, "_establish_connection", lambda self: None)
    monkeypatch.setattr(ssh.SSHEnvironment, "_detect_remote_home", lambda self: "/root")
    monkeypatch.setattr(ssh.SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
    monkeypatch.setattr(ssh.SSHEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(ssh, "iter_sync_files", lambda base: mapping)
    monkeypatch.setattr(ssh.SSHEnvironment, "_ssh_bulk_upload", lambda self, files: None)
    monkeypatch.setattr(ssh.SSHEnvironment, "_ssh_bulk_download", lambda self, dest: manager._bulk_download_fn(dest))

    def run_ssh(self, command, timeout):
        assert command == "date +%s"
        assert timeout == 10
        return SimpleNamespace(returncode=0, stdout="1000000000\n", stderr="")

    monkeypatch.setattr(ssh.SSHEnvironment, "_run_ssh", run_ssh)
    env = ssh.SSHEnvironment(host="test", user="test")
    env._sync_manager.sync_back(hermes_home=tmp_path / "home")
    assert (tmp_path / "skills/cat/new/SKILL.md").read_bytes() == b"remote skill"


@pytest.mark.parametrize("exit_code", [0, 1])
def test_daytona_remote_clock_checks_command_status(exit_code):
    from types import SimpleNamespace
    from tools.environments.daytona import DaytonaEnvironment

    env = object.__new__(DaytonaEnvironment)

    def execute(command):
        assert command == "date +%s"
        return SimpleNamespace(exit_code=exit_code, result="1000000000\n")

    env._sandbox = SimpleNamespace(process=SimpleNamespace(exec=execute))
    if exit_code:
        with pytest.raises(RuntimeError, match="remote clock failed"):
            env._daytona_remote_time()
    else:
        assert env._daytona_remote_time() == 1_000_000_000


def test_modal_remote_clock_uses_checked_capture():
    from tools.environments.modal import ModalEnvironment

    env = object.__new__(ModalEnvironment)

    def execute(command, *, timeout, fail_label, capture):
        assert (command, timeout, fail_label, capture) == ("date +%s", 10, "remote clock", True)
        return "1000000000\n"

    env._exec = execute
    assert env._modal_remote_time() == 1_000_000_000
