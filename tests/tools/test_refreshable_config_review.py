"""Config write-back authority must be checked at teardown, not only upload."""
import tarfile

import pytest
import yaml

from tools import credential_files as credentials
from tools.environments.file_sync import FileSyncManager


@pytest.mark.parametrize("name", ["auth.json", ".env", ".anthropic_oauth.json", "mcp-tokens/server.json", "token.json"])
@pytest.mark.parametrize("revoke", [False, True])
def test_config_sync_back_authority(tmp_path, monkeypatch, name, revoke):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    host = home / name
    host.parent.mkdir(parents=True)
    host.write_bytes(b"original")
    config = home / "config.yaml"

    def declare(refreshable):
        config.write_text(yaml.safe_dump({"terminal": {"credential_files": [
            {"path": name, "refreshable": refreshable},
        ]}}))

    credentials.clear_credential_files()
    declare(True)
    remote = "/root/.hermes/" + name

    def download(destination):
        staged = tmp_path / "staged"
        staged.write_bytes(b"remote refresh")
        with tarfile.open(destination, "w") as archive:
            archive.add(staged, arcname=remote.lstrip("/"))

    manager = FileSyncManager(
        get_files_fn=lambda: [(str(host), remote)],
        upload_fn=lambda *_: None, delete_fn=lambda *_: None,
        bulk_download_fn=download,
    )
    manager.sync(force=True)
    # Populate the authority lookup before changing the real config.
    credentials.get_refreshable_credential_host_paths()
    if revoke:
        declare(False)
        credentials.clear_credential_files()
    manager.sync_back()
    allowed = name == "token.json" and not revoke
    assert host.read_bytes() == (b"remote refresh" if allowed else b"original")
    if name != "token.json":
        assert credentials.get_credential_file_mounts() == []
    credentials.clear_credential_files()


@pytest.mark.parametrize("guard", [None, "raises"])
def test_config_guard_unavailable_fails_closed(tmp_path, monkeypatch, guard):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "token.json").write_text("token")
    (tmp_path / "config.yaml").write_text("terminal:\n  credential_files:\n    - path: token.json\n      refreshable: true\n")
    credentials.clear_credential_files()
    def raises(_):
        raise RuntimeError("guard unavailable")
    monkeypatch.setattr(credentials, "get_read_block_error", raises if guard else None)
    assert credentials.get_refreshable_credential_host_paths() == set()
    assert credentials.get_credential_file_mounts() == []


def test_config_revocation_without_session_reset(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token = tmp_path / "token.json"
    token.write_text("token")
    config = tmp_path / "config.yaml"
    credentials.clear_credential_files()
    config.write_text("terminal:\n  credential_files:\n    - path: token.json\n      refreshable: true\n")
    assert credentials.get_refreshable_credential_host_paths() == {str(token)}
    config.write_text("terminal:\n  credential_files: []\n")
    assert credentials.get_refreshable_credential_host_paths() == set()
