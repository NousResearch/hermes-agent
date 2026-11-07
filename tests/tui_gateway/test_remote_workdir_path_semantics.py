"""Remote SSH POSIX paths must not depend on the gateway host's OS."""

import pytest

from tui_gateway import session_workdir


@pytest.mark.parametrize("remote", ["/home/kali", "/opt/project", "/", "~", "~/project"])
def test_remote_cwd_shape_accepts_remote_posix_absolute_and_tilde(remote):
    assert session_workdir._is_remote_cwd_shape(remote) is True


@pytest.mark.parametrize("remote", ["", "relative/project", "./project", "../project", r"\local\project", r"C:\local\project"])
def test_remote_cwd_shape_rejects_relative_and_host_windows_paths(remote):
    assert session_workdir._is_remote_cwd_shape(remote) is False


def test_local_workdir_still_rejects_missing_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(session_workdir, "_cwd_is_remote", lambda _home: False)
    with pytest.raises(ValueError, match="working directory does not exist"):
        session_workdir._workspace_cwd(None, str(tmp_path / "missing"))
