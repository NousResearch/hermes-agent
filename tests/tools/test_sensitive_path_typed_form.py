"""A POSIX-sensitive path must be refused on every host.

On Windows ``/etc/crontab`` resolves to ``\\etc\\crontab``, so a prefix test against only the
resolved forms never fires and the write/patch tools would accept a destination that is
sensitive on a POSIX host. That is not a theoretical path: the same string crosses a desktop
bridge or a remote backend, which is why the NT-namespace guard in the same function checks the
raw string before anything is joined or resolved. These tests state the invariant in terms of
what the caller typed.
"""

import pytest

from tools.file_tools_write_guards import _check_sensitive_path


@pytest.mark.parametrize("path", [
    "/etc/passwd",
    "/etc/crontab",
    "/private/etc/hosts",
    "/usr/lib/systemd/system/x.service",
    "/private/var/db/secret",
    "/private/var/root/x",
])
def test_posix_system_paths_are_refused_on_any_host(path):
    message = _check_sensitive_path(path)
    assert message is not None, f"{path} was allowed"
    assert "sensitive system path" in message


@pytest.mark.parametrize("path", ["/var/run/docker.sock", "/run/docker.sock"])
def test_docker_socket_is_refused_on_any_host(path):
    assert _check_sensitive_path(path) is not None


def test_relative_spelling_is_not_mistaken_for_the_system_path(tmp_path, monkeypatch):
    """``etc/passwd`` names a file in the task's cwd, not ``/etc/passwd``."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "etc").mkdir()
    assert _check_sensitive_path("etc/passwd") is None


def test_macos_temp_carveouts_stay_allowed():
    """The carve-outs exist so $TMPDIR on macOS is writable; they must not regress."""
    assert _check_sensitive_path("/private/var/folders/xy/T/tmp.txt") is None
    assert _check_sensitive_path("/private/var/tmp/build.log") is None
