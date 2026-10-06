"""Behavioural regression: the login-shell env snapshot must keep Hermes-managed
PATH entries even when a login profile resets PATH.

Debian's ``/etc/profile`` unconditionally resets PATH for non-root shells,
dropping the venv dir the process environment put there — so the snapshot
(``export -p``) serialises the reset PATH and bare ``python3``/``pip`` resolve
to the system interpreter (#56634). These tests read the snapshot back from a
real ``LocalEnvironment`` instead of asserting on injected markers.

See: https://github.com/NousResearch/hermes-agent/issues/56634
"""

import pathlib
import re

import pytest

from tools.environments.local import LocalEnvironment

_PATH_LINE = re.compile(r'^(?:declare -x|export) PATH="(.*)"$', re.M)


def _managed_setup(tmp_path, monkeypatch, *, profile_body: str):
    """A resetting (or no-op) profile plus one fake Hermes-managed bin dir."""
    profile = tmp_path / "profile.sh"
    profile.write_text(profile_body)
    managed = tmp_path / "managed-bin"
    managed.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        "tools.environments.local._resolve_shell_init_files", lambda: [str(profile)]
    )
    monkeypatch.setattr(
        "tools.environments.local._resolve_hermes_bin_dir", lambda: str(managed)
    )
    monkeypatch.setattr(
        "tools.environments.local._managed_runtime_path_entries", lambda: []
    )
    return managed


def _snapshot_path_value(env: LocalEnvironment) -> list[str]:
    content = pathlib.Path(env._snapshot_path).read_text()
    match = _PATH_LINE.search(content)
    assert match, f"no PATH line in snapshot:\n{content}"
    return match.group(1).split(":")


class TestLoginShellPathPreservation:
    @pytest.mark.platforms("posix")
    def test_snapshot_keeps_managed_dir_when_profile_resets_path(
        self, tmp_path, monkeypatch
    ):
        """Debian-style profile reset (PATH=... before the snapshot's export -p)
        must not drop the managed dir from the captured snapshot."""
        managed = _managed_setup(
            tmp_path, monkeypatch, profile_body='PATH="/usr/local/bin:/usr/bin:/bin"\n'
        )

        env = LocalEnvironment(
            cwd=str(tmp_path), env={"PATH": f"{managed}:/usr/local/bin:/usr/bin:/bin"}
        )
        try:
            assert str(managed) in _snapshot_path_value(env)
        finally:
            env.cleanup()

    @pytest.mark.platforms("posix")
    def test_snapshot_does_not_duplicate_managed_dir(self, tmp_path, monkeypatch):
        """A profile that leaves PATH alone: the restore's case guard must not
        prepend a second copy of an entry the shell already has."""
        managed = _managed_setup(
            tmp_path, monkeypatch, profile_body=": # profile that keeps PATH\n"
        )

        env = LocalEnvironment(
            cwd=str(tmp_path), env={"PATH": f"{managed}:/usr/local/bin:/usr/bin:/bin"}
        )
        try:
            entries = _snapshot_path_value(env)
            assert entries.count(str(managed)) == 1
        finally:
            env.cleanup()

    @pytest.mark.platforms("posix")
    def test_non_login_path_keeps_managed_dir_without_profile(
        self, tmp_path, monkeypatch
    ):
        """Non-login commands never source the profile, so the resetting profile
        can't drop the managed dir — the baseline the login path is compared to."""
        managed = _managed_setup(
            tmp_path, monkeypatch, profile_body='PATH="/usr/local/bin:/usr/bin:/bin"\n'
        )

        env = LocalEnvironment(
            cwd=str(tmp_path), env={"PATH": f"{managed}:/usr/local/bin:/usr/bin:/bin"}
        )
        try:
            proc = env._run_bash('printf "%s" "$PATH"', login=False, timeout=30)
            out, _ = proc.communicate(timeout=30)
            assert str(managed) in out.split(":")
        finally:
            env.cleanup()
