"""Tests for login-shell PATH preservation in _run_bash.

On Debian-based systems, /etc/profile unconditionally resets PATH for
non-root shells, discarding venv entries set by the Dockerfile's ENV PATH.
_run_bash must restore PATH at the very start of the login shell body -- so
the snapshot (export -p) captures the correct venv entries -- not merely
somewhere in the constructed script. init_session's bootstrap runs its own
``export -p`` dump as its second statement (see base.py's
``_export_dump_excluding_session_vars``), so a restore appended *after* the
whole cmd_string fixes the live shell's PATH too late: the dump has already
captured and persisted the profile-reset value by then (the bug #56634's
fix PR #56642 originally shipped with, per review).

See: https://github.com/NousResearch/hermes-agent/issues/56634
"""

import os
import sys
from unittest.mock import patch

import pytest

from tools.environments.local import _make_run_env, _path_env_key


class TestLoginShellPathPreservation:
    """Verify that _run_bash restores PATH before anything else in a
    login=True invocation runs, so Debian's /etc/profile reset doesn't reach
    the captured snapshot."""

    def test_path_restore_precedes_rest_of_cmd_string(self, tmp_path, monkeypatch):
        """The PATH restore must be the first statement, before both the
        shell-init prelude and the caller's own cmd_string (which, for the
        real init_session caller, contains the export -p snapshot dump)."""
        from tools.environments.local import LocalEnvironment

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("PATH", "/opt/hermes/venv/bin:/usr/bin:/bin")
        env = LocalEnvironment(cwd=str(tmp_path), env={})

        captured = {}

        def mock_popen(args, **kwargs):
            captured["args"] = args
            captured["env"] = kwargs.get("env")
            raise SystemExit(0)

        # A stand-in for init_session's bootstrap: its own export -p marker
        # appears partway through, exactly like the real snapshot dump does.
        marker_cmd = "umask 077\nexport -p >> /tmp/fake-snapshot\necho done\n"

        with patch("subprocess.Popen", side_effect=mock_popen):
            try:
                env._run_bash(marker_cmd, login=True)
            except SystemExit:
                pass

        cmd = captured["args"][-1]  # bash -l -c <cmd>
        assert "/opt/hermes/venv/bin" in cmd
        restore_pos = cmd.find("/opt/hermes/venv/bin")
        dump_pos = cmd.find("export -p >> /tmp/fake-snapshot")
        assert dump_pos != -1, "test's own marker command must survive unmodified"
        assert restore_pos < dump_pos, (
            "PATH restore must precede the export -p dump, or the dump "
            "captures /etc/profile's reset value instead of the venv PATH"
        )

    def test_no_path_restore_for_non_login(self, tmp_path, monkeypatch):
        """When login=False, no PATH restore should be injected (non-login
        shells never source /etc/profile, so nothing needs restoring)."""
        from tools.environments.local import LocalEnvironment

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("PATH", "/opt/hermes/venv/bin:/usr/bin:/bin")
        env = LocalEnvironment(cwd=str(tmp_path), env={})

        captured = {}

        def mock_popen(args, **kwargs):
            captured["args"] = args
            raise SystemExit(0)

        with patch("subprocess.Popen", side_effect=mock_popen):
            try:
                env._run_bash("echo hello", login=False)
            except SystemExit:
                pass

        cmd = captured["args"][2]
        assert cmd == "echo hello"

    def test_make_run_env_preserves_venv_path(self, tmp_path, monkeypatch):
        """_make_run_env should keep venv entries on PATH."""
        venv_bin = "/opt/hermes/.venv/bin"
        original_path = f"{venv_bin}:/usr/local/bin:/usr/bin:/bin"
        monkeypatch.setenv("PATH", original_path)
        monkeypatch.setenv("HOME", str(tmp_path))

        run_env = _make_run_env({})
        path_key = _path_env_key(run_env)
        assert path_key is not None
        assert venv_bin in run_env[path_key]


@pytest.mark.linux_only
class TestLoginShellPathPreservationE2E:
    """Real bash regression for #56634: reproduces the actual failure mode
    (not a mock) by running init_session's own bootstrap shape -- an
    export -p dump partway through the script, same as base.py's
    ``_export_dump_excluding_session_vars`` -- through a real ``bash -l``
    subprocess, and inspecting the file that dump wrote.

    Self-verifying: skips rather than false-passes on a host whose
    /etc/profile doesn't reset PATH (the precondition this bug depends on),
    instead of assuming every Linux CI runner is Debian-family.
    """

    @staticmethod
    def _etc_profile_resets_path(marker_dir: str) -> bool:
        """True if a plain (unpatched) login shell drops *marker_dir* from
        PATH -- i.e. this host reproduces the /etc/profile precondition."""
        import subprocess

        probe_env = {**os.environ, "PATH": f"{marker_dir}:/usr/bin:/bin"}
        result = subprocess.run(
            ["bash", "-l", "-c", "echo $PATH"],
            env=probe_env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return marker_dir not in result.stdout

    def test_snapshot_dump_captures_venv_path_through_profile_reset(
        self, tmp_path, monkeypatch
    ):
        from tools.environments.local import LocalEnvironment

        marker_dir = str(tmp_path / "fake-hermes-venv" / "bin")
        os.makedirs(marker_dir, exist_ok=True)

        if not self._etc_profile_resets_path(marker_dir):
            pytest.skip(
                "this host's /etc/profile doesn't reset PATH for login "
                "shells -- #56634's precondition doesn't reproduce here"
            )

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("PATH", f"{marker_dir}:/usr/bin:/bin")
        env = LocalEnvironment(cwd=str(tmp_path), env={})

        snapshot_file = tmp_path / "snapshot-dump"
        # Same shape as init_session's real bootstrap: export -p as an
        # early statement, well before the script ends.
        bootstrap = f"umask 077\nexport -p > {snapshot_file}\necho ok\n"

        proc = env._run_bash(bootstrap, login=True, timeout=15)
        stdout, _ = proc.communicate(timeout=15)
        assert proc.returncode == 0, stdout

        dumped = snapshot_file.read_text()
        path_lines = [line for line in dumped.splitlines() if line.startswith("declare -x PATH=")]
        assert path_lines, "export -p dump did not capture a PATH entry at all"
        assert marker_dir in path_lines[0], (
            "the venv PATH entry did not survive into the export -p "
            "snapshot -- the restore ran too late relative to the dump"
        )
