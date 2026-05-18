import os
import stat
from unittest.mock import patch

from tools.environments.local import LocalEnvironment


class TestLocalTempDir:
    def test_uses_os_tmpdir_for_session_artifacts(self, monkeypatch):
        monkeypatch.setenv("TMPDIR", "/data/data/com.termux/files/usr/tmp")
        monkeypatch.delenv("TMP", raising=False)
        monkeypatch.delenv("TEMP", raising=False)

        with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = LocalEnvironment(cwd=".", timeout=10)

        assert env.get_temp_dir() == "/data/data/com.termux/files/usr/tmp"
        assert env._artifact_dir == (
            f"/data/data/com.termux/files/usr/tmp/hermes-session-{env._session_id}"
        )
        assert env._snapshot_path == f"{env._artifact_dir}/snapshot.sh"
        assert env._cwd_file == f"{env._artifact_dir}/cwd.txt"

    def test_prefers_backend_env_tmpdir_override(self, monkeypatch):
        monkeypatch.delenv("TMPDIR", raising=False)
        monkeypatch.delenv("TMP", raising=False)
        monkeypatch.delenv("TEMP", raising=False)

        with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = LocalEnvironment(
                cwd=".",
                timeout=10,
                env={"TMPDIR": "/data/data/com.termux/files/home/.cache/hermes-tmp/"},
            )

        assert env.get_temp_dir() == "/data/data/com.termux/files/home/.cache/hermes-tmp"
        assert env._artifact_dir == (
            f"/data/data/com.termux/files/home/.cache/hermes-tmp/hermes-session-{env._session_id}"
        )
        assert env._snapshot_path == f"{env._artifact_dir}/snapshot.sh"
        assert env._cwd_file == f"{env._artifact_dir}/cwd.txt"

    def test_falls_back_to_tempfile_when_tmp_missing(self, monkeypatch):
        monkeypatch.delenv("TMPDIR", raising=False)
        monkeypatch.delenv("TMP", raising=False)
        monkeypatch.delenv("TEMP", raising=False)

        with patch("tools.environments.local.os.path.isdir", return_value=False), \
             patch("tools.environments.local.os.access", return_value=False), \
             patch("tools.environments.local.tempfile.gettempdir", return_value="/cache/tmp"), \
             patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = LocalEnvironment(cwd=".", timeout=10)
            assert env.get_temp_dir() == "/cache/tmp"
            assert env._artifact_dir == f"/cache/tmp/hermes-session-{env._session_id}"
            assert env._snapshot_path == f"{env._artifact_dir}/snapshot.sh"
            assert env._cwd_file == f"{env._artifact_dir}/cwd.txt"

    def test_session_artifacts_are_private_on_disk(self, tmp_path):
        env = LocalEnvironment(
            cwd=str(tmp_path),
            timeout=10,
            env={"TMPDIR": str(tmp_path), "NONHERMES_SECRET_TOKEN": "dummy-secret"},
        )
        try:
            assert stat.S_IMODE(os.stat(env._artifact_dir).st_mode) == 0o700
            assert stat.S_IMODE(os.stat(env._snapshot_path).st_mode) == 0o600
            assert stat.S_IMODE(os.stat(env._cwd_file).st_mode) == 0o600
        finally:
            env.cleanup()
