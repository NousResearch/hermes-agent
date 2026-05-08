from unittest.mock import patch

from tools.environments.local import LocalEnvironment


class TestLocalTempDir:
    def test_uses_os_tmpdir_for_session_artifacts(self, monkeypatch, tmp_path):
        tmpdir = tmp_path / "termux-tmp"
        tmpdir.mkdir()
        monkeypatch.setenv("TMPDIR", str(tmpdir))
        monkeypatch.delenv("TMP", raising=False)
        monkeypatch.delenv("TEMP", raising=False)

        with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = LocalEnvironment(cwd=".", timeout=10)

        assert env.get_temp_dir() == str(tmpdir)
        assert env._snapshot_path == f"{tmpdir}/hermes-snap-{env._session_id}.sh"
        assert env._cwd_file == f"{tmpdir}/hermes-cwd-{env._session_id}.txt"

    def test_prefers_backend_env_tmpdir_override(self, monkeypatch, tmp_path):
        tmpdir = tmp_path / "backend-tmp"
        tmpdir.mkdir()
        monkeypatch.delenv("TMPDIR", raising=False)
        monkeypatch.delenv("TMP", raising=False)
        monkeypatch.delenv("TEMP", raising=False)

        with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = LocalEnvironment(
                cwd=".",
                timeout=10,
                env={"TMPDIR": f"{tmpdir}/"},
            )

        assert env.get_temp_dir() == str(tmpdir)
        assert env._snapshot_path == f"{tmpdir}/hermes-snap-{env._session_id}.sh"
        assert env._cwd_file == f"{tmpdir}/hermes-cwd-{env._session_id}.txt"

    def test_ignores_stale_env_temp_dir_for_session_artifacts(self, monkeypatch, tmp_path):
        stale_tmpdir = tmp_path / "deleted-tmp"
        usable_tmpdir = tmp_path / "usable-tmp"
        usable_tmpdir.mkdir()
        monkeypatch.setenv("TMPDIR", str(stale_tmpdir))
        monkeypatch.delenv("TMP", raising=False)
        monkeypatch.delenv("TEMP", raising=False)

        with patch("tools.environments.local.os.path.isdir", side_effect=lambda p: str(p) == str(usable_tmpdir)), \
             patch("tools.environments.local.os.access", side_effect=lambda p, mode: str(p) == str(usable_tmpdir)), \
             patch("tools.environments.local.tempfile.gettempdir", return_value=str(usable_tmpdir)), \
             patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = LocalEnvironment(cwd=".", timeout=10)
            assert env.get_temp_dir() == str(usable_tmpdir)
            assert env._snapshot_path == f"{usable_tmpdir}/hermes-snap-{env._session_id}.sh"
            assert env._cwd_file == f"{usable_tmpdir}/hermes-cwd-{env._session_id}.txt"

    def test_falls_back_to_tempfile_when_tmp_missing(self, monkeypatch):
        monkeypatch.delenv("TMPDIR", raising=False)
        monkeypatch.delenv("TMP", raising=False)
        monkeypatch.delenv("TEMP", raising=False)

        with patch("tools.environments.local.os.path.isdir", return_value=False), \
             patch("tools.environments.local.os.access", return_value=False), \
             patch("tools.environments.local.tempfile.gettempdir", return_value="/cache/tmp"), \
             patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = LocalEnvironment(cwd=".", timeout=10)
            assert env.get_temp_dir() == "/tmp"
            assert env._snapshot_path == f"/tmp/hermes-snap-{env._session_id}.sh"
            assert env._cwd_file == f"/tmp/hermes-cwd-{env._session_id}.txt"
