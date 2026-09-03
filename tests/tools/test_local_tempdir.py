from unittest.mock import patch

from tools.environments.local import LocalEnvironment
from tools.environments.snapshot_lifecycle import InodeHeadroom


class TestLocalTempDir:
    def test_uses_os_tmpdir_for_session_artifacts(self, monkeypatch, tmp_path):
        temp_root = tmp_path / "termux-tmp"
        temp_root.mkdir()
        monkeypatch.setenv("TMPDIR", str(temp_root))
        monkeypatch.delenv("TMP", raising=False)
        monkeypatch.delenv("TEMP", raising=False)

        with patch(
            "tools.environments.local.measure_inode_headroom",
            return_value=InodeHeadroom(0.12, 5_000),
        ), \
             patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = LocalEnvironment(cwd=".", timeout=10)

        assert env.get_temp_dir() == str(temp_root)
        assert env._snapshot_path == f"{temp_root}/hermes-snap-{env._session_id}.sh"
        assert env._cwd_file == f"{temp_root}/hermes-cwd-{env._session_id}.txt"

    def test_resolves_symlinked_tmpdir_before_snapshot_paths(self, monkeypatch, tmp_path):
        real_root = tmp_path / "real"
        real_root.mkdir()
        alias = tmp_path / "alias"
        alias.symlink_to(real_root, target_is_directory=True)
        monkeypatch.setenv("TMPDIR", str(alias))
        monkeypatch.delenv("TMP", raising=False)
        monkeypatch.delenv("TEMP", raising=False)

        with patch(
            "tools.environments.local.measure_inode_headroom",
            return_value=InodeHeadroom(0.20, 100_000),
        ), patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = LocalEnvironment(cwd=".", timeout=10)

        assert env.get_temp_dir() == str(real_root)
        assert env._snapshot_path == f"{real_root}/hermes-snap-{env._session_id}.sh"
        assert env._owned_snapshot_artifacts is not None
        env.cleanup()


    def test_falls_back_to_tempfile_when_tmp_missing(self, monkeypatch):
        monkeypatch.delenv("TMPDIR", raising=False)
        monkeypatch.delenv("TMP", raising=False)
        monkeypatch.delenv("TEMP", raising=False)

        with patch(
             "tools.environments.local._resolve_real_temp_root",
             side_effect=lambda value: "/cache/tmp" if value == "/cache/tmp" else None,
        ), \
             patch("tools.environments.local.tempfile.gettempdir", return_value="/cache/tmp"), \
             patch(
                 "tools.environments.local.measure_inode_headroom",
                 return_value=InodeHeadroom(0.12, 5_000),
             ), \
             patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = LocalEnvironment(cwd=".", timeout=10)
            assert env.get_temp_dir() == "/cache/tmp"
            assert env._snapshot_path == f"/cache/tmp/hermes-snap-{env._session_id}.sh"
            assert env._cwd_file == f"/cache/tmp/hermes-cwd-{env._session_id}.txt"
