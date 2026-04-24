from unittest.mock import patch

from tools.environments.local import LocalEnvironment


class TestLocalTempDir:
    def test_uses_os_tmpdir_for_session_artifacts(self, monkeypatch):
        from tools.environments import local

        monkeypatch.setattr(local, "_IS_WINDOWS", False)
        monkeypatch.setenv("TMPDIR", "/data/data/com.termux/files/usr/tmp")
        monkeypatch.delenv("TMP", raising=False)
        monkeypatch.delenv("TEMP", raising=False)

        with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = LocalEnvironment(cwd=".", timeout=10)

        assert env.get_temp_dir() == "/data/data/com.termux/files/usr/tmp"
        assert env._snapshot_path == f"/data/data/com.termux/files/usr/tmp/hermes-snap-{env._session_id}.sh"
        assert env._cwd_file == f"/data/data/com.termux/files/usr/tmp/hermes-cwd-{env._session_id}.txt"

    def test_prefers_backend_env_tmpdir_override(self, monkeypatch):
        from tools.environments import local

        monkeypatch.setattr(local, "_IS_WINDOWS", False)
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
        assert env._snapshot_path == (
            f"/data/data/com.termux/files/home/.cache/hermes-tmp/hermes-snap-{env._session_id}.sh"
        )
        assert env._cwd_file == (
            f"/data/data/com.termux/files/home/.cache/hermes-tmp/hermes-cwd-{env._session_id}.txt"
        )

    def test_falls_back_to_tempfile_when_tmp_missing(self, monkeypatch):
        from tools.environments import local

        monkeypatch.setattr(local, "_IS_WINDOWS", False)
        monkeypatch.delenv("TMPDIR", raising=False)
        monkeypatch.delenv("TMP", raising=False)
        monkeypatch.delenv("TEMP", raising=False)

        with patch("tools.environments.local.os.path.isdir", return_value=False), \
             patch("tools.environments.local.os.access", return_value=False), \
             patch("tools.environments.local.tempfile.gettempdir", return_value="/cache/tmp"), \
             patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = LocalEnvironment(cwd=".", timeout=10)
            assert env.get_temp_dir() == "/cache/tmp"
            assert env._snapshot_path == f"/cache/tmp/hermes-snap-{env._session_id}.sh"
            assert env._cwd_file == f"/cache/tmp/hermes-cwd-{env._session_id}.txt"

    def test_windows_tempdir_uses_msys_for_bash_and_windows_for_python(self, monkeypatch, tmp_path):
        from tools.environments import local
        from tools.platform_compat import windows_path_to_msys

        monkeypatch.setattr(local, "_IS_WINDOWS", True)
        temp_root = tmp_path / "Temp Root"
        expected_msys_temp = windows_path_to_msys(str(temp_root / "hermes"))
        expected_win_temp = str(temp_root / "hermes").replace("\\", "/")

        with patch("tools.platform_compat.tempfile.gettempdir", return_value=str(temp_root)), \
             patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = LocalEnvironment(cwd=r"D:\Code\hermes-agent", timeout=10)
            assert env.get_temp_dir() == expected_msys_temp

        assert env.cwd == "/d/Code/hermes-agent"
        assert env._snapshot_path.startswith(f"{expected_msys_temp}/hermes-snap-")
        assert env._snapshot_path_win.startswith(f"{expected_win_temp}/hermes-snap-")
        assert env._cwd_file_win.startswith(f"{expected_win_temp}/hermes-cwd-")


class TestWindowsBashResolution:
    def test_is_wsl_bash_detects_system32_and_sysnative(self, monkeypatch):
        from tools.environments import local

        monkeypatch.setenv("SystemRoot", r"C:\Windows")

        assert local._is_wsl_bash(r"C:\Windows\System32\bash.exe") is True
        assert local._is_wsl_bash(r"C:\Windows\Sysnative\bash.exe") is True
        assert local._is_wsl_bash(r"C:\Program Files\Git\bin\bash.exe") is False

    def test_find_bash_skips_wsl_path_lookup_when_git_bash_missing(self, monkeypatch):
        from tools.environments import local

        monkeypatch.setattr(local, "_IS_WINDOWS", True)
        monkeypatch.delenv("HERMES_GIT_BASH_PATH", raising=False)
        monkeypatch.setenv("SystemRoot", r"C:\Windows")
        monkeypatch.setattr(local.os.path, "isfile", lambda _path: False)
        monkeypatch.setattr(local.shutil, "which", lambda _name: r"C:\Windows\System32\bash.exe")

        try:
            local._find_bash()
        except RuntimeError as exc:
            assert "WSL bash is not supported" in str(exc)
        else:
            raise AssertionError("_find_bash should reject WSL bash")
