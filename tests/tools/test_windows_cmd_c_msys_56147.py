"""Regression tests for #56147: MSYS must not rewrite ``cmd /c`` before cmd.exe."""

from tools.environments import local as local_mod
from tools.environments.local import (
    LocalEnvironment,
    _apply_msys_cmd_c_arg_exclusion,
    _make_run_env,
    _normalize_windows_cmd_slash_c,
)


class TestNormalizeWindowsCmdSlashC:
    def test_noop_on_non_windows(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", False)
        cmd = "cmd /c start notepad"
        assert _normalize_windows_cmd_slash_c(cmd) == cmd

    def test_doubles_cmd_slash_c(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        assert _normalize_windows_cmd_slash_c("cmd /c start notepad") == (
            "cmd //c start notepad"
        )

    def test_doubles_cmd_exe_uppercase_flag(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        assert _normalize_windows_cmd_slash_c("cmd.exe /C start notepad") == (
            "cmd.exe //C start notepad"
        )

    def test_leaves_posix_paths_untouched(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        cmd = "ls /c/Users/me"
        assert _normalize_windows_cmd_slash_c(cmd) == cmd


class TestMsysCmdCArgExclusion:
    def test_sets_exclusion_on_windows(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        env: dict[str, str] = {}
        _apply_msys_cmd_c_arg_exclusion(env)
        assert env["MSYS2_ARG_CONV_EXCL"] == "/c"

    def test_appends_without_clobbering_existing(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        env = {"MSYS2_ARG_CONV_EXCL": "/foo"}
        _apply_msys_cmd_c_arg_exclusion(env)
        assert env["MSYS2_ARG_CONV_EXCL"] == "/foo;/c"

    def test_noop_on_non_windows(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", False)
        env: dict[str, str] = {}
        _apply_msys_cmd_c_arg_exclusion(env)
        assert "MSYS2_ARG_CONV_EXCL" not in env

    def test_make_run_env_includes_exclusion_on_windows(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        monkeypatch.setenv("MSYS2_ARG_CONV_EXCL", "/foo")
        run_env = _make_run_env({})
        assert run_env["MSYS2_ARG_CONV_EXCL"] == "/foo;/c"


class TestLocalEnvironmentPrepareCommand:
    def test_prepare_command_normalizes_cmd_on_windows(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        env = LocalEnvironment.__new__(LocalEnvironment)
        command, sudo_stdin = env._prepare_command("cmd /c start notepad")
        assert command == "cmd //c start notepad"
        assert sudo_stdin is None
