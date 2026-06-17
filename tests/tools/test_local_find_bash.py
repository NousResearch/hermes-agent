import os

from tools.environments import local


def _windows_bash_env(monkeypatch, existing_paths, path_bash=None):
    monkeypatch.setattr(local, "_IS_WINDOWS", True)
    monkeypatch.delenv("HERMES_GIT_BASH_PATH", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", "/local")
    monkeypatch.setenv("ProgramFiles", "/pf")
    monkeypatch.setenv("ProgramFiles(x86)", "/pf86")
    monkeypatch.setattr(local.os.path, "isfile", lambda path: path in existing_paths)
    monkeypatch.setattr(local.shutil, "which", lambda name: path_bash if name == "bash" else None)


def test_find_bash_prefers_system_git_for_windows_before_path_wsl(monkeypatch):
    system_git = os.path.join("/pf", "Git", "bin", "bash.exe")
    wsl_bash = os.path.join("/windows", "system32", "bash.exe")

    _windows_bash_env(monkeypatch, {system_git}, path_bash=wsl_bash)

    assert local._find_bash() == system_git


def test_find_bash_keeps_explicit_env_override_first(monkeypatch):
    custom = os.path.join("/custom", "bash.exe")
    portable_git = os.path.join("/local", "hermes", "git", "bin", "bash.exe")
    system_git = os.path.join("/pf", "Git", "bin", "bash.exe")

    _windows_bash_env(monkeypatch, {custom, portable_git, system_git}, path_bash="/windows/system32/bash.exe")
    monkeypatch.setenv("HERMES_GIT_BASH_PATH", custom)

    assert local._find_bash() == custom


def test_find_bash_keeps_hermes_portable_git_before_system_git(monkeypatch):
    portable_git = os.path.join("/local", "hermes", "git", "bin", "bash.exe")
    system_git = os.path.join("/pf", "Git", "bin", "bash.exe")

    _windows_bash_env(monkeypatch, {portable_git, system_git}, path_bash="/windows/system32/bash.exe")

    assert local._find_bash() == portable_git


def test_find_bash_uses_path_bash_as_last_resort(monkeypatch):
    path_bash = os.path.join("/other", "bash.exe")

    _windows_bash_env(monkeypatch, set(), path_bash=path_bash)

    assert local._find_bash() == path_bash
