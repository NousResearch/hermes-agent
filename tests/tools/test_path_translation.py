from __future__ import annotations

from pathlib import Path


def test_windows_host_path_normalizer_accepts_msys_and_wsl_paths():
    from tools.path_translation import normalize_windows_host_path

    assert (
        normalize_windows_host_path("/c/Users/Admin/Hermes monitoring", host_is_windows=True)
        == r"C:\Users\Admin\Hermes monitoring"
    )
    assert (
        normalize_windows_host_path("/mnt/c/Users/Admin/Hermes monitoring", host_is_windows=True)
        == r"C:\Users\Admin\Hermes monitoring"
    )
    assert (
        normalize_windows_host_path("/mnt/d/MAPA OPERATOR VAULT/file.txt", host_is_windows=True)
        == r"D:\MAPA OPERATOR VAULT\file.txt"
    )
    assert (
        normalize_windows_host_path(r"C:\Users\Admin\Hermes monitoring", host_is_windows=True)
        == r"C:\Users\Admin\Hermes monitoring"
    )


def test_windows_paths_become_git_bash_paths_for_local_bash_execution():
    from tools.path_translation import windows_path_to_git_bash_path

    assert (
        windows_path_to_git_bash_path(r"C:\Users\Admin\Hermes monitoring", host_is_windows=True)
        == "/c/Users/Admin/Hermes monitoring"
    )
    assert (
        windows_path_to_git_bash_path("/mnt/d/MAPA OPERATOR VAULT/file.txt", host_is_windows=True)
        == "/d/MAPA OPERATOR VAULT/file.txt"
    )
    assert (
        windows_path_to_git_bash_path("/c/Users/Admin/Hermes monitoring", host_is_windows=True)
        == "/c/Users/Admin/Hermes monitoring"
    )


def test_file_tool_resolver_normalizes_wsl_path_to_existing_windows_file(tmp_path):
    from tools.file_tools import _resolve_path_for_task
    from tools.path_translation import windows_path_to_wsl_path

    target = tmp_path / "example.txt"
    target.write_text("ok", encoding="utf-8")
    wsl_path = windows_path_to_wsl_path(str(target), host_is_windows=True)

    assert str(_resolve_path_for_task(wsl_path)).lower() == str(target.resolve()).lower()


def test_local_environment_msys_to_windows_path_handles_mnt_paths():
    from tools.environments.local import _msys_to_windows_path

    assert _msys_to_windows_path("/mnt/c/Users/Admin") == r"C:\Users\Admin"
    assert _msys_to_windows_path("/mnt/d/MAPA OPERATOR VAULT") == r"D:\MAPA OPERATOR VAULT"
    assert _msys_to_windows_path("/c/Users/Admin") == r"C:\Users\Admin"


def test_shell_file_operations_expands_paths_for_local_windows_bash(monkeypatch):
    from tools import path_translation
    from tools.file_operations import ShellFileOperations

    monkeypatch.setattr(path_translation, "is_windows_host", lambda: True)

    class LocalEnvironment:
        __module__ = "tools.environments.local"
        cwd = r"C:\Users\Admin"

        def execute(self, command, cwd=None, **kwargs):
            return {"output": "", "returncode": 0}

    ops = ShellFileOperations(LocalEnvironment())

    assert ops._expand_path(r"C:\Users\Admin\Hermes monitoring") == "/c/Users/Admin/Hermes monitoring"
    assert ops._expand_path("/mnt/d/MAPA OPERATOR VAULT") == "/d/MAPA OPERATOR VAULT"


def test_missing_wsl_path_hint_names_normalized_windows_path():
    from tools.path_translation import missing_path_hint, normalize_windows_host_path

    original = "/mnt/d/definitely/not/here.txt"
    normalized = normalize_windows_host_path(original, host_is_windows=True)

    hint = missing_path_hint(original, normalized)

    assert original in hint
    assert r"D:\definitely\not\here.txt" in hint
    assert "normalized" in hint
