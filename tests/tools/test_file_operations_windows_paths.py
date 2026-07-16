"""Regression tests for Windows paths passed through Git Bash file ops."""

import pytest

from tools.file_operations import ShellFileOperations


class _FakeGitBashEnv:
    cwd = r"C:\Users\alice\project"

    def __init__(self):
        self.commands: list[str] = []
        self.uses_msys_paths = True

    def execute(self, command: str, cwd: str = None, **kwargs) -> dict:
        self.commands.append(command)
        if r"C:\Users\alice\project\notes.txt" in command:
            return {"output": "", "returncode": 1}
        if "/c/Users/alice/project/notes.txt" not in command:
            return {"output": "", "returncode": 1}
        if command.startswith("wc -c"):
            return {"output": "12\n", "returncode": 0}
        if command.startswith("head -c"):
            return {"output": "hello\nworld\n", "returncode": 0}
        if command.startswith("sed -n"):
            return {"output": "hello\nworld\n", "returncode": 0}
        if command.startswith("wc -l"):
            return {"output": "2\n", "returncode": 0}
        return {"output": "", "returncode": 1}


def test_read_file_converts_windows_drive_path_for_git_bash(monkeypatch):
    """Native C:\\ paths must be translated before shelling through Git Bash."""
    from tools import file_operations

    monkeypatch.setattr(file_operations, "_IS_WINDOWS", True)
    env = _FakeGitBashEnv()
    ops = ShellFileOperations(env)

    result = ops.read_file(r"C:\Users\alice\project\notes.txt")

    assert result.error is None
    assert "1|hello" in result.content
    assert all(r"C:\Users\alice\project\notes.txt" not in cmd for cmd in env.commands)
    assert any("/c/Users/alice/project/notes.txt" in cmd for cmd in env.commands)


class _FakeGitBashEnvWithLegacyStyle:
    cwd = r"C:\Users\alice\project"

    def __init__(self):
        self.commands: list[str] = []
        self.uses_msys_paths = True
        self.windows_bash_path_style = "wsl"

    def execute(self, command: str, cwd: str = None, **kwargs) -> dict:
        self.commands.append(command)
        if r"C:\Users\alice\project\notes.txt" in command:
            return {"output": "", "returncode": 1}
        if "/c/Users/alice/project/notes.txt" not in command:
            return {"output": "", "returncode": 1}
        if command.startswith("wc -c"):
            return {"output": "12\n", "returncode": 0}
        if command.startswith("head -c"):
            return {"output": "hello\nworld\n", "returncode": 0}
        if command.startswith("sed -n"):
            return {"output": "hello\nworld\n", "returncode": 0}
        if command.startswith("wc -l"):
            return {"output": "2\n", "returncode": 0}
        return {"output": "", "returncode": 1}


def test_read_file_uses_git_bash_path_despite_legacy_style_attribute(monkeypatch):
    from tools import file_operations
    monkeypatch.setattr(file_operations, "_IS_WINDOWS", True)
    env = _FakeGitBashEnvWithLegacyStyle()
    ops = ShellFileOperations(env)

    result = ops.read_file(r"C:\Users\alice\project\notes.txt")

    assert result.error is None
    assert "1|hello" in result.content
    assert all(r"C:\Users\alice\project\notes.txt" not in cmd for cmd in env.commands)
    assert any("/c/Users/alice/project/notes.txt" in cmd for cmd in env.commands)


def test_windows_drive_path_for_bash_supports_other_drives(monkeypatch):
    """Drive-letter conversion must preserve non-C drives too."""
    from tools import file_operations
    from tools.file_operations import _windows_drive_path_for_bash

    monkeypatch.setattr(file_operations, "_IS_WINDOWS", True)

    result = _windows_drive_path_for_bash(r"D:\hermes jiyi\2026-04-29.md")

    assert result == "/d/hermes jiyi/2026-04-29.md"


@pytest.fixture
def windows_denylist(monkeypatch):
    """Make the denylist block only the native Windows credential path."""
    from tools import file_operations

    blocked = r"C:\Users\alice\.ssh\id_rsa"
    calls: list[str] = []

    def fake_is_write_denied(path: str) -> bool:
        calls.append(path)
        return path == blocked

    monkeypatch.setattr(file_operations, "_IS_WINDOWS", True)
    monkeypatch.setattr(file_operations, "_shared_is_write_denied", fake_is_write_denied)
    return blocked, calls


def test_msys_write_denied_paths_are_checked_as_native_windows(windows_denylist):
    """MSYS path conversion must not bypass credential write-deny checks."""
    from tools.file_operations import _is_write_denied

    _blocked, calls = windows_denylist

    assert _is_write_denied("/c/Users/alice/.ssh/id_rsa") is True
    assert r"C:\Users\alice\.ssh\id_rsa" in calls


def test_write_file_denies_converted_windows_credential_path(windows_denylist):
    blocked, _calls = windows_denylist
    env = _FakeGitBashEnv()
    ops = ShellFileOperations(env)

    result = ops.write_file(blocked, "secret")

    assert result.error and "Write denied" in result.error
    assert env.commands == []


def test_delete_file_denies_converted_windows_credential_path(windows_denylist):
    blocked, _calls = windows_denylist
    env = _FakeGitBashEnv()
    ops = ShellFileOperations(env)

    result = ops.delete_file(blocked)

    assert result.error and "Delete denied" in result.error
    assert env.commands == []


def test_move_file_denies_converted_windows_credential_destination(windows_denylist):
    blocked, _calls = windows_denylist
    env = _FakeGitBashEnv()
    ops = ShellFileOperations(env)

    result = ops.move_file(r"C:\Users\alice\safe.txt", blocked)

    assert result.error and "Move denied" in result.error
    assert env.commands == []


def test_patch_replace_denies_converted_windows_credential_path(windows_denylist):
    blocked, _calls = windows_denylist
    env = _FakeGitBashEnv()
    ops = ShellFileOperations(env)

    result = ops.patch_replace(blocked, "old", "new")

    assert result.error and "Write denied" in result.error
    assert env.commands == []
