"""macOS TCC-safe behavior for broad file searches."""

from pathlib import Path

import tools.file_operations as file_operations
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations, _macos_protected_search_exclusions


class RecordingEnvironment:
    def __init__(self, cwd):
        self.cwd = str(cwd)
        self.commands = []

    def execute(self, command, cwd=None, **kwargs):
        self.commands.append(command)
        if command.startswith("test -e"):
            return {"output": "exists\n", "returncode": 0}
        if command.startswith("command -v"):
            return {"output": "yes\n", "returncode": 0}
        return {"output": "", "returncode": 1}


PROTECTED_NAMES = {
    "Desktop",
    "Documents",
    "Downloads",
    "Library",
    "Movies",
    "Music",
    "Pictures",
}


def test_broad_home_search_excludes_macos_protected_folders(tmp_path):
    home = tmp_path / "Users" / "alice"

    exclusions = _macos_protected_search_exclusions(
        str(home), cwd=str(tmp_path), home=str(home), platform="darwin"
    )

    assert {Path(item).parts[0] for item in exclusions} == PROTECTED_NAMES


def test_explicit_protected_folder_search_is_not_excluded(tmp_path):
    home = tmp_path / "Users" / "alice"

    exclusions = _macos_protected_search_exclusions(
        str(home / "Downloads"), cwd=str(tmp_path), home=str(home), platform="darwin"
    )

    assert exclusions == []


def test_non_macos_search_has_no_implicit_exclusions(tmp_path):
    home = tmp_path / "home" / "alice"

    exclusions = _macos_protected_search_exclusions(
        str(home), cwd=str(tmp_path), home=str(home), platform="linux"
    )

    assert exclusions == []


def test_broad_file_search_passes_protected_globs_to_ripgrep(tmp_path, monkeypatch):
    home = tmp_path / "Users" / "alice"
    home.mkdir(parents=True)
    env = RecordingEnvironment(home)
    ops = ShellFileOperations(env)
    monkeypatch.setattr(file_operations, "_HOME", str(home))
    monkeypatch.setattr(file_operations.sys, "platform", "darwin")

    result = ops.search("*.txt", path=str(home), target="files")

    rg_command = next(command for command in env.commands if command.startswith("rg --files"))
    for dirname in PROTECTED_NAMES:
        assert f"!{dirname}/**" in rg_command
    assert result.warning is not None
    assert "macOS protected folders" in result.warning


def test_broad_content_search_passes_protected_globs_to_ripgrep(tmp_path, monkeypatch):
    home = tmp_path / "Users" / "alice"
    home.mkdir(parents=True)
    env = RecordingEnvironment(home)
    ops = ShellFileOperations(env)
    monkeypatch.setattr(file_operations, "_HOME", str(home))
    monkeypatch.setattr(file_operations.sys, "platform", "darwin")

    ops.search("needle", path=str(home), target="content")

    rg_command = next(command for command in env.commands if command.startswith("set -o pipefail; rg"))
    for dirname in PROTECTED_NAMES:
        assert f"!{dirname}/**" in rg_command


def test_legacy_ripgrep_file_fallback_keeps_protected_globs(tmp_path, monkeypatch):
    home = tmp_path / "Users" / "alice"
    home.mkdir(parents=True)
    env = RecordingEnvironment(home)
    ops = ShellFileOperations(env)
    monkeypatch.setattr(file_operations, "_HOME", str(home))
    monkeypatch.setattr(file_operations.sys, "platform", "darwin")

    ops.search("*.txt", path=str(home), target="files")

    rg_commands = [command for command in env.commands if command.startswith("rg --files")]
    assert len(rg_commands) == 2
    for command in rg_commands:
        assert "!Downloads/**" in command


def test_grep_fallback_excludes_protected_directories(tmp_path, monkeypatch):
    home = tmp_path / "Users" / "alice"
    home.mkdir(parents=True)
    env = RecordingEnvironment(home)
    ops = ShellFileOperations(env)
    monkeypatch.setattr(file_operations, "_HOME", str(home))
    monkeypatch.setattr(file_operations.sys, "platform", "darwin")
    monkeypatch.setattr(ops, "_has_command", lambda command: command == "grep")

    ops.search("needle", path=str(home), target="content")

    grep_command = next(command for command in env.commands if " grep " in command)
    for dirname in PROTECTED_NAMES:
        assert f"--exclude-dir='{dirname}'" in grep_command


def test_find_fallback_prunes_protected_directories(tmp_path, monkeypatch):
    home = tmp_path / "Users" / "alice"
    home.mkdir(parents=True)
    env = RecordingEnvironment(home)
    ops = ShellFileOperations(env)
    monkeypatch.setattr(file_operations, "_HOME", str(home))
    monkeypatch.setattr(file_operations.sys, "platform", "darwin")
    monkeypatch.setattr(ops, "_has_command", lambda command: command == "find")

    ops.search("*.txt", path=str(home), target="files")

    find_commands = [command for command in env.commands if command.startswith("find ")]
    assert find_commands
    for command in find_commands:
        assert str(home / "Downloads") in command
        assert "-prune" in command


def test_real_ripgrep_does_not_descend_into_protected_folder(tmp_path, monkeypatch):
    home = tmp_path / "Users" / "alice"
    safe = home / "safe"
    protected = home / "Downloads"
    safe.mkdir(parents=True)
    protected.mkdir()
    (safe / "visible.txt").write_text("needle")
    (protected / "protected.txt").write_text("needle")
    monkeypatch.setattr(file_operations, "_HOME", str(home))
    monkeypatch.setattr(file_operations.sys, "platform", "darwin")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(home)))

    result = ops.search("needle", path=str(home), target="content")

    paths = [match.path for match in result.matches]
    assert any("visible.txt" in path for path in paths)
    assert all("protected.txt" not in path for path in paths)
