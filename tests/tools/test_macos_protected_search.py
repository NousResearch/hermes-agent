"""macOS TCC-safe behavior for broad file searches."""

from pathlib import Path

import pytest

from tools import file_operations
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.file_operations_search import _macos_protected_search_exclusions


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


def _rg_files_commands(commands):
    return [command for command in commands if "--files" in command]


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


@pytest.mark.platforms("macos")
def test_grep_pruned_search_still_finds_nested_protected_names(tmp_path, monkeypatch):
    """A repo-internal directory literally named 'Downloads' must still be
    searched by the pruned grep path — the exact regression --exclude-dir had."""
    home = tmp_path / "Users" / "alice"
    project = home / "work" / "repo" / "Downloads"
    project.mkdir(parents=True)
    (project / "notes.txt").write_text("needle here\n")
    protected = home / "Downloads"
    protected.mkdir()
    (protected / "secret.txt").write_text("needle protected\n")
    monkeypatch.setattr(file_operations, "_HOME", str(home))
    ops = ShellFileOperations(LocalEnvironment(cwd=str(home)))
    monkeypatch.setattr(ops, "_has_command", lambda command: command == "grep")

    result = ops.search("needle", path=str(home), target="content")

    matched_paths = [m.path for m in (result.matches or [])]
    assert any("work/repo/Downloads/notes.txt" in p for p in matched_paths)
    assert not any(str(protected / "secret.txt") in p for p in matched_paths)


@pytest.mark.platforms("macos")
def test_remote_backend_never_prunes(tmp_path, monkeypatch):
    """Non-local environments get no exclusions: platform facts describe the
    controller, not the execution host (macOS controller + Linux SSH backend
    must not prune the remote's Downloads)."""
    home = tmp_path / "Users" / "alice"
    home.mkdir(parents=True)
    env = RecordingEnvironment(home)
    env.is_local = False  # remote/container-shaped backend
    ops = ShellFileOperations(env)
    monkeypatch.setattr(file_operations, "_HOME", str(home))

    result = ops.search("*.txt", path=str(home), target="files")

    rg_command = _rg_files_commands(env.commands)[0]
    assert "!Downloads/**" not in rg_command
    assert result.warning is None


@pytest.mark.platforms("macos")
def test_rg_multi_root_scopes_protected_globs_and_restores_absolute_paths(monkeypatch):
    env = RecordingEnvironment("/")
    ops = ShellFileOperations(env)
    monkeypatch.setattr(file_operations, "_HOME", "/Users/alice")

    def execute(command, cwd=None, **kwargs):
        env.commands.append(command)
        if command.startswith("test -e"):
            output = "not_found\n" if "'/Users/alice /repo'" in command else "exists\n"
            return {"output": output, "returncode": 0}
        if command.startswith("command -v rg"):
            return {"output": "/usr/bin/rg\n", "returncode": 0}
        if "--version" in command:
            return {"output": "ripgrep 14.1.1\n", "returncode": 0}
        if "--files" in command:
            return {
                "output": "repo/Downloads/visible.txt\nUsers/alice/safe.txt\n",
                "returncode": 0,
            }
        raise AssertionError(command)

    env.execute = execute
    result = ops.search(
        "*.txt", path="/Users/alice /repo", target="files", order="modified"
    )

    commands = _rg_files_commands(env.commands)
    assert len(commands) == 1
    command = commands[0]
    assert command.startswith("set -o pipefail; cd '/' && ")
    assert "--sortr=modified" in command
    assert "'!Users/alice/Downloads/**'" in command
    assert "'!repo/Downloads/**'" not in command
    assert "'Users/alice' 'repo'" in command
    assert result.files == [
        "/repo/Downloads/visible.txt",
        "/Users/alice/safe.txt",
    ]


@pytest.mark.platforms("macos")
def test_rg_scoped_multi_root_terminates_options_before_dash_prefixed_root(monkeypatch):
    env = RecordingEnvironment("/Users/alice")
    ops = ShellFileOperations(env)
    monkeypatch.setattr(file_operations, "_HOME", "/Users/alice")

    def execute(command, cwd=None, **kwargs):
        env.commands.append(command)
        if command.startswith("test -e"):
            output = "not_found\n" if "'., --version'" in command else "exists\n"
            return {"output": output, "returncode": 0}
        if command.startswith("command -v rg"):
            return {"output": "/usr/bin/rg\n", "returncode": 0}
        if "--files" in command:
            return {"output": "", "returncode": 0}
        raise AssertionError(command)

    env.execute = execute
    result = ops.search("*.txt", path="., --version", target="files")

    command = _rg_files_commands(env.commands)[0]
    assert "cd '/Users/alice' &&" in command
    assert " -- '.' '--version' 2>/dev/null" in command
    assert result.error is None


@pytest.mark.platforms("macos")
def test_real_ripgrep_does_not_descend_into_protected_folder(tmp_path, monkeypatch):
    home = tmp_path / "Users" / "alice"
    safe = home / "safe"
    protected = home / "Downloads"
    safe.mkdir(parents=True)
    protected.mkdir()
    (safe / "visible.txt").write_text("needle")
    (protected / "protected.txt").write_text("needle")
    monkeypatch.setattr(file_operations, "_HOME", str(home))
    ops = ShellFileOperations(LocalEnvironment(cwd=str(home)))

    result = ops.search("needle", path=str(home), target="content")

    paths = [match.path for match in result.matches]
    assert any("visible.txt" in path for path in paths)
    assert all("protected.txt" not in path for path in paths)


@pytest.mark.platforms("macos")
def test_broad_rg_files_search_warns_folders_are_still_opened(monkeypatch):
    """rg's '!<dir>/**' globs exclude only the folder's CONTENTS — the folder
    entry itself is still opened, so the TCC prompt can fire mid-search. The
    warning must not claim the prompt was avoided (#134775)."""
    env = RecordingEnvironment("/Users/alice")
    ops = ShellFileOperations(env)
    monkeypatch.setattr(file_operations, "_HOME", "/Users/alice")

    def stub_exec(command, cwd=None, **kwargs):
        env.commands.append(command)
        if command.startswith("test -e"):
            return {"output": "exists\n", "returncode": 0}
        if command.startswith("command -v rg"):
            return {"output": "/usr/bin/rg\n", "returncode": 0}
        if "--files" in command:
            return {"output": "safe/visible.txt\n", "returncode": 0}
        return {"output": "", "returncode": 1}

    env.execute = stub_exec
    result = ops.search("*.txt", path="/Users/alice", target="files")

    command = _rg_files_commands(env.commands)[0]
    assert "'!Downloads/**'" in command
    assert result.warning is not None
    assert "Excluded the contents of macOS protected folders" in result.warning
    assert "still opens those folders" in result.warning
    assert "Skipped macOS protected folders" not in result.warning


@pytest.mark.platforms("macos")
def test_real_ripgrep_broad_content_search_warns_folders_are_still_opened(tmp_path, monkeypatch):
    home = tmp_path / "Users" / "alice"
    safe = home / "safe"
    protected = home / "Downloads"
    safe.mkdir(parents=True)
    protected.mkdir()
    (safe / "visible.txt").write_text("needle")
    (protected / "protected.txt").write_text("needle")
    monkeypatch.setattr(file_operations, "_HOME", str(home))
    ops = ShellFileOperations(LocalEnvironment(cwd=str(home)))

    result = ops.search("needle", path=str(home), target="content")

    assert result.warning is not None
    assert "Excluded the contents of macOS protected folders" in result.warning
    assert "still opens those folders" in result.warning
    assert "Skipped macOS protected folders" not in result.warning


@pytest.mark.platforms("macos")
def test_grep_pruned_broad_search_keeps_skipped_warning(tmp_path, monkeypatch):
    """find -prune genuinely never descends into the protected dirs, so that
    transport keeps the strong 'Skipped …' wording."""
    home = tmp_path / "Users" / "alice"
    safe = home / "safe"
    protected = home / "Downloads"
    safe.mkdir(parents=True)
    protected.mkdir()
    (safe / "visible.txt").write_text("needle here\n")
    (protected / "secret.txt").write_text("needle protected\n")
    monkeypatch.setattr(file_operations, "_HOME", str(home))
    ops = ShellFileOperations(LocalEnvironment(cwd=str(home)))
    monkeypatch.setattr(ops, "_has_command", lambda command: command == "grep")

    result = ops.search("needle", path=str(home), target="content")

    matched_paths = [m.path for m in (result.matches or [])]
    assert any("visible.txt" in p for p in matched_paths)
    assert not any("secret.txt" in p for p in matched_paths)
    assert result.warning is not None
    assert result.warning.startswith("Skipped macOS protected folders")
    assert "to avoid an unattended privacy prompt" in result.warning


@pytest.mark.platforms("macos")
def test_multi_path_search_warns_folders_are_still_opened(monkeypatch):
    env = RecordingEnvironment("/Users/alice")
    ops = ShellFileOperations(env)
    monkeypatch.setattr(file_operations, "_HOME", "/Users/alice")

    def stub_exec(command, cwd=None, **kwargs):
        env.commands.append(command)
        if command.startswith("test -e"):
            output = "not_found\n" if "'/Users/alice /repo'" in command else "exists\n"
            return {"output": output, "returncode": 0}
        if command.startswith("command -v rg"):
            return {"output": "/usr/bin/rg\n", "returncode": 0}
        if "--files" in command:
            return {"output": "safe.txt\n", "returncode": 0}
        raise AssertionError(command)

    env.execute = stub_exec
    result = ops.search("*.txt", path="/Users/alice /repo", target="files")

    assert result.error is None
    assert result.warning is not None
    assert "path contained 2 entries; searched 2 that exist" in result.warning
    assert "Excluded the contents of macOS protected folders" in result.warning
    assert "Skipped macOS protected folders" not in result.warning
