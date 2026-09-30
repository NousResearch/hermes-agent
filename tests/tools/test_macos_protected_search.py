"""macOS TCC-safe behavior for broad file searches."""

from pathlib import Path

import pytest

import tools.file_operations as file_operations
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
    assert "!Downloads" not in rg_command
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
    assert command.startswith("set -o pipefail; (cd '/' && ")
    assert "--sortr=modified" in command
    assert "'!/Users/alice/Downloads'" in command
    assert "'!repo/Downloads'" not in command
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
    assert " -- '.' '--version') 2>/dev/null" in command
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
@pytest.mark.parametrize("native", ["0", "1"])
def test_rg_prunes_only_protected_entries_from_any_cwd(tmp_path, monkeypatch, native):
    """#129733: prune before opendir, without hiding project namesakes."""
    home = tmp_path.resolve() / "home [test]"
    project = home / "Code" / "Downloads"
    project.mkdir(parents=True)
    visible = project / "keep.txt"
    visible.write_text("needle\n")
    protected = home / "Downloads"
    protected.mkdir()
    secret = protected / "private.txt"
    secret.write_text("needle\n")
    other = tmp_path.resolve() / "other"
    other.mkdir()
    monkeypatch.setattr(file_operations, "_HOME", str(home))
    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", native)
    env = LocalEnvironment(cwd=str(project))
    ops = ShellFileOperations(env)
    assert ops._has_command("rg"), "real ripgrep required"

    protected.chmod(0)
    try:
        for root in (str(home), "../..", f"{home}, {other}"):
            for target, pattern in (("files", "*.txt"), ("content", "needle")):
                result = ops.search(pattern, path=root, target=target, file_glob="*.txt")
                assert not result.error, result.to_dict()
                paths = result.files if target == "files" else [m.path for m in result.matches]
                assert {Path(p).resolve() for p in paths} == {visible}, result.to_dict()
                assert "Skipped macOS protected" in result.warning
        assert env.cwd == str(project), "search must not move the session cwd"
    finally:
        protected.chmod(0o700)

    # A folder explicitly named by the caller is still searchable, even when
    # another operand is its broad parent.
    for root in (str(protected), f"{home}, {protected}"):
        result = ops.search("*.txt", path=root, target="files")
        assert not result.error, result.to_dict()
        assert str(secret) in result.files


@pytest.mark.platforms("macos")
@pytest.mark.parametrize("native", ["0", "1"])
def test_rg_zero_match_hints_share_protected_scope(tmp_path, monkeypatch, native):
    """Follow-up probes cannot undo the broad search's protected-folder policy."""
    home = tmp_path.resolve() / "home"
    protected = home / "Downloads"
    protected.mkdir(parents=True)
    (protected / "private.txt").write_text("NEEDLE\nliteral[dot]\n")
    project = home / "Code" / "Downloads"
    project.mkdir(parents=True)
    monkeypatch.setattr(file_operations, "_HOME", str(home))
    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", native)
    ops = ShellFileOperations(LocalEnvironment(cwd=str(project)))
    assert ops._has_command("rg"), "real ripgrep required"

    for pattern in ("needle", "literal[dot]"):
        result = ops.search(pattern, path=str(home), target="content")
        assert not result.error, result.to_dict()
        assert result.total_count == 0
        assert ops._zero_match_probe(pattern, str(home), None) is None
    hidden = project / ".hint.txt"
    hidden.write_text("needle\n")
    hint = ops._zero_match_probe("needle", str(home), None)
    assert "hidden or gitignored" in hint
    assert str(hidden) in hint
