"""macOS TCC-safe behavior for broad file searches."""

import re
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


def _rg_files_commands(commands):
    return [command for command in commands if "--files" in command]


def _find_commands(commands):
    return [
        command for command in commands
        if command.startswith("find ") or "; find " in command
    ]


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

    rg_command = _rg_files_commands(env.commands)[0]
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


def test_empty_ripgrep_file_search_is_one_scan_with_protected_globs(tmp_path, monkeypatch):
    home = tmp_path / "Users" / "alice"
    home.mkdir(parents=True)
    env = RecordingEnvironment(home)
    ops = ShellFileOperations(env)
    monkeypatch.setattr(file_operations, "_HOME", str(home))
    monkeypatch.setattr(file_operations.sys, "platform", "darwin")

    ops.search("*.txt", path=str(home), target="files")

    rg_commands = _rg_files_commands(env.commands)
    assert len(rg_commands) == 1
    for command in rg_commands:
        assert "!Downloads/**" in command


def test_grep_fallback_prunes_by_path_not_basename(tmp_path, monkeypatch):
    """The grep fallback must NOT use --exclude-dir (basename-wide: it would
    skip every nested dir named Downloads anywhere under the root). It routes
    through find's path-scoped -prune instead."""
    home = tmp_path / "Users" / "alice"
    home.mkdir(parents=True)
    env = RecordingEnvironment(home)
    ops = ShellFileOperations(env)
    monkeypatch.setattr(file_operations, "_HOME", str(home))
    monkeypatch.setattr(file_operations.sys, "platform", "darwin")
    monkeypatch.setattr(ops, "_has_command", lambda command: command == "grep")

    ops.search("needle", path=str(home), target="content")

    pruned_command = next(command for command in env.commands if "-prune" in command)
    for dirname in PROTECTED_NAMES:
        # Path-scoped pruning: full protected path present, no basename-wide
        # --exclude-dir for protected names.
        assert ops._escape_shell_arg(str(home / dirname)) in pruned_command
        assert f"--exclude-dir={dirname}" not in pruned_command
        assert f"--exclude-dir='{dirname}'" not in pruned_command


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
    monkeypatch.setattr(file_operations.sys, "platform", "darwin")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(home)))
    monkeypatch.setattr(ops, "_has_command", lambda command: command == "grep")

    result = ops.search("needle", path=str(home), target="content")

    matched_paths = [m.path for m in (result.matches or [])]
    assert any("work/repo/Downloads/notes.txt" in p for p in matched_paths)
    assert not any(str(protected / "secret.txt") in p for p in matched_paths)


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
    monkeypatch.setattr(file_operations.sys, "platform", "darwin")

    result = ops.search("*.txt", path=str(home), target="files")

    rg_command = _rg_files_commands(env.commands)[0]
    assert "!Downloads/**" not in rg_command
    assert result.warning is None


def test_find_fallback_prunes_protected_directories(tmp_path, monkeypatch):
    home = tmp_path / "Users" / "alice"
    home.mkdir(parents=True)
    env = RecordingEnvironment(home)
    ops = ShellFileOperations(env)
    monkeypatch.setattr(file_operations, "_HOME", str(home))
    monkeypatch.setattr(file_operations.sys, "platform", "darwin")
    monkeypatch.setattr(ops, "_has_command", lambda command: command == "find")

    ops.search("*.txt", path=str(home), target="files")

    find_commands = _find_commands(env.commands)
    assert find_commands
    for command in find_commands:
        assert ops._escape_shell_arg(str(home / "Downloads")) in command
        assert "-prune" in command


def _multi_root_protected_search(tmp_path, monkeypatch, engine):
    home = tmp_path / "Users" / "alice"
    downloads = home / "Downloads"
    downloads.mkdir(parents=True)
    env = RecordingEnvironment(home)
    ops = ShellFileOperations(env)
    monkeypatch.setattr(file_operations, "_HOME", str(home))
    monkeypatch.setattr(file_operations.sys, "platform", "darwin")
    monkeypatch.setattr(ops, "_has_command", lambda command: command == engine)
    path_checks = 0

    def execute(command, cwd=None, **kwargs):
        nonlocal path_checks
        env.commands.append(command)
        if command.startswith("test -e"):
            path_checks += 1
            output = "not_found\n" if path_checks == 1 else "exists\n"
            return {"output": output, "returncode": 0}
        if "--files" in command or command.startswith("set -o pipefail; find "):
            return {"output": "", "returncode": 0}
        return {"output": "yes\n", "returncode": 0}

    env.execute = execute
    result = ops.search("*.txt", path=f"{home} {downloads}", target="files")
    return ops, env, result, downloads


def test_rg_multi_root_keeps_explicit_protected_root_and_reports_actual_skips(
    tmp_path, monkeypatch
):
    ops, env, result, downloads = _multi_root_protected_search(
        tmp_path, monkeypatch, "rg"
    )

    command = _rg_files_commands(env.commands)[0]
    absolute_operand = downloads.as_posix() in command
    anchored_operand = (
        f"cd {ops._escape_shell_arg(downloads.parent.as_posix())} &&" in command
        and " -- '.' 'Downloads' 2>/dev/null" in command
    )
    assert absolute_operand or anchored_operand
    assert "!Downloads/**" not in command
    assert "path contained 2 entries" in (result.warning or "")
    assert "macOS protected folders" in (result.warning or "")
    protected_warning = result.warning.split("macOS protected folders", 1)[1]
    assert "Desktop" in protected_warning
    assert "Downloads" not in protected_warning


def test_find_multi_root_keeps_explicit_protected_root_and_reports_actual_skips(
    tmp_path, monkeypatch
):
    ops, env, result, downloads = _multi_root_protected_search(
        tmp_path, monkeypatch, "find"
    )

    command = _find_commands(env.commands)[0]
    assert "Downloads" in command
    assert re.search(r"(?<!! )-path '[^']*/Downloads'", command) is None
    assert "path contained 2 entries" in (result.warning or "")
    assert "macOS protected folders" in (result.warning or "")
    protected_warning = result.warning.split("macOS protected folders", 1)[1]
    assert "Desktop" in protected_warning
    assert "Downloads" not in protected_warning


def test_rg_multi_root_scopes_protected_globs_and_restores_absolute_paths(monkeypatch):
    env = RecordingEnvironment("/")
    ops = ShellFileOperations(env)
    monkeypatch.setattr(file_operations, "_HOME", "/Users/alice")
    monkeypatch.setattr(file_operations.sys, "platform", "darwin")

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


def test_rg_scoped_multi_root_handles_dot_spaces_and_overlapping_roots(monkeypatch):
    env = RecordingEnvironment("/Users/alice/work space")
    ops = ShellFileOperations(env)
    monkeypatch.setattr(file_operations, "_HOME", "/Users/alice")
    monkeypatch.setattr(file_operations.sys, "platform", "darwin")

    def execute(command, cwd=None, **kwargs):
        env.commands.append(command)
        if command.startswith("test -e"):
            output = "not_found\n" if "'., /Users/alice'" in command else "exists\n"
            return {"output": output, "returncode": 0}
        if command.startswith("command -v rg"):
            return {"output": "/usr/bin/rg\n", "returncode": 0}
        if "--files" in command:
            return {"output": "work space/local.txt\n", "returncode": 0}
        raise AssertionError(command)

    env.execute = execute
    result = ops.search("*.txt", path="., /Users/alice", target="files")

    command = _rg_files_commands(env.commands)[0]
    assert "cd '/Users/alice' &&" in command
    assert "'work space' '.'" in command
    assert "'!Downloads/**'" in command
    assert result.files == ["/Users/alice/work space/local.txt"]


def test_rg_scoped_multi_root_terminates_options_before_dash_prefixed_root(monkeypatch):
    env = RecordingEnvironment("/Users/alice")
    ops = ShellFileOperations(env)
    monkeypatch.setattr(file_operations, "_HOME", "/Users/alice")
    monkeypatch.setattr(file_operations.sys, "platform", "darwin")

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
