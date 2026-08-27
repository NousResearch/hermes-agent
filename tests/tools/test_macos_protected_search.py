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
        assert str(home / dirname) in pruned_command
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

    rg_command = next(command for command in env.commands if command.startswith("rg --files"))
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


def test_zero_match_probes_keep_protected_globs(tmp_path, monkeypatch):
    """The zero-match probes re-scan the root (the hidden probe with
    --hidden --no-ignore) — they must carry the same protected exclusions
    as the primary engines, or a zero-result broad search still descends
    into protected app data and fires the unattended TCC prompt."""
    home = tmp_path / "Users" / "alice"
    home.mkdir(parents=True)
    env = RecordingEnvironment(home)
    ops = ShellFileOperations(env)
    monkeypatch.setattr(file_operations, "_HOME", str(home))
    monkeypatch.setattr(file_operations.sys, "platform", "darwin")
    monkeypatch.setattr(ops, "_has_command", lambda command: command == "rg")

    ops._zero_match_probe("needle.[0-9]", str(home), None)

    probe_commands = [c for c in env.commands if c.startswith("rg ")]
    assert probe_commands, "probes issued no rg commands"
    assert any("--hidden --no-ignore" in c for c in probe_commands)
    for command in probe_commands:
        for dirname in PROTECTED_NAMES:
            assert f"!{dirname}/**" in command, (
                f"probe missing exclusion for {dirname}: {command}"
            )
