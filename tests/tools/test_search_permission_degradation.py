"""Regression tests: search_files degrades on unreadable (EACCES) paths.

t_8f08d82f: a broad file/content search over a tree containing a permission-
restricted directory made rg exit 2. The file-discovery branch discarded stderr
(``2>/dev/null``) and failed closed with an opaque message (or, for
``order='modified'``, the misleading "ripgrep 14+ is required" even though the
version gate had already passed). Content search hard-failed the same way when
no readable file matched.

Contract after the fix:
- rg exit 2 whose diagnostics are ALL permission-denied lines is a
  partial-success condition: results already on stdout are returned, and a
  warning names the skipped paths. With no payload, the result degrades to an
  explicit empty result + actionable warning (never a hard error).
- Any OTHER exit-2 diagnostic (bad glob, bad regex) still fails closed, now
  including the diagnostic text.
- The modified-order invocation-failure message no longer claims a version
  problem the capability gate already ruled out.
"""

import os
import shutil

import pytest

from tools.file_operations import ShellFileOperations
from tools.file_operations_search import (
    _extract_unreadable_paths,
    _is_permission_only_error,
)
from tools.environments.local import LocalEnvironment


# --- unit coverage for the helpers ----------------------------------------------


def test_permission_only_true_for_all_denied_lines():
    diagnostics = (
        "rg: /srv/locked: Permission denied (os error 13)\n"
        "rg: /srv/other: Permission denied (os error 13)\n"
    )
    assert _is_permission_only_error(diagnostics) is True


def test_permission_only_false_when_any_other_diagnostic():
    diagnostics = (
        "rg: /srv/locked: Permission denied (os error 13)\n"
        "rg: error parsing glob '[': unclosed character class\n"
    )
    assert _is_permission_only_error(diagnostics) is False


def test_permission_only_false_for_empty_diagnostics():
    assert _is_permission_only_error("") is False


def test_extract_paths_handles_both_rg_shapes_and_dedups():
    diagnostics = (
        "rg: /srv/a: Permission denied (os error 13)\n"
        "rg: /srv/b: IO error for operation on /srv/b: Permission denied (os error 13)\n"
        "rg: /srv/a: Permission denied (os error 13)\n"
    )
    assert _extract_unreadable_paths(diagnostics) == ["/srv/a", "/srv/b"]


def test_extract_paths_ignores_other_diagnostics():
    assert _extract_unreadable_paths("rg: regex parse error:\n^") == []


# --- fake environments -----------------------------------------------------------


class RecordingEnvironment:
    """Minimal non-local env fake matching the engine-selection test harness."""

    is_local = False
    cwd = "/repo"

    def __init__(self, *, rg_output="", rg_code=0, rg_version="ripgrep 14.1.0\n"):
        self.commands = []
        self.rg_output = rg_output
        self.rg_code = rg_code
        self.rg_version = rg_version

    def execute(self, command, **kwargs):
        self.commands.append(command)
        if command.startswith("test -e "):
            return {"output": "exists\n", "returncode": 0}
        if command.startswith("command -v rg"):
            return {"output": "/usr/bin/rg\n", "returncode": 0}
        if "--version" in command:
            return {"output": self.rg_version, "returncode": 0}
        if "--files" in command:
            return {"output": self.rg_output, "returncode": self.rg_code}
        if "--line-number" in command:
            # rg content-search invocation (goes through the same canned output).
            return {"output": self.rg_output, "returncode": self.rg_code}
        if "--count-matches" in command:
            return {"output": "", "returncode": 1}
        return {"output": "", "returncode": 1}

    @property
    def rg_commands(self):
        return [command for command in self.commands if "--files" in command]


class GrepOnlyEnvironment(RecordingEnvironment):
    """Same harness but without rg, forcing the bounded-find fallback."""

    def execute(self, command, **kwargs):
        self.commands.append(command)
        if command.startswith("test -e "):
            return {"output": "exists\n", "returncode": 0}
        if command.startswith("command -v rg"):
            return {"output": "", "returncode": 1}
        if command.startswith("command -v find"):
            return {"output": "yes\n", "returncode": 0}
        if "find " in command:
            return {"output": self.rg_output, "returncode": self.rg_code}
        return {"output": "", "returncode": 1}


EACCES_DIR = "rg: /srv/locked: Permission denied (os error 13)\n"
EACCES_DIR_SORTED = (
    "rg: /srv/locked: IO error for operation on /srv/locked: "
    "Permission denied (os error 13)\n"
)


# --- file discovery (rg lane) ----------------------------------------------------


@pytest.mark.parametrize("order", ["discovery", "modified"])
def test_file_search_with_unreadable_dir_returns_partial_results(order):
    output = EACCES_DIR + "/repo/partial.py\n"
    env = RecordingEnvironment(rg_output=output, rg_code=2)

    result = ShellFileOperations(env).search(
        "*.py", path="/repo", target="files", order=order)

    assert result.error is None
    assert result.files == ["/repo/partial.py"]
    assert result.total_count == 1
    assert "unreadable path" in (result.warning or "")
    assert "/srv/locked" in (result.warning or "")


def test_file_search_modified_uses_io_error_diagnostic_shape():
    env = RecordingEnvironment(rg_output=EACCES_DIR_SORTED + "/repo/partial.py\n", rg_code=2)

    result = ShellFileOperations(env).search(
        "*.py", path="/repo", target="files", order="modified")

    assert result.error is None
    assert result.files == ["/repo/partial.py"]
    assert "/srv/locked" in (result.warning or "")


@pytest.mark.parametrize("order", ["discovery", "modified"])
def test_file_search_fully_unreadable_degrades_to_actionable_empty(order):
    env = RecordingEnvironment(rg_output=EACCES_DIR, rg_code=2)

    result = ShellFileOperations(env).search(
        "*.py", path="/srv", target="files", order=order)

    assert result.error is None
    assert result.files == []
    assert result.total_count == 0
    assert "unreadable path" in (result.warning or "")
    assert "fix permissions" in (result.warning or "")


def test_file_search_non_permission_exit2_still_fails_closed_with_detail():
    env = RecordingEnvironment(
        rg_output="rg: error parsing glob '[': unclosed character class\n", rg_code=2)

    result = ShellFileOperations(env).search("*.py", path="/repo", target="files")

    assert result.error is not None
    assert result.files == []
    assert "File search failed while running ripgrep" in result.error
    assert "unclosed character class" in result.error


def test_file_search_modified_non_permission_error_no_longer_blames_version():
    env = RecordingEnvironment(
        rg_output="rg: /srv/broken: IO error: disk exploded\n", rg_code=2)

    result = ShellFileOperations(env).search(
        "*.py", path="/repo", target="files", order="modified")

    assert result.error is not None
    assert "exact modification-time order" in result.error.lower()
    assert "ripgrep" in result.error
    assert "invocation failure" in result.error
    assert "disk exploded" in result.error
    assert "Upgrade ripgrep" not in result.error


def test_file_search_permission_degrade_keeps_timeout_limit_reason():
    output = EACCES_DIR + "/repo/partial.py\n[Command timed out after 60s]\n"
    env = RecordingEnvironment(rg_output=output, rg_code=124)

    result = ShellFileOperations(env).search("*.py", path="/repo", target="files")

    assert result.error is None
    assert result.files == ["/repo/partial.py"]
    assert result.limit_reason == "search_timeout"
    assert "unreadable path" in (result.warning or "")


# --- content search --------------------------------------------------------------


def test_content_search_with_unreadable_dir_returns_partial_matches():
    output = EACCES_DIR + "/repo/a.py:3:needle\n"
    env = RecordingEnvironment(rg_output=output, rg_code=2)

    result = ShellFileOperations(env).search(
        "needle", path="/repo", target="content")

    assert result.error is None
    assert [(m.path, m.line_number, m.content) for m in result.matches] == [
        ("/repo/a.py", 3, "needle")]
    assert "unreadable path" in (result.warning or "")
    assert "/srv/locked" in (result.warning or "")


def test_content_search_fully_unreadable_degrades_to_actionable_empty():
    env = RecordingEnvironment(rg_output=EACCES_DIR, rg_code=2)

    result = ShellFileOperations(env).search("needle", path="/srv", target="content")

    assert result.error is None
    assert result.total_count == 0
    assert "unreadable path" in (result.warning or "")
    assert "fix permissions" in (result.warning or "")


def test_content_search_hard_error_regex_still_surfaced():
    env = RecordingEnvironment(
        rg_output="rg: regex parse error:\n    (?:[\n       ^\nerror: unclosed character class\n",
        rg_code=2)

    result = ShellFileOperations(env).search("[", path="/repo", target="content")

    assert result.error is not None
    assert result.error.startswith("Search failed: rg: regex parse error:")
    assert "unclosed character class" in result.error
    assert result.warning is None


# --- bounded-find fallback --------------------------------------------------------


@pytest.mark.parametrize("order", ["discovery", "modified"])
def test_find_exit1_with_payload_degrades_to_partial(order):
    output = "/narrow/partial.py\n" if order == "discovery" else "10 /narrow/partial.py\n"
    env = GrepOnlyEnvironment(rg_output=output, rg_code=1)

    result = ShellFileOperations(env).search(
        "*.py", path="/narrow", target="files", order=order)

    assert result.error is None
    assert result.files == ["/narrow/partial.py"]
    assert result.total_count == 1
    assert result.warning is not None
    assert "unreadable" in (result.warning or "")
    assert "partial" in (result.warning or "")


@pytest.mark.parametrize("order", ["discovery", "modified"])
def test_find_exit1_with_empty_payload_still_fails_closed(order):
    env = GrepOnlyEnvironment(rg_output="", rg_code=1)

    result = ShellFileOperations(env).search(
        "*.py", path="/narrow", target="files", order=order)

    assert result.error is not None
    assert result.files == []


def test_find_exit2_still_fails_closed():
    env = GrepOnlyEnvironment(rg_output="/narrow/partial.py\n", rg_code=2)

    result = ShellFileOperations(env).search("*.py", path="/narrow", target="files")

    assert result.error is not None
    assert result.files == []


# --- live end-to-end (real rg, real filesystem) -----------------------------------


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
def test_live_file_search_skips_unreadable_dir(tmp_path):
    (tmp_path / "open").mkdir()
    (tmp_path / "open" / "a.py").write_text("x = 1\n")
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "secret.py").write_text("x = 2\n")
    os.chmod(locked, 0o000)
    try:
        ops = ShellFileOperations(LocalEnvironment(str(tmp_path)), cwd=str(tmp_path))
        result = ops.search("*.py", path=str(tmp_path), target="files")
        assert result.error is None
        assert result.files == [str(tmp_path / "open" / "a.py")]
        assert result.total_count == 1
        assert str(locked) in (result.warning or "")
    finally:
        os.chmod(locked, 0o755)


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
def test_live_content_search_skips_unreadable_dir(tmp_path):
    (tmp_path / "open").mkdir()
    (tmp_path / "open" / "a.py").write_text("needle here\n")
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "secret.py").write_text("needle hidden\n")
    os.chmod(locked, 0o000)
    try:
        ops = ShellFileOperations(LocalEnvironment(str(tmp_path)), cwd=str(tmp_path))
        result = ops.search("needle", path=str(tmp_path), target="content")
        assert result.error is None
        assert [m.path for m in result.matches] == [str(tmp_path / "open" / "a.py")]
        assert str(locked) in (result.warning or "")
    finally:
        os.chmod(locked, 0o755)