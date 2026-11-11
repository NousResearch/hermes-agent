"""Regression tests: search_files degrades on unreadable (EACCES) paths.

A broad file/content search over a tree containing a permission-
restricted directory made rg exit 2. The file-discovery branch discarded stderr
(``2>/dev/null``) and failed closed with an opaque message (or, for
``order='modified'``, the misleading "ripgrep 14+ is required" even though the
version gate had already passed). Content search hard-failed the same way when
no readable file matched.

Contract after the fix:
- rg/grep exit 2 whose diagnostics are ALL permission-denied lines is a
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


# GNU grep spells the same condition without ripgrep's "(os error N)" suffix
# (captured from grep 3.11: ``grep -rnHE needle <tree with a chmod 000 dir>``).
GREP_EACCES = (
    "grep: /srv/locked: Permission denied\n"
    "grep: /srv/unread.txt: Permission denied\n"
)


def test_permission_only_true_for_grep_denied_lines():
    assert _is_permission_only_error(GREP_EACCES) is True


def test_permission_only_false_for_grep_other_diagnostic():
    diagnostics = GREP_EACCES + "grep: /srv/gone: No such file or directory\n"
    assert _is_permission_only_error(diagnostics) is False


def test_extract_paths_handles_grep_shape():
    assert _extract_unreadable_paths(GREP_EACCES) == ["/srv/locked", "/srv/unread.txt"]


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


class GrepContentEnvironment(RecordingEnvironment):
    """No rg, grep present: content search takes the plain ``grep -r`` lane."""

    def execute(self, command, **kwargs):
        self.commands.append(command)
        if command.startswith("test -e "):
            return {"output": "exists\n", "returncode": 0}
        if command.startswith("command -v rg"):
            return {"output": "", "returncode": 1}
        if command.startswith("command -v grep"):
            return {"output": "yes\n", "returncode": 0}
        if "grep -rnH " in command and " -E " in command:
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


@pytest.mark.parametrize("other", ["", "rg: /repo/gone: No such file or directory (os error 2)\n"])
def test_content_search_with_unreadable_dir_returns_partial_matches(other):
    output = EACCES_DIR + other + "/repo/a.py:3:needle\n"
    env = RecordingEnvironment(rg_output=output, rg_code=2)

    result = ShellFileOperations(env).search(
        "needle", path="/repo", target="content")

    assert result.error is None
    assert [(m.path, m.line_number, m.content) for m in result.matches] == [
        ("/repo/a.py", 3, "needle")]
    assert "unreadable path" in (result.warning or "")
    assert "/srv/locked" in (result.warning or "")
    if other:
        assert other.strip() in (result.warning or "")


class CountEnvironment(RecordingEnvironment):
    """rg lane where ``--count-matches`` returns the canned output."""

    def execute(self, command, **kwargs):
        if "--count-matches" in command:
            self.commands.append(command)
            return {"output": self.rg_output, "returncode": self.rg_code}
        return super().execute(command, **kwargs)


def test_count_search_with_unreadable_dir_keeps_partial_warning():
    env = CountEnvironment(rg_output=EACCES_DIR + "/repo/a.py:2\n", rg_code=2)

    result = ShellFileOperations(env).search(
        "needle", path="/repo", target="content", output_mode="count")

    assert result.error is None
    assert result.counts == {"/repo/a.py": 2}
    assert "unreadable path" in (result.warning or "")
    assert "/srv/locked" in (result.warning or "")


def test_count_search_clean_has_no_warning():
    env = CountEnvironment(rg_output="/repo/a.py:2\n", rg_code=0)

    result = ShellFileOperations(env).search(
        "needle", path="/repo", target="content", output_mode="count")

    assert result.counts == {"/repo/a.py": 2}
    assert result.warning is None


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


@pytest.mark.parametrize("code", [126, 2])
def test_content_search_engine_binary_not_executable_is_an_error(code):
    """The shell refusing to run rg itself is a broken invocation, not a skipped path."""
    env = RecordingEnvironment(rg_output="bash: /usr/bin/rg: Permission denied\n", rg_code=code)

    result = ShellFileOperations(env).search("needle", path="/repo", target="content")

    assert result.error is not None
    assert "/usr/bin/rg: Permission denied" in result.error
    assert "unreadable path" not in (result.warning or "")


# --- content search, grep lane (rg absent) ---------------------------------------


def test_grep_content_search_with_unreadable_paths_returns_partial_matches():
    output = "grep: /repo/locked: Permission denied\n/repo/a.py:3:needle\n"
    env = GrepContentEnvironment(rg_output=output, rg_code=2)

    result = ShellFileOperations(env).search("needle", path="/repo", target="content")

    assert any("grep -rnH " in command and " -E " in command for command in env.commands)
    assert result.error is None
    assert [(m.path, m.line_number, m.content) for m in result.matches] == [
        ("/repo/a.py", 3, "needle")]
    assert "unreadable path" in (result.warning or "")
    assert "/repo/locked" in (result.warning or "")


def test_grep_content_search_fully_unreadable_degrades_to_actionable_empty():
    env = GrepContentEnvironment(rg_output=GREP_EACCES, rg_code=2)

    result = ShellFileOperations(env).search("needle", path="/srv", target="content")

    assert result.error is None
    assert result.total_count == 0
    assert "/srv/locked" in (result.warning or "")
    assert "fix permissions" in (result.warning or "")


def test_grep_content_search_non_permission_error_still_fails_closed():
    env = GrepContentEnvironment(
        rg_output="grep: Unmatched ( or \\(\n", rg_code=2)

    result = ShellFileOperations(env).search("(", path="/repo", target="content")

    assert result.error is not None
    assert "Unmatched" in result.error


# --- bounded-find fallback --------------------------------------------------------


@pytest.mark.parametrize("order", ["discovery", "modified"])
@pytest.mark.parametrize("diagnostic,expected", [
    ("", ["unreadable", "partial"]),
    ("find: '/narrow/gone': No such file or directory\n",
     ["Some paths could not be read: find: '/narrow/gone': No such file or directory"]),
])
def test_find_exit1_with_payload_degrades_to_partial(order, diagnostic, expected):
    output = "/narrow/partial.py\n" if order == "discovery" else "10 /narrow/partial.py\n"
    env = GrepOnlyEnvironment(rg_output=diagnostic + output, rg_code=1)

    result = ShellFileOperations(env).search(
        "*.py", path="/narrow", target="files", order=order)

    assert result.error is None
    assert result.files == ["/narrow/partial.py"]
    assert result.total_count == 1
    for text in expected:
        assert text in (result.warning or "")
    if diagnostic:
        assert "permission" not in (result.warning or "").lower()


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


@pytest.mark.platforms("posix")
@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads chmod 000 dirs")
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


@pytest.mark.platforms("posix")
@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads chmod 000 dirs")
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

@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs POSIX non-root chmod")
def test_pruned_grep_permission_only_empty_keeps_skip_warning(tmp_path):
    """find folds a permission failure into exit 1; that empty search is not an error."""
    root = tmp_path / ".hidden" / "repo"
    (root / "locked").mkdir(parents=True)
    (root / "a.py").write_text("nothing here\n")
    (root / "locked").chmod(0)
    try:
        ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))
        ops._has_command = lambda name, _has=ops._has_command: name != "rg" and _has(name)
        result = ops.search("needle", path=str(root), target="content")
    finally:
        (root / "locked").chmod(0o755)
    assert result.error is None and result.total_count == 0
    assert result.warning == (
        f"Skipped 1 unreadable path(s) with restricted permissions: {root}/locked. Results may be "
        "incomplete; fix permissions or run as a user that can read them to include them.")


def _pruned_grep_ops(tmp_path, find_output=None):
    """A grep-only backend whose root sits under a dot-dir (the find -exec grep lane).
    ``find_output`` replaces the find run with an exit-1 result carrying that text."""
    from tools.file_operations_common import ExecuteResult

    root = tmp_path / ".hidden" / "repo"
    root.mkdir(parents=True)
    (root / "a.py").write_text("nothing here\n")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))
    ops._has_command = lambda name, _has=ops._has_command: name != "rg" and _has(name)
    if find_output is not None:
        real_exec = ops._exec

        def fake_exec(command, **kwargs):
            if " find -H " in command:
                return ExecuteResult(stdout=find_output, exit_code=1)
            return real_exec(command, **kwargs)
        ops._exec = fake_exec
    return ops, root


@pytest.mark.skipif(os.name != "posix", reason="find -exec grep lane is POSIX")
@pytest.mark.parametrize("diagnostic", [
    "find: '{root}/locked': Keine Berechtigung",
    "find: {root}/locked: Operation not permitted",
], ids=["german-locale", "eperm"])
def test_pruned_grep_traversal_diagnostics_stay_non_fatal(tmp_path, diagnostic):
    """find's own traversal complaints, in any wording, leave an empty search empty."""
    ops, root = _pruned_grep_ops(tmp_path, find_output=diagnostic.format(root=tmp_path / ".hidden" / "repo") + "\n")
    result = ops.search("needle", path=str(root), target="content")
    assert result.error is None and result.total_count == 0, result


@pytest.mark.skipif(os.name != "posix", reason="find -exec grep lane is POSIX")
def test_pruned_grep_invalid_regex_is_a_search_error(tmp_path):
    ops, root = _pruned_grep_ops(tmp_path)
    result = ops.search("foo(", path=str(root), target="content")
    assert result.error and result.error.startswith("Search failed: grep: "), result


def _pruned_grep_ops_with_probe(tmp_path, probe):
    """Pruned grep lane whose find run reports a traversal diagnostic and no matches;
    ``probe`` stands in for the /dev/null validation rerun (a result or an exception)."""
    from tools.file_operations_common import ExecuteResult

    ops, root = _pruned_grep_ops(tmp_path)
    real_exec = ops._exec
    seen = {}

    def fake_exec(command, **kwargs):
        if " find -H " in command:
            return ExecuteResult(stdout=f"find: {root}/locked: Operation not permitted\n", exit_code=1)
        if command.startswith("grep ") and "/dev/null" in command:
            seen["timeout"] = kwargs.get("timeout")
            if isinstance(probe, BaseException):
                raise probe
            return probe
        return real_exec(command, **kwargs)
    ops._exec = fake_exec
    return ops, root, seen


@pytest.mark.skipif(os.name != "posix", reason="find -exec grep lane is POSIX")
@pytest.mark.parametrize("probe", [
    ("timeout", 124, "Command timed out"),
    ("transport", 255, "ssh: connect to host: Connection refused"),
    ("raised", None, TimeoutError("probe timed out")),
    ("raised-other", None, RuntimeError("backend gone")),
], ids=lambda p: p[0])
def test_pruned_grep_inconclusive_validation_keeps_empty_result(tmp_path, probe):
    """Only grep's own exit 2 marks the invocation broken; a validation rerun that times
    out, loses its transport or raises leaves the completed empty search as it was."""
    from tools.file_operations_common import ExecuteResult

    _, code, payload = probe
    fake = payload if isinstance(payload, BaseException) else ExecuteResult(stdout=payload, exit_code=code)
    ops, root, seen = _pruned_grep_ops_with_probe(tmp_path, fake)
    result = ops.search("needle", path=str(root), target="content")
    assert result.error is None and result.total_count == 0, result
    assert seen["timeout"] is not None and seen["timeout"] <= 10


@pytest.mark.skipif(os.name != "posix", reason="find -exec grep lane is POSIX")
def test_pruned_grep_validation_exit_2_is_a_search_error(tmp_path):
    from tools.file_operations_common import ExecuteResult

    ops, root, _ = _pruned_grep_ops_with_probe(
        tmp_path, ExecuteResult(stdout="grep: Unmatched ( or \\(", exit_code=2))
    result = ops.search("needle", path=str(root), target="content")
    assert result.error == "Search failed: grep: Unmatched ( or \\(", result


@pytest.mark.platforms("posix")
def test_pruned_grep_find_usage_error_is_a_search_error(tmp_path):
    """find's own refusal (no operand, nothing searched) is not reported as zero matches."""
    message = "find: Only one instance of {} is supported with -exec ... +"
    ops, root = _pruned_grep_ops(tmp_path, find_output=message + "\n")
    result = ops.search("needle", path=str(root), target="content")
    assert result.error == f"Search failed: {message}", result
