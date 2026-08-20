"""Tests for tools/file_operations.py — deny list, result dataclasses, helpers."""

import os
import re
import pytest
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

from tools.file_operations import (
    _is_write_denied,
    ReadResult,
    WriteResult,
    PatchResult,
    SearchResult,
    SearchMatch,
    LintResult,
    ShellFileOperations,
    MAX_LINE_LENGTH,
    normalize_read_pagination,
    normalize_search_pagination,
)


# =========================================================================
# Write deny list
# =========================================================================

class TestIsWriteDenied:
    def test_ssh_authorized_keys_denied(self):
        path = os.path.join(str(Path.home()), ".ssh", "authorized_keys")
        assert _is_write_denied(path) is True


    def test_netrc_denied(self):
        path = os.path.join(str(Path.home()), ".netrc")
        assert _is_write_denied(path) is True

    @pytest.mark.parametrize("name", [".pgpass", ".npmrc", ".pypirc"])
    def test_credential_config_files_denied(self, name):
        path = os.path.join(str(Path.home()), name)
        assert _is_write_denied(path) is True

    def test_aws_prefix_denied(self):
        path = os.path.join(str(Path.home()), ".aws", "credentials")
        assert _is_write_denied(path) is True


    @pytest.mark.parametrize(
        "path",
        [
            "./.anthropic_oauth.json",
        ],
    )
    def test_oauth_traversal_denied(self, path):
        """Path traversal attempts to protected OAuth files must be blocked."""
        from hermes_constants import get_hermes_home
        hermes_home = get_hermes_home()
        full_path = str(hermes_home / path)
        assert _is_write_denied(full_path) is True


    def test_mcp_tokens_dir_protected_in_profile_mode(self, tmp_path, monkeypatch):
        """mcp-tokens/ under profile AND under root must both be denied."""
        root = tmp_path / "hermes"
        profile = root / "profiles" / "coder"
        profile.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(profile))

        assert _is_write_denied(str(profile / "mcp-tokens" / "tok.json")) is True
        assert _is_write_denied(str(root / "mcp-tokens" / "tok.json")) is True
        # The directory itself must also be denied (not just files inside)
        assert _is_write_denied(str(root / "mcp-tokens")) is True

    def test_pairing_dir_denied(self, tmp_path, monkeypatch):
        """Regression: pairing/ must be write-denied under both profile and root.

        PR #30383 introduced ~/.hermes/pairing/{platform}-approved.json as the
        gateway access-control list. Without this block, a prompt-injected agent
        can write arbitrary user IDs into an approved file, granting persistent
        gateway access without going through the pairing code flow — the same
        threat class that motivated protecting webhook_subscriptions.json.
        """
        root = tmp_path / "hermes"
        profile = root / "profiles" / "coder"
        profile.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(profile))

        # Active profile pairing entries
        assert _is_write_denied(str(profile / "pairing" / "telegram-approved.json")) is True
        assert _is_write_denied(str(profile / "pairing" / "discord-pending.json")) is True
        # The directory itself
        assert _is_write_denied(str(profile / "pairing")) is True
        # Root pairing entries (profile mode — same shape as mcp-tokens gap)
        assert _is_write_denied(str(root / "pairing" / "telegram-approved.json")) is True
        assert _is_write_denied(str(root / "pairing")) is True


# =========================================================================
# Result dataclasses
# =========================================================================

class TestReadResult:
    def test_to_dict_omits_defaults(self):
        r = ReadResult()
        d = r.to_dict()
        assert "error" not in d    # None omitted
        assert "similar_files" not in d  # empty list omitted


    def test_binary_fields(self):
        r = ReadResult(is_binary=True, is_image=True, mime_type="image/png")
        d = r.to_dict()
        assert d["is_binary"] is True
        assert d["is_image"] is True
        assert d["mime_type"] == "image/png"


class TestWriteResult:
    def test_to_dict_omits_none(self):
        r = WriteResult(bytes_written=100)
        d = r.to_dict()
        assert d["bytes_written"] == 100
        assert "error" not in d
        assert "warning" not in d

    def test_to_dict_includes_error(self):
        r = WriteResult(error="Permission denied")
        d = r.to_dict()
        assert d["error"] == "Permission denied"


class TestPatchResult:
    def test_to_dict_success(self):
        r = PatchResult(success=True, diff="--- a\n+++ b", files_modified=["a.py"])
        d = r.to_dict()
        assert d["success"] is True
        assert d["diff"] == "--- a\n+++ b"
        assert d["files_modified"] == ["a.py"]

    def test_to_dict_error(self):
        r = PatchResult(error="File not found")
        d = r.to_dict()
        assert d["success"] is False
        assert d["error"] == "File not found"


class TestSearchResult:
    def test_to_dict_with_matches(self):
        m = SearchMatch(path="a.py", line_number=10, content="hello")
        r = SearchResult(matches=[m], total_count=1)
        d = r.to_dict()
        assert d["total_count"] == 1
        assert len(d["matches"]) == 1
        assert d["matches"][0]["path"] == "a.py"


    def test_truncated_flag(self):
        r = SearchResult(total_count=100, truncated=True)
        d = r.to_dict()
        assert d["truncated"] is True


class TestSearchResultDensify:
    """Path-grouped densification of content-mode matches (lossless)."""

    def _matches(self, n, paths=None):
        # Real ripgrep output is path-ordered: all matches in a file are
        # consecutive (verified against live search_files corpus). The fixture
        # mirrors that — group by path, then enumerate lines within each.
        paths = paths or ["a.py"]
        out = []
        per = max(1, n // len(paths))
        ln = 0
        for p in paths:
            for _ in range(per):
                ln += 1
                out.append(SearchMatch(path=p, line_number=ln,
                                       content=f"line content {ln}"))
        # pad remainder onto the last path
        while len(out) < n:
            ln += 1
            out.append(SearchMatch(path=paths[-1], line_number=ln,
                                   content=f"line content {ln}"))
        return out

    def test_densify_off_by_default(self):
        # The model-facing default must be unchanged for callers that don't
        # opt in: verbose array, no matches_text key.
        r = SearchResult(matches=self._matches(10), total_count=10)
        d = r.to_dict()
        assert "matches" in d
        assert "matches_text" not in d

    def test_densify_below_threshold_keeps_verbose(self):
        # Too few matches: the grouping header would cost more than it saves,
        # so we fall back to the verbose array even with densify=True.
        r = SearchResult(matches=self._matches(4), total_count=4)
        d = r.to_dict(densify=True)
        assert "matches" in d
        assert "matches_text" not in d


    def test_densify_paths_with_spaces(self):
        matches = [SearchMatch(path="my dir/a b.py", line_number=i + 1, content=f"x{i}")
                   for i in range(6)]
        text = SearchResult(matches=matches, total_count=6).to_dict(densify=True)["matches_text"]
        # path with spaces survives as a header line verbatim
        assert "my dir/a b.py" in text.split("\n")[0]


class TestLintResult:
    def test_skipped(self):
        r = LintResult(skipped=True, message="No linter for .md files")
        d = r.to_dict()
        assert d["status"] == "skipped"
        assert d["message"] == "No linter for .md files"


    def test_error(self):
        r = LintResult(success=False, output="SyntaxError line 5")
        d = r.to_dict()
        assert d["status"] == "error"
        assert "SyntaxError" in d["output"]


# =========================================================================
# ShellFileOperations helpers
# =========================================================================

@pytest.fixture()
def mock_env():
    """Create a mock terminal environment."""
    env = MagicMock()
    env.cwd = "/tmp/test"
    env.execute.return_value = {"output": "", "returncode": 0}
    return env


@pytest.fixture()
def file_ops(mock_env):
    return ShellFileOperations(mock_env)


def make_real_subprocess_env(cwd: str, include_stderr: bool = False) -> MagicMock:
    """Mock env whose execute() runs the command in a real subprocess.

    For tests that need the generated shell scripts to actually run
    (search fallback, atomic-write permissions) instead of being
    intercepted by a bare MagicMock.  ``include_stderr`` folds stderr
    into ``output`` for tests that surface shell error text; leave it
    off for tests that parse structured stdout (e.g. find results).
    """
    env = MagicMock()
    env.cwd = cwd

    def execute(command, **kwargs):
        completed = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
            input=kwargs.get("stdin_data"),
        )
        output = completed.stdout
        if include_stderr:
            output += completed.stderr
        return {
            "output": output,
            "returncode": completed.returncode,
        }

    env.execute = execute
    return env


class TestShellFileOpsHelpers:
    def test_normalize_read_pagination_clamps_invalid_values(self):
        assert normalize_read_pagination(offset=0, limit=0) == (1, 1)
        assert normalize_read_pagination(offset=-10, limit=-5) == (1, 1)
        assert normalize_read_pagination(offset="bad", limit="bad") == (1, 2000)
        assert normalize_read_pagination(offset=2, limit=999999) == (2, 2000)


    def test_escape_shell_arg_simple(self, file_ops):
        assert file_ops._escape_shell_arg("hello") == "'hello'"


    @pytest.mark.windows_only
    def test_escape_shell_arg_rewrites_forward_slash_native_paths(self, file_ops):
        """Windows-only: ``_bash_safe_path`` only rewrites drive paths to the
        Git Bash form on Windows, where the MSYS path mangling it works around
        actually happens."""
        assert file_ops._escape_shell_arg(
            "C:/Users/alice/notes.txt"
        ) == "'/c/Users/alice/notes.txt'"

    @pytest.mark.windows_only
    def test_read_file_uses_bash_safe_windows_paths(self, mock_env):
        """Windows-only: proves read_file's shell commands carry the MSYS path
        form Git Bash needs — a translation that is a no-op off Windows."""
        commands = []

        def side_effect(command, **kwargs):
            commands.append(command)
            # The size probe gates `wc -c` behind `[ -f ]` so a FIFO or device
            # cannot block the read; it still reports a plain byte count.
            if command.startswith("if [ -f ") or command.startswith("wc -c"):
                return {"output": "5\n", "returncode": 0}
            if command.startswith("head -c") and "| base64" in command:
                import base64 as b64
                return {"output": b64.b64encode(b"hello").decode(), "returncode": 0}
            if command.startswith("head -c"):
                return {"output": "hello", "returncode": 0}
            if command.startswith("sed -n"):
                return {"output": "hello\n", "returncode": 0}
            if command.startswith("wc -l"):
                return {"output": "1\n", "returncode": 0}
            return {"output": "", "returncode": 0}

        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.read_file(r"C:\Users\alice\notes.txt")

        assert result.error is None
        assert commands[0] == (
            "if [ -f '/c/Users/alice/notes.txt' ]; "
            "then wc -c < '/c/Users/alice/notes.txt' 2>/dev/null; "
            "elif [ -e '/c/Users/alice/notes.txt' ]; "
            "then echo __hermes_not_regular__; "
            "else exit 1; fi"
        )
        assert commands[1] == "head -c 1000 '/c/Users/alice/notes.txt' 2>/dev/null | base64"
        assert commands[2] == "sed -n '1,2000p' '/c/Users/alice/notes.txt' | cut -b1-8001"
        assert commands[3] == "wc -l < '/c/Users/alice/notes.txt'"

    def test_is_likely_binary_by_extension(self, file_ops):
        assert file_ops._is_likely_binary("photo.png") is True
        assert file_ops._is_likely_binary("data.db") is True
        assert file_ops._is_likely_binary("code.py") is False
        assert file_ops._is_likely_binary("readme.md") is False


    def test_cwd_fallback_to_slash(self):
        env = MagicMock(spec=[])  # no cwd attribute
        ops = ShellFileOperations(env)
        assert ops.cwd == "/"

    def test_read_file_strips_leaked_terminal_fence_markers(self, mock_env):
        leaked = (
            "'\x07__HERMES_FENCE_a9f7b3__\x1b]0;cat "
            "'/tmp/test/a.py' 2> /dev/null\x07\n"
            "print('ok')\n"
            "__HERMES_FENCE_a9f7b3__\x07'\n"
        )

        def side_effect(command, **kwargs):
            if command.startswith("if [ -f ") or command.startswith("wc -c"):
                return {"output": "12\n", "returncode": 0}
            if command.startswith("head -c"):
                return {"output": "print('ok')\n", "returncode": 0}
            if command.startswith("sed -n"):
                return {"output": leaked, "returncode": 0}
            if command.startswith("wc -l"):
                return {"output": "1\n", "returncode": 0}
            return {"output": "", "returncode": 0}

        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.read_file("/tmp/test/a.py")

        assert result.error is None
        assert "HERMES_FENCE" not in result.content
        assert "\x1b]" not in result.content
        assert "\x07" not in result.content
        assert "1|print('ok')" in result.content

    def test_read_file_raw_strips_leaked_terminal_fence_markers(self, mock_env):
        leaked = (
            "__HERMES_FENCE_a9f7b3__\x07'\n"
            "alpha\n"
            "\x1b]0;cat '/tmp/test/a.txt'\x07__HERMES_FENCE_a9f7b3__\n"
        )

        def side_effect(command, **kwargs):
            if command.startswith("if [ -f ") or command.startswith("wc -c"):
                return {"output": "6\n", "returncode": 0}
            if command.startswith("head -c"):
                return {"output": "alpha\n", "returncode": 0}
            if command.startswith("cat "):
                return {"output": leaked, "returncode": 0}
            return {"output": "", "returncode": 0}

        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.read_file_raw("/tmp/test/a.txt")

        assert result.error is None
        assert result.content == "alpha\n"


class TestSearchPathValidation:
    """Test that search() returns an error for non-existent paths."""

    def test_search_nonexistent_path_returns_error(self, mock_env):
        """search() should return an error when the path doesn't exist."""
        def side_effect(command, **kwargs):
            if "test -e" in command:
                return {"output": "not_found", "returncode": 1}
            if "command -v" in command:
                return {"output": "yes", "returncode": 0}
            return {"output": "", "returncode": 0}
        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.search("pattern", path="/nonexistent/path")
        assert result.error is not None
        assert "not found" in result.error.lower() or "Path not found" in result.error


    def test_search_rg_error_exit_code(self, mock_env):
        """search() should report error when rg returns exit code 2."""
        call_count = {"n": 0}
        def side_effect(command, **kwargs):
            call_count["n"] += 1
            if "test -e" in command:
                return {"output": "exists", "returncode": 0}
            if "command -v" in command:
                return {"output": "yes", "returncode": 0}
            # rg returns exit 2 (error) with empty output
            return {"output": "", "returncode": 2}
        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.search("pattern", path="/some/path")
        assert result.error is not None
        assert "search failed" in result.error.lower() or "Search error" in result.error


@pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "drives a real POSIX sh script: the fallback shells out to POSIX "
        "`find`, and `_has_command` probes with `command -v`. "
        "make_real_subprocess_env runs it through subprocess shell=True, "
        "which is cmd.exe here, where `find` resolves to the unrelated "
        "string-search find.exe"
    ),
)
class TestSearchFilesFallbackHiddenPaths:
    def _make_env(self):
        return make_real_subprocess_env("/")

    def test_hidden_root_with_hidden_ancestor_includes_files(self, tmp_path, monkeypatch):
        """Fallback find should include visible files when path is inside hidden root."""
        root = tmp_path / ".hermes" / "logs"
        root.mkdir(parents=True)
        visible_file = root / "agent.log"
        hidden_dir_file = root / ".hidden" / "secret.log"
        nested_hidden_file = root / "nested" / ".secret.log"
        visible_nested_file = root / "nested" / "visible.log"

        for p in [visible_file, nested_hidden_file, visible_nested_file, hidden_dir_file]:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x")

        ops = ShellFileOperations(self._make_env())
        monkeypatch.setattr(ops, "_has_command", lambda command: command == "find")
        result = ops._search_files("*.log", str(root), limit=50, offset=0)

        assert result.error is None
        assert set(result.files) == {str(visible_file), str(visible_nested_file)}

    def test_normal_root_still_excludes_hidden_descendants(self, tmp_path, monkeypatch):
        """Fallback find should still exclude hidden descendant paths for normal roots."""
        root = tmp_path / "repo"
        root.mkdir()
        visible_file = root / "agent.log"
        visible_nested_file = root / "nested" / "visible.log"
        hidden_dir_file = root / ".hidden" / "secret.log"

        for p in [visible_file, visible_nested_file, hidden_dir_file]:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x")

        ops = ShellFileOperations(self._make_env())
        monkeypatch.setattr(ops, "_has_command", lambda command: command == "find")
        result = ops._search_files("*.log", str(root), limit=50, offset=0)

        assert result.error is None
        assert set(result.files) == {str(visible_file), str(visible_nested_file)}


class TestShellFileOpsWriteDenied:
    def test_write_file_denied_path(self, file_ops):
        result = file_ops.write_file("~/.ssh/authorized_keys", "evil key")
        assert result.error is not None
        assert "denied" in result.error.lower()


    def test_move_file_failure_path(self, mock_env):
        mock_env.execute.return_value = {"output": "No such file or directory", "returncode": 1}
        ops = ShellFileOperations(mock_env)
        result = ops.move_file("/tmp/nonexistent.txt", "/tmp/dest.txt")
        assert result.error is not None
        assert "Failed to move" in result.error


class TestPatchReplacePostWriteVerification:
    """Tests for the post-write verification added in patch_replace.

    Confirms that a silent persistence failure (where write_file's command
    appears to succeed but the bytes on disk don't match new_content) is
    surfaced as an error instead of being reported as a successful patch.
    """

    def test_patch_replace_fails_when_file_not_persisted(self, mock_env):
        """write_file reports success but the re-read returns old content:
        patch_replace must return an error, not success-with-diff."""
        file_contents = {"/tmp/test/a.py": "hello world\n"}

        def side_effect(command, **kwargs):
            # cat reads the file — both the initial read and the verify read
            if command.startswith("cat "):
                # Extract path from cat command (strip quotes)
                for path in file_contents:
                    if path in command:
                        return {"output": file_contents[path], "returncode": 0}
                return {"output": "", "returncode": 1}
            # mkdir for parent dir
            if command.startswith("mkdir "):
                return {"output": "", "returncode": 0}
            # wc -c for byte count after write
            if command.startswith("if [ -f ") or command.startswith("wc -c"):
                for path in file_contents:
                    if path in command:
                        return {"output": str(len(file_contents[path].encode())), "returncode": 0}
                return {"output": "0", "returncode": 0}
            # Everything else (including the write itself) pretends to succeed
            # but DOESN'T update file_contents — simulates silent failure
            return {"output": "", "returncode": 0}

        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.patch_replace("/tmp/test/a.py", "hello", "hi")
        assert result.error is not None, (
            "Silent persistence failure must surface as error, got: "
            f"success={result.success}, diff={result.diff}"
        )
        assert "verification failed" in result.error.lower()
        assert "did not persist" in result.error.lower()


    def test_patch_replace_fails_when_verify_read_errors(self, mock_env):
        """If the verify-read step itself fails (exit code != 0), return an error."""
        call_count = {"cat": 0}
        state = {"content": "hello world\n"}

        def side_effect(command, stdin_data=None, **kwargs):
            if stdin_data is not None:  # write (atomic temp-file + mv script)
                state["content"] = stdin_data
                return {"output": "", "returncode": 0}
            if command.startswith("cat "):  # read
                call_count["cat"] += 1
                # First read (initial fetch) succeeds; second read (verify) fails
                if call_count["cat"] == 1:
                    return {"output": state["content"], "returncode": 0}
                return {"output": "", "returncode": 1}
            if command.startswith("mkdir "):
                return {"output": "", "returncode": 0}
            if command.startswith("if [ -f ") or command.startswith("wc -c"):
                return {"output": str(len(state["content"].encode())), "returncode": 0}
            return {"output": "", "returncode": 0}

        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.patch_replace("/tmp/test/a.py", "hello", "hi")
        assert result.error is not None
        assert "could not re-read" in result.error.lower()


# =========================================================================
# Git baseline check for write_file warning
# =========================================================================

class _DeletedTestGitBaselineCheck:
    """Removed May 2026 — these tests asserted on a ``_check_git_baseline``
    method that doesn't exist on ``ShellFileOperations`` (regression intro
    by a separate refactor). All 6 tests in the class fail with
    AttributeError on origin/main. Deleted wholesale per Teknium's
    instruction to keep CI green; reinstate them when the underlying
    helper is restored or replaced.
    """
    pass


# =========================================================================
# Atomic write: umask-default permissions for new files
# =========================================================================

@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="POSIX file modes are not enforced on NTFS, and os.umask has no effect there",
)
class TestAtomicWriteNewFilePermissions:
    """_atomic_write should apply umask-default perms to new files (not 0600)."""

    @pytest.mark.parametrize("test_umask", [0o022, 0o002, 0o077])
    def test_new_file_gets_umask_default_permissions(self, tmp_path, test_umask):
        """Newly created file should get umask-computed perms, not mktemp's 0600.

        Uses a real subprocess so the shell script actually runs.
        """
        ops = ShellFileOperations(make_real_subprocess_env(str(tmp_path)))
        dest = tmp_path / "new_file.txt"
        assert not dest.exists()

        old_umask = os.umask(test_umask)
        try:
            result = ops.write_file(str(dest), "test content\n")
        finally:
            os.umask(old_umask)

        assert result.error is None, f"write failed: {result.error}"
        assert dest.read_text() == "test content\n"
        expected_mode = 0o666 & ~test_umask
        actual_mode = dest.stat().st_mode & 0o777
        assert actual_mode == expected_mode, (
            f"Expected mode {expected_mode:04o} (umask {test_umask:04o}), "
            f"got {actual_mode:04o}"
        )

    def test_overwrite_still_preserves_existing_mode(self, tmp_path):
        """The new-file branch must not disturb the overwrite path's
        mode preservation (e.g. an executable script stays 0755)."""
        ops = ShellFileOperations(make_real_subprocess_env(str(tmp_path)))
        dest = tmp_path / "existing.sh"
        dest.write_text("#!/bin/sh\n")
        dest.chmod(0o755)

        result = ops.write_file(str(dest), "#!/bin/sh\necho updated\n")

        assert result.error is None, f"write failed: {result.error}"
        assert dest.read_text() == "#!/bin/sh\necho updated\n"
        assert dest.stat().st_mode & 0o777 == 0o755


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Symlinks require elevated privileges on Windows",
)
class TestAtomicWriteThroughSymlink:
    """_atomic_write must edit a symlink's target, not replace the link.

    Regression: the temp-file + ``mv`` swap replaced the symlink itself with a
    plain file, orphaning the real target and destroying the link (data-loss).
    """

    def test_write_follows_symlink_and_preserves_link(self, tmp_path):
        ops = ShellFileOperations(make_real_subprocess_env(str(tmp_path)))
        real = tmp_path / "real.txt"
        link = tmp_path / "link.txt"
        real.write_text("original\n")
        link.symlink_to(real)

        result = ops.write_file(str(link), "newcontent\n")

        assert result.error is None, f"write failed: {result.error}"
        # The link must survive as a symlink...
        assert link.is_symlink(), "symlink was replaced by a plain file"
        # ...and the real target must carry the new content.
        assert real.read_text() == "newcontent\n"
        assert os.path.realpath(link) == str(real)

    def test_write_through_broken_symlink_falls_back(self, tmp_path):
        """A broken link resolves through readlink -f and creates the target."""
        ops = ShellFileOperations(make_real_subprocess_env(str(tmp_path)))
        target = tmp_path / "target.txt"
        link = tmp_path / "broken.lnk"
        link.symlink_to(target)  # target does not exist yet

        result = ops.write_file(str(link), "data\n")

        assert result.error is None, f"write failed: {result.error}"
        assert target.exists()
        assert target.read_text() == "data\n"


class TestReadNonUtf8IsBinary:
    """Non-UTF-8 content must be flagged binary, not returned as lossy text.

    Regression: the terminal env decodes stdout with errors="replace", turning
    every non-UTF-8 byte into U+FFFD before _is_likely_binary sees it. U+FFFD is
    "printable", so the non-printable ratio never caught it, and a
    read→edit→write round-trip would overwrite the original bytes with mojibake.
    """

    def test_replacement_char_sample_flagged_binary(self, tmp_path):
        ops = ShellFileOperations(make_real_subprocess_env(str(tmp_path)))
        # A latin-1 file decoded with errors="replace" yields U+FFFD chars.
        lossy_sample = "caf\ufffd r\ufffdsum\ufffd\n"
        assert ops._is_likely_binary("notes.txt", lossy_sample) is True

    def test_plain_utf8_text_not_flagged(self, tmp_path):
        ops = ShellFileOperations(make_real_subprocess_env(str(tmp_path)))
        # Proper UTF-8 (including non-ASCII) must still read as text.
        assert ops._is_likely_binary("notes.txt", "café résumé\nsecond\n") is False

# =========================================================================
# Byte-layer binary detection (#80308 class: CJK/multibyte text flagged
# binary because the byte-boundary sample manufactured U+FFFD in transit)
# =========================================================================

class TestByteLayerBinaryDetection:
    """Regression suite for the misclassification class behind #80308.

    Fragment reports/fixes each caught one member: #80261, #80250, #80188,
    #80349, #79834, #79534, #79408. The boundary contract: text = valid
    UTF-8 allowing one incomplete multibyte sequence at the sample's end;
    NUL or mid-stream invalid UTF-8 = read-only.
    """

    # --- unit: _is_likely_binary_bytes -----------------------------------

    def test_cjk_text_cut_mid_character_is_text(self, file_ops):
        # 999 ASCII bytes + a 3-byte CJK char cut after its first byte —
        # exactly what `head -c 1000` does to a CJK file.
        sample = (b"a" * 999 + "中".encode("utf-8"))[:1000]
        assert sample[-1:] != b"a"  # the cut really is mid-character
        assert file_ops._is_likely_binary_bytes(sample) is False

    def test_pure_cjk_text_cut_mid_character_is_text(self, file_ops):
        sample = ("汉字" * 400).encode("utf-8")[:1000]
        assert file_ops._is_likely_binary_bytes(sample) is False

    def test_emoji_cut_at_boundary_is_text(self, file_ops):
        # 4-byte sequence cut after 2 bytes.
        sample = (b"x" * 998 + "🎉".encode("utf-8"))[:1000]
        assert file_ops._is_likely_binary_bytes(sample) is False

    def test_utf8_bom_is_text(self, file_ops):
        assert file_ops._is_likely_binary_bytes(b"\xef\xbb\xbfhello") is False

    def test_file_containing_real_replacement_char_is_text(self, file_ops):
        # A log file that legitimately stores U+FFFD is valid UTF-8. The old
        # text-layer check could not tell it from transport damage.
        assert file_ops._is_likely_binary_bytes("log: \ufffd bad byte\n".encode("utf-8")) is False

    def test_nul_byte_is_binary(self, file_ops):
        assert file_ops._is_likely_binary_bytes(b"MZ\x00\x01text") is True

    def test_elf_header_is_binary(self, file_ops):
        assert file_ops._is_likely_binary_bytes(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 8) is True

    def test_latin1_text_stays_read_only(self, file_ops):
        # Mid-stream invalid UTF-8 (0xE9 = latin-1 é). Reading it through the
        # replace-decoding transport would mojibake a read→edit→write
        # round-trip, so it must stay flagged (the old check's guarantee).
        assert file_ops._is_likely_binary_bytes(b"caf\xe9 au lait, plus padding") is True

    def test_empty_sample_is_text(self, file_ops):
        assert file_ops._is_likely_binary_bytes(b"") is False

    def test_short_ascii_is_text(self, file_ops):
        assert file_ops._is_likely_binary_bytes(b"hello\n") is False

    def test_truncated_garbage_tail_after_invalid_prefix_is_binary(self, file_ops):
        # Error near the end but the prefix itself is not clean UTF-8.
        assert file_ops._is_likely_binary_bytes(b"\xff\xfe" + b"a" * 10 + b"\xe4") is True

    # --- transport: _sample_file_bytes ------------------------------------

    def test_sample_decodes_base64_transport(self, mock_env):
        import base64 as b64
        payload = ("汉字" * 400).encode("utf-8")[:1000]
        mock_env.execute.return_value = {
            "output": b64.b64encode(payload).decode() + "\n",
            "returncode": 0,
        }
        ops = ShellFileOperations(mock_env)
        assert ops._sample_file_bytes("/tmp/x.txt") == payload

    def test_sample_falls_back_on_non_base64_output(self, mock_env):
        mock_env.execute.return_value = {"output": "not base64 at all!!", "returncode": 0}
        ops = ShellFileOperations(mock_env)
        assert ops._sample_file_bytes("/tmp/x.txt") is None

    def test_sample_falls_back_on_nonzero_exit(self, mock_env):
        mock_env.execute.return_value = {"output": "", "returncode": 127}
        ops = ShellFileOperations(mock_env)
        assert ops._sample_file_bytes("/tmp/x.txt") is None

    # --- integration: read_file over the mocked terminal ------------------

    def _dispatch(self, cjk_bytes):
        import base64 as b64

        def side_effect(command, **kwargs):
            if command.startswith("if [ -f ") or command.startswith("wc -c"):
                return {"output": f"{len(cjk_bytes)}\n", "returncode": 0}
            if command.startswith("head -c") and "| base64" in command:
                return {"output": b64.b64encode(cjk_bytes[:1000]).decode(), "returncode": 0}
            if command.startswith("sed -n"):
                return {"output": cjk_bytes.decode("utf-8", errors="replace"), "returncode": 0}
            if command.startswith("wc -l"):
                return {"output": "1\n", "returncode": 0}
            return {"output": "", "returncode": 0}

        return side_effect

    def test_read_file_returns_cjk_content_instead_of_binary_error(self, mock_env):
        content = ("汉字测试" * 300).encode("utf-8")  # > 1000 bytes, cut mid-char
        mock_env.execute.side_effect = self._dispatch(content)
        ops = ShellFileOperations(mock_env)
        result = ops.read_file("/tmp/notes-中文.txt")
        assert result.is_binary is False
        assert result.error is None
        assert "汉字测试" in (result.content or "")

    def test_read_file_still_blocks_nul_binaries(self, mock_env):
        content = b"\x7fELF\x00\x00binarybinary" + b"\x00" * 100
        mock_env.execute.side_effect = self._dispatch(content)
        ops = ShellFileOperations(mock_env)
        result = ops.read_file("/tmp/a.out")
        assert result.is_binary is True



class TestEscapeNativeToolArg:
    """Regression tests for _escape_native_tool_arg (Windows native-binary paths).

    Live failure (Windows, Aug 2026): search_files passed rg the MSYS form
    (/c/Users/...) that _escape_shell_arg produces, but Hermes sets
    MSYS_NO_PATHCONV=1 / MSYS2_ARG_CONV_EXCL=* for its bash subprocesses,
    so nothing converted the path back for the native (winget) ripgrep
    binary — every search on a drive-letter path failed with
    "The system cannot find the path specified. (os error 3)". Native
    Windows binaries need C:/... (forward-slash native), which bash also
    passes through untouched.
    """

    def _ops(self, mock_env):
        return ShellFileOperations(mock_env)

    def test_windows_native_path_kept_native(self, mock_env, monkeypatch):
        import tools.environments.local as local_mod
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        ops = self._ops(mock_env)
        out = ops._escape_native_tool_arg(r"C:\Users\alice\project")
        assert out == "'C:/Users/alice/project'"

    def test_msys_path_translated_back_to_native(self, mock_env, monkeypatch):
        import tools.environments.local as local_mod
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        ops = self._ops(mock_env)
        out = ops._escape_native_tool_arg("/c/Users/alice/project")
        assert out == "'C:/Users/alice/project'"

    def test_posix_path_untouched_on_windows(self, mock_env, monkeypatch):
        """Multi-segment POSIX paths (/home/x, /tmp/y) are not drive paths."""
        import tools.environments.local as local_mod
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        ops = self._ops(mock_env)
        assert ops._escape_native_tool_arg("/tmp/workdir") == "'/tmp/workdir'"

    def test_non_windows_behaves_like_escape_shell_arg(self, mock_env, monkeypatch):
        import tools.environments.local as local_mod
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", False)
        ops = self._ops(mock_env)
        assert ops._escape_native_tool_arg("/home/u/it's here") == (
            ops._escape_shell_arg("/home/u/it's here")
        )

    def test_rg_content_search_uses_native_form(self, mock_env, monkeypatch):
        """_search_with_rg must pass the path in native C:/ form to rg."""
        import tools.environments.local as local_mod
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        commands = []

        def side_effect(command, **kwargs):
            commands.append(command)
            if "test -e" in command:
                return {"output": "exists", "returncode": 0}
            if "command -v" in command:
                return {"output": "yes", "returncode": 0}
            return {"output": "", "returncode": 0}

        mock_env.execute.side_effect = side_effect
        ops = self._ops(mock_env)
        ops.search("needle", path=r"C:\Users\alice\project")
        rg_cmds = [c for c in commands if "rg " in c or c.startswith("rg")]
        assert rg_cmds, f"no rg command captured in: {commands}"
        assert any("'C:/Users/alice/project'" in c for c in rg_cmds), rg_cmds
        assert all("/c/Users" not in c for c in rg_cmds), rg_cmds

    def test_shell_linter_uses_native_form(self, mock_env, monkeypatch):
        """_check_lint must hand node/python/etc. the native C:/ path.

        Regression for the double-prefix failure (#84303): node given the
        MSYS /c/Users/... form resolves it as C:\\c\\Users\\... and every
        .js write reports a phantom ENOENT lint error.
        """
        import tools.environments.local as local_mod
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        commands = []

        def side_effect(command, **kwargs):
            commands.append(command)
            if "command -v" in command:
                return {"output": "yes", "returncode": 0}
            return {"output": "", "returncode": 0}

        mock_env.execute.side_effect = side_effect
        ops = self._ops(mock_env)
        result = ops._check_lint(r"C:\Users\alice\app\main.js")
        assert result.skipped is False
        node_cmds = [c for c in commands if "node --check" in c]
        assert node_cmds, f"no node command captured in: {commands}"
        assert "'C:/Users/alice/app/main.js'" in node_cmds[0]
        assert "/c/Users" not in node_cmds[0]
