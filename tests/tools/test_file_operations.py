"""Tests for tools/file_operations.py — deny list, result dataclasses, helpers."""

import os
import re
import pytest
import subprocess
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


    def test_escape_shell_arg_rewrites_forward_slash_native_paths(self, monkeypatch, file_ops):
        import tools.environments.local as local_mod

        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        assert file_ops._escape_shell_arg(
            "C:/Users/alice/notes.txt"
        ) == "'/c/Users/alice/notes.txt'"

    def test_read_file_uses_bash_safe_windows_paths(self, mock_env, monkeypatch):
        import tools.environments.local as local_mod

        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        commands = []

        def side_effect(command, **kwargs):
            commands.append(command)
            if command.startswith("wc -c"):
                return {"output": "5\n", "returncode": 0}
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
        assert commands[0] == "wc -c < '/c/Users/alice/notes.txt' 2>/dev/null"
        assert commands[1] == "head -c 1000 '/c/Users/alice/notes.txt' 2>/dev/null"
        assert commands[2] == "sed -n '1,2000p' '/c/Users/alice/notes.txt'"
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
            if command.startswith("wc -c"):
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
            if command.startswith("wc -c"):
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
            if command.startswith("wc -c"):
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
            if command.startswith("wc -c"):
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


class TestPatchRejectsRedactedPlaceholders:
    """Regression tests for #30962 — refuse patch input that looks like it
    was copy-pasted from redacted ``read_file`` output.

    The patch tool must not try to match (or write back) masked secret
    placeholders like ``sk-exa...hars``: matching fails silently against
    the raw file, and writing would corrupt the real secret.
    """

    def test_patch_replace_rejects_redacted_old_string(self, file_ops):
        result = file_ops.patch_replace(
            "/tmp/cfg.yaml",
            old_string="api_key: sk-exa...hars\nmodel: title-generator",
            new_string="api_key: sk-exa...hars\nmodel: new-model",
        )
        assert result.error is not None
        assert "redacted" in result.error.lower()
        assert "old_string" in result.error

    def test_patch_replace_rejects_redacted_new_string(self, file_ops):
        result = file_ops.patch_replace(
            "/tmp/cfg.yaml",
            old_string="model: title-generator",
            new_string="model: title-generator\napi_key: ghp_ab...wxyz",
        )
        assert result.error is not None
        assert "redacted" in result.error.lower()
        assert "new_string" in result.error

    def test_patch_replace_rejects_redacted_private_key_marker(self, file_ops):
        result = file_ops.patch_replace(
            "/tmp/secrets.yaml",
            old_string="ssh_key: [REDACTED PRIVATE KEY]",
            new_string="ssh_key: replacement",
        )
        assert result.error is not None
        assert "redacted" in result.error.lower()

    def test_patch_replace_allows_normal_ellipsis_in_code(self, mock_env):
        """``...`` in real code (Python Ellipsis, docstrings) must not trip
        the check — only known-prefix + ellipsis patterns are rejected."""
        state = {"content": "def stub(): ...\n"}

        def side_effect(command, stdin_data=None, **kwargs):
            # Writes stream the new content over stdin. The legacy path was a
            # bare ``cat > file``; the atomic path (#35252) is a ``set -e; …;
            # cat > "$tmp"; mv -f "$tmp" "$t"`` script. Capture either by
            # keying on stdin_data, which only the content-write carries.
            if stdin_data is not None and "cat >" in command:
                state["content"] = stdin_data
                return {"output": "", "returncode": 0}
            if command.startswith("cat "):
                return {"output": state["content"], "returncode": 0}
            if command.startswith("mkdir "):
                return {"output": "", "returncode": 0}
            if command.startswith("wc -c"):
                return {"output": str(len(state["content"].encode())), "returncode": 0}
            return {"output": "", "returncode": 0}

        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.patch_replace(
            "/tmp/test/stub.py",
            old_string="def stub(): ...",
            new_string="def stub(): return None",
        )
        assert result.error is None, f"Unexpected rejection: {result.error}"
        assert result.success is True

    def test_patch_v4a_rejects_redacted_payload(self, file_ops):
        payload = (
            "*** Begin Patch\n"
            "*** Update File: /tmp/cfg.yaml\n"
            "@@ auxiliary @@\n"
            " title_generation:\n"
            "    api_key: sk-exa...hars\n"
            "-    model: title-generator\n"
            "+    model: new-model\n"
            "*** End Patch\n"
        )
        result = file_ops.patch_v4a(payload)
        assert result.error is not None
        assert "redacted" in result.error.lower()
        assert "V4A" in result.error


class TestWriteFileRejectsRedactedPlaceholders:
    """Regression tests for #30962 — `write_file` is the more obvious footgun
    than `patch_replace`: an agent that reads a redacted file, edits in-memory,
    and writes the whole thing back overwrites every masked credential on
    disk with its placeholder."""

    def test_write_file_rejects_prefix_masked_content(self, file_ops):
        result = file_ops.write_file(
            "/tmp/cfg.yaml",
            "auxiliary:\n  api_key: sk-exa...hars\n  model: title-generator\n",
        )
        assert result.error is not None
        assert "redacted" in result.error.lower()

    def test_write_file_rejects_opaque_masked_content(self, file_ops):
        """Opaque secret (no vendor prefix) behind a sensitive key — the
        original detector missed this; v2 catches it."""
        result = file_ops.write_file(
            "/tmp/cfg.yaml",
            "password: abc123...wxyz\nuser: alice\n",
        )
        assert result.error is not None
        assert "redacted" in result.error.lower()

    def test_write_file_rejects_triple_star_in_db_connstring(self, file_ops):
        result = file_ops.write_file(
            "/tmp/.env",
            "DATABASE_URL=postgres://user:***@host/db\nLOG_LEVEL=info\n",
        )
        assert result.error is not None
        assert "redacted" in result.error.lower()

    def test_write_file_allows_env_example_placeholders(self, mock_env):
        """`OPENAI_API_KEY=***` is how real `.env.example` files document
        the placeholder shape; we deliberately do NOT flag it (the
        corruption window for sub-floor-length redacted values is narrow,
        and refusing to edit template files would be worse)."""
        state = {"content": ""}

        def side_effect(command, stdin_data=None, **kwargs):
            # Writes stream the new content over stdin. The legacy path was a
            # bare ``cat > file``; the atomic path (#35252) is a ``set -e; …;
            # cat > "$tmp"; mv -f "$tmp" "$t"`` script. Capture either by
            # keying on stdin_data, which only the content-write carries.
            if stdin_data is not None and "cat >" in command:
                state["content"] = stdin_data
                return {"output": "", "returncode": 0}
            if command.startswith("cat "):
                return {"output": state["content"], "returncode": 0}
            if command.startswith("mkdir "):
                return {"output": "", "returncode": 0}
            if command.startswith("wc -c"):
                return {"output": str(len(state["content"].encode())), "returncode": 0}
            return {"output": "", "returncode": 0}

        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.write_file(
            "/tmp/test/.env.example",
            "OPENAI_API_KEY=***\nLOG_LEVEL=info\n",
        )
        assert result.error is None, f"Template placeholder rejected: {result.error}"

    def test_write_file_error_message_says_restart_required(self, file_ops):
        """BLOCKER 3: error message must not imply an in-session config flip
        will fix things — the redaction flag is read at process start."""
        result = file_ops.write_file(
            "/tmp/cfg.yaml",
            "api_key: sk-exa...hars\n",
        )
        assert result.error is not None
        assert "restart" in result.error.lower()

    def test_write_file_allows_normal_content(self, mock_env):
        """Sanity check: clean content still passes through."""
        state = {"content": ""}

        def side_effect(command, stdin_data=None, **kwargs):
            # Writes stream the new content over stdin. The legacy path was a
            # bare ``cat > file``; the atomic path (#35252) is a ``set -e; …;
            # cat > "$tmp"; mv -f "$tmp" "$t"`` script. Capture either by
            # keying on stdin_data, which only the content-write carries.
            if stdin_data is not None and "cat >" in command:
                state["content"] = stdin_data
                return {"output": "", "returncode": 0}
            if command.startswith("cat "):
                return {"output": state["content"], "returncode": 0}
            if command.startswith("mkdir "):
                return {"output": "", "returncode": 0}
            if command.startswith("wc -c"):
                return {"output": str(len(state["content"].encode())), "returncode": 0}
            return {"output": "", "returncode": 0}

        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.write_file("/tmp/test/clean.py", "def foo():\n    return 42\n")
        assert result.error is None, f"Unexpected rejection: {result.error}"


class TestPatchRejectsRedactedPlaceholdersV2:
    """#30962 round 2 — coverage for the gaps the first round of integration
    tests missed: V4A `Add File` (no `old_string` matching → corruption risk
    is highest), `replace_all=True`, and the opaque / `***` placeholder
    shapes added to the detector."""

    def test_patch_v4a_rejects_add_file_with_placeholder(self, file_ops):
        """`*** Add File` blocks have no context-matching step — if the
        placeholder slipped through it would land directly on disk."""
        payload = (
            "*** Begin Patch\n"
            "*** Add File: /tmp/new-cfg.yaml\n"
            "+auxiliary:\n"
            "+  api_key: sk-exa...hars\n"
            "+  model: title-generator\n"
            "*** End Patch\n"
        )
        result = file_ops.patch_v4a(payload)
        assert result.error is not None
        assert "redacted" in result.error.lower()

    def test_patch_v4a_rejects_multi_file_payload_with_placeholder(self, file_ops):
        """Multi-file V4A: even if only one of several files carries a
        placeholder, the whole payload must be refused."""
        payload = (
            "*** Begin Patch\n"
            "*** Update File: /tmp/clean.yaml\n"
            "@@ section @@\n"
            "-old: 1\n"
            "+new: 2\n"
            "*** Update File: /tmp/dirty.yaml\n"
            "@@ section @@\n"
            " api_key: sk-exa...hars\n"
            "-other: a\n"
            "+other: b\n"
            "*** End Patch\n"
        )
        result = file_ops.patch_v4a(payload)
        assert result.error is not None
        assert "redacted" in result.error.lower()

    def test_patch_replace_with_replace_all_still_rejects(self, file_ops):
        """`replace_all=True` must not bypass the guard."""
        result = file_ops.patch_replace(
            "/tmp/cfg.yaml",
            old_string="api_key: sk-exa...hars",
            new_string="api_key: new-value",
            replace_all=True,
        )
        assert result.error is not None
        assert "redacted" in result.error.lower()

    def test_patch_replace_rejects_opaque_masked(self, file_ops):
        """Opaque token (no vendor prefix) detected via sensitive-key context."""
        result = file_ops.patch_replace(
            "/tmp/cfg.yaml",
            old_string="password: abc123...wxyz",
            new_string="password: newvalue",
        )
        assert result.error is not None
        assert "redacted" in result.error.lower()

    def test_patch_replace_rejects_triple_star_in_new_string(self, file_ops):
        """The DB-connstring `***` round-trip — agent reads a redacted env,
        constructs new_string to update an adjacent var, accidentally leaves
        the `***`-masked URL in. Without this guard, the real password gets
        overwritten with `***` on disk."""
        result = file_ops.patch_replace(
            "/tmp/.env",
            old_string="LOG_LEVEL=info",
            new_string=(
                "DATABASE_URL=postgres://user:***@host/db\nLOG_LEVEL=debug"
            ),
        )
        assert result.error is not None
        assert "redacted" in result.error.lower()

    def test_error_message_says_restart_required(self, file_ops):
        """BLOCKER 3: error must surface that the redaction flag is
        process-start, not in-session toggleable."""
        result = file_ops.patch_replace(
            "/tmp/cfg.yaml",
            old_string="api_key: sk-exa...hars",
            new_string="api_key: x",
        )
        assert result.error is not None
        assert "restart" in result.error.lower()


class TestRedactedGuardFileReadEndToEnd:
    """#30962 round 3 — exercise the PRODUCTION read→write flow.

    ``read_file_tool`` / ``search_tool`` redact file content with
    ``redact_sensitive_text(..., file_read=True)`` (see tools/file_tools.py),
    which since #35519 emits the non-reusable sentinel ``«redacted:sk-…»``
    for prefix credentials instead of the head/tail ASCII mask. These tests
    feed that *actual* redacted output — not a hand-written fixture — into
    ``write_file``, replace-mode patching, and V4A patching, so they fail
    loudly if the redactor's file-read format and the guard ever drift
    apart again.
    """

    RAW_CONFIG = (
        "auxiliary:\n"
        "  api_key: sk-proj-abcdefghijklmnopqrstuvwxyz123456\n"
        "  model: title-generator\n"
    )

    @staticmethod
    def _redacted_config():
        from agent.redact import redact_sensitive_text
        redacted = redact_sensitive_text(
            TestRedactedGuardFileReadEndToEnd.RAW_CONFIG,
            force=True,
            file_read=True,
        )
        # Contract with the redactor: the secret is gone and the file-read
        # sentinel is present. If this assertion fails, the redactor's
        # file-read format changed and the guard needs re-alignment.
        assert "sk-proj-abcdefghijklmnopqrstuvwxyz123456" not in redacted
        assert "«redacted:" in redacted
        return redacted

    def test_write_file_rejects_actual_file_read_output(self, file_ops):
        """Read redacted config → write it back verbatim (the #35519
        corruption flow) → must be refused."""
        result = file_ops.write_file("/tmp/cfg.yaml", self._redacted_config())
        assert result.error is not None
        assert "redacted" in result.error.lower()

    def test_patch_replace_rejects_actual_file_read_output(self, file_ops):
        """old_string copied from redacted read_file output — would never
        match the raw bytes on disk; must be refused, not silently miss."""
        redacted = self._redacted_config()
        key_line = next(l for l in redacted.splitlines() if "api_key" in l)
        result = file_ops.patch_replace(
            "/tmp/cfg.yaml",
            old_string=key_line,
            new_string="  api_key: replacement",
        )
        assert result.error is not None
        assert "redacted" in result.error.lower()
        assert "old_string" in result.error

    def test_patch_replace_rejects_sentinel_in_new_string(self, file_ops):
        """new_string carrying the sentinel — writing it would corrupt the
        stored credential into the literal marker."""
        redacted = self._redacted_config()
        key_line = next(l for l in redacted.splitlines() if "api_key" in l)
        result = file_ops.patch_replace(
            "/tmp/cfg.yaml",
            old_string="model: title-generator",
            new_string=f"model: new-model\n{key_line}",
        )
        assert result.error is not None
        assert "redacted" in result.error.lower()
        assert "new_string" in result.error

    def test_patch_v4a_rejects_actual_file_read_output(self, file_ops):
        redacted = self._redacted_config()
        key_line = next(l for l in redacted.splitlines() if "api_key" in l)
        payload = (
            "*** Begin Patch\n"
            "*** Update File: /tmp/cfg.yaml\n"
            "@@ auxiliary @@\n"
            f" {key_line}\n"
            "-  model: title-generator\n"
            "+  model: new-model\n"
            "*** End Patch\n"
        )
        result = file_ops.patch_v4a(payload)
        assert result.error is not None
        assert "redacted" in result.error.lower()

    def test_patch_v4a_rejects_add_file_with_sentinel(self, file_ops):
        """`*** Add File` has no context-match step — a sentinel here lands
        directly on disk if the guard misses it."""
        redacted = self._redacted_config()
        body = "\n".join(f"+{l}" for l in redacted.splitlines())
        payload = (
            "*** Begin Patch\n"
            "*** Add File: /tmp/new-cfg.yaml\n"
            f"{body}\n"
            "*** End Patch\n"
        )
        result = file_ops.patch_v4a(payload)
        assert result.error is not None
        assert "redacted" in result.error.lower()
# =========================================================================
# Atomic write: umask-default permissions for new files
# =========================================================================

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
