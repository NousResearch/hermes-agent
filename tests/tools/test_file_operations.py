"""Tests for tools/file_operations.py — deny list, result dataclasses, helpers."""

import os
import pytest
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch
import subprocess

from tools.file_operations import (
    _is_write_denied,
    WRITE_DENIED_PATHS,
    WRITE_DENIED_PREFIXES,
    ReadResult,
    WriteResult,
    PatchResult,
    SearchResult,
    SearchMatch,
    LintResult,
    ShellFileOperations,
    BINARY_EXTENSIONS,
    IMAGE_EXTENSIONS,
    MAX_LINE_LENGTH,
    normalize_read_pagination,
    normalize_search_pagination,
)


class LocalShellEnv:
    """Tiny local terminal-env shim for ShellFileOperations integration tests."""

    def __init__(self, cwd: str):
        self.cwd = cwd

    def execute(self, command, cwd=None, timeout=None, stdin_data=None):
        completed = subprocess.run(
            command,
            shell=True,
            cwd=cwd or self.cwd,
            input=stdin_data,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "output": completed.stdout + completed.stderr,
            "returncode": completed.returncode,
        }


# =========================================================================
# Write deny list
# =========================================================================

class TestIsWriteDenied:
    def test_ssh_authorized_keys_denied(self):
        path = os.path.join(str(Path.home()), ".ssh", "authorized_keys")
        assert _is_write_denied(path) is True

    def test_ssh_id_rsa_denied(self):
        path = os.path.join(str(Path.home()), ".ssh", "id_rsa")
        assert _is_write_denied(path) is True

    def test_netrc_denied(self):
        path = os.path.join(str(Path.home()), ".netrc")
        assert _is_write_denied(path) is True

    def test_aws_prefix_denied(self):
        path = os.path.join(str(Path.home()), ".aws", "credentials")
        assert _is_write_denied(path) is True

    def test_kube_prefix_denied(self):
        path = os.path.join(str(Path.home()), ".kube", "config")
        assert _is_write_denied(path) is True

    def test_normal_file_allowed(self, tmp_path):
        path = str(tmp_path / "safe_file.txt")
        assert _is_write_denied(path) is False

    def test_project_file_allowed(self):
        assert _is_write_denied("/tmp/project/main.py") is False

    def test_tilde_expansion(self):
        assert _is_write_denied("~/.ssh/authorized_keys") is True



# =========================================================================
# Result dataclasses
# =========================================================================

class TestReadResult:
    def test_to_dict_omits_defaults(self):
        r = ReadResult()
        d = r.to_dict()
        assert "error" not in d    # None omitted
        assert "similar_files" not in d  # empty list omitted

    def test_to_dict_preserves_empty_content(self):
        """Empty file should still have content key in the dict."""
        r = ReadResult(content="", total_lines=0, file_size=0)
        d = r.to_dict()
        assert "content" in d
        assert d["content"] == ""
        assert d["total_lines"] == 0
        assert d["file_size"] == 0

    def test_to_dict_includes_values(self):
        r = ReadResult(content="hello", total_lines=10, file_size=50, truncated=True)
        d = r.to_dict()
        assert d["content"] == "hello"
        assert d["total_lines"] == 10
        assert d["truncated"] is True

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

    def test_to_dict_empty(self):
        r = SearchResult()
        d = r.to_dict()
        assert d["total_count"] == 0
        assert "matches" not in d

    def test_to_dict_files_mode(self):
        r = SearchResult(files=["a.py", "b.py"], total_count=2)
        d = r.to_dict()
        assert d["files"] == ["a.py", "b.py"]

    def test_to_dict_count_mode(self):
        r = SearchResult(counts={"a.py": 3, "b.py": 1}, total_count=4)
        d = r.to_dict()
        assert d["counts"]["a.py"] == 3

    def test_truncated_flag(self):
        r = SearchResult(total_count=100, truncated=True)
        d = r.to_dict()
        assert d["truncated"] is True


class TestLintResult:
    def test_skipped(self):
        r = LintResult(skipped=True, message="No linter for .md files")
        d = r.to_dict()
        assert d["status"] == "skipped"
        assert d["message"] == "No linter for .md files"

    def test_success(self):
        r = LintResult(success=True, output="")
        d = r.to_dict()
        assert d["status"] == "ok"

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


class TestShellFileOpsHelpers:
    def test_normalize_read_pagination_clamps_invalid_values(self):
        assert normalize_read_pagination(offset=0, limit=0) == (1, 1)
        assert normalize_read_pagination(offset=-10, limit=-5) == (1, 1)
        assert normalize_read_pagination(offset="bad", limit="bad") == (1, 500)
        assert normalize_read_pagination(offset=2, limit=999999) == (2, 2000)

    def test_normalize_search_pagination_clamps_invalid_values(self):
        assert normalize_search_pagination(offset=-10, limit=-5) == (0, 1)
        assert normalize_search_pagination(offset="bad", limit="bad") == (0, 50)
        assert normalize_search_pagination(offset=3, limit=0) == (3, 1)

    def test_escape_shell_arg_simple(self, file_ops):
        assert file_ops._escape_shell_arg("hello") == "'hello'"

    def test_escape_shell_arg_with_quotes(self, file_ops):
        result = file_ops._escape_shell_arg("it's")
        assert "'" in result
        # Should be safely escaped
        assert result.count("'") >= 4  # wrapping + escaping

    def test_is_likely_binary_by_extension(self, file_ops):
        assert file_ops._is_likely_binary("photo.png") is True
        assert file_ops._is_likely_binary("data.db") is True
        assert file_ops._is_likely_binary("code.py") is False
        assert file_ops._is_likely_binary("readme.md") is False

    def test_is_likely_binary_by_content(self, file_ops):
        # High ratio of non-printable chars -> binary
        binary_content = "\x00\x01\x02\x03" * 250
        assert file_ops._is_likely_binary("unknown", binary_content) is True

        # Normal text -> not binary
        assert file_ops._is_likely_binary("unknown", "Hello world\nLine 2\n") is False

    def test_is_image(self, file_ops):
        assert file_ops._is_image("photo.png") is True
        assert file_ops._is_image("pic.jpg") is True
        assert file_ops._is_image("icon.ico") is True
        assert file_ops._is_image("data.pdf") is False
        assert file_ops._is_image("code.py") is False

    def test_add_line_numbers(self, file_ops):
        content = "line one\nline two\nline three"
        result = file_ops._add_line_numbers(content)
        assert "     1|line one" in result
        assert "     2|line two" in result
        assert "     3|line three" in result

    def test_add_line_numbers_with_offset(self, file_ops):
        content = "continued\nmore"
        result = file_ops._add_line_numbers(content, start_line=50)
        assert "    50|continued" in result
        assert "    51|more" in result

    def test_add_line_numbers_truncates_long_lines(self, file_ops):
        long_line = "x" * (MAX_LINE_LENGTH + 100)
        result = file_ops._add_line_numbers(long_line)
        assert "[truncated]" in result

    def test_unified_diff(self, file_ops):
        old = "line1\nline2\nline3\n"
        new = "line1\nchanged\nline3\n"
        diff = file_ops._unified_diff(old, new, "test.py")
        assert "-line2" in diff
        assert "+changed" in diff
        assert "test.py" in diff

    def test_cwd_from_env(self, mock_env):
        mock_env.cwd = "/custom/path"
        ops = ShellFileOperations(mock_env)
        assert ops.cwd == "/custom/path"

    def test_cwd_fallback_to_slash(self):
        env = MagicMock(spec=[])  # no cwd attribute
        ops = ShellFileOperations(env)
        assert ops.cwd == "/"


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

    def test_search_nonexistent_path_files_mode(self, mock_env):
        """search(target='files') should also return error for bad paths."""
        def side_effect(command, **kwargs):
            if "test -e" in command:
                return {"output": "not_found", "returncode": 1}
            if "command -v" in command:
                return {"output": "yes", "returncode": 0}
            return {"output": "", "returncode": 0}
        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.search("*.py", path="/nonexistent/path", target="files")
        assert result.error is not None
        assert "not found" in result.error.lower() or "Path not found" in result.error

    def test_search_existing_path_proceeds(self, mock_env):
        """search() should proceed normally when the path exists."""
        def side_effect(command, **kwargs):
            if "test -e" in command:
                return {"output": "exists", "returncode": 0}
            if "command -v" in command:
                return {"output": "yes", "returncode": 0}
            # rg returns exit 1 (no matches) with empty output
            return {"output": "", "returncode": 1}
        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.search("pattern", path="/existing/path")
        assert result.error is None
        assert result.total_count == 0  # No matches but no error

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


class TestShellFileOpsWriteDenied:
    def test_write_file_denied_path(self, file_ops):
        result = file_ops.write_file("~/.ssh/authorized_keys", "evil key")
        assert result.error is not None
        assert "denied" in result.error.lower()

    def test_patch_replace_denied_path(self, file_ops):
        result = file_ops.patch_replace("~/.ssh/authorized_keys", "old", "new")
        assert result.error is not None
        assert "denied" in result.error.lower()

    def test_delete_file_denied_path(self, file_ops):
        result = file_ops.delete_file("~/.ssh/authorized_keys")
        assert result.error is not None
        assert "denied" in result.error.lower()

    def test_move_file_src_denied(self, file_ops):
        result = file_ops.move_file("~/.ssh/id_rsa", "/tmp/dest.txt")
        assert result.error is not None
        assert "denied" in result.error.lower()

    def test_move_file_dst_denied(self, file_ops):
        result = file_ops.move_file("/tmp/src.txt", "~/.aws/credentials")
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

    def test_patch_replace_succeeds_when_file_persisted(self, mock_env):
        """Normal success path: write persists, verify read returns new bytes."""
        state = {"content": "hello world\n"}

        def side_effect(command, stdin_data=None, **kwargs):
            # Write is `cat > path` — detect by the `>` redirect, NOT just `cat `
            if command.startswith("cat >"):
                if stdin_data is not None:
                    state["content"] = stdin_data
                return {"output": "", "returncode": 0}
            if command.startswith("cat "):  # read
                return {"output": state["content"], "returncode": 0}
            if command.startswith("mkdir "):
                return {"output": "", "returncode": 0}
            if command.startswith("wc -c"):
                return {"output": str(len(state["content"].encode())), "returncode": 0}
            return {"output": "", "returncode": 0}

        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.patch_replace("/tmp/test/a.py", "hello", "hi")
        assert result.error is None, f"Unexpected error: {result.error}"
        assert result.success is True
        assert state["content"] == "hi world\n", f"File not actually updated: {state['content']!r}"

    def test_patch_replace_fails_when_verify_read_errors(self, mock_env):
        """If the verify-read step itself fails (exit code != 0), return an error."""
        call_count = {"cat": 0}
        state = {"content": "hello world\n"}

        def side_effect(command, stdin_data=None, **kwargs):
            if command.startswith("cat >"):  # write
                if stdin_data is not None:
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


class TestHashlinePatchEditing:
    def test_patch_replace_accepts_valid_hashline_old_string(self, tmp_path):
        file_path = tmp_path / "sample.txt"
        file_path.write_text("alpha\nbeta\n", encoding="utf-8")
        ops = ShellFileOperations(LocalShellEnv(str(tmp_path)))

        with patch.dict(os.environ, {"HERMES_HASHLINE_EDIT": "1"}, clear=False):
            anchored_old = ops.read_file(str(file_path), offset=1, limit=1).content
            result = ops.patch_replace(str(file_path), anchored_old, "ALPHA")

        assert result.success is True
        assert result.error is None
        assert file_path.read_text(encoding="utf-8") == "ALPHA\nbeta\n"

    def test_patch_replace_rejects_stale_hashline_old_string(self, tmp_path):
        file_path = tmp_path / "sample.txt"
        file_path.write_text("alpha\nbeta\n", encoding="utf-8")
        ops = ShellFileOperations(LocalShellEnv(str(tmp_path)))

        with patch.dict(os.environ, {"HERMES_HASHLINE_EDIT": "1"}, clear=False):
            anchored_old = ops.read_file(str(file_path), offset=1, limit=1).content
            file_path.write_text("omega\nbeta\n", encoding="utf-8")
            result = ops.patch_replace(str(file_path), anchored_old, "ALPHA")

        assert result.success is False
        assert result.error is not None
        assert "re-read" in result.error.lower() or "reread" in result.error.lower()

    def test_patch_v4a_accepts_valid_hashline_context_and_remove_lines(self, tmp_path):
        file_path = tmp_path / "sample.txt"
        file_path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        ops = ShellFileOperations(LocalShellEnv(str(tmp_path)))

        with patch.dict(os.environ, {"HERMES_HASHLINE_EDIT": "1"}, clear=False):
            anchored = ops.read_file(str(file_path), offset=1, limit=3).content.splitlines()
            patch_text = "\n".join([
                "*** Begin Patch",
                f"*** Update File: {file_path}",
                "@@ sample @@",
                f" {anchored[0]}",
                f"-{anchored[1]}",
                "+BETA",
                f" {anchored[2]}",
                "*** End Patch",
            ])
            result = ops.patch_v4a(patch_text)

        assert result.success is True
        assert result.error is None
        assert file_path.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"

    def test_patch_replace_rejects_truncated_hashline_old_string(self, tmp_path):
        file_path = tmp_path / "sample.txt"
        long_line = "a" * (MAX_LINE_LENGTH + 20)
        file_path.write_text(long_line + "\n", encoding="utf-8")
        ops = ShellFileOperations(LocalShellEnv(str(tmp_path)))

        with patch.dict(os.environ, {"HERMES_HASHLINE_EDIT": "1"}, clear=False):
            anchored_old = ops.read_file(str(file_path), offset=1, limit=1).content
            result = ops.patch_replace(str(file_path), anchored_old, "SHORT")

        assert result.success is False
        assert result.error is not None
        assert "truncated line" in result.error.lower()

    def test_hashline_anchor_detects_change_beyond_visible_truncation(self, tmp_path):
        file_path = tmp_path / "sample.txt"
        original = "a" * (MAX_LINE_LENGTH + 20)
        changed = original[:MAX_LINE_LENGTH + 10] + "b" + original[MAX_LINE_LENGTH + 11:]
        file_path.write_text(original + "\n", encoding="utf-8")
        ops = ShellFileOperations(LocalShellEnv(str(tmp_path)))

        with patch.dict(os.environ, {"HERMES_HASHLINE_EDIT": "1"}, clear=False):
            anchored_old = ops.read_file(str(file_path), offset=1, limit=1).content
            file_path.write_text(changed + "\n", encoding="utf-8")
            result = ops.patch_replace(str(file_path), anchored_old, "SHORT")

        assert result.success is False
        assert result.error is not None
        assert "truncated line" in result.error.lower() or "stale" in result.error.lower()


class TestPythonAstAwareReadChunking:
    def test_read_inside_function_snaps_to_function_boundaries(self, tmp_path):
        file_path = tmp_path / "sample.py"
        file_path.write_text(textwrap.dedent("""
            before = 1

            def demo():
                first = 1
                second = 2
                return first + second

            after = 2
        """).lstrip(), encoding="utf-8")
        ops = ShellFileOperations(LocalShellEnv(str(tmp_path)))

        result = ops.read_file(str(file_path), offset=5, limit=1)

        assert result.content.splitlines() == [
            "     3|def demo():",
            "     4|    first = 1",
            "     5|    second = 2",
            "     6|    return first + second",
        ]
        assert result.hint == "Use offset=7 to continue reading (showing 3-6 of 8 lines)"

    def test_requested_end_inside_function_expands_to_complete_boundary(self, tmp_path):
        file_path = tmp_path / "sample.py"
        file_path.write_text(textwrap.dedent("""
            preface = 0

            def demo():
                a = 1
                b = 2
                return a + b

            tail = 3
        """).lstrip(), encoding="utf-8")
        ops = ShellFileOperations(LocalShellEnv(str(tmp_path)))

        result = ops.read_file(str(file_path), offset=1, limit=5)

        assert result.content.splitlines() == [
            "     1|preface = 0",
            "     2|",
            "     3|def demo():",
            "     4|    a = 1",
            "     5|    b = 2",
            "     6|    return a + b",
        ]
        assert result.hint == "Use offset=7 to continue reading (showing 1-6 of 8 lines)"

    def test_parse_failure_falls_back_to_exact_raw_pagination(self, tmp_path):
        file_path = tmp_path / "broken.py"
        file_path.write_text("x = 1\ndef broken(:\n    pass\ny = 2\n", encoding="utf-8")
        ops = ShellFileOperations(LocalShellEnv(str(tmp_path)))

        result = ops.read_file(str(file_path), offset=3, limit=1)

        assert result.content == "     3|    pass"
        assert result.hint == "Use offset=4 to continue reading (showing 3-3 of 4 lines)"

    def test_too_large_ast_expansion_falls_back_to_exact_range(self, tmp_path):
        file_path = tmp_path / "large.py"
        body = "\n".join(f"    line_{i} = {i}" for i in range(1, 360))
        file_path.write_text(f"def giant():\n{body}\n", encoding="utf-8")
        ops = ShellFileOperations(LocalShellEnv(str(tmp_path)))

        result = ops.read_file(str(file_path), offset=200, limit=10)

        lines = result.content.splitlines()
        assert lines[0] == "   200|    line_199 = 199"
        assert lines[-1] == "   209|    line_208 = 208"
        assert len(lines) == 10
        assert result.hint == "Use offset=210 to continue reading (showing 200-209 of 360 lines)"


class TestTreeSitterAwareReadChunking:
    def test_non_python_supported_extension_surfaces_graceful_fallback_metadata(self, tmp_path):
        file_path = tmp_path / "sample.js"
        file_path.write_text(
            "const before = 1;\nfunction demo() {\n  return before + 1;\n}\n",
            encoding="utf-8",
        )
        ops = ShellFileOperations(LocalShellEnv(str(tmp_path)))

        result = ops.read_file(str(file_path), offset=3, limit=1)

        assert result.content == "     3|  return before + 1;"
        assert result.returned_start_line == 3
        assert result.returned_end_line == 3
        assert result.chunking_strategy in {"line", "tree_sitter"}
        if result.chunking_strategy == "line":
            assert result.chunking_fallback_reason == "tree_sitter_unavailable"

    def test_tree_sitter_expansion_preserves_actual_line_numbers_and_hint(self, tmp_path):
        file_path = tmp_path / "sample.ts"
        file_path.write_text(
            "const before = 1;\nfunction demo() {\n  return before + 1;\n}\nconst after = 2;\n",
            encoding="utf-8",
        )
        ops = ShellFileOperations(LocalShellEnv(str(tmp_path)))

        with patch(
            "tools.file_operations.maybe_expand_syntax_read_window",
            return_value={
                "content": "function demo() {\n  return before + 1;\n}",
                "start_line": 2,
                "end_line": 4,
                "total_lines": 5,
                "strategy": "tree_sitter",
                "language": "typescript",
                "fallback_reason": None,
            },
        ):
            result = ops.read_file(str(file_path), offset=3, limit=1)

        assert result.content.splitlines() == [
            "     2|function demo() {",
            "     3|  return before + 1;",
            "     4|}",
        ]
        assert result.returned_start_line == 2
        assert result.returned_end_line == 4
        assert result.hint == "Use offset=5 to continue reading (showing 2-4 of 5 lines)"
        assert result.chunking_strategy == "tree_sitter"
        assert result.chunking_language == "typescript"

    def test_hashline_ast_expanded_read_preserves_actual_line_numbers_and_anchors(self, tmp_path):
        file_path = tmp_path / "sample.py"
        file_path.write_text(textwrap.dedent("""
            prefix = 1

            def demo():
                value = 1
                return value

            suffix = 2
        """).lstrip(), encoding="utf-8")
        ops = ShellFileOperations(LocalShellEnv(str(tmp_path)))

        with patch.dict(os.environ, {"HERMES_HASHLINE_EDIT": "1"}, clear=False):
            read_result = ops.read_file(str(file_path), offset=5, limit=1)
            anchored_lines = read_result.content.splitlines()
            assert anchored_lines[0].startswith("     3#")
            assert anchored_lines[-1].startswith("     5#")

            result = ops.patch_replace(str(file_path), anchored_lines[0], "def renamed():")

        assert result.success is True
        assert file_path.read_text(encoding="utf-8").splitlines()[2] == "def renamed():"

    def test_ast_expansion_hint_uses_actual_returned_end_line(self, tmp_path):
        file_path = tmp_path / "sample.py"
        file_path.write_text(textwrap.dedent("""
            start = 1

            def demo():
                one = 1
                two = 2
                three = 3
                return one + two + three

            finish = 9
        """).lstrip(), encoding="utf-8")
        ops = ShellFileOperations(LocalShellEnv(str(tmp_path)))

        result = ops.read_file(str(file_path), offset=5, limit=1)

        assert result.hint == "Use offset=8 to continue reading (showing 3-7 of 9 lines)"

    def test_read_inside_class_method_prefers_method_not_containing_class(self, tmp_path):
        file_path = tmp_path / "nested.py"
        file_path.write_text(textwrap.dedent("""
            class Outer:
                padding0 = 0
                padding1 = 1
                padding2 = 2
                def method(self):
                    target = 1
                    return target
                def second(self):
                    return 2
                padding3 = 3
        """).lstrip(), encoding="utf-8")
        ops = ShellFileOperations(LocalShellEnv(str(tmp_path)))

        result = ops.read_file(str(file_path), offset=6, limit=1)

        assert result.content.splitlines() == [
            "     5|    def method(self):",
            "     6|        target = 1",
            "     7|        return target",
        ]
        assert result.returned_start_line == 5
        assert result.returned_end_line == 7

    def test_read_inside_nested_function_prefers_innermost_function(self, tmp_path):
        file_path = tmp_path / "nested_func.py"
        file_path.write_text(textwrap.dedent("""
            def outer():
                before = 1
                def inner():
                    target = 1
                    return target
                after = 2
                return inner()
        """).lstrip(), encoding="utf-8")
        ops = ShellFileOperations(LocalShellEnv(str(tmp_path)))

        result = ops.read_file(str(file_path), offset=4, limit=1)

        assert result.content.splitlines() == [
            "     3|    def inner():",
            "     4|        target = 1",
            "     5|        return target",
        ]
        assert result.returned_start_line == 3
        assert result.returned_end_line == 5

