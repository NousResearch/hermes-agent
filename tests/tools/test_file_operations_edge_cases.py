"""Tests for edge cases in tools/file_operations.py.

Covers:
- ``_is_likely_binary()`` content-analysis branch (dead-code removal regression guard)
- ``_check_lint()`` robustness against file paths containing curly braces
"""

import os
import sys

import pytest
from unittest.mock import MagicMock, patch

from tools.file_operations import ShellFileOperations
from tools.file_operations_search import _parse_search_context_line


# =========================================================================
# _is_likely_binary edge cases
# =========================================================================


class TestIsLikelyBinary:
    """Verify content-analysis logic after dead-code removal."""

    @pytest.fixture()
    def ops(self):
        return ShellFileOperations.__new__(ShellFileOperations)

    def test_binary_extension_returns_true(self, ops):
        """Known binary extensions should short-circuit without content analysis."""
        assert ops._is_likely_binary("image.png") is True
        assert ops._is_likely_binary("archive.tar.gz", content_sample="hello") is True

    def test_text_content_returns_false(self, ops):
        """Normal printable text should not be classified as binary."""
        sample = "Hello, world!\nThis is a normal text file.\n"
        assert ops._is_likely_binary("unknown.xyz", content_sample=sample) is False


    def test_just_above_threshold(self, ops):
        """301/1000 = 30.1% non-printable → should be binary."""
        sample = "\x00" * 301 + "a" * 699
        assert ops._is_likely_binary("data.xyz", content_sample=sample) is True

    def test_tabs_and_newlines_excluded(self, ops):
        """Tabs, carriage returns, and newlines should not count as non-printable."""
        sample = "\t" * 400 + "\n" * 300 + "\r" * 200 + "a" * 100
        assert ops._is_likely_binary("file.txt", content_sample=sample) is False

    def test_content_sample_longer_than_1000(self, ops):
        """Only the first 1000 characters should be analysed."""
        # First 1000 chars: 200 NUL + 800 printable = 20% → not binary
        # Remaining 1000 chars: all NUL → ignored by [:1000] slice
        sample = "\x00" * 200 + "a" * 800 + "\x00" * 1000
        assert ops._is_likely_binary("file.xyz", content_sample=sample) is False


# =========================================================================
# _check_lint edge cases
# =========================================================================


class TestCheckLintBracePaths:
    """Verify _check_lint handles file paths with curly braces safely.

    Uses ``.js`` to exercise the shell-linter path since ``.py`` now goes
    through the in-process ast.parse linter (see TestCheckLintInproc).
    """

    @pytest.fixture()
    def ops(self):
        obj = ShellFileOperations.__new__(ShellFileOperations)
        obj._command_cache = {}
        return obj


    def test_path_with_curly_braces(self, ops):
        """Path containing ``{`` and ``}`` must not raise KeyError/ValueError."""
        with patch.object(ops, "_has_command", return_value=True), \
             patch.object(ops, "_exec") as mock_exec:
            mock_exec.return_value = MagicMock(exit_code=0, stdout="")
            # This would raise KeyError with .format() but works with .replace()
            result = ops._check_lint("/tmp/{test}_file.js")

        assert result.success is True
        cmd_arg = mock_exec.call_args[0][0]
        assert "{test}" in cmd_arg


    def test_unsupported_extension_skipped(self, ops):
        """Extensions without a linter should return a skipped result."""
        result = ops._check_lint("/tmp/file.unknown_ext")
        assert result.skipped is True

    def test_missing_linter_skipped(self, ops):
        """When the linter binary is not installed, skip gracefully."""
        with patch.object(ops, "_has_command", return_value=False):
            result = ops._check_lint("/tmp/test.js")
        assert result.skipped is True

    def test_lint_failure_returns_output(self, ops):
        """When the linter exits non-zero, result should capture output."""
        with patch.object(ops, "_has_command", return_value=True), \
             patch.object(ops, "_exec") as mock_exec:
            mock_exec.return_value = MagicMock(
                exit_code=1,
                stdout="SyntaxError: invalid syntax",
            )
            result = ops._check_lint("/tmp/bad.js")

        assert result.success is False
        assert "SyntaxError" in result.output


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX sh stand-ins for node")
def test_local_js_lint_runs_pm_node_never_the_users(tmp_path, monkeypatch):
    """Hermes's own post-write lint uses PM's Node even when the user's node sorts first on
    PATH, and skips (never falls back to the user's) when PM has no Node."""
    import hermes_constants
    from tools.environments.local import LocalEnvironment

    def node_stand_in(directory, label):
        directory.mkdir()
        node = directory / "node"
        node.write_text(f'#!/bin/sh\necho {label} "$@"\nexit 1\n', encoding="utf-8")
        node.chmod(0o755)
        return str(directory)

    user_bin = node_stand_in(tmp_path / "user-bin", "user-node")
    store_dirs = [node_stand_in(tmp_path / "store-node", "pm-node")]
    monkeypatch.setenv("PATH", os.pathsep.join([user_bin, os.environ.get("PATH", "")]))
    monkeypatch.setattr(hermes_constants, "with_hermes_node_path", lambda env: {
        **env, "PATH": os.pathsep.join([*store_dirs, env.get("PATH", "")]).strip(os.pathsep)})
    target = tmp_path / "a.js"
    target.write_text("x\n", encoding="utf-8")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))

    result = ops._check_lint(str(target))
    assert result.output == f"pm-node --check {target}"

    store_dirs.clear()
    result = ops._check_lint(str(target))
    assert result.skipped and "Hermes-managed Node" in result.message


class TestCheckLintInproc:
    """Verify in-process linters (.py via ast.parse, .json, .yaml, .toml).

    These bypass the shell linter table entirely and parse content
    directly in Python — no subprocess, no toolchain dependency.
    """

    @pytest.fixture()
    def ops(self):
        obj = ShellFileOperations.__new__(ShellFileOperations)
        obj._command_cache = {}
        return obj

    def test_python_inproc_clean(self, ops):
        """Valid Python content passes in-process ast.parse."""
        result = ops._check_lint("/tmp/ok.py", content="x = 1\n")
        assert result.success is True
        assert not result.skipped
        assert result.output == ""


    def test_json_inproc_clean(self, ops):
        result = ops._check_lint("/tmp/a.json", content='{"a": 1}')
        assert result.success is True


    def test_toml_inproc_error(self, ops):
        result = ops._check_lint("/tmp/b.toml", content='[section\nk = "v"')
        assert result.success is False
        assert "TOMLDecodeError" in result.output


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_json_write_refuses_only_introduced_nonstandard_constants(tmp_path, token):
    """NaN/Infinity are refused when the write introduces them (strict JSON
    consumers reject them), but a file already holding one still takes edits."""
    from tools.environments.local import LocalEnvironment

    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))
    clean = tmp_path / "clean.json"
    clean.write_text('{"value": 1}\n', encoding="utf-8")
    result = ops.write_file(str(clean), f'{{"value": {token}}}\n')
    assert result.error and token in result.error
    assert clean.read_text(encoding="utf-8") == '{"value": 1}\n'

    quoted = f'{{"value": ["{token}", 1.5, null]}}\n'
    assert ops.write_file(str(clean), quoted).error is None
    assert clean.read_text(encoding="utf-8") == quoted

    legacy = tmp_path / "legacy.json"
    legacy.write_text(f'{{"value": {token}, "n": 1}}\n', encoding="utf-8")
    edited = f'{{"value": {token}, "n": 2}}\n'
    assert ops.write_file(str(legacy), edited).error is None
    assert legacy.read_text(encoding="utf-8") == edited


class TestCheckLintDelta:
    """Verify _check_lint_delta() filters pre-existing errors from post-edit output."""

    @pytest.fixture()
    def ops(self):
        obj = ShellFileOperations.__new__(ShellFileOperations)
        obj._command_cache = {}
        return obj



    def test_pre_existing_remains_flagged_but_not_new(self, ops):
        """Single-error parsers (ast) may miss that post is OK — be cautious."""
        # Pre has line-1 error, post keeps it (and doesn't add anything new)
        pre = 'def a(:\n    pass\n'
        post = 'def a(:\n    pass\n\nprint(42)\n'  # still line 1 broken
        r = ops._check_lint_delta("/tmp/d.py", pre_content=pre, post_content=post)
        # File is still broken — don't lie and claim success — but flag it as pre-existing
        assert r.success is False
        assert "pre-existing" in (r.message or "").lower()


# =========================================================================
# Pagination bounds
# =========================================================================




# =========================================================================
# Search context parsing
# =========================================================================


class TestSearchContextParsing:

    def test_parse_search_context_line_prefers_rightmost_numeric_separator(self):
        parsed = _parse_search_context_line("dir/file-12-name.py-8-context here")

        assert parsed == ("dir/file-12-name.py", 8, "context here")


    def test_search_with_grep_context_handles_filename_with_dash_digits(self):
        env = MagicMock()
        env.cwd = "/tmp"
        ops = ShellFileOperations(env)

        with patch.object(ops, "_exec") as mock_exec:
            mock_exec.return_value = MagicMock(
                exit_code=0,
                stdout="dir/file-12-name.py-8-context here\n",
            )
            result = ops._search_with_grep(
                "needle",
                path=".",
                file_glob=None,
                limit=10,
                offset=0,
                output_mode="content",
                context=1,
            )

        assert result.error is None
        assert result.total_count == 1
        assert result.matches[0].path == "dir/file-12-name.py"
        assert result.matches[0].line_number == 8
        assert result.matches[0].content == "context here"


# =========================================================================
# total_lines for files without a trailing newline (#3907)
# =========================================================================


class TestNoTrailingNewlineTotalLines:
    """``wc -l`` counts newlines, not lines: a final unterminated line must
    still count. Covers the live read paths plus the assembler contract."""

    @pytest.fixture()
    def ops(self):
        from tools.environments.local import LocalEnvironment

        return ShellFileOperations(LocalEnvironment())


    def test_pagination_admits_final_unterminated_line(self, tmp_path, ops):
        target = tmp_path / "no_trailing.txt"
        target.write_bytes(b"line1\nline2\nline3")

        first = ops.read_file(str(target), offset=1, limit=2)
        assert first.truncated is True
        assert "of 3 lines" in (first.hint or "")

        last = ops.read_file(str(target), offset=3)
        assert last.error is None
        assert last.content == "3|line3"
        assert last.total_lines == 3

    def test_terminated_and_empty_files_unchanged(self, tmp_path, ops):
        terminated = tmp_path / "terminated.txt"
        terminated.write_bytes(b"a\nb\nc\n")
        assert ops.read_file(str(terminated)).total_lines == 3

        empty = tmp_path / "empty.txt"
        empty.write_bytes(b"")
        result = ops.read_file(str(empty))
        assert result.total_lines == 0

    def test_native_path_counts_final_unterminated_line(self, tmp_path, ops):
        target = tmp_path / "no_trailing.txt"
        target.write_bytes(b"line1\nline2\nline3")

        result = ops._read_file_native(str(target), 1, 2000)

        assert result.error is None
        assert result.total_lines == 3

    def test_assembler_bumps_count_only_on_proven_missing_newline(self):
        ops = ShellFileOperations.__new__(ShellFileOperations)

        proved = ops._assemble_read_result(
            "a\nb\nc\n", offset=1, end_line=2000, total_lines=2,
            file_size=5, file_ends_with_newline=False)
        assert proved.total_lines == 3

        unknown = ops._assemble_read_result(
            "a\nb\nc\n", offset=1, end_line=2000, total_lines=2,
            file_size=5, file_ends_with_newline=None)
        assert unknown.total_lines == 2
